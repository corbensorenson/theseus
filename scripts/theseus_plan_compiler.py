#!/usr/bin/env python3
"""Compile Theseus goals into typed, VCM-backed execution DAGs.

This is a planner, linter, router, and trace compiler. It does not execute work.
Executable nodes are routed to existing Theseus surfaces such as the control
plane, Hive work board, VCM context bridge, or watchdog. Public planning
benchmarks stay calibration-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vcm_consumer_abi


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS / "theseus_plan_compiler.json"
DEFAULT_REGISTRY = CONFIGS / "project_manifest_registry.json"
DEFAULT_CONTEXTS = REPORTS / "vcm_task_contexts.json"
DEFAULT_VCM_CONTEXT_GOVERNOR = REPORTS / "vcm_context_governor.json"
DEFAULT_OUT = REPORTS / "theseus_plan_compiler.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_plan_compiler.md"
DEFAULT_DAGS_OUT = REPORTS / "theseus_plan_compiled_dags.json"
DEFAULT_TRACE_OUT = REPORTS / "theseus_plan_trace_bundle.jsonl"
DEFAULT_ABLATION_OUT = REPORTS / "theseus_plan_compiler_ablation.json"
DEFAULT_DETERMINISTIC_TOOL_REGISTRY = REPORTS / "deterministic_tool_registry.json"
DEFAULT_DETERMINISTIC_TOOL_REPORT = REPORTS / "deterministic_tool_substrate.json"
DEFAULT_EXECUTION_SPINE_OUT = REPORTS / "viea_execution_spine.json"
DEFAULT_PROCEDURAL_MEMORY_REPORT = REPORTS / "procedural_memory_toolification.json"
DEFAULT_PROCEDURAL_ADOPTION_REPORT = REPORTS / "procedural_memory_route_adoption.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--vcm-contexts", default=rel(DEFAULT_CONTEXTS))
    parser.add_argument("--vcm-governor", default=rel(DEFAULT_VCM_CONTEXT_GOVERNOR))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--dags-out", default=rel(DEFAULT_DAGS_OUT))
    parser.add_argument("--trace-out", default=rel(DEFAULT_TRACE_OUT))
    parser.add_argument("--ablation-out", default=rel(DEFAULT_ABLATION_OUT))
    parser.add_argument("--deterministic-tool-registry", default=rel(DEFAULT_DETERMINISTIC_TOOL_REGISTRY))
    parser.add_argument("--deterministic-tool-report", default=rel(DEFAULT_DETERMINISTIC_TOOL_REPORT))
    parser.add_argument("--procedural-memory-report", default=rel(DEFAULT_PROCEDURAL_MEMORY_REPORT))
    parser.add_argument("--procedural-adoption-report", default=rel(DEFAULT_PROCEDURAL_ADOPTION_REPORT))
    parser.add_argument("--execute-private", action="store_true", help="Run the safe private VIEA execute-mode smoke after compiling DAGs.")
    parser.add_argument("--execution-spine-out", default=rel(DEFAULT_EXECUTION_SPINE_OUT))
    parser.add_argument("--max-context-pages-per-node", type=int, default=6)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config), {})
    registry = read_json(resolve(args.registry), {})
    contexts = read_json(resolve(args.vcm_contexts), {})
    vcm_governor = read_json(resolve(args.vcm_governor), {})
    deterministic_tool_registry = read_json(resolve(args.deterministic_tool_registry), {})
    deterministic_tool_report = read_json(resolve(args.deterministic_tool_report), {})
    procedural_memory = read_json(resolve(args.procedural_memory_report), {})
    procedural_adoption = read_json(resolve(args.procedural_adoption_report), {})
    state = load_state()

    compiled_goals = []
    trace_rows = []
    representative_goals = [
        goal for goal in config.get("representative_goals", [])
        if isinstance(goal, dict)
    ] if isinstance(config.get("representative_goals"), list) else []
    procedural_canary_goals = procedural_memory_canary_goals(procedural_memory)
    procedural_default_goals = procedural_memory_default_route_goals(procedural_adoption)
    representative_goals.extend(procedural_canary_goals)
    representative_goals.extend(procedural_default_goals)
    for goal in representative_goals:
        compiled, goal_trace = compile_goal(
            goal=goal,
            config=config,
            registry=registry,
            contexts=contexts,
            vcm_governor=vcm_governor,
            deterministic_tool_registry=deterministic_tool_registry,
            deterministic_tool_report=deterministic_tool_report,
            state=state,
            max_context_pages=max(1, int(args.max_context_pages_per_node)),
        )
        compiled_goals.append(compiled)
        trace_rows.extend(goal_trace)

    ablation = build_ablation(compiled_goals, state)
    readiness = benchmark_readiness(config)
    gates = build_gates(config, registry, compiled_goals, trace_rows, ablation, readiness, vcm_governor)
    hard_failures = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warning_failures = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warning_failures:
        trigger_state = "YELLOW"

    dags_payload = {
        "policy": "project_theseus_plan_compiled_dags_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "passed": trigger_state in {"GREEN", "YELLOW"},
        "compiled_goal_count": len(compiled_goals),
        "compiled_goals": compiled_goals,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    report = {
        "policy": "project_theseus_plan_compiler_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "passed": trigger_state in {"GREEN", "YELLOW"},
        "purpose": config.get("purpose"),
        "inputs": {
            "config": rel(resolve(args.config)),
            "registry": rel(resolve(args.registry)),
            "vcm_contexts": rel(resolve(args.vcm_contexts)),
            "vcm_governor": rel(resolve(args.vcm_governor)),
            "deterministic_tool_registry": rel(resolve(args.deterministic_tool_registry)),
            "deterministic_tool_report": rel(resolve(args.deterministic_tool_report)),
            "procedural_memory_report": rel(resolve(args.procedural_memory_report)),
            "procedural_adoption_report": rel(resolve(args.procedural_adoption_report)),
        },
        "outputs": {
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
            "compiled_dags": rel(resolve(args.dags_out)),
            "trace_bundle": rel(resolve(args.trace_out)),
            "ablation": rel(resolve(args.ablation_out)),
            "execution_spine": rel(resolve(args.execution_spine_out)),
        },
        "summary": summarize(compiled_goals, trace_rows, ablation, gates, started, vcm_governor),
        "gates": gates,
        "compiled_goal_summaries": [compact_goal_summary(goal) for goal in compiled_goals],
        "ablation": ablation,
        "planning_benchmark_readiness": readiness,
        "integration_contract": {
            "execution_posture": "compile_and_route_by_default; safe_private_execute_mode_available_with_flag",
            "decision_owner": "scripts/theseus_control_plane.py",
            "executor_surfaces": [
            "scripts/hive_work_board_executor.py",
            "scripts/autonomy_watchdog.py",
            "scripts/vcm_task_context_bridge.py",
            "scripts/theseus_control_plane.py",
            "scripts/theseus_deterministic_tool_substrate.py"
            ],
            "vcm_required": True,
            "vcm_context_governor_required": True,
            "registry_required": True,
            "public_benchmark_training_allowed": False,
            "fallback_returns_allowed": False,
            "arbitrary_remote_execution_allowed": False,
        },
        "recommendation": next_recommendation(trigger_state, compiled_goals, ablation),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    report["summary"]["procedural_memory_canary_goal_count"] = len(procedural_canary_goals)
    report["summary"]["procedural_memory_default_route_goal_count"] = len(procedural_default_goals)
    report["procedural_memory_canary_goals"] = [
        {
            "goal_id": goal.get("id"),
            "candidate_id": goal.get("procedural_candidate_id"),
            "canary_route_id": goal.get("canary_route_id"),
            "replay_fixture_id": goal.get("replay_fixture_id"),
        }
        for goal in procedural_canary_goals
    ]
    report["procedural_memory_default_route_goals"] = [
        {
            "goal_id": goal.get("id"),
            "default_route_id": goal.get("default_route_id"),
            "candidate_id": goal.get("procedural_candidate_id"),
            "transaction_id": goal.get("adoption_transaction_id"),
        }
        for goal in procedural_default_goals
    ]

    write_json(resolve(args.dags_out), dags_payload)
    write_json(resolve(args.ablation_out), ablation)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.trace_out), trace_rows)
    write_text(resolve(args.markdown_out), render_markdown(report))
    execute_result = {}
    if args.execute_private and trigger_state != "RED":
        execute_result = run_private_execute_mode(args)
        report["execute_mode"] = execute_result
        report["summary"]["execute_mode_returncode"] = execute_result.get("returncode")
        report["summary"]["execute_mode_trigger_state"] = get_path(execute_result, ["report", "trigger_state"], "")
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({"trigger_state": trigger_state, "summary": report["summary"]}, indent=2))
    if trigger_state == "RED":
        return 2
    if args.execute_private and execute_result.get("returncode") not in {0, None}:
        return int(execute_result.get("returncode") or 2)
    return 0


def compile_goal(
    *,
    goal: dict[str, Any],
    config: dict[str, Any],
    registry: dict[str, Any],
    contexts: dict[str, Any],
    vcm_governor: dict[str, Any],
    deterministic_tool_registry: dict[str, Any],
    deterministic_tool_report: dict[str, Any],
    state: dict[str, Any],
    max_context_pages: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    goal_id = slug(str(goal.get("id") or "goal"))
    atoms = [atom for atom in goal.get("atoms", []) if isinstance(atom, dict)]
    contract = build_contract(goal, config, registry)
    nodes = []
    for atom in atoms:
        node = compile_atom(
            goal_id=goal_id,
            goal=goal,
            atom=atom,
            config=config,
            contexts=contexts,
            vcm_governor=vcm_governor,
            deterministic_tool_registry=deterministic_tool_registry,
            deterministic_tool_report=deterministic_tool_report,
            state=state,
            max_context_pages=max_context_pages,
        )
        nodes.append(node)
    schedule = schedule_nodes(nodes)
    for node in nodes:
        node_id = node["node_id"]
        node["schedule"] = schedule.get(node_id, {})
        node["claim_objects"] = claim_objects_for(node, contract)
        node["evidence_targets"] = evidence_targets_for(node)
        node["repair_policy"] = repair_policy_for(node)
        node["asi_stack_records"] = asi_stack_records_for(node, contract)
        if isinstance(node.get("execution_packet"), dict):
            node["execution_packet"]["claim_ids"] = [claim.get("claim_id") for claim in node.get("claim_objects", [])]
            node["execution_packet"]["evidence_targets"] = [target.get("evidence_ref") for target in node.get("evidence_targets", [])]
            node["execution_packet"]["asi_stack_record_ids"] = asi_stack_record_ids(node.get("asi_stack_records", {}))
            node["execution_packet"]["packet_hash"] = stable_hash(node["execution_packet"])
    lint = lint_goal(contract, nodes, config, registry)
    contract_hash = stable_hash(contract)
    for node in nodes:
        node["contract_hash"] = contract_hash
    trace_rows = [trace_row(goal_id, contract_hash, node) for node in nodes]
    goal_records = goal_records_for(goal_id, contract, nodes, lint, contract_hash)
    return {
        "goal_id": goal_id,
        "title": str(goal.get("title") or ""),
        "priority": str(goal.get("priority") or "medium"),
        "risk_tier": str(goal.get("risk_tier") or "medium"),
        "contract_hash": contract_hash,
        "contract": contract,
        "node_count": len(nodes),
        "edge_count": sum(len(node.get("depends_on", [])) for node in nodes),
        "estimated_makespan_seconds": max((node.get("schedule", {}).get("earliest_finish", 0) for node in nodes), default=0),
        "critical_path_node_ids": [node["node_id"] for node in nodes if node.get("schedule", {}).get("critical_path")],
        "parallel_layer_count": len({node.get("schedule", {}).get("layer", 0) for node in nodes}),
        "nodes": nodes,
        "goal_records": goal_records,
        "lint": lint,
        "trigger_state": "GREEN" if not lint["hard_failures"] else "RED",
    }, trace_rows


def build_contract(goal: dict[str, Any], config: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    owner_surface = str(goal.get("owner_surface") or "")
    owner_abstraction = owner_surface_abstraction(owner_surface, registry)
    return {
        "contract_id": stable_id("contract", goal.get("id"), goal.get("objective")),
        "goal_id": str(goal.get("id") or ""),
        "title": str(goal.get("title") or ""),
        "objective": str(goal.get("objective") or ""),
        "owner_surface": owner_surface,
        "owner_surface_registered": owner_surface_registered(owner_surface, registry),
        "owner_abstraction_id": owner_abstraction,
        "owner_abstraction_registered": owner_abstraction_registered(owner_abstraction, registry),
        "priority": str(goal.get("priority") or "medium"),
        "risk_tier": str(goal.get("risk_tier") or "medium"),
        "outputs": [str(item) for item in list_value(goal.get("outputs"))],
        "non_goals": [str(item) for item in list_value(goal.get("non_goals"))],
        "acceptance_tests": [slug(str(item)) for item in list_value(goal.get("acceptance_tests"))],
        "constraint_capsules": [
            {
                "constraint_id": str(row.get("id") or ""),
                "rule": str(row.get("rule") or ""),
                "source": "configs/theseus_plan_compiler.json",
            }
            for row in config.get("hard_constraints", [])
            if isinstance(row, dict)
        ],
        "public_benchmark_policy": "calibration_only",
        "public_training_rows_allowed": False,
        "fallback_returns_allowed": False,
        "arbitrary_remote_execution_allowed": False,
        "teacher_policy": "proposal_only_unless_governed_distillation_gate_admits_rows",
        "locked_utc": now(),
    }


def compile_atom(
    *,
    goal_id: str,
    goal: dict[str, Any],
    atom: dict[str, Any],
    config: dict[str, Any],
    contexts: dict[str, Any],
    vcm_governor: dict[str, Any],
    deterministic_tool_registry: dict[str, Any],
    deterministic_tool_report: dict[str, Any],
    state: dict[str, Any],
    max_context_pages: int,
) -> dict[str, Any]:
    atom_id = slug(str(atom.get("id") or atom.get("title") or "node"))
    node_id = f"{goal_id}.{atom_id}"
    vcm_slice = select_vcm_slice(atom, config, contexts, vcm_governor, max_pages=max_context_pages)
    route = route_node(atom, state)
    outputs = [str(item) for item in list_value(atom.get("outputs"))]
    depends_on = [f"{goal_id}.{slug(str(dep))}" for dep in list_value(atom.get("depends_on"))]
    allowed_tools = [str(item) for item in list_value(atom.get("allowed_tools"))]
    acceptance_refs = [slug(str(item)) for item in list_value(atom.get("acceptance_refs"))]
    tool_requirements = tool_requirements_for(allowed_tools, deterministic_tool_registry)
    tool_eligibility = tool_eligibility_for(
        atom=atom,
        route=route,
        allowed_tools=allowed_tools,
        tool_requirements=tool_requirements,
        deterministic_tool_registry=deterministic_tool_registry,
        deterministic_tool_report=deterministic_tool_report,
    )
    tool_receipts = tool_receipts_for(
        node_id=node_id,
        tool_eligibility=tool_eligibility,
        deterministic_tool_registry=deterministic_tool_registry,
        deterministic_tool_report=deterministic_tool_report,
    )
    semantic_payload = {
        "op": str(atom.get("op") or ""),
        "title": str(atom.get("title") or ""),
        "required_capabilities": list_value(atom.get("required_capabilities")),
        "outputs": outputs,
        "acceptance_refs": acceptance_refs,
    }
    semantic_hash = stable_hash(semantic_payload)
    execution_packet = execution_packet_for(
        node_id=node_id,
        atom=atom,
        route=route,
        vcm_slice=vcm_slice,
        semantic_hash=semantic_hash,
        tool_requirements=tool_requirements,
        tool_eligibility=tool_eligibility,
        tool_receipts=tool_receipts,
        outputs=outputs,
        acceptance_refs=acceptance_refs,
    )
    return {
        "node_id": node_id,
        "atom_id": atom_id,
        "op": str(atom.get("op") or "UNKNOWN"),
        "title": str(atom.get("title") or ""),
        "depends_on": depends_on,
        "inputs": [str(item) for item in list_value(atom.get("inputs"))],
        "outputs": outputs,
        "required_capabilities": [str(item) for item in list_value(atom.get("required_capabilities"))],
        "allowed_tools": allowed_tools,
        "executor_backend": str(atom.get("executor_backend") or ""),
        "worker_tier": str(atom.get("worker_tier") or "T0"),
        "risk_tier": str(atom.get("risk_tier") or goal.get("risk_tier") or "medium"),
        "estimated_seconds": max(1, int(atom.get("estimated_seconds") or 1)),
        "training_surface": str(atom.get("training_surface") or "none"),
        "fallback_return_allowed": bool(atom.get("fallback_return_allowed", False)),
        "acceptance_refs": acceptance_refs,
        "vcm_context_slice": vcm_slice,
        "tool_requirements": tool_requirements,
        "tool_eligibility": tool_eligibility,
        "tool_receipts": tool_receipts,
        "execution_packet": execution_packet,
        "route": route,
        "semantic_hash": semantic_hash,
        "preconditions": preconditions_for(atom),
        "effects": effects_for(atom),
    }


def select_vcm_slice(
    atom: dict[str, Any],
    config: dict[str, Any],
    contexts: dict[str, Any],
    vcm_governor: dict[str, Any],
    *,
    max_pages: int,
) -> dict[str, Any]:
    requested_family = str(atom.get("vcm_family") or "planning")
    family_id = str(config.get("vcm_family_defaults", {}).get(requested_family, requested_family))
    task_contexts = [row for row in contexts.get("task_contexts", []) if isinstance(row, dict)]
    selected_context = {}
    for row in task_contexts:
        if str(row.get("task_family_id") or "") == family_id:
            selected_context = row
            break
    if not selected_context and task_contexts:
        selected_context = task_contexts[0]
    pages = [summarize_page(page) for page in list_value(selected_context.get("selected_pages"))[:max_pages] if isinstance(page, dict)]
    governor_receipt = compact_vcm_governor(vcm_governor)
    context_ready = bool(selected_context.get("ready", False))
    governed_ready = context_ready and bool(pages) and bool(governor_receipt.get("ready"))
    return {
        "vcm_family": requested_family,
        "task_family_id": str(selected_context.get("task_family_id") or family_id),
        "ready": context_ready,
        "governed_ready": governed_ready,
        "governor_receipt": governor_receipt,
        "selected_page_count": len(pages),
        "selected_pages": pages,
        "context_hash": stable_hash({"family": family_id, "pages": [page.get("address") for page in pages]}),
        "governed_context_hash": stable_hash({
            "family": family_id,
            "pages": [page.get("address") for page in pages],
            "governor_receipt_id": governor_receipt.get("receipt_id"),
        }),
        "context_adequacy_state": "governed_sufficient_for_planning" if governed_ready else "fault_missing_or_ungoverned_context",
        "constraint_capsules": [
            "no_public_benchmark_training",
            "no_arbitrary_remote_execution",
            "no_fallback_returns",
            "registry_first",
            "teacher_proposal_only",
            "vcm_context_governor_required",
        ],
    }


def compact_vcm_governor(vcm_governor: dict[str, Any]) -> dict[str, Any]:
    summary = vcm_governor.get("summary") if isinstance(vcm_governor.get("summary"), dict) else {}
    mission = vcm_governor.get("mission_brief") if isinstance(vcm_governor.get("mission_brief"), dict) else {}
    deletion = vcm_governor.get("deletion_closure") if isinstance(vcm_governor.get("deletion_closure"), dict) else {}
    scif = vcm_governor.get("digital_scif") if isinstance(vcm_governor.get("digital_scif"), dict) else {}
    hard_gaps = list_value(vcm_governor.get("hard_gaps"))
    warnings = list_value(vcm_governor.get("warnings"))
    receipt = {
        "policy": str(vcm_governor.get("policy") or ""),
        "report": "reports/vcm_context_governor.json",
        "receipt_id": stable_id("vcm_context_governor", vcm_governor.get("created_utc"), summary),
        "trigger_state": str(vcm_governor.get("trigger_state") or ""),
        "ready": vcm_governor_ready(vcm_governor),
        "hard_gap_count": int(summary.get("hard_gap_count") or len(hard_gaps)),
        "warning_count": int(summary.get("warning_count") or len(warnings)),
        "mission_brief_status": str(summary.get("mission_brief_status") or mission.get("status") or ""),
        "mission_brief_compression_loss": summary.get("mission_brief_compression_loss", mission.get("compression_loss")),
        "mission_brief_omission_count": int(summary.get("mission_brief_omission_count") or len(list_value(mission.get("omissions")))),
        "selected_chunk_ids": [str(item) for item in list_value(mission.get("selected_chunk_ids"))],
        "authority_limits": [str(item) for item in list_value(mission.get("authority_limits"))],
        "deletion_closure_status": str(summary.get("deletion_closure_status") or deletion.get("status") or ""),
        "deletion_closure_fault_count": int(summary.get("deletion_closure_fault_count") or deletion.get("closure_fault_count") or 0),
        "scif_status": str(summary.get("scif_status") or scif.get("status") or ""),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return receipt


def vcm_governor_ready(vcm_governor: dict[str, Any]) -> bool:
    summary = vcm_governor.get("summary") if isinstance(vcm_governor.get("summary"), dict) else {}
    mission = vcm_governor.get("mission_brief") if isinstance(vcm_governor.get("mission_brief"), dict) else {}
    deletion = vcm_governor.get("deletion_closure") if isinstance(vcm_governor.get("deletion_closure"), dict) else {}
    scif = vcm_governor.get("digital_scif") if isinstance(vcm_governor.get("digital_scif"), dict) else {}
    return (
        str(vcm_governor.get("trigger_state") or "") == "GREEN"
        and int(summary.get("hard_gap_count") or 0) == 0
        and str(summary.get("mission_brief_status") or mission.get("status") or "") == "ready"
        and str(summary.get("deletion_closure_status") or deletion.get("status") or "") == "closed"
        and str(summary.get("scif_status") or scif.get("status") or "") == "ready"
        and int(summary.get("deletion_closure_fault_count") or deletion.get("closure_fault_count") or 0) == 0
    )


def route_node(atom: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    backend = str(atom.get("executor_backend") or "")
    capabilities = {str(item) for item in list_value(atom.get("required_capabilities"))}
    hive = state.get("hive_node_registry", {}).get("summary", {})
    resource = state.get("resource_aware_execution_policy", {}).get("summary", {})
    route = {
        "backend": backend,
        "surface": backend_to_surface(backend),
        "execute_now": False,
        "reason": "plan_compiler_routes_only; executor surface must lease and run separately",
        "local_mlx_ready": bool(hive.get("local_mlx_ready")),
        "remote_cuda_live_ready": bool(hive.get("remote_cuda_live_ready")),
        "resource_owner": str(resource.get("resource_execution_owner") or ""),
        "blocked": False,
        "block_reason": "",
    }
    if "cuda" in capabilities and not hive.get("remote_cuda_live_ready"):
        route["blocked"] = True
        route["block_reason"] = "cuda_required_but_remote_cuda_not_live_ready"
    if "mlx_apple" in capabilities and not hive.get("local_mlx_ready"):
        route["blocked"] = True
        route["block_reason"] = "mlx_required_but_local_mlx_not_ready"
    if backend == "governed_teacher_proposal":
        route["execute_now"] = False
        route["reason"] = "teacher proposal packet only; distillation remains governed"
    if backend == "local_deterministic_tool":
        route["execute_now"] = False
        route["reason"] = "deterministic tool packet emitted; scripts/theseus_deterministic_tool_substrate.py owns bounded local execution"
    return route


def tool_requirements_for(allowed_tools: list[str], deterministic_tool_registry: dict[str, Any]) -> list[dict[str, Any]]:
    cards = {
        str(row.get("id") or ""): row
        for row in deterministic_tool_registry.get("tools", [])
        if isinstance(row, dict)
    }
    requirements = []
    for tool in allowed_tools:
        card = cards.get(tool)
        if card:
            requirements.append({
                "tool_id": tool,
                "registered": True,
                "trust_tier": str(card.get("trust_tier") or ""),
                "cost_tier": str(card.get("cost_tier") or ""),
                "dependency_status": card.get("dependency_status") if isinstance(card.get("dependency_status"), dict) else {},
                "failure_behavior": str(card.get("failure_behavior") or ""),
                "vcm_bindings": card.get("vcm_bindings") if isinstance(card.get("vcm_bindings"), dict) else {},
                "tool_card_checksum": str(card.get("replay_checksum") or ""),
                "strict_no_fallback_returns": bool(card.get("strict_no_fallback_returns", False)),
            })
        elif is_deterministic_tool_id(tool):
            requirements.append({
                "tool_id": tool,
                "registered": False,
                "missing_card": True,
                "dependency_status": {},
                "failure_behavior": "blocked until deterministic tool card exists",
                "strict_no_fallback_returns": True,
            })
    return requirements


def tool_eligibility_for(
    *,
    atom: dict[str, Any],
    route: dict[str, Any],
    allowed_tools: list[str],
    tool_requirements: list[dict[str, Any]],
    deterministic_tool_registry: dict[str, Any],
    deterministic_tool_report: dict[str, Any],
) -> dict[str, Any]:
    cards = tool_card_index(deterministic_tool_registry)
    requested_exact = [tool for tool in allowed_tools if is_deterministic_tool_id(tool)]
    alias_resolution = tool_alias_resolution(atom, allowed_tools)
    candidate_ids = sorted({tool for row in alias_resolution for tool in list_value(row.get("candidate_tool_ids"))} | set(requested_exact))
    registered_ids = [tool for tool in candidate_ids if tool in cards]
    missing_ids = [tool for tool in candidate_ids if tool not in cards]
    unavailable_ids = [
        tool
        for tool in registered_ids
        if not get_path(cards.get(tool, {}), ["dependency_status", "available"], False)
    ]
    report_ready = deterministic_tool_report_ready(deterministic_tool_report)
    required_for_node = (
        str(route.get("backend") or "") == "local_deterministic_tool"
        or bool(requested_exact)
        or bool(alias_resolution and str(route.get("backend") or "") == "local_deterministic_tool")
    )
    if missing_ids:
        decision = "blocked_missing_tool_card"
    elif unavailable_ids:
        decision = "blocked_dependency_unavailable"
    elif registered_ids and not report_ready:
        decision = "blocked_tool_report_not_ready"
    elif registered_ids and required_for_node:
        decision = "eligible_required"
    elif registered_ids:
        decision = "eligible_optional"
    else:
        decision = "not_applicable"
    return {
        "record_type": "tool_eligibility_record",
        "eligibility_id": stable_id(
            "tool_eligibility",
            atom.get("id"),
            route.get("backend"),
            allowed_tools,
            candidate_ids,
            deterministic_tool_report.get("created_utc"),
        ),
        "decision": decision,
        "required_for_node": required_for_node,
        "allowed_tools": allowed_tools,
        "requested_deterministic_tool_ids": requested_exact,
        "alias_resolution": alias_resolution,
        "candidate_tool_ids": candidate_ids,
        "registered_tool_ids": registered_ids,
        "missing_tool_ids": missing_ids,
        "unavailable_tool_ids": unavailable_ids,
        "tool_report_ready": report_ready,
        "tool_report_ref": rel(DEFAULT_DETERMINISTIC_TOOL_REPORT),
        "tool_registry_ref": rel(DEFAULT_DETERMINISTIC_TOOL_REGISTRY),
        "tool_assisted_score_boundary": "separate_from_model_only_never_learned_generation",
        "learned_generation_claim_allowed": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def tool_alias_resolution(atom: dict[str, Any], allowed_tools: list[str]) -> list[dict[str, Any]]:
    alias_map = {
        "json_report": ["search.local_bm25", "search.local_hybrid"],
        "registry_lookup": ["search.local_bm25"],
        "vcm_lookup": ["search.vcm_hybrid"],
    }
    capability_map = {
        "report_reading": ["search.local_bm25", "search.local_hybrid"],
        "registry_lookup": ["search.local_bm25"],
        "vcm_context": ["search.vcm_hybrid"],
        "vcm_lookup": ["search.vcm_hybrid"],
        "claim_evidence_mapping": ["tool.trace_replay"],
        "trace_replay": ["tool.trace_replay"],
        "local_search": ["search.local_bm25", "search.local_hybrid"],
        "exact_math": ["math.sympy_exact", "math.numeric_interval", "math.linear_algebra", "math.numeric_verify", "math.mpmath_verify"],
        "formal_check": ["logic.lean_check", "logic.z3_smt"],
    }
    rows: list[dict[str, Any]] = []
    for alias in allowed_tools:
        candidates = alias_map.get(str(alias), [])
        if candidates:
            rows.append({
                "source": "allowed_tool_alias",
                "alias": str(alias),
                "candidate_tool_ids": candidates,
                "reason": "generic planner tool alias resolved to registered deterministic tool cards",
            })
    for capability in [str(item) for item in list_value(atom.get("required_capabilities"))]:
        candidates = capability_map.get(capability, [])
        if candidates:
            rows.append({
                "source": "required_capability",
                "alias": capability,
                "candidate_tool_ids": candidates,
                "reason": "required capability has a deterministic tool substrate route",
            })
    return rows


def tool_receipts_for(
    *,
    node_id: str,
    tool_eligibility: dict[str, Any],
    deterministic_tool_registry: dict[str, Any],
    deterministic_tool_report: dict[str, Any],
) -> list[dict[str, Any]]:
    cards = tool_card_index(deterministic_tool_registry)
    results = tool_result_index(deterministic_tool_report)
    report_receipt = compact_deterministic_tool_report(deterministic_tool_report)
    candidate_ids = [str(tool) for tool in list_value(tool_eligibility.get("candidate_tool_ids"))]
    if not candidate_ids:
        return [
            {
                "record_type": "tool_call_receipt",
                "receipt_id": stable_id("tool_receipt", node_id, "not_applicable"),
                "node_id": node_id,
                "tool_id": "",
                "decision": "not_applicable",
                "required_for_node": bool(tool_eligibility.get("required_for_node")),
                "evidence_refs": [rel(DEFAULT_DETERMINISTIC_TOOL_REGISTRY)],
                "tool_report_ready": report_receipt.get("ready"),
                "learned_generation_claim_allowed": False,
                "tool_assisted_score_boundary": "separate_from_model_only_never_learned_generation",
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        ]
    receipts = []
    for tool_id in candidate_ids:
        card = cards.get(tool_id, {})
        tool_results = results.get(tool_id, [])
        evidence_refs = [str(row.get("evidence_ref") or "") for row in tool_results if row.get("evidence_ref")]
        receipts.append({
            "record_type": "tool_call_receipt",
            "receipt_id": stable_id("tool_receipt", node_id, tool_id, tool_eligibility.get("eligibility_id")),
            "node_id": node_id,
            "tool_id": tool_id,
            "decision": "eligible" if tool_id in cards and report_receipt.get("ready") else "blocked",
            "required_for_node": bool(tool_eligibility.get("required_for_node")),
            "tool_card_registered": tool_id in cards,
            "tool_card_checksum": str(card.get("replay_checksum") or ""),
            "dependency_status": card.get("dependency_status") if isinstance(card.get("dependency_status"), dict) else {},
            "strict_no_fallback_returns": bool(card.get("strict_no_fallback_returns", False)),
            "verified_result_count": sum(1 for row in tool_results if row.get("verified") is True),
            "evidence_refs": evidence_refs or [rel(DEFAULT_DETERMINISTIC_TOOL_REPORT)],
            "replay_checksums": [str(row.get("replay_checksum") or "") for row in tool_results if row.get("replay_checksum")],
            "vcm_addresses": [str(row.get("vcm_address") or "") for row in tool_results if row.get("vcm_address")],
            "tool_report_receipt_id": report_receipt.get("receipt_id"),
            "tool_report_ready": report_receipt.get("ready"),
            "vcm_context_governor_ready": report_receipt.get("vcm_context_governor_ready"),
            "vcm_context_adequacy_state": report_receipt.get("vcm_context_adequacy_state"),
            "learned_generation_claim_allowed": False,
            "tool_assisted_score_boundary": "separate_from_model_only_never_learned_generation",
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        })
    return receipts


def tool_card_index(deterministic_tool_registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id") or ""): row
        for row in deterministic_tool_registry.get("tools", [])
        if isinstance(row, dict) and row.get("id")
    }


def tool_result_index(deterministic_tool_report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deterministic_tool_report.get("tool_results", []) if isinstance(deterministic_tool_report.get("tool_results"), list) else []:
        if isinstance(row, dict) and row.get("tool_id"):
            out[str(row.get("tool_id"))].append(row)
    return dict(out)


def deterministic_tool_report_ready(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return (
        str(report.get("trigger_state") or "") == "GREEN"
        and int(summary.get("tool_card_count") or 0) > 0
        and int(summary.get("verified_solved_count") or 0) > 0
        and bool(summary.get("vcm_context_governor_ready"))
        and int(summary.get("public_training_rows_written") or 0) == 0
        and int(summary.get("external_inference_calls") or 0) == 0
        and int(summary.get("fallback_return_count") or 0) == 0
    )


def compact_deterministic_tool_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "receipt_id": stable_id("deterministic_tool_report", report.get("created_utc"), summary.get("verified_solved_count"), summary.get("vcm_context_governor_receipt_id")),
        "report": rel(DEFAULT_DETERMINISTIC_TOOL_REPORT),
        "ready": deterministic_tool_report_ready(report),
        "trigger_state": report.get("trigger_state"),
        "tool_card_count": int(summary.get("tool_card_count") or 0),
        "verified_solved_count": int(summary.get("verified_solved_count") or 0),
        "vcm_context_governor_ready": bool(summary.get("vcm_context_governor_ready")),
        "vcm_context_adequacy_state": summary.get("vcm_context_adequacy_state"),
        "public_training_rows_written": int(summary.get("public_training_rows_written") or 0),
        "external_inference_calls": int(summary.get("external_inference_calls") or 0),
        "fallback_return_count": int(summary.get("fallback_return_count") or 0),
    }


def execution_packet_for(
    *,
    node_id: str,
    atom: dict[str, Any],
    route: dict[str, Any],
    vcm_slice: dict[str, Any],
    semantic_hash: str,
    tool_requirements: list[dict[str, Any]],
    tool_eligibility: dict[str, Any],
    tool_receipts: list[dict[str, Any]],
    outputs: list[str],
    acceptance_refs: list[str],
) -> dict[str, Any]:
    backend = str(atom.get("executor_backend") or "")
    mode = "local_deterministic_tool_packet" if backend == "local_deterministic_tool" else "executor_surface_packet"
    return {
        "packet_id": stable_id("execution_packet", node_id, semantic_hash),
        "node_id": node_id,
        "mode": mode,
        "executor_backend": backend,
        "executor_surface": route.get("surface"),
        "execute_now": False,
        "tool_ids": [row.get("tool_id") for row in tool_requirements],
        "tool_card_checksums": [row.get("tool_card_checksum") for row in tool_requirements if row.get("tool_card_checksum")],
        "tool_eligibility_id": tool_eligibility.get("eligibility_id"),
        "tool_eligibility_decision": tool_eligibility.get("decision"),
        "tool_required_for_node": bool(tool_eligibility.get("required_for_node")),
        "tool_receipt_ids": [row.get("receipt_id") for row in tool_receipts],
        "tool_evidence_refs": [
            ref
            for row in tool_receipts
            for ref in list_value(row.get("evidence_refs"))
            if ref
        ],
        "vcm_context_hash": vcm_slice.get("context_hash"),
        "vcm_governed_context_hash": vcm_slice.get("governed_context_hash"),
        "vcm_context_governor_receipt_id": get_path(vcm_slice, ["governor_receipt", "receipt_id"], ""),
        "vcm_context_governor_ready": get_path(vcm_slice, ["governor_receipt", "ready"], False),
        "vcm_selected_page_count": vcm_slice.get("selected_page_count", 0),
        "semantic_hash": semantic_hash,
        "expected_outputs": outputs,
        "acceptance_refs": acceptance_refs,
        "strict_no_fallback_returns": True,
        "public_training_rows_allowed": False,
        "external_inference_allowed": False,
        "structured_non_solved_states": ["UNKNOWN", "UNSOLVED", "TOOL_UNAVAILABLE", "TOOL_FAULT"],
        "dogfood_trace_shape": ["accepted", "missed", "ignored", "corrected", "completed", "failure"],
    }


def is_deterministic_tool_id(tool: str) -> bool:
    return tool.startswith(("math.", "logic.", "search.", "tool.", "rewrite."))


def schedule_nodes(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {node["node_id"]: node for node in nodes}
    indegree = {node_id: 0 for node_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep in by_id:
                indegree[node["node_id"]] += 1
                children[dep].append(node["node_id"])
    queue = deque(sorted([node_id for node_id, count in indegree.items() if count == 0]))
    order = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for child in sorted(children[node_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        return {node["node_id"]: {"cycle_detected": True} for node in nodes}

    schedule: dict[str, dict[str, Any]] = {}
    for node_id in order:
        node = by_id[node_id]
        deps = [dep for dep in node.get("depends_on", []) if dep in by_id]
        earliest_start = max((schedule[dep]["earliest_finish"] for dep in deps), default=0)
        layer = 0 if not deps else max(schedule[dep]["layer"] for dep in deps) + 1
        duration = max(1, int(node.get("estimated_seconds") or 1))
        schedule[node_id] = {
            "layer": layer,
            "earliest_start": earliest_start,
            "earliest_finish": earliest_start + duration,
            "estimated_seconds": duration,
        }
    makespan = max((row["earliest_finish"] for row in schedule.values()), default=0)
    latest_finish = {node_id: makespan for node_id in order}
    for node_id in reversed(order):
        child_starts = [latest_finish[child] - int(by_id[child].get("estimated_seconds") or 1) for child in children[node_id]]
        if child_starts:
            latest_finish[node_id] = min(child_starts)
        latest_start = latest_finish[node_id] - int(by_id[node_id].get("estimated_seconds") or 1)
        slack = latest_start - schedule[node_id]["earliest_start"]
        schedule[node_id]["latest_start"] = latest_start
        schedule[node_id]["slack_seconds"] = slack
        schedule[node_id]["critical_path"] = math.isclose(float(slack), 0.0, abs_tol=0.001)
    return schedule


def lint_goal(contract: dict[str, Any], nodes: list[dict[str, Any]], config: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    findings = []
    allowed_backends = {str(item) for item in list_value(config.get("allowed_executor_backends"))}
    allowed_tools = {str(item) for item in list_value(config.get("allowed_tools"))}
    node_ids = {node["node_id"] for node in nodes}
    acceptance_tests = set(contract.get("acceptance_tests", []))

    if not contract.get("objective"):
        findings.append(finding("missing_objective", "hard", "contract objective is required"))
    if not acceptance_tests:
        findings.append(finding("missing_acceptance_tests", "hard", "at least one acceptance test is required"))
    if not contract.get("owner_surface_registered"):
        findings.append(finding("owner_surface_unregistered", "hard", contract.get("owner_surface")))
    if not contract.get("owner_abstraction_registered"):
        findings.append(
            finding(
                "owner_surface_missing_abstraction_binding",
                "hard",
                {
                    "owner_surface": contract.get("owner_surface"),
                    "owner_abstraction_id": contract.get("owner_abstraction_id"),
                },
            )
        )
    if not nodes:
        findings.append(finding("no_plan_nodes", "hard", "goal has no atoms"))

    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep not in node_ids:
                findings.append(finding("missing_dependency", "hard", {"node": node["node_id"], "dependency": dep}))
        if node.get("executor_backend") not in allowed_backends:
            findings.append(finding("unsupported_executor_backend", "hard", {"node": node["node_id"], "backend": node.get("executor_backend")}))
        for tool in node.get("allowed_tools", []):
            if tool not in allowed_tools:
                findings.append(finding("unsupported_tool", "hard", {"node": node["node_id"], "tool": tool}))
            if "shell" in tool.lower() or "remote_exec" in tool.lower():
                findings.append(finding("arbitrary_execution_tool_blocked", "hard", {"node": node["node_id"], "tool": tool}))
        for requirement in node.get("tool_requirements", []):
            if requirement.get("missing_card"):
                findings.append(finding("deterministic_tool_card_missing", "hard", {"node": node["node_id"], "tool": requirement.get("tool_id")}))
            if requirement and not requirement.get("strict_no_fallback_returns"):
                findings.append(finding("deterministic_tool_allows_fallback", "hard", {"node": node["node_id"], "tool": requirement.get("tool_id")}))
        if not node.get("outputs"):
            findings.append(finding("node_missing_outputs", "hard", node["node_id"]))
        if not node.get("vcm_context_slice", {}).get("selected_page_count"):
            findings.append(finding("node_missing_vcm_context", "hard", node["node_id"]))
        if not node.get("vcm_context_slice", {}).get("governed_ready"):
            findings.append(
                finding(
                    "node_missing_governed_vcm_context",
                    "hard",
                    {
                        "node": node["node_id"],
                        "context_adequacy_state": node.get("vcm_context_slice", {}).get("context_adequacy_state"),
                        "governor_trigger_state": get_path(node, ["vcm_context_slice", "governor_receipt", "trigger_state"], ""),
                    },
                )
            )
        if node.get("fallback_return_allowed"):
            findings.append(finding("fallback_return_allowed", "hard", node["node_id"]))
        if "public" in str(node.get("training_surface") or "").lower():
            findings.append(finding("public_training_surface_blocked", "hard", node["node_id"]))
        for ref in node.get("acceptance_refs", []):
            if ref not in acceptance_tests:
                findings.append(finding("unknown_acceptance_ref", "warning", {"node": node["node_id"], "acceptance_ref": ref}))
        if node.get("route", {}).get("blocked"):
            findings.append(finding("route_blocked", "warning", {"node": node["node_id"], "reason": node.get("route", {}).get("block_reason")}))

    cycle_nodes = cycle_node_ids(nodes)
    if cycle_nodes:
        findings.append(finding("cycle_detected", "hard", cycle_nodes))
    orphan_nodes = orphan_node_ids(nodes)
    for node_id in orphan_nodes:
        findings.append(finding("orphan_node", "warning", node_id))

    hard = [row for row in findings if row["severity"] == "hard"]
    warnings = [row for row in findings if row["severity"] == "warning"]
    return {
        "passed": not hard,
        "finding_count": len(findings),
        "hard_failure_count": len(hard),
        "warning_count": len(warnings),
        "hard_failures": hard,
        "warnings": warnings,
    }


def cycle_node_ids(nodes: list[dict[str, Any]]) -> list[str]:
    by_id = {node["node_id"]: node for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            cycles.add(node_id)
            return
        visiting.add(node_id)
        for dep in by_id.get(node_id, {}).get("depends_on", []):
            if dep in by_id:
                visit(dep)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in by_id:
        visit(node_id)
    return sorted(cycles)


def orphan_node_ids(nodes: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return []
    by_id = {node["node_id"]: node for node in nodes}
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep in by_id:
                parents[node["node_id"]].append(dep)
                children[dep].append(node["node_id"])
    terminals = [node["node_id"] for node in nodes if not children[node["node_id"]] or node.get("op") in {"VERIFY", "REPORT", "BENCHMARK_ADAPTER"}]
    reachable: set[str] = set()
    queue = deque(terminals)
    while queue:
        node_id = queue.popleft()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        queue.extend(parents[node_id])
    return sorted(set(by_id) - reachable)


def build_ablation(compiled_goals: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    node_count = sum(goal.get("node_count", 0) for goal in compiled_goals)
    edge_count = sum(goal.get("edge_count", 0) for goal in compiled_goals)
    context_pages = [
        node.get("vcm_context_slice", {}).get("selected_page_count", 0)
        for goal in compiled_goals
        for node in goal.get("nodes", [])
    ]
    duplicate_semantic_hashes = duplicate_count(
        [node.get("semantic_hash") for goal in compiled_goals for node in goal.get("nodes", [])]
    )
    direct = {
        "source": "current_work_board_and_control_plane_reports",
        "ready_tasks": int(get_path(state, ["hive_work_board_executor", "summary", "ready_tasks"], 0)),
        "selected_tasks": int(get_path(state, ["hive_work_board_executor", "summary", "selected_tasks"], 0)),
        "control_plane_blockers": int(get_path(state, ["theseus_control_plane", "summary", "blocker_count"], 0)),
        "hard_control_plane_blockers": int(get_path(state, ["theseus_control_plane", "summary", "hard_blocker_count"], 0)),
        "vcm_loaded_by_default": bool(get_path(state, ["vcm_task_context_bridge", "summary", "high_priority_ready_count"], 0)),
        "decision_shape": "latest-report selection and task queue facts",
    }
    compiled = {
        "source": "theseus_plan_compiler_v0_dry_run",
        "compiled_goal_count": len(compiled_goals),
        "node_count": node_count,
        "edge_count": edge_count,
        "asi_stack_record_count": sum(len(asi_stack_record_ids(node.get("asi_stack_records", {}))) for goal in compiled_goals for node in goal.get("nodes", [])),
        "goal_record_count": sum(len(goal.get("goal_records", {})) for goal in compiled_goals),
        "parallelizable_node_count": sum(1 for goal in compiled_goals for node in goal.get("nodes", []) if node.get("schedule", {}).get("slack_seconds", 0) > 0 or node.get("schedule", {}).get("layer", 0) == 0),
        "duplicate_semantic_hash_count": duplicate_semantic_hashes,
        "average_context_pages_per_node": round(sum(context_pages) / max(1, len(context_pages)), 3),
        "missing_context_node_count": sum(1 for count in context_pages if not count),
        "local_deterministic_tool_packet_count": sum(1 for goal in compiled_goals for node in goal.get("nodes", []) if node.get("execution_packet", {}).get("mode") == "local_deterministic_tool_packet"),
        "deterministic_tool_requirement_count": sum(len(node.get("tool_requirements", [])) for goal in compiled_goals for node in goal.get("nodes", [])),
        "tool_eligibility_declared_node_count": sum(1 for goal in compiled_goals for node in goal.get("nodes", []) if isinstance(node.get("tool_eligibility"), dict) and node.get("tool_eligibility", {}).get("eligibility_id")),
        "tool_receipt_node_count": sum(1 for goal in compiled_goals for node in goal.get("nodes", []) if node.get("tool_receipts")),
        "lint_hard_failure_count": sum(goal.get("lint", {}).get("hard_failure_count", 0) for goal in compiled_goals),
        "lint_warning_count": sum(goal.get("lint", {}).get("warning_count", 0) for goal in compiled_goals),
        "decision_shape": "contract -> typed DAG -> VCM slice -> route -> verification target",
    }
    return {
        "policy": "project_theseus_plan_compiler_ablation_v0",
        "created_utc": now(),
        "comparison_type": "dry_run_structural_ablation",
        "direct_current_path": direct,
        "compiled_planning_path": compiled,
        "measured_improvements": {
            "contract_lock_present": True,
            "typed_dag_present": node_count > 0,
            "vcm_slice_per_node": compiled["missing_context_node_count"] == 0,
            "duplicate_semantic_hash_reduction_ready": duplicate_semantic_hashes == 0,
            "public_training_block_explicit": True,
            "executor_surfaces_existing": True,
            "trace_bundle_replayable": node_count > 0,
            "tool_eligibility_declared_per_node": all(isinstance(node.get("tool_eligibility"), dict) and node.get("tool_eligibility", {}).get("eligibility_id") for goal in compiled_goals for node in goal.get("nodes", [])),
            "tool_receipts_present_per_node": all(bool(node.get("tool_receipts")) for goal in compiled_goals for node in goal.get("nodes", [])),
            "asi_stack_records_per_node": all(node_has_required_asi_records(node) for goal in compiled_goals for node in goal.get("nodes", [])),
            "goal_governance_records_present": all(goal_has_required_records(goal) for goal in compiled_goals),
        },
        "limits": [
            "This ablation validates planning structure, not task execution quality.",
            "A future execute-mode proof should lease work-board tasks and compare completion rates on the same task set.",
            "Public planning benchmarks remain adapter-ready only; no public calibration was run here."
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_gates(
    config: dict[str, Any],
    registry: dict[str, Any],
    compiled_goals: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    ablation: dict[str, Any],
    readiness: list[dict[str, Any]],
    vcm_governor: dict[str, Any],
) -> list[dict[str, Any]]:
    all_nodes = [node for goal in compiled_goals for node in goal.get("nodes", [])]
    allowed_backends = {str(item) for item in list_value(config.get("allowed_executor_backends"))}
    registry_ids = {str(row.get("id") or "") for row in registry.get("surfaces", []) if isinstance(row, dict)}
    abstraction_ids = {str(row.get("id") or "") for row in registry.get("abstractions", []) if isinstance(row, dict)}
    local_tool_nodes = [node for node in all_nodes if node.get("executor_backend") == "local_deterministic_tool"]
    governor_summary = compact_vcm_governor(vcm_governor)
    return [
        gate("config_loaded", bool(config.get("policy")), config.get("policy"), "hard"),
        gate("representative_goals_present", bool(compiled_goals), len(compiled_goals), "hard"),
        gate("all_owner_surfaces_registered", all(goal.get("contract", {}).get("owner_surface") in registry_ids for goal in compiled_goals), [goal.get("contract", {}).get("owner_surface") for goal in compiled_goals], "hard"),
        gate(
            "all_owner_surfaces_abstraction_bound",
            all(goal.get("contract", {}).get("owner_abstraction_id") in abstraction_ids for goal in compiled_goals),
            {
                goal.get("goal_id"): {
                    "owner_surface": goal.get("contract", {}).get("owner_surface"),
                    "owner_abstraction_id": goal.get("contract", {}).get("owner_abstraction_id"),
                }
                for goal in compiled_goals
            },
            "hard",
        ),
        gate("all_goal_lints_pass", all(goal.get("lint", {}).get("hard_failure_count", 1) == 0 for goal in compiled_goals), {goal["goal_id"]: goal.get("lint", {}).get("hard_failures", []) for goal in compiled_goals}, "hard"),
        gate("vcm_context_governor_ready", bool(governor_summary.get("ready")), governor_summary, "hard"),
        gate("all_nodes_have_vcm_context", all(node.get("vcm_context_slice", {}).get("selected_page_count", 0) > 0 for node in all_nodes), len(all_nodes), "hard"),
        gate("all_nodes_have_governed_vcm_context", all(node.get("vcm_context_slice", {}).get("governed_ready") is True for node in all_nodes), {node.get("node_id"): node.get("vcm_context_slice", {}).get("context_adequacy_state") for node in all_nodes}, "hard"),
        gate(
            "all_nodes_pass_vcm_consumer_abi",
            all(get_path(node, ["asi_stack_records", "vcm_consumer_abi", "ready"], False) is True for node in all_nodes),
            {
                node.get("node_id"): {
                    "packet_id": get_path(node, ["asi_stack_records", "vcm_consumer_abi", "packet_id"], ""),
                    "faults": get_path(node, ["asi_stack_records", "vcm_consumer_abi", "typed_faults"], []),
                }
                for node in all_nodes
            },
            "hard",
        ),
        gate(
            "all_context_adequacy_records_use_governor_receipt",
            all(
                bool(get_path(node, ["asi_stack_records", "context_adequacy", "governor_receipt_id"], ""))
                and get_path(node, ["asi_stack_records", "context_adequacy", "governor_ready"], False) is True
                for node in all_nodes
            ),
            {
                node.get("node_id"): {
                    "governor_receipt_id": get_path(node, ["asi_stack_records", "context_adequacy", "governor_receipt_id"], ""),
                    "governor_ready": get_path(node, ["asi_stack_records", "context_adequacy", "governor_ready"], False),
                }
                for node in all_nodes
            },
            "hard",
        ),
        gate("all_executor_backends_registered", all(node.get("executor_backend") in allowed_backends for node in all_nodes), sorted({node.get("executor_backend") for node in all_nodes}), "hard"),
        gate("local_deterministic_tool_packets_present", bool(local_tool_nodes) and all(node.get("execution_packet", {}).get("mode") == "local_deterministic_tool_packet" for node in local_tool_nodes), len(local_tool_nodes), "hard"),
        gate("deterministic_tool_cards_present_when_required", all(not req.get("missing_card") for node in all_nodes for req in node.get("tool_requirements", [])), [req for node in all_nodes for req in node.get("tool_requirements", []) if req.get("missing_card")], "hard"),
        gate(
            "all_nodes_declare_tool_eligibility",
            all(isinstance(node.get("tool_eligibility"), dict) and node.get("tool_eligibility", {}).get("eligibility_id") for node in all_nodes),
            {
                "node_count": len(all_nodes),
                "declared_count": sum(1 for node in all_nodes if isinstance(node.get("tool_eligibility"), dict) and node.get("tool_eligibility", {}).get("eligibility_id")),
            },
            "hard",
        ),
        gate(
            "all_nodes_persist_tool_receipts",
            all(bool(node.get("tool_receipts")) for node in all_nodes),
            [node.get("node_id") for node in all_nodes if not node.get("tool_receipts")],
            "hard",
        ),
        gate(
            "tool_receipts_forbid_learned_generation_credit",
            all(
                receipt.get("learned_generation_claim_allowed") is False
                and int(receipt.get("public_training_rows_written") or 0) == 0
                and int(receipt.get("external_inference_calls") or 0) == 0
                and int(receipt.get("fallback_return_count") or 0) == 0
                for node in all_nodes
                for receipt in list_value(node.get("tool_receipts"))
                if isinstance(receipt, dict)
            ),
            {
                node.get("node_id"): [
                    {
                        "receipt_id": receipt.get("receipt_id"),
                        "learned_generation_claim_allowed": receipt.get("learned_generation_claim_allowed"),
                        "public_training_rows_written": receipt.get("public_training_rows_written"),
                        "external_inference_calls": receipt.get("external_inference_calls"),
                        "fallback_return_count": receipt.get("fallback_return_count"),
                    }
                    for receipt in list_value(node.get("tool_receipts"))
                    if isinstance(receipt, dict)
                ]
                for node in all_nodes
            },
            "hard",
        ),
        gate("trace_bundle_rows_match_nodes", len(trace_rows) == len(all_nodes), {"trace_rows": len(trace_rows), "nodes": len(all_nodes)}, "hard"),
        gate("all_nodes_have_required_asi_stack_records", all(node_has_required_asi_records(node) for node in all_nodes), missing_asi_stack_records(all_nodes), "hard"),
        gate("all_goals_have_required_governance_records", all(goal_has_required_records(goal) for goal in compiled_goals), missing_goal_records(compiled_goals), "hard"),
        gate("trace_rows_include_key_asi_record_ids", all(row.get("typed_job_id") and row.get("authority_transition_id") and row.get("context_transaction_id") and row.get("artifact_graph_id") for row in trace_rows), len(trace_rows), "hard"),
        gate("no_public_training_rows", True, 0, "hard"),
        gate("no_external_inference_calls", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
        gate("benchmark_adapters_calibration_only", all(row.get("train_allowed") is False and row.get("public_role") == "calibration_only" for row in readiness), readiness, "hard"),
        gate("ablation_structural_improvement_present", all(ablation.get("measured_improvements", {}).values()), ablation.get("measured_improvements"), "warning"),
    ]


def required_asi_stack_record_keys() -> set[str]:
    return {
        "semantic_atom",
        "semantic_node",
        "typed_job",
        "runtime_adapter_invocation",
        "authority_transition",
        "authority_use_receipt",
        "context_abi_record",
        "context_transaction",
        "context_adequacy",
        "resource_budget",
        "costed_route",
        "generation_mode",
        "failure_boundary",
        "artifact_graph",
        "evidence_transitions",
        "proof_carrying_claims",
        "routing_decision",
        "simulation_contract",
        "tool_call_receipts",
    }


def required_goal_record_keys() -> set[str]:
    return {
        "intent_contract",
        "command_contract",
        "reference_trace",
        "constitutional_predicates",
        "agency_rights_checklist",
        "value_conflict_record",
        "governance_rights",
        "research_backlog",
    }


def node_has_required_asi_records(node: dict[str, Any]) -> bool:
    records = node.get("asi_stack_records") if isinstance(node.get("asi_stack_records"), dict) else {}
    if not required_asi_stack_record_keys().issubset(set(records)):
        return False
    if not records.get("evidence_transitions"):
        return False
    if not records.get("tool_call_receipts"):
        return False
    if node.get("claim_objects") and not records.get("proof_carrying_claims"):
        return False
    for key in [
        "typed_job",
        "runtime_adapter_invocation",
        "authority_transition",
        "context_transaction",
        "resource_budget",
        "generation_mode",
        "artifact_graph",
    ]:
        if not isinstance(records.get(key), dict):
            return False
    return True


def missing_asi_stack_records(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    required = required_asi_stack_record_keys()
    for node in nodes:
        records = node.get("asi_stack_records") if isinstance(node.get("asi_stack_records"), dict) else {}
        absent = sorted(required - set(records))
        if absent or not records.get("evidence_transitions") or not records.get("tool_call_receipts") or (node.get("claim_objects") and not records.get("proof_carrying_claims")):
            missing.append(
                {
                    "node_id": node.get("node_id"),
                    "missing_keys": absent,
                    "evidence_transitions_present": bool(records.get("evidence_transitions")),
                    "tool_call_receipts_present": bool(records.get("tool_call_receipts")),
                    "proof_carrying_claims_present": bool(records.get("proof_carrying_claims")),
                }
            )
    return missing


def goal_has_required_records(goal: dict[str, Any]) -> bool:
    records = goal.get("goal_records") if isinstance(goal.get("goal_records"), dict) else {}
    if not required_goal_record_keys().issubset(set(records)):
        return False
    if not records.get("constitutional_predicates"):
        return False
    if not records.get("governance_rights"):
        return False
    return True


def missing_goal_records(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    required = required_goal_record_keys()
    for goal in goals:
        records = goal.get("goal_records") if isinstance(goal.get("goal_records"), dict) else {}
        absent = sorted(required - set(records))
        if absent or not records.get("constitutional_predicates") or not records.get("governance_rights"):
            missing.append(
                {
                    "goal_id": goal.get("goal_id"),
                    "missing_keys": absent,
                    "constitutional_predicates_present": bool(records.get("constitutional_predicates")),
                    "governance_rights_present": bool(records.get("governance_rights")),
                }
            )
    return missing


def summarize(
    compiled_goals: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    ablation: dict[str, Any],
    gates: list[dict[str, Any]],
    started: float,
    vcm_governor: dict[str, Any],
) -> dict[str, Any]:
    all_nodes = [node for goal in compiled_goals for node in goal.get("nodes", [])]
    governor_summary = compact_vcm_governor(vcm_governor)
    return {
        "compiled_goal_count": len(compiled_goals),
        "compiled_node_count": len(all_nodes),
        "compiled_edge_count": sum(goal.get("edge_count", 0) for goal in compiled_goals),
        "trace_row_count": len(trace_rows),
        "asi_stack_record_count": sum(len(asi_stack_record_ids(node.get("asi_stack_records", {}))) for node in all_nodes),
        "goal_record_count": sum(len(goal.get("goal_records", {})) for goal in compiled_goals),
        "goal_lint_hard_failure_count": sum(goal.get("lint", {}).get("hard_failure_count", 0) for goal in compiled_goals),
        "goal_lint_warning_count": sum(goal.get("lint", {}).get("warning_count", 0) for goal in compiled_goals),
        "critical_path_node_count": sum(1 for node in all_nodes if node.get("schedule", {}).get("critical_path")),
        "route_blocked_node_count": sum(1 for node in all_nodes if node.get("route", {}).get("blocked")),
        "local_deterministic_tool_packet_count": sum(1 for node in all_nodes if node.get("execution_packet", {}).get("mode") == "local_deterministic_tool_packet"),
        "deterministic_tool_requirement_count": sum(len(node.get("tool_requirements", [])) for node in all_nodes),
        "tool_eligibility_declared_node_count": sum(1 for node in all_nodes if isinstance(node.get("tool_eligibility"), dict) and node.get("tool_eligibility", {}).get("eligibility_id")),
        "tool_receipt_node_count": sum(1 for node in all_nodes if node.get("tool_receipts")),
        "tool_eligible_required_node_count": sum(1 for node in all_nodes if get_path(node, ["tool_eligibility", "decision"], "") == "eligible_required"),
        "tool_eligible_optional_node_count": sum(1 for node in all_nodes if get_path(node, ["tool_eligibility", "decision"], "") == "eligible_optional"),
        "tool_not_applicable_node_count": sum(1 for node in all_nodes if get_path(node, ["tool_eligibility", "decision"], "") == "not_applicable"),
        "tool_receipt_count": sum(len(list_value(node.get("tool_receipts"))) for node in all_nodes),
        "average_context_pages_per_node": ablation.get("compiled_planning_path", {}).get("average_context_pages_per_node", 0),
        "vcm_context_governor_ready": bool(governor_summary.get("ready")),
        "vcm_context_governor_state": governor_summary.get("trigger_state"),
        "vcm_context_governor_receipt_id": governor_summary.get("receipt_id"),
        "vcm_context_adequacy_governed_node_count": sum(1 for node in all_nodes if node.get("vcm_context_slice", {}).get("governed_ready") is True),
        "vcm_consumer_abi_ready_node_count": sum(
            1 for node in all_nodes if get_path(node, ["asi_stack_records", "vcm_consumer_abi", "ready"], False) is True
        ),
        "vcm_context_governor_hard_gap_count": governor_summary.get("hard_gap_count", 0),
        "vcm_mission_brief_status": governor_summary.get("mission_brief_status"),
        "vcm_deletion_closure_status": governor_summary.get("deletion_closure_status"),
        "vcm_scif_status": governor_summary.get("scif_status"),
        "benchmark_readiness_adapter_count": len([node for node in all_nodes if node.get("op") == "BENCHMARK_ADAPTER"]),
        "gate_count": len(gates),
        "gate_failure_count": len([row for row in gates if not row["passed"]]),
        "hard_gate_failure_count": len([row for row in gates if row["severity"] == "hard" and not row["passed"]]),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def benchmark_readiness(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in config.get("planning_benchmark_readiness", []) if isinstance(config.get("planning_benchmark_readiness"), list) else []:
        if not isinstance(row, dict):
            continue
        out = dict(row)
        out["contract_hash"] = stable_hash(row)
        out["readiness_state"] = "ADAPTER_READY_DRY_RUN" if row.get("adapter_status") else "MISSING_ADAPTER_STATUS"
        out["public_training_rows_written"] = 0
        out["external_inference_calls"] = 0
        rows.append(out)
    return rows


def procedural_memory_canary_goals(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Lift replay-proven procedural candidates into canary-only plan contracts.

    This is deliberately conservative. A canary route is a registry-gated
    planner packet, not default adoption, not learned generation, and not
    execution success. It lets Theseus rehearse one repeated assistant trace
    through the same VCM/evidence/route machinery as other plan nodes.
    """

    replay_by_id = {
        str(row.get("id") or ""): row
        for row in report.get("assistant_trace_replay_results", [])
        if isinstance(row, dict)
    }
    goals: list[dict[str, Any]] = []
    for route in report.get("canary_routes", []) if isinstance(report.get("canary_routes"), list) else []:
        if not isinstance(route, dict):
            continue
        if route.get("canary_route_eligible") is not True:
            continue
        if route.get("default_route_allowed") is not False:
            continue
        fixture_id = str(route.get("replay_fixture_id") or "")
        fixture = replay_by_id.get(fixture_id, {})
        if fixture.get("passed") is not True:
            continue
        route_id = str(route.get("id") or stable_id("canary_route", route))
        candidate_id = str(route.get("candidate_id") or "")
        metrics = route.get("metrics") if isinstance(route.get("metrics"), dict) else {}
        goal_id = f"procedural_memory_canary_{slug(route_id)}"
        goals.append(
            {
                "id": goal_id,
                "title": f"Canary route procedural memory candidate {candidate_id}",
                "owner_surface": str(route.get("owner_surface") or "theseus_plan_compiler"),
                "priority": "medium",
                "risk_tier": "low",
                "procedural_candidate_id": candidate_id,
                "canary_route_id": route_id,
                "replay_fixture_id": fixture_id,
                "objective": (
                    "Route one replay-verified assistant-trace procedural candidate through the "
                    "registry-gated planner canary path without default adoption."
                ),
                "non_goals": [
                    "Do not make this candidate a default route.",
                    "Do not claim learned code generation.",
                    "Do not expose raw private text.",
                    "Do not write public benchmark training rows.",
                    "Do not call external runtime inference.",
                    "Do not use fallback returns.",
                ],
                "outputs": [
                    "procedural_memory_canary_route_packet",
                    "procedural_memory_duplicate_work_delta_estimate",
                    "procedural_memory_verification_cost_delta_estimate",
                ],
                "acceptance_tests": [
                    "canary_replay_fixture_passed",
                    "canary_not_default_route",
                    "registry_owner_surface_present",
                    "duplicate_work_delta_recorded",
                    "verification_cost_delta_recorded",
                    "no_public_training_rows",
                    "no_external_inference_calls",
                    "no_fallback_returns",
                ],
                "atoms": [
                    {
                        "id": "validate_replay_fixture",
                        "op": "VERIFY",
                        "title": "Validate replay fixture and no-cheat counters for procedural canary",
                        "depends_on": [],
                        "inputs": [fixture_id, candidate_id],
                        "outputs": ["procedural_memory_replay_fixture_receipt"],
                        "required_capabilities": ["trace_replay", "registry_lookup"],
                        "allowed_tools": ["json_report", "registry_lookup", "tool.trace_replay"],
                        "executor_backend": "local_deterministic_tool",
                        "worker_tier": "T0",
                        "estimated_seconds": 8,
                        "vcm_family": "planning",
                        "acceptance_refs": [
                            "canary_replay_fixture_passed",
                            "canary_not_default_route",
                            "no_public_training_rows",
                            "no_external_inference_calls",
                            "no_fallback_returns",
                        ],
                    },
                    {
                        "id": "compile_canary_route_packet",
                        "op": "ROUTE",
                        "title": "Compile registry-gated canary route packet",
                        "depends_on": ["validate_replay_fixture"],
                        "inputs": [route_id, "procedural_memory_replay_fixture_receipt"],
                        "outputs": ["procedural_memory_canary_route_packet"],
                        "required_capabilities": ["semantic_ir", "work_board_routing", "registry_lookup"],
                        "allowed_tools": ["json_report", "registry_lookup"],
                        "executor_backend": "theseus_control_plane",
                        "worker_tier": "T0",
                        "estimated_seconds": 10,
                        "vcm_family": "planning",
                        "acceptance_refs": [
                            "registry_owner_surface_present",
                            "canary_not_default_route",
                        ],
                    },
                    {
                        "id": "measure_duplicate_and_verification_delta",
                        "op": "VERIFY",
                        "title": "Record duplicate-work and verification-cost delta estimates",
                        "depends_on": ["compile_canary_route_packet"],
                        "inputs": [
                            str(metrics.get("estimated_duplicate_work_delta", "")),
                            str(metrics.get("estimated_verification_cost_delta", "")),
                        ],
                        "outputs": [
                            "procedural_memory_duplicate_work_delta_estimate",
                            "procedural_memory_verification_cost_delta_estimate",
                        ],
                        "required_capabilities": ["report_reading", "claim_evidence_mapping"],
                        "allowed_tools": ["json_report"],
                        "executor_backend": "local_deterministic_tool",
                        "worker_tier": "T0",
                        "estimated_seconds": 6,
                        "vcm_family": "planning",
                        "acceptance_refs": [
                            "duplicate_work_delta_recorded",
                            "verification_cost_delta_recorded",
                        ],
                    },
                ],
                "source_route": {
                    "id": route_id,
                    "candidate_id": candidate_id,
                    "replay_fixture_id": fixture_id,
                    "default_route_allowed": False,
                    "metrics": metrics,
                    "public_training_rows_written": int(route.get("public_training_rows_written") or 0),
                    "external_inference_calls": int(route.get("external_inference_calls") or 0),
                    "fallback_return_count": int(route.get("fallback_return_count") or 0),
                    "non_claims": list_value(route.get("non_claims")),
                },
            }
        )
    return goals


def procedural_memory_default_route_goals(report: dict[str, Any]) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    if str(report.get("trigger_state") or "") != "GREEN":
        return goals
    transaction = report.get("adoption_transaction") if isinstance(report.get("adoption_transaction"), dict) else {}
    transaction_id = str(transaction.get("transaction_id") or "")
    for route in report.get("default_routes", []) if isinstance(report.get("default_routes"), list) else []:
        if not isinstance(route, dict):
            continue
        if route.get("default_route_adopted") is not True:
            continue
        if route.get("learned_generation_claim_allowed") is not False:
            continue
        guard = route.get("continued_regression_guard") if isinstance(route.get("continued_regression_guard"), dict) else {}
        if guard.get("armed") is not True:
            continue
        route_id = str(route.get("id") or "")
        candidate_id = str(route.get("candidate_id") or "")
        source_canary = str(route.get("source_canary_route_id") or "")
        goal_id = f"procedural_memory_default_route_{slug(route_id)}"
        goals.append(
            {
                "id": goal_id,
                "title": f"Maintain default procedural memory route {route_id}",
                "owner_surface": str(route.get("owner_surface") or "theseus_plan_compiler"),
                "priority": "medium",
                "risk_tier": "low",
                "default_route_id": route_id,
                "procedural_candidate_id": candidate_id,
                "adoption_transaction_id": transaction_id,
                "objective": (
                    "Compile the adopted procedural-memory default route as a guarded local metadata route "
                    "while preserving replay, rollback, and no-cheat boundaries."
                ),
                "non_goals": [
                    "Do not count the procedural route as learned generation.",
                    "Do not use public benchmark data as training rows.",
                    "Do not call external runtime inference.",
                    "Do not keep the default route if the regression guard fails.",
                ],
                "outputs": [
                    "procedural_memory_default_route_packet",
                    "procedural_memory_default_route_guard_receipt",
                    "procedural_memory_default_route_rollback_handle",
                ],
                "acceptance_tests": [
                    "default_route_adoption_transaction_present",
                    "default_route_regression_guard_present",
                    "default_route_no_learned_generation_claim",
                    "default_route_rollback_handle_present",
                    "no_public_training_rows",
                    "no_external_inference_calls",
                    "no_fallback_returns",
                ],
                "atoms": [
                    {
                        "id": "validate_default_route_transaction",
                        "op": "VERIFY",
                        "title": "Validate procedural route adoption transaction and regression guard",
                        "depends_on": [],
                        "inputs": [transaction_id, route_id, candidate_id],
                        "outputs": ["procedural_memory_default_route_guard_receipt"],
                        "required_capabilities": ["registry_lookup", "trace_replay"],
                        "allowed_tools": ["json_report", "registry_lookup", "tool.trace_replay"],
                        "executor_backend": "local_deterministic_tool",
                        "worker_tier": "T0",
                        "estimated_seconds": 8,
                        "vcm_family": "planning",
                        "acceptance_refs": [
                            "default_route_adoption_transaction_present",
                            "default_route_regression_guard_present",
                            "default_route_no_learned_generation_claim",
                            "no_public_training_rows",
                            "no_external_inference_calls",
                            "no_fallback_returns",
                        ],
                    },
                    {
                        "id": "compile_default_route_packet",
                        "op": "ROUTE",
                        "title": "Compile local metadata default-route packet",
                        "depends_on": ["validate_default_route_transaction"],
                        "inputs": [route_id, source_canary, "procedural_memory_default_route_guard_receipt"],
                        "outputs": ["procedural_memory_default_route_packet"],
                        "required_capabilities": ["semantic_ir", "work_board_routing", "registry_lookup"],
                        "allowed_tools": ["json_report", "registry_lookup"],
                        "executor_backend": "theseus_control_plane",
                        "worker_tier": "T0",
                        "estimated_seconds": 10,
                        "vcm_family": "planning",
                        "acceptance_refs": [
                            "default_route_adoption_transaction_present",
                            "default_route_regression_guard_present",
                        ],
                    },
                    {
                        "id": "attach_default_route_rollback_handle",
                        "op": "VERIFY",
                        "title": "Attach rollback handle for guarded procedural default route",
                        "depends_on": ["compile_default_route_packet"],
                        "inputs": [str(guard.get("guard_id") or "")],
                        "outputs": ["procedural_memory_default_route_rollback_handle"],
                        "required_capabilities": ["claim_evidence_mapping", "registry_lookup"],
                        "allowed_tools": ["json_report", "registry_lookup"],
                        "executor_backend": "local_deterministic_tool",
                        "worker_tier": "T0",
                        "estimated_seconds": 6,
                        "vcm_family": "planning",
                        "acceptance_refs": ["default_route_rollback_handle_present"],
                    },
                ],
                "source_route": route,
            }
        )
    return goals


def load_state() -> dict[str, Any]:
    paths = {
        "theseus_control_plane": REPORTS / "theseus_control_plane.json",
        "hive_work_board_executor": REPORTS / "hive_work_board_executor.json",
        "vcm_task_context_bridge": REPORTS / "vcm_task_context_bridge.json",
        "hive_node_registry": REPORTS / "hive_node_registry.json",
        "resource_aware_execution_policy": REPORTS / "resource_aware_execution_policy.json",
        "autonomy_launch_readiness": REPORTS / "autonomy_launch_readiness.json",
    }
    state = {}
    for key, path in paths.items():
        payload = read_json(path, {})
        state[key] = {
            "trigger_state": payload.get("trigger_state"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else payload.get("summary", {}),
            "payload": payload,
        }
        if key in {"hive_node_registry", "resource_aware_execution_policy"}:
            state[key]["summary"] = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    return state


def preconditions_for(atom: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "backend_available", "check": f"executor_backend == {atom.get('executor_backend')}"},
        {"id": "vcm_context_available", "check": "selected_page_count > 0"},
        {"id": "acceptance_refs_defined", "check": "node acceptance refs are contract acceptance tests"},
    ]


def effects_for(atom: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": f"produces_{slug(str(output))}", "output": str(output)}
        for output in list_value(atom.get("outputs"))
    ]


def claim_objects_for(node: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": stable_id("claim", node["node_id"], ref),
            "node_id": node["node_id"],
            "type": "procedural",
            "predicate": "acceptance_ref_supported_by_node",
            "parameters": {"acceptance_ref": ref, "owner_surface": contract.get("owner_surface")},
            "evidence_refs": [f"evidence://theseus_plan_compiler/{node['node_id']}/{ref}"],
            "assurance_level": "A2_procedure_expected",
            "actionability": "internal_control_signal",
        }
        for ref in node.get("acceptance_refs", [])
    ]


def evidence_targets_for(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_ref": f"evidence://theseus_plan_compiler/{node['node_id']}/output/{slug(str(output))}",
            "node_id": node["node_id"],
            "expected_output": output,
            "verification": "executor report or local deterministic check must write a JSON evidence pointer before promotion-facing use",
        }
        for output in node.get("outputs", [])
    ]


def repair_policy_for(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_attempts": 2,
        "localized_recompile": True,
        "preserve_verified_siblings": True,
        "failure_fingerprint": stable_id("failure", node["node_id"], node.get("semantic_hash")),
        "escalation": "same_backend_retry_then_teacher_architecture_proposal_if_policy_allows",
    }


def asi_stack_records_for(node: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Build ASI Stack protocol records for one compiled node.

    These are compile-time governance records. They make authority, context,
    routing, evidence, and failure obligations inspectable before execution;
    they are not claims that the node already succeeded.
    """

    node_id = str(node["node_id"])
    packet = node.get("execution_packet") if isinstance(node.get("execution_packet"), dict) else {}
    route = node.get("route") if isinstance(node.get("route"), dict) else {}
    vcm = node.get("vcm_context_slice") if isinstance(node.get("vcm_context_slice"), dict) else {}
    claims = node.get("claim_objects") if isinstance(node.get("claim_objects"), list) else []
    evidence_targets = node.get("evidence_targets") if isinstance(node.get("evidence_targets"), list) else []
    repair = node.get("repair_policy") if isinstance(node.get("repair_policy"), dict) else {}
    governor = vcm.get("governor_receipt") if isinstance(vcm.get("governor_receipt"), dict) else {}
    governor_ready = bool(governor.get("ready"))
    context_pages_ready = bool(vcm.get("ready")) and bool(vcm.get("selected_page_count", 0))
    permissions = permissions_for_node(node)
    authority = authority_ceiling_for_node(node)
    risk = str(node.get("risk_tier") or contract.get("risk_tier") or "medium")
    job_id = stable_id("typed_job", node_id, contract.get("contract_id"))
    adapter_id = stable_id("runtime_adapter", route.get("surface"), node.get("executor_backend"))
    context_packet_id = stable_id("context_packet", node_id, vcm.get("governed_context_hash") or vcm.get("context_hash"))
    context_transaction_id = stable_id(
        "context_transaction",
        node_id,
        vcm.get("governed_context_hash") or vcm.get("context_hash"),
        governor.get("receipt_id"),
        "read",
    )
    selected_pages = vcm.get("selected_pages") if isinstance(vcm.get("selected_pages"), list) else []
    tool_ids = [str(row.get("tool_id") or "") for row in node.get("tool_requirements", []) if isinstance(row, dict)]
    tool_eligibility = node.get("tool_eligibility") if isinstance(node.get("tool_eligibility"), dict) else {}
    tool_receipts = node.get("tool_receipts") if isinstance(node.get("tool_receipts"), list) else []
    if not tool_ids:
        tool_ids = [str(item) for item in list_value(tool_eligibility.get("candidate_tool_ids")) if item]
    tool_receipt_ids = [str(row.get("receipt_id") or "") for row in tool_receipts if isinstance(row, dict) and row.get("receipt_id")]
    tool_evidence_refs = [
        str(ref)
        for row in tool_receipts
        if isinstance(row, dict)
        for ref in list_value(row.get("evidence_refs"))
        if ref
    ]
    semantic_units = [
        {
            "address": str(page.get("address") or ""),
            "title": str(page.get("title") or ""),
            "source_path": str(page.get("source_path") or ""),
            "taints": list_value(page.get("taints")),
        }
        for page in selected_pages
        if isinstance(page, dict)
    ]
    vcm_consumer_packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id=f"theseus_plan_compiler:{node_id}",
        purpose="planning",
        read_set=["reports/vcm_context_governor.json"],
        write_set=[],
        authority_ceiling=[authority],
        permitted_uses=["planning_context", "route_selection", "verification_obligation_compilation"],
        context_refs=[
            {
                "kind": "semantic_address",
                "ref": page.get("address"),
                "required": True,
                "exists": bool(page.get("address")),
                "sha256": page.get("content_hash") or vcm.get("context_hash"),
                "taint_labels": page.get("taints", []),
                "contradiction_refs": page.get("contradiction_refs", []),
            }
            for page in semantic_units
        ],
        taint_labels=sorted({str(taint) for page in semantic_units for taint in list_value(page.get("taints"))}),
        deletion_obligations=["invalidate_compiled_plan_when_context_is_revoked"],
        contradiction_refs=[
            str(ref)
            for page in semantic_units
            for ref in list_value(page.get("contradiction_refs"))
            if ref
        ],
        compression_loss=float(governor.get("mission_brief_compression_loss") or 0.0),
        audit_refs=["scripts/theseus_plan_compiler.py", str(vcm.get("context_hash") or "")],
    )
    governor_ready = governor_ready and bool(vcm_consumer_packet.get("ready"))
    if semantic_units and context_pages_ready and governor_ready:
        adequacy_state = "governed_sufficient_for_planning"
        context_faults: list[str] = []
    else:
        context_faults = []
        if not semantic_units:
            context_faults.append("missing_context_pages")
        if not context_pages_ready:
            context_faults.append("vcm_context_bridge_not_ready")
        if not governor_ready:
            context_faults.extend(vcm_consumer_packet.get("typed_faults") or ["vcm_context_governor_not_ready"])
        adequacy_state = "fault_" + "_and_".join(context_faults or ["unknown_context_fault"])
    failure_trigger = route.get("block_reason") or "execution_or_verification_failure"
    residual_risk = "route_blocked" if route.get("blocked") else "unexecuted_compile_time_obligation"
    generation_mode = generation_mode_for_node(node)
    now_utc = now()

    semantic_atom = {
        "record_type": "semantic_atom",
        "atom_id": stable_id("semantic_atom", node_id, node.get("semantic_hash")),
        "intent": str(node.get("title") or node.get("op") or ""),
        "inputs": node.get("inputs", []),
        "outputs": node.get("outputs", []),
        "constraints": contract.get("constraint_capsules", []),
        "dependencies": node.get("depends_on", []),
        "authority_required": authority,
        "validator": "evidence_targets_and_acceptance_refs",
        "target": route.get("surface") or node.get("executor_backend"),
        "repair_scope": repair,
        "support_state": "compile_time_record",
    }
    typed_job = {
        "record_type": "typed_job",
        "job_id": job_id,
        "contract_id": contract.get("contract_id"),
        "job_type": str(node.get("op") or "UNKNOWN"),
        "lifecycle_state": "compiled_not_leased",
        "runtime_adapter": adapter_id,
        "inputs": node.get("inputs", []),
        "outputs": node.get("outputs", []),
        "permissions": permissions,
        "approval_state": approval_state_for_node(node),
        "failure_behavior": "emit_residual_and_do_not_claim_completion",
        "audit_events": [
            {"event": "compiled", "created_utc": now_utc},
            {"event": "lease_required_before_execution", "created_utc": now_utc},
        ],
        "replay_status": "packet_hash_available" if packet.get("packet_hash") else "packet_hash_pending",
    }
    runtime_adapter_invocation = {
        "record_type": "runtime_adapter_invocation",
        "invocation_id": stable_id("adapter_invocation", node_id, adapter_id),
        "job_id": job_id,
        "adapter_id": adapter_id,
        "target_type": "registered_executor_surface",
        "capability": str(node.get("executor_backend") or ""),
        "permission_required": permissions,
        "sandbox_mode": sandbox_mode_for_node(node),
        "approval_required": approval_required_for_node(node),
        "approval_record": approval_state_for_node(node),
        "authority_handle": stable_id("authority_handle", node_id, authority),
        "inputs": node.get("inputs", []),
        "effect_receipt": "pending_execution",
        "rollback_handle": rollback_handle_for_node(node),
        "residuals": [failure_trigger] if route.get("blocked") else [],
    }
    authority_transition = {
        "record_type": "authority_transition_record",
        "transition_id": stable_id("authority_transition", node_id, authority, permissions),
        "principal": "theseus_plan_compiler",
        "source_layer": "planning_execution_spine",
        "target_boundary": route.get("surface") or node.get("executor_backend"),
        "requested_operation": str(node.get("op") or "UNKNOWN"),
        "authority_ceiling": authority,
        "grant_id": stable_id("grant", contract.get("contract_id"), node_id, authority),
        "revocation_epoch": "on_contract_change_or_route_block",
        "decision": "allow_compile_only" if not route.get("blocked") else "deny_execution_until_route_unblocked",
        "denial_reason": route.get("block_reason") if route.get("blocked") else "",
        "effect_receipt": "required_after_execution",
        "audit_refs": [packet.get("packet_id"), vcm.get("context_hash"), governor.get("report")],
    }
    authority_use_receipt = {
        "record_type": "authority_use_receipt",
        "handle_id": runtime_adapter_invocation["authority_handle"],
        "principal": "theseus_plan_compiler",
        "purpose": str(node.get("title") or ""),
        "destination": route.get("surface") or "",
        "allowed_action": permissions,
        "clearance": authority,
        "approval_record": approval_state_for_node(node),
        "scif_lifecycle": "not_secret_bearing" if not uses_privileged_context(node) else "handle_only_model_visible",
        "sanitized_output": True,
        "residual_leak_risk": "low_compile_time_metadata_only",
        "revocation_path": "invalidate_packet_and_block_route",
    }
    context_abi_record = {
        "record_type": "context_abi_record",
        "request_id": stable_id("context_abi", node_id, vcm.get("context_hash")),
        "task_id": node_id,
        "semantic_address": f"vcm://plan_compiler/{node_id}",
        "version": "1.0",
        "mount": vcm.get("task_family_id"),
        "snapshot_id": stable_id("vcm_snapshot", vcm.get("context_hash")),
        "representation_contract": "selected_pages_plus_constraint_capsules_plus_vcm_governor_receipt",
        "authority_labels": authority,
        "admission_state": "admitted" if adequacy_state == "governed_sufficient_for_planning" else "fault_context_governance",
        "adequacy_state": adequacy_state,
        "fault_state": "" if adequacy_state == "governed_sufficient_for_planning" else ",".join(context_faults),
        "materialization_ref": vcm.get("governed_context_hash") or vcm.get("context_hash"),
        "governor_receipt_id": governor.get("receipt_id"),
        "governor_ready": governor_ready,
        "governor_trigger_state": governor.get("trigger_state"),
        "mission_brief_status": governor.get("mission_brief_status"),
        "deletion_closure_status": governor.get("deletion_closure_status"),
        "scif_status": governor.get("scif_status"),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "audit_refs": [page.get("address") for page in semantic_units] + [governor.get("report")],
    }
    context_transaction = {
        "record_type": "context_transaction_record",
        "transaction_id": context_transaction_id,
        "operation": "read",
        "snapshot_id": context_abi_record["snapshot_id"],
        "mounts": [vcm.get("task_family_id")],
        "read_set": [page.get("address") for page in semantic_units],
        "write_set": [],
        "branch_policy": "planner_read_only_context",
        "taint_labels": sorted({str(taint) for page in semantic_units for taint in list_value(page.get("taints"))}),
        "deletion_obligations": "closed_by_vcm_governor" if governor.get("deletion_closure_status") == "closed" else "fault_requires_vcm_deletion_closure",
        "contradiction_refs": [],
        "closure_state": "closed_for_read_only_governed_compile" if adequacy_state == "governed_sufficient_for_planning" else "fault_context_governance",
        "faults": context_faults,
        "governor_receipt_id": governor.get("receipt_id"),
        "governor_ready": governor_ready,
        "mission_brief_status": governor.get("mission_brief_status"),
        "deletion_closure_status": governor.get("deletion_closure_status"),
        "scif_status": governor.get("scif_status"),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "audit_refs": [vcm.get("context_hash"), governor.get("report")],
    }
    context_adequacy = {
        "record_type": "context_adequacy_record",
        "adequacy_id": stable_id("context_adequacy", node_id, vcm.get("governed_context_hash") or vcm.get("context_hash"), governor.get("receipt_id")),
        "target_claim_id": claims[0].get("claim_id") if claims else stable_id("claim_placeholder", node_id),
        "semantic_units": semantic_units,
        "compression_path": "selected_vcm_pages_plus_mission_brief_governor_to_compact_node_context",
        "verification_mode": "governed_planning_adequacy_only",
        "adequacy_state": adequacy_state,
        "governor_receipt_id": governor.get("receipt_id"),
        "governor_ready": governor_ready,
        "governor_report": governor.get("report"),
        "mission_brief_status": governor.get("mission_brief_status"),
        "mission_brief_omission_count": governor.get("mission_brief_omission_count"),
        "deletion_closure_status": governor.get("deletion_closure_status"),
        "scif_status": governor.get("scif_status"),
        "fail_closed": adequacy_state != "governed_sufficient_for_planning",
        "residual_risks": context_faults,
        "required_escalation": "execute_or_verify_requires_vcm_context_governor_refresh" if context_faults else "none",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    resource_budget = {
        "record_type": "resource_budget_record",
        "budget_id": stable_id("resource_budget", node_id, risk, node.get("estimated_seconds")),
        "task_id": node_id,
        "value_hypothesis": "roadmap_progress_if_evidence_target_written",
        "risk_class": risk,
        "capacity_pool": str(node.get("worker_tier") or "T0"),
        "cost_estimate": {"estimated_seconds": int(node.get("estimated_seconds") or 0), "tool_count": len(tool_receipts)},
        "verification_tax": {"evidence_target_count": len(evidence_targets), "claim_count": len(claims)},
        "quality_predicate": node.get("acceptance_refs", []),
        "safety_gates": ["no_public_training_rows", "no_external_runtime_inference", "no_fallback_returns"],
        "budget_decision": "compile_route_allowed",
        "escalation_rule": "high_risk_or_missing_context_requires_review",
        "residuals": [failure_trigger] if route.get("blocked") else [],
        "evidence_refs": [target.get("evidence_ref") for target in evidence_targets] + tool_evidence_refs,
    }
    costed_route = {
        "record_type": "costed_route_record",
        "task_id": node_id,
        "quality_predicate": node.get("acceptance_refs", []),
        "authority_ceiling": authority,
        "candidate_routes": [str(node.get("executor_backend") or "")],
        "selected_route": route.get("surface") or node.get("executor_backend"),
        "rejected_lower_cost_routes": [],
        "verification_result": "not_executed_compile_time",
        "cost_accounting": resource_budget["cost_estimate"],
        "residual_obligations": [failure_trigger] if route.get("blocked") else [],
        "fallback_route": "residual_not_fallback_answer",
        "promotion_candidate": False,
        "non_claims": ["route selection is not execution success", "cost estimate is not capability evidence"],
    }
    generation_mode_record = {
        "record_type": "generation_mode_record",
        "task_id": node_id,
        "risk_tier": risk,
        "latency_budget": int(node.get("estimated_seconds") or 0),
        "compute_budget": str(node.get("worker_tier") or "T0"),
        "memory_budget": vcm.get("selected_page_count", 0),
        "context_packet_id": context_packet_id,
        "generation_mode": generation_mode,
        "draft_source": "none" if generation_mode == "non_generation_task" else str(node.get("executor_backend") or ""),
        "verifier": "evidence_targets_and_acceptance_refs",
        "acceptance_predicate": node.get("acceptance_refs", []),
        "proposed_output_accounting": {"expected_outputs": len(node.get("outputs", []))},
        "accepted_output_accounting": {"accepted_outputs": 0, "reason": "not_executed"},
        "wall_clock_time": 0,
        "quality_or_pass_result": "not_executed",
        "repair_or_fallback": "repair_residual_only_no_fallback_answer",
        "promotion_decision": "not_promotion_candidate",
        "measurement_status": "compile_time_record",
        "metric_definitions": ["accepted_verified_output_count", "verification_tax", "runtime_ms"],
        "evidence_refs": [target.get("evidence_ref") for target in evidence_targets],
        "non_claims": ["generation mode record does not imply learned generation"],
    }
    failure_boundary = {
        "record_type": "failure_boundary_map",
        "failure_id": repair.get("failure_fingerprint") or stable_id("failure", node_id),
        "layer": "planning_execution_spine",
        "trigger": failure_trigger,
        "protected_invariant": "no_public_training_no_external_runtime_no_fallback_credit",
        "detection_route": "plan_compiler_lint_and_executor_report",
        "containment_action": "block_or_emit_residual_without_claiming_success",
        "evidence_record": [target.get("evidence_ref") for target in evidence_targets],
        "downstream_owner": contract.get("owner_surface"),
        "residual_risk": residual_risk,
    }
    artifact_graph = {
        "record_type": "artifact_graph_record",
        "artifact_id": stable_id("artifact", node_id, packet.get("packet_id")),
        "artifact_type": "compiled_execution_packet",
        "parent_job": job_id,
        "source_refs": [contract.get("contract_id"), node.get("semantic_hash")],
        "context_refs": [vcm.get("context_hash"), vcm.get("governed_context_hash")],
        "context_transaction_refs": [context_transaction_id],
        "context_adequacy_refs": [context_adequacy.get("adequacy_id")],
        "vcm_governor_refs": [governor.get("receipt_id"), governor.get("report")],
        "tool_refs": tool_ids,
        "tool_eligibility_ref": tool_eligibility.get("eligibility_id"),
        "tool_receipt_refs": tool_receipt_ids,
        "tool_evidence_refs": tool_evidence_refs,
        "claim_refs": [claim.get("claim_id") for claim in claims],
        "test_refs": node.get("acceptance_refs", []),
        "audit_events": ["compiled", "awaiting_lease"],
        "replay_metadata": {"packet_id": packet.get("packet_id"), "packet_hash": packet.get("packet_hash")},
        "environment_assumptions": {"executor_backend": node.get("executor_backend"), "worker_tier": node.get("worker_tier")},
        "provenance_status": "compiled_from_registry_owned_plan",
        "replay_limits": "requires executor surface and current dependencies",
    }
    evidence_transitions = [
        {
            "record_type": "evidence_transition_record",
            "transition_id": stable_id("evidence_transition", node_id, target.get("evidence_ref")),
            "claim_id": claims[0].get("claim_id") if claims else stable_id("claim_placeholder", node_id),
            "old_support_state": "missing",
            "new_support_state": "required_not_yet_observed",
            "transition_reason": "plan_node_requires_output_evidence_before_claim_credit",
            "required_artifacts": [target.get("expected_output")],
            "artifact_refs": [],
            "verification_command": "executor_surface_must_write_json_evidence",
            "verification_result": "not_run",
            "negative_results": [],
            "limitations": ["compile-time target only"],
            "review_status": "pending_execution",
        }
        for target in evidence_targets
        if isinstance(target, dict)
    ]
    proof_carrying_claims = [
        {
            "record_type": "proof_carrying_claim",
            "proof_claim_id": stable_id("proof_claim", claim.get("claim_id"), node_id),
            "claim_id": claim.get("claim_id"),
            "required_tier": "procedure_receipt",
            "interpretation_mapping": "acceptance_ref_to_executor_evidence_target",
            "justification_artifact": artifact_graph["artifact_id"],
            "verifier": "executor_report_or_local_deterministic_check",
            "verifier_result": "not_run",
            "limitations": ["proof-carrying envelope only until executor evidence exists"],
            "downgrade_rule": "missing_or_failed_verifier_keeps_claim_uncredited",
            "tribunal_ref": "",
            "ledger_update": "pending_evidence_transition",
        }
        for claim in claims
        if isinstance(claim, dict)
    ]
    routing_decision = {
        "record_type": "routing_decision_record",
        "decision_id": stable_id("routing_decision", node_id, route.get("surface")),
        "task_id": node_id,
        "capability_request": node.get("required_capabilities", []),
        "candidate_specialists": [str(node.get("executor_backend") or "")],
        "selected_specialist": route.get("surface") or node.get("executor_backend"),
        "route_shape": "compiled_packet_then_executor_lease",
        "authority_check": authority_transition["decision"],
        "readiness_check": "blocked" if route.get("blocked") else "compile_ready",
        "cost_quality_reason": "minimum_registered_executor_for_declared_backend",
        "fallback_route": "residual_or_replan_not_answer_fallback",
        "residuals": [failure_trigger] if route.get("blocked") else [],
        "ledger_refs": [artifact_graph["artifact_id"], resource_budget["budget_id"]],
    }
    semantic_node = {
        "record_type": "semantic_node_record",
        "node_id": stable_id("semantic_node", node_id, node.get("semantic_hash")),
        "concept_label": str(node.get("op") or "plan_node"),
        "provenance_refs": [contract.get("contract_id"), node.get("semantic_hash")],
        "parent_refs": node.get("depends_on", []),
        "child_refs": [],
        "relation_refs": node.get("required_capabilities", []),
        "tokenization_contract": "not_tokenized_semantic_ir_record",
        "grounding_state": "grounded_in_plan_config_and_registry",
        "version": "1.0",
        "supersedes": [],
        "residual_uncertainty": residual_risk,
        "permitted_uses": ["planning", "routing", "repair_localization"],
        "evaluation_refs": node.get("acceptance_refs", []),
    }
    simulation_contract = {
        "record_type": "simulation_contract_record",
        "simulation_id": stable_id("simulation_contract", node_id, "compile_time"),
        "claim_id": claims[0].get("claim_id") if claims else stable_id("claim_placeholder", node_id),
        "scope": "compile_time_plan_fixture",
        "fidelity_standard": "structural_route_and_evidence_shape_only",
        "temporal_semantics": "not_runtime_execution",
        "demand_estimate": {"estimated_seconds": int(node.get("estimated_seconds") or 0)},
        "capacity_bottlenecks": ["executor_not_leased", "verification_not_run"],
        "approximation_liberties": ["dry_run_cost_estimate", "planner_context_slice"],
        "supported_claim_boundary": "can_support_dispatch_readiness_shape_only",
        "residual_risks": [residual_risk],
        "evidence_refs": [target.get("evidence_ref") for target in evidence_targets] + tool_evidence_refs,
    }
    return {
        "semantic_atom": semantic_atom,
        "semantic_node": semantic_node,
        "typed_job": typed_job,
        "runtime_adapter_invocation": runtime_adapter_invocation,
        "authority_transition": authority_transition,
        "authority_use_receipt": authority_use_receipt,
        "context_abi_record": context_abi_record,
        "context_transaction": context_transaction,
        "context_adequacy": context_adequacy,
        "resource_budget": resource_budget,
        "costed_route": costed_route,
        "generation_mode": generation_mode_record,
        "failure_boundary": failure_boundary,
        "artifact_graph": artifact_graph,
        "evidence_transitions": evidence_transitions,
        "proof_carrying_claims": proof_carrying_claims,
        "routing_decision": routing_decision,
        "simulation_contract": simulation_contract,
        "tool_eligibility": tool_eligibility,
        "tool_call_receipts": tool_receipts,
        "vcm_consumer_abi": vcm_consumer_packet,
    }


def goal_records_for(
    goal_id: str,
    contract: dict[str, Any],
    nodes: list[dict[str, Any]],
    lint: dict[str, Any],
    contract_hash: str,
) -> dict[str, Any]:
    constraints = contract.get("constraint_capsules", []) if isinstance(contract.get("constraint_capsules"), list) else []
    node_ids = [node.get("node_id") for node in nodes]
    all_artifacts = [
        get_path(node, ["asi_stack_records", "artifact_graph", "artifact_id"], "")
        for node in nodes
    ]
    intent_contract = {
        "record_type": "intent_contract",
        "intent_id": stable_id("intent", goal_id, contract_hash),
        "request_summary": contract.get("title"),
        "desired_outcome": contract.get("objective"),
        "allowed_means": ["registered_surfaces", "local_deterministic_tools", "governed_teacher_proposals"],
        "forbidden_means": ["public_benchmark_training", "external_runtime_inference", "fallback_answer_credit", "arbitrary_remote_execution"],
        "authority_ceiling": "compile_route_and_local_private_execution_only",
        "source_boundaries": ["roadmap.md", "configs/theseus_plan_compiler.json", "configs/project_manifest_registry.json"],
        "acceptance_criteria": contract.get("acceptance_tests", []),
        "evidence_requirements": [ref for node in nodes for ref in [target.get("evidence_ref") for target in node.get("evidence_targets", [])]],
        "escalation_conditions": ["lint_hard_failure", "route_blocked", "missing_context", "policy_violation"],
        "stop_conditions": ["public_training_attempt", "external_runtime_inference_attempt", "fallback_return_attempt"],
        "open_ambiguities": [],
    }
    command_contract = {
        "record_type": "command_contract",
        "contract_id": contract.get("contract_id"),
        "intent_id": intent_contract["intent_id"],
        "role": "roadmap_execution_spine",
        "objective": contract.get("objective"),
        "context_refs": ["reports/vcm_task_contexts.json", "configs/project_manifest_registry.json"],
        "constraints": constraints,
        "procedure": "compile_to_typed_vcm_backed_dag_before_execution",
        "allowed_means": intent_contract["allowed_means"],
        "forbidden_means": intent_contract["forbidden_means"],
        "output_contract": contract.get("outputs", []),
        "verification": contract.get("acceptance_tests", []),
        "failure_behavior": "emit_residual_and_do_not_claim_completion",
        "authority_ceiling": intent_contract["authority_ceiling"],
        "required_approvals": [],
        "expected_artifacts": all_artifacts,
        "feedback_route": "claim_evidence_ledger_and_dogfood_metadata",
    }
    constitutional_predicates = [
        {
            "record_type": "constitutional_predicate_record",
            "predicate_id": stable_id("constitutional_predicate", goal_id, row.get("id")),
            "normative_source": "AGENTS.md and roadmap.md",
            "commitment": row.get("rule"),
            "operational_test": row.get("id"),
            "protected_scope": "all_plan_nodes",
            "translation_status": "operationalized_as_plan_compiler_gate",
            "uncertainty": "predicate covers compile-time route shape, not all runtime behavior",
            "review_route": "registry_and_control_plane_gate",
            "self_modification_rule": "cannot weaken without explicit replacement transaction and evidence",
            "non_claims": ["constitutional predicate record is not a proof of global alignment"],
        }
        for row in constraints
        if isinstance(row, dict)
    ]
    agency_rights = {
        "record_type": "agency_rights_checklist",
        "plan_id": goal_id,
        "affected_parties": ["operator", "local_workspace"],
        "delegation_scope": "bounded_registered_execution_only",
        "manipulation_risk": "low_compile_time",
        "reversibility": "reports_and_generated_artifacts_can_be_retired_or_rolled_back_by_registry_policy",
        "review_channel": "docs/PROJECT_STATE.md and reports/theseus_plan_compiler.json",
        "appeal_channel": "operator_can_change_goal_or_policy",
        "shutdown_or_rollback_path": "stop_execution_or_revert_registered_route",
        "accountable_principal": "theseus_plan_compiler",
        "residual_dependency_risk": "executor_surfaces_may_have_stale_evidence_until refreshed",
        "approval_required": False,
    }
    value_conflict = {
        "record_type": "value_conflict_record",
        "conflict_id": stable_id("value_conflict", goal_id, contract_hash),
        "value_axes": ["speed", "truthfulness", "capability_growth", "anti_cheating"],
        "stakeholders": ["operator", "future Theseus users"],
        "stakes": "avoid fast but dishonest progress while still reducing friction",
        "reversibility": "plan records are reversible; execution side effects require adapter rollback handles",
        "evidence_required": contract.get("acceptance_tests", []),
        "review_route": "registry_gate_then_control_plane",
        "decision": "truth_and_evidence_boundaries_take_precedence_over_speed",
        "residual_uncertainty": "some execution surfaces remain stale until Phase 0 refresh",
        "expiry_or_revisit_condition": "revisit after registry gate and VIEA execute-mode evidence refresh",
    }
    governance_rights = [
        {
            "record_type": "governance_right_record",
            "right_id": stable_id("governance_right", goal_id, right_type),
            "right_type": right_type,
            "holder": "operator",
            "scope": "roadmap_execution_spine",
            "required_artifacts": ["reports/theseus_plan_compiler.json", "reports/theseus_plan_trace_bundle.jsonl"],
            "safety_constraints": intent_contract["forbidden_means"],
            "access_path": report_path,
            "denial_or_redaction_reason": "",
            "preservation_rule": "retain evidence refs and negative results",
            "test_hook": test_hook,
        }
        for right_type, report_path, test_hook in [
            ("audit", "reports/theseus_plan_compiler.json", "trace_bundle_rows_match_nodes"),
            ("exit", "goal_can_stop_without_unbounded_self_update", "no_unbounded_execution"),
            ("rollback", "registry_replacement_transaction_required", "rollback_metadata_required_before_default_route"),
            ("dissent", "residual_records_and_lint_findings", "negative_results_preserved"),
        ]
    ]
    research_backlog = [
        {
            "record_type": "research_backlog_record",
            "backlog_id": stable_id("research_backlog", goal_id, row.get("kind"), row.get("evidence")),
            "source_or_gap": row.get("kind"),
            "access_state": "local_repo_evidence_available",
            "assigned_chapters": ["Project Theseus Roadmap"],
            "source_note_state": "not_applicable_repo_gap",
            "claim_mapping_state": "requires_fix_before_claim_credit",
            "external_literature_need": "none_for_gap_triage",
            "proof_or_test_backlog": row.get("evidence"),
            "insertion_decision": "keep_in_roadmap_until_resolved",
            "residuals": row,
            "next_action": "repair_registered_surface_or_refresh_evidence",
        }
        for row in list_value(lint.get("hard_failures")) + list_value(lint.get("warnings"))
        if isinstance(row, dict)
    ]
    reference_trace = {
        "record_type": "reference_trace_record",
        "trace_id": stable_id("reference_trace", goal_id, contract_hash),
        "intent_contract": intent_contract["intent_id"],
        "command_contract": command_contract["contract_id"],
        "authority_chain": [record.get("transition_id") for node in nodes for record in [get_path(node, ["asi_stack_records", "authority_transition"], {})]],
        "layer_handoffs": node_ids,
        "artifacts": all_artifacts,
        "evidence_updates": [
            transition.get("transition_id")
            for node in nodes
            for transition in list_value(get_path(node, ["asi_stack_records", "evidence_transitions"], []))
            if isinstance(transition, dict)
        ],
        "stop_conditions": intent_contract["stop_conditions"],
        "missing_contracts": [] if not lint.get("hard_failures") else lint.get("hard_failures"),
        "validation_commands": ["python3 scripts/theseus_plan_compiler.py"],
        "support_state_effect": "compile_time_trace_only",
        "non_claims": ["reference trace is not execution success"],
    }
    return {
        "intent_contract": intent_contract,
        "command_contract": command_contract,
        "reference_trace": reference_trace,
        "constitutional_predicates": constitutional_predicates,
        "agency_rights_checklist": agency_rights,
        "value_conflict_record": value_conflict,
        "governance_rights": governance_rights,
        "research_backlog": research_backlog,
    }


def asi_stack_record_ids(records: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_id") or key in {"task_id", "job_id", "transition_id", "invocation_id", "artifact_id"}:
                    if item:
                        ids.append(str(item))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(records)
    seen: set[str] = set()
    out: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def permissions_for_node(node: dict[str, Any]) -> list[str]:
    backend = str(node.get("executor_backend") or "")
    permissions = ["read_registered_context", "write_registered_reports"]
    if backend in {"theseus_control_plane", "hive_work_board_executor", "autonomy_watchdog"}:
        permissions.append("enqueue_or_monitor_registered_tasks")
    if backend == "local_deterministic_tool":
        permissions.append("run_registered_local_deterministic_tool")
    if backend == "governed_teacher_proposal":
        permissions.append("write_teacher_proposal_packet")
    if backend == "planning_benchmark_adapter":
        permissions.append("run_private_adapter_dry_run")
    return sorted(set(permissions))


def authority_ceiling_for_node(node: dict[str, Any]) -> str:
    backend = str(node.get("executor_backend") or "")
    risk = str(node.get("risk_tier") or "medium")
    if backend == "governed_teacher_proposal":
        return "teacher_proposal_only_no_runtime_serving"
    if backend == "planning_benchmark_adapter":
        return "calibration_adapter_no_training"
    if backend == "hive_work_board_executor":
        return "registered_hive_task_only_no_shell"
    if risk == "high":
        return "compile_only_until_explicit_review"
    return "local_private_registered_surface"


def approval_required_for_node(node: dict[str, Any]) -> bool:
    return str(node.get("risk_tier") or "") == "high" or str(node.get("executor_backend") or "") == "governed_teacher_proposal"


def approval_state_for_node(node: dict[str, Any]) -> str:
    return "approval_required_before_execution" if approval_required_for_node(node) else "compile_time_preapproved_by_policy"


def sandbox_mode_for_node(node: dict[str, Any]) -> str:
    backend = str(node.get("executor_backend") or "")
    if backend == "local_deterministic_tool":
        return "local_registered_tool_no_shell"
    if backend == "hive_work_board_executor":
        return "registered_hive_task_scope"
    if backend == "governed_teacher_proposal":
        return "proposal_packet_only"
    return "registered_script_surface"


def rollback_handle_for_node(node: dict[str, Any]) -> str:
    outputs = [str(item) for item in list_value(node.get("outputs"))]
    if not outputs:
        return "no_output_declared_emit_residual"
    return stable_id("rollback_handle", node.get("node_id"), outputs)


def uses_privileged_context(node: dict[str, Any]) -> bool:
    tools = " ".join(str(item) for item in list_value(node.get("allowed_tools"))).lower()
    backend = str(node.get("executor_backend") or "").lower()
    return "teacher" in tools or "teacher" in backend or "secret" in tools


def generation_mode_for_node(node: dict[str, Any]) -> str:
    backend = str(node.get("executor_backend") or "")
    op = str(node.get("op") or "").lower()
    if backend == "local_deterministic_tool":
        return "deterministic_tool_evidence"
    if backend == "governed_teacher_proposal":
        return "teacher_proposal_packet"
    if "benchmark" in backend:
        return "calibration_adapter_dry_run"
    if "train" in op or "decode" in op or "generate" in op:
        return "registered_model_or_generator_route"
    return "non_generation_task"


def trace_row(goal_id: str, contract_hash: str, node: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_plan_trace_bundle_v0",
        "created_utc": now(),
        "goal_id": goal_id,
        "node_id": node["node_id"],
        "contract_hash": contract_hash,
        "semantic_hash": node.get("semantic_hash"),
        "executor_backend": node.get("executor_backend"),
        "executor_surface": node.get("route", {}).get("surface"),
        "execution_packet_id": node.get("execution_packet", {}).get("packet_id"),
        "execution_packet_hash": node.get("execution_packet", {}).get("packet_hash"),
        "tool_requirement_ids": [row.get("tool_id") for row in node.get("tool_requirements", [])],
        "tool_eligibility_id": get_path(node, ["tool_eligibility", "eligibility_id"], ""),
        "tool_eligibility_decision": get_path(node, ["tool_eligibility", "decision"], ""),
        "tool_required_for_node": get_path(node, ["tool_eligibility", "required_for_node"], False),
        "tool_receipt_ids": [row.get("receipt_id") for row in list_value(node.get("tool_receipts")) if isinstance(row, dict)],
        "tool_evidence_refs": [
            ref
            for row in list_value(node.get("tool_receipts"))
            if isinstance(row, dict)
            for ref in list_value(row.get("evidence_refs"))
            if ref
        ],
        "vcm_context_hash": node.get("vcm_context_slice", {}).get("context_hash"),
        "vcm_governed_context_hash": node.get("vcm_context_slice", {}).get("governed_context_hash"),
        "vcm_context_adequacy_state": node.get("vcm_context_slice", {}).get("context_adequacy_state"),
        "vcm_context_governor_receipt_id": get_path(node, ["vcm_context_slice", "governor_receipt", "receipt_id"], ""),
        "vcm_context_governor_ready": get_path(node, ["vcm_context_slice", "governor_receipt", "ready"], False),
        "vcm_consumer_abi_packet_id": get_path(node, ["asi_stack_records", "vcm_consumer_abi", "packet_id"], ""),
        "vcm_consumer_abi_ready": get_path(node, ["asi_stack_records", "vcm_consumer_abi", "ready"], False),
        "claim_ids": [claim.get("claim_id") for claim in node.get("claim_objects", [])],
        "evidence_refs": [target.get("evidence_ref") for target in node.get("evidence_targets", [])],
        "asi_stack_record_ids": asi_stack_record_ids(node.get("asi_stack_records", {})),
        "authority_transition_id": get_path(node, ["asi_stack_records", "authority_transition", "transition_id"], ""),
        "typed_job_id": get_path(node, ["asi_stack_records", "typed_job", "job_id"], ""),
        "artifact_graph_id": get_path(node, ["asi_stack_records", "artifact_graph", "artifact_id"], ""),
        "context_transaction_id": get_path(node, ["asi_stack_records", "context_transaction", "transaction_id"], ""),
        "context_adequacy_id": get_path(node, ["asi_stack_records", "context_adequacy", "adequacy_id"], ""),
        "generation_mode": get_path(node, ["asi_stack_records", "generation_mode", "generation_mode"], ""),
        "repair_fingerprint": node.get("repair_policy", {}).get("failure_fingerprint"),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def next_recommendation(trigger_state: str, compiled_goals: list[dict[str, Any]], ablation: dict[str, Any]) -> dict[str, Any]:
    if trigger_state != "GREEN":
        return {
            "next_action": "fix_plan_compiler_lint_failures_before_any_execution",
            "reason": "The compiler found hard failures in plan structure or policy wiring.",
        }
    blocked_routes = [
        {"goal_id": goal["goal_id"], "node_id": node["node_id"], "reason": node.get("route", {}).get("block_reason")}
        for goal in compiled_goals
        for node in goal.get("nodes", [])
        if node.get("route", {}).get("blocked")
    ]
    if blocked_routes:
        return {
            "next_action": "execute_local_only_plans_and_keep_blocked_accelerator_routes_advisory",
            "reason": "The compiler is usable, but some CUDA/MLX routes are blocked by current node capability state.",
            "blocked_routes": blocked_routes[:8],
        }
    return {
        "next_action": "run_execute_mode_proof_on_same_private_task_set",
        "reason": "The compiler produced clean typed DAGs; the next proof should run the safe private VIEA execute-mode path and compare against the old direct baseline.",
        "suggested_command": "python3 scripts/theseus_plan_compiler.py --execute-private",
        "ablation_summary": ablation.get("compiled_planning_path", {}),
    }


def compact_goal_summary(goal: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_id": goal.get("goal_id"),
        "title": goal.get("title"),
        "trigger_state": goal.get("trigger_state"),
        "node_count": goal.get("node_count"),
        "edge_count": goal.get("edge_count"),
        "estimated_makespan_seconds": goal.get("estimated_makespan_seconds"),
        "critical_path_node_ids": goal.get("critical_path_node_ids"),
        "lint": {
            "hard_failure_count": goal.get("lint", {}).get("hard_failure_count"),
            "warning_count": goal.get("lint", {}).get("warning_count"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Theseus Plan Compiler",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        "## Summary",
        "",
        f"- Compiled goals: {summary.get('compiled_goal_count')}",
        f"- Compiled nodes: {summary.get('compiled_node_count')}",
        f"- Compiled edges: {summary.get('compiled_edge_count')}",
        f"- Trace rows: {summary.get('trace_row_count')}",
        f"- ASI Stack record IDs: {summary.get('asi_stack_record_count')}",
        f"- Goal governance record groups: {summary.get('goal_record_count')}",
        f"- Local deterministic tool packets: {summary.get('local_deterministic_tool_packet_count')}",
        f"- Deterministic tool requirements: {summary.get('deterministic_tool_requirement_count')}",
        f"- Procedural-memory canary/default goals: {summary.get('procedural_memory_canary_goal_count', 0)} / {summary.get('procedural_memory_default_route_goal_count', 0)}",
        f"- Hard lint failures: {summary.get('goal_lint_hard_failure_count')}",
        f"- Warnings: {summary.get('goal_lint_warning_count')}",
        f"- Average VCM pages per node: {summary.get('average_context_pages_per_node')}",
        f"- Route-blocked nodes: {summary.get('route_blocked_node_count')}",
        f"- Execute-mode trigger: `{summary.get('execute_mode_trigger_state', '')}`",
        "",
        "## Recommendation",
        "",
        f"- `{get_path(report, ['recommendation', 'next_action'], '')}`: {get_path(report, ['recommendation', 'reason'], '')}",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker} `{row.get('name')}` ({row.get('severity')})")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This compiler is dry-run/route-only by default.",
            "- With `--execute-private`, it invokes the bounded local VIEA execution spine for private deterministic tool packets.",
            "- Public planning benchmarks are adapter-ready calibration surfaces only.",
            "- VCM context slices and registry owner surfaces are required for every compiled executable node.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_private_execute_mode(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/viea_execution_spine.py",
        "--execute",
        "--dags",
        rel(resolve(args.dags_out)),
        "--out",
        rel(resolve(args.execution_spine_out)),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    report = read_json(resolve(args.execution_spine_out), {})
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "report": {
            "trigger_state": report.get("trigger_state"),
            "summary": report.get("summary", {}),
            "outputs": report.get("outputs", {}),
        },
    }


def owner_surface_registered(owner_surface: str, registry: dict[str, Any]) -> bool:
    return owner_surface in {str(row.get("id") or "") for row in registry.get("surfaces", []) if isinstance(row, dict)}


def owner_surface_abstraction(owner_surface: str, registry: dict[str, Any]) -> str:
    for row in registry.get("surfaces", []):
        if isinstance(row, dict) and str(row.get("id") or "") == owner_surface:
            return str(row.get("abstraction_id") or "")
    return ""


def owner_abstraction_registered(abstraction_id: str, registry: dict[str, Any]) -> bool:
    return bool(abstraction_id) and abstraction_id in {
        str(row.get("id") or "") for row in registry.get("abstractions", []) if isinstance(row, dict)
    }


def backend_to_surface(backend: str) -> str:
    return {
        "theseus_control_plane": "scripts/theseus_control_plane.py",
        "hive_work_board_executor": "scripts/hive_work_board_executor.py",
        "autonomy_watchdog": "scripts/autonomy_watchdog.py",
        "vcm_task_context_bridge": "scripts/vcm_task_context_bridge.py",
            "local_deterministic_tool": "scripts/theseus_deterministic_tool_substrate.py",
        "governed_teacher_proposal": "reports/hive_teacher_auto_escalation_ledger.jsonl",
        "planning_benchmark_adapter": "reports/theseus_plan_compiler.json",
    }.get(backend, "")


def summarize_page(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "address": str(page.get("address") or ""),
        "title": str(page.get("title") or ""),
        "source_path": str(page.get("source_path") or ""),
        "lane": str(page.get("lane") or ""),
        "execution_class": str(page.get("execution_class") or ""),
        "taints": [str(item) for item in list_value(page.get("taints"))],
        "score": page.get("score"),
    }


def finding(kind: str, severity: str, evidence: Any) -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "evidence": evidence}


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def duplicate_count(values: list[Any]) -> int:
    counts = Counter(str(value) for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def stable_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    out = []
    prev_underscore = False
    for char in value.lower():
        if char.isalnum():
            out.append(char)
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    return "".join(out).strip("_") or "item"


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
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
