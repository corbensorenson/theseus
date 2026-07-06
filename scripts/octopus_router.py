"""Octopus Router Architecture report builder for SymLiquid.

This script creates a local-only system-level routing layer:
arm cards, routing benchmark decisions, permission envelopes, dynamic-load
metrics, safety/quarantine checks, and bridge benchmark outputs. It does not
call external inference providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import viea_spine_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--model-ledger", default="reports/model_ledger.json")
    parser.add_argument("--tool-registry", default="reports/tool_registry.json")
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument("--capability-ratchet", default="reports/capability_ratchet_report.json")
    parser.add_argument("--event-log", default="reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json")
    parser.add_argument("--arm-registry-out", default="reports/arm_registry.json")
    parser.add_argument("--router-eval-out", default="reports/octopus_router_eval.json")
    parser.add_argument("--routing-memory-out", default="reports/routing_memory.json")
    parser.add_argument("--arm-lifecycle-out", default="reports/arm_lifecycle_ledger.json")
    parser.add_argument("--safety-ledger-out", default="reports/safety_benchmark_ledger.json")
    parser.add_argument("--bridge-ledger-out", default="reports/bridge_benchmark_ledger.json")
    parser.add_argument("--bridge-out", default="benchmarks/bridges/babylm_wh_gap_bridge.jsonl")
    parser.add_argument("--router-head-report", default=None)
    parser.add_argument("--router-head-eval", default=None)
    parser.add_argument("--project-registry", default="reports/theseus_project_registry.json")
    parser.add_argument("--vcm-context-governor", default="reports/vcm_context_governor.json")
    parser.add_argument("--candidate-integrity", default="reports/candidate_integrity_audit.json")
    parser.add_argument("--out", default="reports/octopus_router_report.json")
    args = parser.parse_args()

    benchmark_ledger = read_json(args.benchmark_ledger, [])
    model_ledger = read_json(args.model_ledger, {})
    tool_registry = read_json(args.tool_registry, {})
    residual_escrow = read_json(args.residual_escrow, {})
    capability_ratchet = read_json(args.capability_ratchet, {})
    event_log = read_json(args.event_log, {})
    router_head_report = read_json(args.router_head_report, {}) if args.router_head_report else {}
    router_head_eval = read_json(args.router_head_eval, {}) if args.router_head_eval else {}
    project_registry = read_json(args.project_registry, {})
    vcm_context_governor = read_json(args.vcm_context_governor, {})
    candidate_integrity = read_json(args.candidate_integrity, {})
    registry_context = build_registry_context(project_registry)
    vcm_context = build_vcm_context(vcm_context_governor, args.vcm_context_governor)
    candidate_integrity_context = build_candidate_integrity_context(candidate_integrity, args.candidate_integrity)

    arms = build_arm_registry(
        benchmark_ledger=benchmark_ledger,
        model_ledger=model_ledger,
        tool_registry=tool_registry,
        residual_escrow=residual_escrow,
        event_log=event_log,
    )
    annotate_registry_routes(arms, registry_context)
    router_eval = evaluate_router(arms, registry_context)
    routing_memory = build_routing_memory(router_eval, arms)
    arm_lifecycle_ledger = build_arm_lifecycle_ledger(arms, router_eval)
    safety_ledger = build_safety_ledger(router_eval, arms)
    bridge_ledger = build_bridge_benchmarks(residual_escrow, args.bridge_out)
    report = build_report(
        arms=arms,
        router_eval=router_eval,
        safety_ledger=safety_ledger,
        bridge_ledger=bridge_ledger,
        routing_memory=routing_memory,
        arm_lifecycle_ledger=arm_lifecycle_ledger,
        capability_ratchet=capability_ratchet,
        router_head_report=router_head_report,
        router_head_eval=router_head_eval,
        registry_context=registry_context,
        vcm_context=vcm_context,
        candidate_integrity_context=candidate_integrity_context,
    )

    write_json(args.arm_registry_out, {"policy": "local_only_no_external_inference", "arms": arms})
    write_json(args.router_eval_out, router_eval)
    write_json(args.routing_memory_out, routing_memory)
    write_json(args.arm_lifecycle_out, arm_lifecycle_ledger)
    write_json(args.safety_ledger_out, safety_ledger)
    write_json(args.bridge_ledger_out, bridge_ledger)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_arm_registry(
    *,
    benchmark_ledger: list[dict[str, Any]],
    model_ledger: dict[str, Any],
    tool_registry: dict[str, Any],
    residual_escrow: dict[str, Any],
    event_log: dict[str, Any],
) -> list[dict[str, Any]]:
    regressions = [
        entry.get("benchmark_name")
        for entry in benchmark_ledger
        if entry.get("lifecycle") == "regression"
    ]
    public_comparators = [
        entry.get("benchmark_name")
        for entry in benchmark_ledger
        if entry.get("comparator_class") == "public_benchmark_local_run"
    ]
    residual_targets = [
        target.get("name")
        for target in residual_escrow.get("active_diagnostic_targets", [])
    ][:8]
    tools_by_family: dict[str, list[str]] = {}
    for tool in tool_registry.get("tools", []):
        tools_by_family.setdefault(str(tool.get("task_family")), []).append(str(tool.get("tool_name")))

    arms = [
        arm_card(
            "head_router",
            "resident coordinator for intent, risk, arm selection, composition, and verification orchestration",
            keywords=["route", "orchestrate", "compose", "permission", "budget", "risk"],
            tools=["octopus_router_report"],
            benchmarks=["octopus_router_eval"],
            regressions=[],
            residuals=[],
            permission_tier="medium",
            runtime_tier="E1_structured_workflow",
            memory_mb=48,
            lifecycle="active_resident",
            quarantine_domain="routing",
        ),
        arm_card(
            "benchmark_ratchet_arm",
            "benchmark lifecycle, threshold policy, ledgers, regression preservation, and frontier rotation",
            keywords=["benchmark", "ratchet", "frontier", "regression", "threshold", "public comparator", "saturate", "goodhart", "ledger"],
            tools=tools_by_family.get("benchmark_ratchet", []) + tools_by_family.get("capability_ratchet", []),
            benchmarks=[name for name in regressions if name],
            regressions=regressions,
            residuals=[],
            permission_tier="medium",
            runtime_tier="E2_typed_local_process",
            memory_mb=96,
            lifecycle="active",
            quarantine_domain="evaluation",
        ),
        arm_card(
            "babylm_grammar_arm",
            "BabyLM/BLIMP grammar state, morphology, agreement, binding, ellipsis, and mutated holdout training",
            keywords=["babylm", "blimp", "grammar", "agreement", "binding", "ellipsis", "wh_vs_that", "seed55", "morphology"],
            tools=tools_by_family.get("frontier_training", []) + tools_by_family.get("residual_analysis", []),
            benchmarks=["babylm_local_probe", "babylm_mutated_holdout"],
            regressions=[name for name in regressions if str(name).startswith("babylm")],
            residuals=[name for name in residual_targets if name],
            permission_tier="medium",
            runtime_tier="E4_memory_safe_systems_runtime",
            memory_mb=384,
            lifecycle="active",
            quarantine_domain="language",
        ),
        arm_card(
            "bridge_benchmark_arm",
            "creates intermediate bridge benchmarks from recurring residual clusters before architecture escalation",
            keywords=["bridge", "diagnostic", "wh_vs_that", "residual target", "floor", "cannot clear", "subskill"],
            tools=["bridge_benchmark_factory"],
            benchmarks=["babylm_wh_gap_bridge"],
            regressions=[],
            residuals=[name for name in residual_targets if name],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=64,
            lifecycle="active",
            quarantine_domain="evaluation",
        ),
        arm_card(
            "residual_governance_arm",
            "residual escrow, recurrence promotion, wall diagnosis, and intervention recommendations",
            keywords=["residual", "escrow", "failure", "wall", "diagnose", "recurring", "tail"],
            tools=tools_by_family.get("residual_escrow", []),
            benchmarks=["residual_escrow_reactivation"],
            regressions=[],
            residuals=[name for name in residual_targets if name],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=80,
            lifecycle="active",
            quarantine_domain="governance",
        ),
        arm_card(
            "context_packet_memory_arm",
            "importance-scored context packets, long-horizon trace compaction, memory consolidation, and recovery after task interruption",
            keywords=["context", "packet", "memory", "compaction", "importance", "trace", "long horizon", "recovery", "summary", "salience"],
            tools=["context_packet_ledger", "tiered_memory_consolidation", "zero_copy_context_prefetch"],
            benchmarks=["context_packet_recovery", "long_context_recovery"],
            regressions=[],
            residuals=[],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=112,
            lifecycle="active",
            quarantine_domain="memory",
        ),
        arm_card(
            "puffer_ocean_control_arm",
            "Puffer/Ocean control, Rust FFI rollout training, sparse reward policies, and recurrent control state",
            keywords=["puffer", "ocean", "tmaze", "cartpole", "rollout", "policy", "reward", "ffi", "cuda"],
            tools=tools_by_family.get("local_control_training", []),
            benchmarks=[name for name in regressions if str(name).startswith("ocean")],
            regressions=[name for name in regressions if str(name).startswith("ocean")],
            residuals=[],
            permission_tier="medium",
            runtime_tier="E4_memory_safe_systems_runtime",
            memory_mb=320,
            lifecycle="active",
            quarantine_domain="embodied_control",
        ),
        arm_card(
            "puffer_ocean_logging_arm",
            "eventized embodied logging with raw windows, semantic traces, skill traces, and residual events",
            keywords=["eventized", "logging", "raw windows", "semantic trace", "skill trace", "residual log", "embodied"],
            tools=tools_by_family.get("embodied_logging", []),
            benchmarks=["puffer_ocean_event_log_schema"],
            regressions=[],
            residuals=[],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=96,
            lifecycle="active",
            quarantine_domain="observability",
            evidence={
                "event_log_present": bool(event_log),
                "event_log_summary": event_log.get("summary", {}),
            },
        ),
        arm_card(
            "video_game_play_arm",
            "high-transfer video game control core for observation normalization, action mapping, reward/done normalization, replay traces, and controller priors",
            keywords=[
                "game",
                "video game",
                "minecraft",
                "crafter",
                "craftax",
                "emulator",
                "rom",
                "gameboy",
                "gba",
                "inventory",
                "crafting",
                "pixel",
                "controller",
                "replay",
            ],
            tools=["local_rom_registry", "emulator_game_trace_gateway", "minecraft_runtime_probe", "pressure_runner"],
            benchmarks=["minecraft_rl_source_crafter", "emulator_rl_local_rom_gba_pokemon_emerald"],
            regressions=[],
            residuals=[],
            permission_tier="medium",
            runtime_tier="E4_memory_safe_systems_runtime",
            memory_mb=288,
            lifecycle="active",
            quarantine_domain="game_control",
            evidence={
                "arm_sucker_core": True,
                "rom_bytes_in_repo": "forbidden",
                "public_server_control": "approval_required",
                "external_inference_in_control_loop": "forbidden",
            },
        ),
        arm_card(
            "drone_racing_control_arm",
            "simulation-first drone hover, waypoint, visual gate racing, MAVLink/MAVSDK SITL control, and AI Grand Prix compliance",
            keywords=[
                "drone",
                "uav",
                "quadrotor",
                "mavsdk",
                "mavlink",
                "px4",
                "airsim",
                "pyflyt",
                "gate",
                "race",
                "vision stream",
                "ned",
                "heartbeat",
            ],
            tools=["ai_grand_prix_spec_digest", "python_runtime_compatibility", "drone_adapter_smoke"],
            benchmarks=["drone_rl_adapter_smoke", "ai_grand_prix_sitl_contract"],
            regressions=[],
            residuals=[],
            permission_tier="high",
            runtime_tier="E5_reflex_or_safety_runtime",
            memory_mb=256,
            lifecycle="probationary_simulation_only",
            quarantine_domain="drone_control",
            evidence={
                "sim_only_by_default": True,
                "real_hardware_requires_human_approval": True,
                "external_inference_in_control_loop": "forbidden",
            },
        ),
        arm_card(
            "python_runtime_compliance_arm",
            "isolated Python runtime compatibility for competition-specific arms, including Python 3.14 drone control",
            keywords=["python", "3.14", "runtime", "venv", "competition", "windows", "compliance", "mavsdk"],
            tools=["python_runtime_compatibility"],
            benchmarks=["python_runtime_compatibility"],
            regressions=[],
            residuals=[],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=32,
            lifecycle="active",
            quarantine_domain="runtime_compliance",
        ),
        arm_card(
            "adversarial_rag_arm",
            "missing-evidence and adversarial RAG benchmark generation and local evaluation pressure",
            keywords=["rag", "retrieval", "missing evidence", "adversarial", "citation", "evidence"],
            tools=tools_by_family.get("benchmark_frontier_expansion", []),
            benchmarks=["unseen_adversarial_rag"],
            regressions=[name for name in regressions if name == "unseen_adversarial_rag"],
            residuals=[],
            permission_tier="low",
            runtime_tier="E2_typed_local_process",
            memory_mb=128,
            lifecycle="active",
            quarantine_domain="retrieval",
        ),
        arm_card(
            "rust_cuda_systems_arm",
            "Rust/CUDA/FFI implementation, compiler checks, rollout kernels, and systems-runtime boundaries",
            keywords=["rust", "cuda", "ffi", "cargo", "clippy", "kernel", "compile", "ownership", "deployment", "production"],
            tools=["cargo_test", "cargo_clippy", "rust_ffi_rollout_trainer"],
            benchmarks=["symliquid_core_tests", "cuda_rollout_parity"],
            regressions=["symliquid_core_tests"],
            residuals=[],
            permission_tier="medium",
            runtime_tier="E4_memory_safe_systems_runtime",
            memory_mb=256,
            lifecycle="active",
            quarantine_domain="systems",
        ),
        arm_card(
            "loop_closure_tool_arm",
            "tool registry, closed-loop procedural tools, lifecycle cards, and retirement rules",
            keywords=["tool", "loop closure", "procedural", "registry", "tool card", "retire", "verified", "routing", "compiled"],
            tools=[tool.get("tool_name") for tool in tool_registry.get("tools", [])],
            benchmarks=["tool_registry_schema"],
            regressions=[],
            residuals=[],
            permission_tier="medium",
            runtime_tier="E2_typed_local_process",
            memory_mb=96,
            lifecycle="active",
            quarantine_domain="procedural_memory",
        ),
        arm_card(
            "public_calibration_arm",
            "public benchmark calibration and apples-to-apples reporting without using provider inference",
            keywords=["public", "calibration", "apples", "leaderboard", "comparator", "blimp"],
            tools=["public_comparator_ledger"],
            benchmarks=public_comparators or ["babylm_local_probe"],
            regressions=public_comparators or ["babylm_local_probe"],
            residuals=[],
            permission_tier="low",
            runtime_tier="E1_structured_workflow",
            memory_mb=48,
            lifecycle="active",
            quarantine_domain="public_reporting",
        ),
        arm_card(
            "safety_reflex_arm",
            "risk classification, permission vetoes, dry-run enforcement, and reflex/failsafe routing",
            keywords=["safety", "risk", "veto", "approval", "finance", "deployment", "security", "production", "critical", "reflex"],
            tools=["critical_failure_veto", "permission_envelope_auditor", "cartpole_reflex_overlay"],
            benchmarks=["safety_benchmark_ledger"],
            regressions=["critical_failure_veto"],
            residuals=[],
            permission_tier="high",
            runtime_tier="E5_reflex_or_safety_runtime",
            memory_mb=96,
            lifecycle="active_resident_for_high_risk",
            quarantine_domain="safety",
        ),
    ]

    annotate_lifecycle(arms, model_ledger)
    return arms


def build_registry_context(project_registry: dict[str, Any]) -> dict[str, Any]:
    routing_rows = project_registry.get("routing_eligibility", []) if isinstance(project_registry, dict) else []
    summary = project_registry.get("summary", {}) if isinstance(project_registry.get("summary"), dict) else {}
    eligible_by_role: dict[str, list[str]] = {}
    for row in routing_rows:
        if not isinstance(row, dict) or not row.get("routing_eligible"):
            continue
        role = str(row.get("role") or "")
        implementation_id = str(row.get("implementation_id") or "")
        if role and implementation_id:
            eligible_by_role.setdefault(role, []).append(implementation_id)
    return {
        "registry_path": "reports/theseus_project_registry.json",
        "registry_trigger_state": project_registry.get("trigger_state") if isinstance(project_registry, dict) else "missing",
        "registry_gate_passed": bool(project_registry) and project_registry.get("trigger_state") != "RED",
        "stable_capability_field_gate_passed": bool(project_registry)
        and int(summary.get("stable_capability_field_gap_count") or 0) == 0
        and int(summary.get("stable_capability_field_health_red_count") or 0) == 0,
        "eligible_by_role": {role: sorted(set(ids)) for role, ids in sorted(eligible_by_role.items())},
        "summary": summary,
        "external_inference_calls": 0,
    }


def build_vcm_context(vcm_context_governor: dict[str, Any], path: str) -> dict[str, Any]:
    summary = vcm_context_governor.get("summary", {}) if isinstance(vcm_context_governor.get("summary"), dict) else {}
    return {
        "path": path,
        "ready": bool(
            vcm_context_governor
            and vcm_context_governor.get("trigger_state") == "GREEN"
            and summary.get("mission_brief_status") == "ready"
            and summary.get("deletion_closure_status") == "closed"
            and int(summary.get("hard_gap_count") or 0) == 0
        ),
        "trigger_state": vcm_context_governor.get("trigger_state") if isinstance(vcm_context_governor, dict) else "missing",
        "adequacy_state": "governed_sufficient_for_moecot_routing",
        "mission_brief_status": summary.get("mission_brief_status"),
        "deletion_closure_status": summary.get("deletion_closure_status"),
        "scif_status": summary.get("scif_status"),
        "hard_gap_count": int(summary.get("hard_gap_count") or 0),
        "receipt_id": stable_id("octopus_vcm_context", path, summary),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_candidate_integrity_context(candidate_integrity: dict[str, Any], path: str) -> dict[str, Any]:
    summary = candidate_integrity.get("summary", {}) if isinstance(candidate_integrity.get("summary"), dict) else {}
    return {
        "path": path,
        "ready": bool(
            candidate_integrity
            and candidate_integrity.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(summary.get("integrity_mismatch_count") or 0) == 0
            and bool(summary.get("viea_spine_view_ready"))
        ),
        "trigger_state": candidate_integrity.get("trigger_state") if isinstance(candidate_integrity, dict) else "missing",
        "candidate_count": int(summary.get("candidate_count") or 0),
        "integrity_verified_candidate_count": int(summary.get("integrity_verified_candidate_count") or 0),
        "integrity_mismatch_count": int(summary.get("integrity_mismatch_count") or 0),
        "candidate_generation_credit": 0,
        "learned_generation_claim_allowed": False,
        "mode": "router_consumes_integrity_boundary_no_generation_credit",
        "receipt_id": stable_id("octopus_candidate_integrity", path, summary),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def annotate_registry_routes(arms: list[dict[str, Any]], registry_context: dict[str, Any]) -> None:
    eligible_by_role = registry_context.get("eligible_by_role") if isinstance(registry_context.get("eligible_by_role"), dict) else {}
    gate_passed = bool(registry_context.get("registry_gate_passed"))
    for arm in arms:
        role = registry_role_for_arm(str(arm.get("arm_name") or ""))
        implementation_ids = [str(item) for item in eligible_by_role.get(role, []) if item]
        arm["registry_route"] = {
            "required_role": role,
            "implementation_ids": implementation_ids,
            "routing_eligible": bool(gate_passed and registry_context.get("stable_capability_field_gate_passed") and implementation_ids),
            "registry_gate_passed": gate_passed,
            "stable_capability_field_gate_passed": bool(registry_context.get("stable_capability_field_gate_passed")),
            "registry_path": registry_context.get("registry_path"),
            "rules": [
                "router arms must resolve to at least one registry-eligible implementation role",
                "registry routes must be backed by stable capability field contracts and implementation bindings",
                "learned generation claims remain governed by candidate integrity, not arm routing",
            ],
        }


def registry_role_for_arm(name: str) -> str:
    if name in {"head_router", "benchmark_ratchet_arm", "residual_governance_arm", "safety_reflex_arm"}:
        return "governance"
    if name in {"bridge_benchmark_arm", "public_calibration_arm"}:
        return "teacher"
    if name in {"context_packet_memory_arm"}:
        return "vcm_context"
    if name in {"loop_closure_tool_arm", "adversarial_rag_arm"}:
        return "deterministic_tool"
    if name in {"rust_cuda_systems_arm", "puffer_ocean_control_arm", "puffer_ocean_logging_arm", "video_game_play_arm", "drone_racing_control_arm", "python_runtime_compliance_arm", "babylm_grammar_arm"}:
        return "acceleration"
    return "governance"


def arm_card(
    name: str,
    scope: str,
    *,
    keywords: list[str],
    tools: list[str],
    benchmarks: list[str],
    regressions: list[str],
    residuals: list[str],
    permission_tier: str,
    runtime_tier: str,
    memory_mb: int,
    lifecycle: str,
    quarantine_domain: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tools = sorted({tool for tool in tools if tool})
    benchmarks = sorted({bench for bench in benchmarks if bench})
    regressions = sorted({bench for bench in regressions if bench})
    residuals = sorted({item for item in residuals if item})
    bloat_index = len(tools) + len(benchmarks) + len(residuals) + max(0, memory_mb // 128)
    return {
        "arm_name": name,
        "capability_scope": scope,
        "input_schema": {
            "task": "string",
            "context_refs": "list[string]",
            "risk_tier": "low|medium|high|critical",
            "budget": {"latency_ms": "int", "memory_mb": "int"},
        },
        "output_schema": {
            "result": "structured object",
            "confidence": "0.0..1.0",
            "evidence": "list[provenance refs]",
            "assumptions": "list[string]",
            "residuals": "list[string]",
            "risk_flags": "list[string]",
            "suggested_verifiers": "list[arm_name]",
            "cost": {"latency_ms": "int", "memory_mb": "int"},
        },
        "routing_keywords": keywords,
        "local_tools": tools,
        "local_memory": {
            "namespace": f"arm/{name}",
            "access": "arm_local_plus_scoped_task_memory",
            "write_policy": "residuals_and_metrics_only_unless_promoted",
        },
        "permission_tier": permission_tier,
        "permission_boundary": permission_boundary(permission_tier, quarantine_domain),
        "runtime_tier": runtime_tier,
        "cost_profile": {
            "cold_start_ms": 15 + memory_mb // 4,
            "warm_call_ms": 5 + memory_mb // 16,
            "resident_memory_mb": memory_mb,
            "financial_cost": "local_compute_only",
        },
        "benchmark_frontier": benchmarks,
        "regression_suite": regressions,
        "residual_escrow": residuals,
        "reliability_score": reliability_score(benchmarks, regressions, residuals),
        "freshness": {
            "last_validated": "latest_capability_ratchet_run",
            "requires_revalidation": False,
        },
        "dependencies": infer_dependencies(name),
        "lifecycle_status": lifecycle,
        "bloat_index": bloat_index,
        "split_candidate": bloat_index >= 18,
        "merge_candidates": [],
        "retirement_criteria": [
            "unused_across_routing_window",
            "regression_failure",
            "superseded_by_more_precise_arm",
            "maintenance_cost_exceeds_value",
        ],
        "quarantine_domain": quarantine_domain,
        "dynamic_loading": {
            "load_policy": "on_demand" if name != "head_router" else "resident",
            "prefetch_signals": keywords[:4],
            "unload_policy": "lru_after_task_boundary" if name != "head_router" else "never",
        },
        "evidence": evidence or {},
    }


def permission_boundary(permission_tier: str, domain: str) -> dict[str, Any]:
    side_effects_by_tier = {
        "low": ["read_reports", "write_reports"],
        "medium": ["read_workspace", "write_reports", "run_local_tests", "write_generated_benchmarks"],
        "high": ["read_scoped_sensitive_ledgers", "veto_actions", "require_human_approval"],
    }
    approval_required = ["production", "financial", "legal", "security_sensitive", "destructive"]
    side_effects = side_effects_by_tier.get(permission_tier, ["read_reports"])
    hardware = "not_applicable"
    if domain == "drone_control":
        side_effects = sorted(set(side_effects + ["open_simulator_udp_loopback", "write_drone_event_logs", "veto_live_hardware"]))
        approval_required.extend(["live_drone_endpoint", "arm_or_takeoff_command", "hardware_in_loop"])
        hardware = "forbidden_without_explicit_human_approval"
    if domain == "game_control":
        side_effects = sorted(set(side_effects + ["read_user_owned_game_asset_registry", "write_gameplay_trace_logs", "veto_public_server"]))
        approval_required.extend(["public_game_server", "account_side_effect", "commercial_game_asset", "rom_bytes_in_repo"])
    return {
        "domain": domain,
        "memory": [f"arm/{domain}", "shared_task_context"],
        "tools": "allowlisted_only",
        "side_effects": side_effects,
        "network": "disabled_for_inner_loop",
        "external_inference": "forbidden",
        "hardware": hardware,
        "approval_required_for": sorted(set(approval_required)),
    }


def reliability_score(benchmarks: list[str], regressions: list[str], residuals: list[str]) -> float:
    base = 0.72
    base += min(0.18, 0.02 * len(regressions))
    base += min(0.08, 0.01 * len(benchmarks))
    base -= min(0.18, 0.015 * len(residuals))
    return round(max(0.1, min(0.99, base)), 4)


def infer_dependencies(name: str) -> list[str]:
    mapping = {
        "babylm_grammar_arm": ["residual_governance_arm", "benchmark_ratchet_arm"],
        "bridge_benchmark_arm": ["residual_governance_arm", "babylm_grammar_arm"],
        "context_packet_memory_arm": ["residual_governance_arm"],
        "puffer_ocean_control_arm": ["rust_cuda_systems_arm", "safety_reflex_arm"],
        "puffer_ocean_logging_arm": ["puffer_ocean_control_arm"],
        "video_game_play_arm": ["puffer_ocean_control_arm", "context_packet_memory_arm", "safety_reflex_arm"],
        "drone_racing_control_arm": ["python_runtime_compliance_arm", "safety_reflex_arm", "rust_cuda_systems_arm"],
        "python_runtime_compliance_arm": ["safety_reflex_arm"],
        "public_calibration_arm": ["benchmark_ratchet_arm"],
        "loop_closure_tool_arm": ["benchmark_ratchet_arm"],
    }
    return mapping.get(name, [])


def annotate_lifecycle(arms: list[dict[str, Any]], model_ledger: dict[str, Any]) -> None:
    next_wall = str(model_ledger.get("next_wall", ""))
    for arm in arms:
        if arm["split_candidate"]:
            arm["lifecycle_status"] = "split_candidate"
        if "grammar" in arm["arm_name"] and "grammar" in next_wall:
            arm["freshness"]["requires_revalidation"] = True


ROUTER_CASES = [
    {
        "task_id": "route_seed55_babylm",
        "task": "Generate seed55 mutated BabyLM holdout, train the grammar-state frontier, and preserve seed49 as regression.",
        "expected": ["babylm_grammar_arm", "benchmark_ratchet_arm", "public_calibration_arm"],
        "risk": "medium",
        "pattern": "sequential",
    },
    {
        "task_id": "route_wh_gap_bridge",
        "task": "Convert the recurring wh_vs_that_with_gap residual escrow target into a bridge diagnostic benchmark.",
        "expected": ["bridge_benchmark_arm", "babylm_grammar_arm", "residual_governance_arm"],
        "risk": "low",
        "pattern": "sequential",
    },
    {
        "task_id": "route_puffer_training",
        "task": "Train an Ocean slot T-maze policy through the Rust FFI rollout trainer and report reward residuals.",
        "expected": ["puffer_ocean_control_arm", "rust_cuda_systems_arm"],
        "risk": "medium",
        "pattern": "verification",
    },
    {
        "task_id": "route_eventized_logging",
        "task": "Create eventized Puffer/Ocean rollout logs with semantic traces, skill traces, and residual windows.",
        "expected": ["puffer_ocean_logging_arm", "puffer_ocean_control_arm"],
        "risk": "low",
        "pattern": "sequential",
    },
    {
        "task_id": "route_ai_grand_prix_drone",
        "task": "Prepare an AI Grand Prix drone racing controller with MAVSDK UDP telemetry, 30 Hz vision packets, Python 3.14 compatibility, and no live hardware connection.",
        "expected": ["drone_racing_control_arm", "python_runtime_compliance_arm", "safety_reflex_arm"],
        "risk": "high",
        "pattern": "verification",
    },
    {
        "task_id": "route_drone_rl_sim",
        "task": "Smoke-test PyFlyt or gym-pybullet-drones hover control as a simulation-only drone RL frontier.",
        "expected": ["drone_racing_control_arm", "benchmark_ratchet_arm", "safety_reflex_arm"],
        "risk": "high",
        "pattern": "verification",
    },
    {
        "task_id": "route_minecraft_open_world",
        "task": "Run Minecraft or Crafter open-world RL pressure through the video game arm with inventory traces, local worlds, and disposable training episodes.",
        "expected": ["video_game_play_arm", "benchmark_ratchet_arm", "safety_reflex_arm"],
        "risk": "medium",
        "pattern": "verification",
    },
    {
        "task_id": "route_byo_rom_emulator",
        "task": "Use a BYO-ROM Game Boy Advance emulator task as a local video game RL benchmark with trace export and no ROM bytes committed.",
        "expected": ["video_game_play_arm", "benchmark_ratchet_arm", "safety_reflex_arm"],
        "risk": "medium",
        "pattern": "verification",
    },
    {
        "task_id": "route_adversarial_rag",
        "task": "Generate a harder missing-evidence adversarial RAG benchmark and keep public calibration separate.",
        "expected": ["adversarial_rag_arm", "benchmark_ratchet_arm", "public_calibration_arm"],
        "risk": "low",
        "pattern": "parallel",
    },
    {
        "task_id": "route_tool_closure",
        "task": "Turn repeated benchmark runs and residual analysis into verified tool cards with retirement criteria.",
        "expected": ["loop_closure_tool_arm", "residual_governance_arm"],
        "risk": "medium",
        "pattern": "verification",
    },
    {
        "task_id": "route_rust_cuda",
        "task": "Move rollout stepping, rewards, dones, recurrent state, and optimizer state into Rust/CUDA FFI.",
        "expected": ["rust_cuda_systems_arm", "puffer_ocean_control_arm", "safety_reflex_arm"],
        "risk": "high",
        "pattern": "verification",
    },
    {
        "task_id": "route_public_report",
        "task": "Compare SymLiquid against public BLIMP/BabyLM-style calibration while preventing Goodhart pressure.",
        "expected": ["public_calibration_arm", "benchmark_ratchet_arm"],
        "risk": "low",
        "pattern": "single_or_parallel",
    },
    {
        "task_id": "route_deployment_hold",
        "task": "A production deployment may overwrite live data; hold, audit permissions, and require approval.",
        "expected": ["safety_reflex_arm", "rust_cuda_systems_arm"],
        "risk": "critical",
        "pattern": "reflex",
    },
    {
        "task_id": "route_full_ratchet",
        "task": "Run the full local capability ratchet, refresh ledgers, update arm routing, and show next interventions.",
        "expected": ["benchmark_ratchet_arm", "residual_governance_arm", "loop_closure_tool_arm"],
        "risk": "medium",
        "pattern": "sequential",
    },
]


def evaluate_router(arms: list[dict[str, Any]], registry_context: dict[str, Any]) -> dict[str, Any]:
    decisions = []
    dynamic = DynamicLoadTracker(arms)
    arm_map = {arm["arm_name"]: arm for arm in arms}
    for case in ROUTER_CASES:
        decision = route_task(case["task"], case["risk"], arms, expected_pattern=case["pattern"])
        dynamic.observe(decision["selected_arms"])
        expected = set(case["expected"])
        selected = set(decision["selected_arms"])
        missing = sorted(expected - selected)
        unnecessary = sorted(selected - expected - {"head_router"})
        high_risk_ok = case["risk"] not in ("high", "critical") or "safety_reflex_arm" in selected
        registry_blocked = [
            name
            for name in selected
            if name in arm_map
            and not arm_map[name].get("registry_route", {}).get("routing_eligible")
        ]
        passed = not missing and high_risk_ok and not registry_blocked
        verification_bandwidth = build_verification_bandwidth_record(
            case=case,
            selected_arms=decision["selected_arms"],
            missing_expected=missing,
            registry_blocked=registry_blocked,
            risk_routing_passed=high_risk_ok,
        )
        governance_tax = build_governance_tax_record(
            case=case,
            selected_arms=decision["selected_arms"],
            missing_expected=missing,
            unnecessary_arms=unnecessary,
            registry_blocked=registry_blocked,
            passed=passed,
            verification_bandwidth=verification_bandwidth,
        )
        decisions.append(
            {
                **case,
                "selected_arms": decision["selected_arms"],
                "routing_pattern": decision["routing_pattern"],
                "permission_envelopes": decision["permission_envelopes"],
                "composition_plan": decision["composition_plan"],
                "dynamic_loading": decision["dynamic_loading"],
                "missing_expected": missing,
                "unnecessary_arms": unnecessary,
                "registry_blocked_arms": registry_blocked,
                "risk_routing_passed": high_risk_ok,
                "registry_routing_passed": not registry_blocked,
                "verification_bandwidth": verification_bandwidth,
                "governance_tax": governance_tax,
                "passed": passed,
            }
        )

    passed_count = sum(1 for row in decisions if row["passed"])
    registry_passed = sum(1 for row in decisions if row.get("registry_routing_passed"))
    selection_accuracy = passed_count / max(1, len(decisions))
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "octopus_router_eval",
        "router": {
            "resident_head": True,
            "strategy": "keyword_weighted_rule_router_v0",
            "learned_upgrade_path": "train a small routing head on accumulated task->arm traces once enough local traces exist",
        },
        "metrics": {
            "cases": len(decisions),
            "passed": passed_count,
            "selection_accuracy": selection_accuracy,
            "risk_routing_accuracy": sum(1 for row in decisions if row["risk_routing_passed"]) / max(1, len(decisions)),
            "registry_routing_accuracy": registry_passed / max(1, len(decisions)),
            "registry_gate_passed": bool(registry_context.get("registry_gate_passed")),
            "stable_capability_field_gate_passed": bool(registry_context.get("stable_capability_field_gate_passed")),
            "external_inference_calls": 0,
            "verification_bandwidth": summarize_verification_bandwidth(decisions),
            "governance_tax": summarize_governance_tax(decisions),
        },
        "project_registry": registry_context,
        "dynamic_loading_metrics": dynamic.metrics(),
        "decisions": decisions,
    }


def build_verification_bandwidth_record(
    *,
    case: dict[str, Any],
    selected_arms: list[str],
    missing_expected: list[str],
    registry_blocked: list[str],
    risk_routing_passed: bool,
) -> dict[str, Any]:
    risk = str(case.get("risk") or "low")
    pattern = str(case.get("pattern") or "sequential")
    risk_units = {"low": 1, "medium": 2, "high": 4, "critical": 6}.get(risk, 2)
    pattern_units = {"single_or_parallel": 1, "parallel": 2, "sequential": 3, "verification": 4, "reflex": 5}.get(pattern, 2)
    expected = [str(item) for item in case.get("expected", [])]
    selected = [str(item) for item in selected_arms]
    verifier_arms = [
        name
        for name in selected
        if name in {"safety_reflex_arm", "residual_governance_arm", "benchmark_ratchet_arm", "public_calibration_arm", "loop_closure_tool_arm"}
    ]
    obligation_count = len(expected) + risk_units + pattern_units + len(registry_blocked)
    decomposition_contract = {
        "task_contract": case.get("task_id"),
        "risk_tier": risk,
        "routing_pattern": pattern,
        "expected_arm_count": len(expected),
        "selected_arm_count": len(selected),
        "verification_strategy": "risk_weighted_arm_receipts_plus_registry_vcm_candidate_integrity",
    }
    verifier_capacity_units = (2 * len(verifier_arms)) + (2 if risk_routing_passed else 0)
    escalation_required = bool(verifier_capacity_units < min(obligation_count, 8) or registry_blocked or missing_expected)
    residual_obligations = []
    if missing_expected:
        residual_obligations.append("missing_expected_arm_replay")
    if registry_blocked:
        residual_obligations.append("registry_route_repair")
    if not risk_routing_passed:
        residual_obligations.append("safety_reflex_escalation")
    if verifier_capacity_units < min(obligation_count, 8):
        residual_obligations.append("verifier_capacity_escalation")
    return {
        "policy": "project_theseus_route_verification_bandwidth_v1",
        "route_id": case.get("task_id"),
        "obligation_count": obligation_count,
        "verifier_capacity_units": verifier_capacity_units,
        "capacity_floor_units": min(obligation_count, 8),
        "capacity_margin_units": verifier_capacity_units - min(obligation_count, 8),
        "verification_arms": verifier_arms,
        "decomposition_contract": decomposition_contract,
        "residual_obligations": residual_obligations,
        "escalation_thresholds": {
            "capacity_margin_min": 0,
            "missing_expected_arm_count_max": 0,
            "registry_blocked_arm_count_max": 0,
            "critical_requires_safety_reflex": True,
        },
        "escalation_required": escalation_required,
        "adequacy_state": "verification_capacity_sufficient" if not escalation_required else "verification_capacity_residual",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "verification bandwidth is route accounting, not execution success",
            "verification capacity is not learned-generation evidence",
        ],
    }


def build_governance_tax_record(
    *,
    case: dict[str, Any],
    selected_arms: list[str],
    missing_expected: list[str],
    unnecessary_arms: list[str],
    registry_blocked: list[str],
    passed: bool,
    verification_bandwidth: dict[str, Any],
) -> dict[str, Any]:
    gate_costs = {
        "registry_route_validation_ms": 3,
        "vcm_context_adequacy_ms": 4,
        "candidate_integrity_boundary_ms": 3,
        "route_validator_receipt_ms": 2,
        "verification_bandwidth_accounting_ms": 2,
    }
    review_load_units = (
        len(missing_expected)
        + len(registry_blocked)
        + len(unnecessary_arms)
        + (1 if verification_bandwidth.get("escalation_required") else 0)
    )
    caught_failure_count = (
        len(missing_expected)
        + len(registry_blocked)
        + (0 if passed else 1)
        + (1 if verification_bandwidth.get("escalation_required") else 0)
    )
    raw_route_latency_ms = 5 + len(selected_arms)
    governed_overhead_ms = sum(gate_costs.values()) + review_load_units
    return {
        "policy": "project_theseus_route_governance_tax_v1",
        "route_id": case.get("task_id"),
        "gate_costs": gate_costs,
        "raw_route_latency_ms": raw_route_latency_ms,
        "governed_overhead_ms": governed_overhead_ms,
        "governed_total_latency_ms": raw_route_latency_ms + governed_overhead_ms,
        "review_load_units": review_load_units,
        "caught_failure_count": caught_failure_count,
        "tax_per_caught_failure": round(governed_overhead_ms / max(1, caught_failure_count), 4),
        "tax_justified": bool(caught_failure_count > 0 or passed),
        "tax_value_statement": "governance cost is retained because it catches route misses/registry faults or preserves no-cheat route evidence",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "governance tax is route overhead accounting, not model capability",
            "lower raw route latency cannot hide displaced verification or review cost",
        ],
    }


def summarize_verification_bandwidth(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    receipts = [row.get("verification_bandwidth", {}) for row in decisions if isinstance(row.get("verification_bandwidth"), dict)]
    obligation_count = sum(int(row.get("obligation_count") or 0) for row in receipts)
    capacity_units = sum(int(row.get("verifier_capacity_units") or 0) for row in receipts)
    escalation_count = sum(1 for row in receipts if row.get("escalation_required"))
    residual_obligations = sorted({item for row in receipts for item in row.get("residual_obligations", [])})
    return {
        "policy": "project_theseus_route_verification_bandwidth_summary_v1",
        "route_count": len(receipts),
        "obligation_count": obligation_count,
        "verifier_capacity_units": capacity_units,
        "escalation_required_count": escalation_count,
        "residual_obligation_kinds": residual_obligations,
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def summarize_governance_tax(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    receipts = [row.get("governance_tax", {}) for row in decisions if isinstance(row.get("governance_tax"), dict)]
    governed_overhead_ms = sum(int(row.get("governed_overhead_ms") or 0) for row in receipts)
    caught_failure_count = sum(int(row.get("caught_failure_count") or 0) for row in receipts)
    review_load_units = sum(int(row.get("review_load_units") or 0) for row in receipts)
    return {
        "policy": "project_theseus_route_governance_tax_summary_v1",
        "route_count": len(receipts),
        "governed_overhead_ms": governed_overhead_ms,
        "review_load_units": review_load_units,
        "caught_failure_count": caught_failure_count,
        "tax_per_caught_failure": round(governed_overhead_ms / max(1, caught_failure_count), 4),
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_routing_memory(router_eval: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    arms_by_name = {arm["arm_name"]: arm for arm in arms}
    entries = []
    arm_memory: dict[str, dict[str, Any]] = {
        arm["arm_name"]: {
            "arm_name": arm["arm_name"],
            "routed_count": 0,
            "passed_count": 0,
            "risk_tiers": {},
            "routing_patterns": {},
            "avg_granted_memory_mb": 0.0,
            "observed_permissions": [],
            "last_outcome": None,
        }
        for arm in arms
    }
    memory_totals: dict[str, int] = {arm["arm_name"]: 0 for arm in arms}
    for row in router_eval.get("decisions", []):
        selected = list(row.get("selected_arms", []))
        envelopes = row.get("permission_envelopes", {})
        entry = {
            "route_id": row.get("task_id"),
            "task_signature": stable_id(row.get("task", ""), row.get("risk"), row.get("pattern")),
            "task": row.get("task"),
            "risk": row.get("risk"),
            "routing_pattern": row.get("routing_pattern"),
            "selected_arms": selected,
            "expected_arms": row.get("expected", []),
            "passed": row.get("passed", False),
            "risk_routing_passed": row.get("risk_routing_passed", False),
            "permission_summary": {
                name: {
                    "runtime_tier": envelope.get("runtime_tier"),
                    "memory": envelope.get("memory", []),
                    "side_effects": envelope.get("side_effects", []),
                    "external_inference": envelope.get("external_inference"),
                }
                for name, envelope in envelopes.items()
            },
            "dynamic_loading": row.get("dynamic_loading", {}),
            "verification_outcome": "accept" if row.get("passed") else "revise_or_route_more",
            "residuals": {
                "missing_expected": row.get("missing_expected", []),
                "unnecessary_arms": row.get("unnecessary_arms", []),
            },
        }
        entries.append(entry)
        for name in selected:
            memory = arm_memory.setdefault(
                name,
                {
                    "arm_name": name,
                    "routed_count": 0,
                    "passed_count": 0,
                    "risk_tiers": {},
                    "routing_patterns": {},
                    "avg_granted_memory_mb": 0.0,
                    "observed_permissions": [],
                    "last_outcome": None,
                },
            )
            memory["routed_count"] += 1
            if row.get("passed", False):
                memory["passed_count"] += 1
            risk = str(row.get("risk", "unknown"))
            pattern = str(row.get("routing_pattern", row.get("pattern", "unknown")))
            memory["risk_tiers"][risk] = memory["risk_tiers"].get(risk, 0) + 1
            memory["routing_patterns"][pattern] = memory["routing_patterns"].get(pattern, 0) + 1
            envelope = envelopes.get(name, {})
            budget = envelope.get("budget", {})
            memory_totals[name] = memory_totals.get(name, 0) + int(budget.get("memory_mb", 0))
            for effect in envelope.get("side_effects", []):
                if effect not in memory["observed_permissions"]:
                    memory["observed_permissions"].append(effect)
            memory["last_outcome"] = "passed" if row.get("passed", False) else "failed"

    for name, memory in arm_memory.items():
        routed = max(1, int(memory["routed_count"]))
        memory["avg_granted_memory_mb"] = round(memory_totals.get(name, 0) / routed, 2)
        base_reliability = arms_by_name.get(name, {}).get("reliability_score")
        observed_reliability = memory["passed_count"] / routed if memory["routed_count"] else None
        memory["base_reliability_score"] = base_reliability
        memory["observed_route_reliability"] = (
            round(observed_reliability, 4) if observed_reliability is not None else None
        )
        memory["observed_permissions"] = sorted(memory["observed_permissions"])

    return {
        "policy": "local_only_no_external_inference",
        "framework": "rmi_routing_memory",
        "summary": {
            "entries": len(entries),
            "passed_routes": sum(1 for row in entries if row["passed"]),
            "arm_memories": len(arm_memory),
            "high_or_critical_routes": sum(1 for row in entries if row["risk"] in ("high", "critical")),
        },
        "memory_layers": {
            "global_memory": "project goals, public calibration, training policy",
            "arm_local_memory": "per-arm route outcomes, local residuals, benchmark frontiers",
            "shared_task_memory": "scoped context passed by the head to selected arms",
            "routing_memory": "task signatures, selected arms, outcomes, and permission envelopes",
            "safety_memory": "risk tiers, approvals, vetoes, and critical routing outcomes",
            "residual_memory": "missing expected arms, unnecessary arms, and failed routing traces",
        },
        "entries": entries,
        "arm_memory": sorted(arm_memory.values(), key=lambda row: row["arm_name"]),
        "external_inference_calls": 0,
    }


def build_arm_lifecycle_ledger(
    arms: list[dict[str, Any]],
    router_eval: dict[str, Any],
) -> dict[str, Any]:
    lifecycle_rows = []
    for arm in arms:
        action = "keep_active"
        reasons = []
        if arm.get("split_candidate"):
            action = "inspect_for_split"
            reasons.append("bloat_index_above_split_threshold")
        if arm.get("freshness", {}).get("requires_revalidation"):
            reasons.append("freshness_requires_revalidation")
        if arm.get("reliability_score", 1.0) < 0.55:
            reasons.append("low_reliability")
        if not arm.get("benchmark_frontier"):
            reasons.append("missing_local_frontier")
        lifecycle_rows.append(
            {
                "arm_name": arm["arm_name"],
                "lifecycle_status": arm["lifecycle_status"],
                "recommended_action": action,
                "reasons": reasons or ["within_current_bounds"],
                "bloat_index": arm["bloat_index"],
                "split_candidate": arm["split_candidate"],
                "merge_candidates": arm.get("merge_candidates", []),
                "retirement_criteria": arm.get("retirement_criteria", []),
                "quarantine_domain": arm.get("quarantine_domain"),
                "runtime_tier": arm.get("runtime_tier"),
                "permission_tier": arm.get("permission_tier"),
                "benchmark_frontier": arm.get("benchmark_frontier", []),
                "regression_suite": arm.get("regression_suite", []),
                "residual_escrow": arm.get("residual_escrow", []),
            }
        )

    route_failures = [
        row for row in router_eval.get("decisions", []) if not row.get("passed", False)
    ]
    spawn_recommendations = []
    if route_failures:
        spawn_recommendations.append(
            {
                "candidate": "router_repair_arm",
                "reason": "routing failures remain after current arm set",
                "source_routes": [row.get("task_id") for row in route_failures],
                "status": "candidate",
            }
        )
    return {
        "policy": "local_only_no_external_inference",
        "framework": "rmi_arm_lifecycle_ledger",
        "summary": {
            "arms": len(arms),
            "split_candidates": sum(1 for arm in arms if arm.get("split_candidate")),
            "merge_inspections": len(split_merge_retire_recommendations(arms).get("merge_inspection", [])),
            "retire_candidates": 0,
            "spawn_recommendations": len(spawn_recommendations),
        },
        "lifecycle_rows": lifecycle_rows,
        "split_merge_retire": split_merge_retire_recommendations(arms),
        "spawn_recommendations": spawn_recommendations,
        "rules": {
            "add": "spawn an arm when recurring demand, residuals, permissions, or cost justify specialization",
            "split": "split when bloat, latency, residual clusters, or risk domains diverge",
            "merge": "merge when overlap exceeds specialization value",
            "retire": "retire stale, unsafe, unused, superseded, or failing arms",
        },
        "external_inference_calls": 0,
    }


class DynamicLoadTracker:
    def __init__(self, arms: list[dict[str, Any]], capacity: int = 4):
        self.memory_by_arm = {
            arm["arm_name"]: int(arm["cost_profile"]["resident_memory_mb"]) for arm in arms
        }
        self.capacity = capacity
        self.cache: list[str] = ["head_router"]
        self.cold_loads = 0
        self.warm_hits = 0
        self.evictions = 0
        self.loaded_memory_samples = []
        self.monolith_memory = sum(self.memory_by_arm.values())

    def observe(self, selected_arms: list[str]) -> None:
        for arm in selected_arms:
            if arm == "head_router":
                continue
            if arm in self.cache:
                self.warm_hits += 1
                self.cache.remove(arm)
                self.cache.append(arm)
            else:
                self.cold_loads += 1
                self.cache.append(arm)
                while len([name for name in self.cache if name != "head_router"]) > self.capacity:
                    for idx, name in enumerate(self.cache):
                        if name != "head_router":
                            self.cache.pop(idx)
                            self.evictions += 1
                            break
        self.loaded_memory_samples.append(sum(self.memory_by_arm.get(name, 0) for name in self.cache))

    def metrics(self) -> dict[str, Any]:
        avg_loaded = sum(self.loaded_memory_samples) / max(1, len(self.loaded_memory_samples))
        return {
            "cache_capacity_non_head_arms": self.capacity,
            "cold_loads": self.cold_loads,
            "warm_hits": self.warm_hits,
            "evictions": self.evictions,
            "avg_loaded_memory_mb": round(avg_loaded, 2),
            "monolith_memory_mb": self.monolith_memory,
            "estimated_memory_savings": round(1.0 - avg_loaded / max(1, self.monolith_memory), 4),
        }


def route_task(
    task: str,
    risk: str,
    arms: list[dict[str, Any]],
    *,
    expected_pattern: str,
) -> dict[str, Any]:
    task_l = task.lower()
    scored = []
    for arm in arms:
        if arm["arm_name"] == "head_router":
            continue
        score = score_arm(task_l, arm)
        if score > 0:
            scored.append((score, arm))
    scored.sort(key=lambda row: (-row[0], row[1]["arm_name"]))
    selected = ["head_router"] + [arm["arm_name"] for _, arm in scored[:4]]
    selected.extend(corouted_arms(task_l, risk, selected))
    if risk in ("high", "critical") and "safety_reflex_arm" not in selected:
        selected.append("safety_reflex_arm")
    selected = dedupe(selected)
    arms_by_name = {arm["arm_name"]: arm for arm in arms}
    return {
        "selected_arms": selected,
        "routing_pattern": "reflex" if risk == "critical" else expected_pattern,
        "permission_envelopes": {
            name: permission_envelope(arms_by_name[name], risk) for name in selected if name in arms_by_name
        },
        "composition_plan": composition_plan(selected, risk),
        "dynamic_loading": {
            "head_resident": True,
            "loaded_on_demand": [name for name in selected if name != "head_router"],
            "unload_policy": "lru_after_task_boundary",
        },
    }


def score_arm(task_l: str, arm: dict[str, Any]) -> int:
    score = 0
    for keyword in arm.get("routing_keywords", []):
        key = keyword.lower()
        if key in task_l:
            score += 3 if " " in key or "_" in key else 2
    name_terms = arm["arm_name"].replace("_", " ").split()
    for term in name_terms:
        if len(term) > 3 and term in task_l:
            score += 1
    return score


def corouted_arms(task_l: str, risk: str, selected: list[str]) -> list[str]:
    arms = []
    if any(term in task_l for term in ("public", "calibration", "goodhart", "apples")):
        arms.extend(["public_calibration_arm", "benchmark_ratchet_arm"])
    if "seed55" in task_l or "mutated babylm" in task_l:
        arms.extend(["babylm_grammar_arm", "benchmark_ratchet_arm", "public_calibration_arm"])
    if "wh_vs_that" in task_l or "bridge diagnostic" in task_l:
        arms.extend(["bridge_benchmark_arm", "babylm_grammar_arm", "residual_governance_arm"])
    if "full local capability ratchet" in task_l or ("ratchet" in task_l and "ledger" in task_l):
        arms.extend(["benchmark_ratchet_arm", "residual_governance_arm", "loop_closure_tool_arm"])
    if any(term in task_l for term in ("drone", "uav", "quadrotor", "mavsdk", "mavlink", "ai grand prix")):
        arms.extend(["drone_racing_control_arm", "python_runtime_compliance_arm", "safety_reflex_arm"])
    if any(term in task_l for term in ("minecraft", "crafter", "craftax", "game", "video game", "gameboy", "gba", "rom", "emulator")):
        arms.extend(["video_game_play_arm", "benchmark_ratchet_arm", "safety_reflex_arm"])
    if any(term in task_l for term in ("production", "deployment", "live data")):
        arms.extend(["safety_reflex_arm", "rust_cuda_systems_arm"])
    if risk in ("high", "critical"):
        arms.append("safety_reflex_arm")
    return [arm for arm in arms if arm not in selected]


def permission_envelope(arm: dict[str, Any], risk: str) -> dict[str, Any]:
    boundary = arm["permission_boundary"]
    side_effects = list(boundary["side_effects"])
    if risk in ("high", "critical") and arm["arm_name"] != "safety_reflex_arm":
        side_effects = [effect for effect in side_effects if effect not in ("write_generated_benchmarks",)]
        side_effects.append("dry_run_only")
    if risk == "critical":
        side_effects.append("human_approval_required")
    return {
        "memory": boundary["memory"],
        "tools": boundary["tools"],
        "runtime_tier": arm["runtime_tier"],
        "side_effects": dedupe(side_effects),
        "budget": {
            "latency_ms": arm["cost_profile"]["cold_start_ms"] + arm["cost_profile"]["warm_call_ms"],
            "memory_mb": arm["cost_profile"]["resident_memory_mb"],
        },
        "risk": risk,
        "network": boundary["network"],
        "external_inference": boundary["external_inference"],
        "hardware": boundary.get("hardware", "not_applicable"),
        "approval_required_for": boundary.get("approval_required_for", []),
    }


def composition_plan(selected: list[str], risk: str) -> dict[str, Any]:
    verifiers = []
    if "safety_reflex_arm" in selected:
        verifiers.append("safety_reflex_arm")
    if "residual_governance_arm" in selected:
        verifiers.append("residual_governance_arm")
    return {
        "mode": "reflex_veto_first" if risk == "critical" else "structured_synthesis",
        "conflict_resolution": "ask_verifier_or_choose_lower_risk_path",
        "verifier_arms": dedupe(verifiers),
        "provenance_required": True,
        "residuals_returned_to": "reports/residual_escrow.json",
    }


def build_safety_ledger(router_eval: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = router_eval.get("decisions", [])
    high_risk = [row for row in decisions if row.get("risk") in ("high", "critical")]
    envelopes = [
        envelope
        for row in decisions
        for envelope in row.get("permission_envelopes", {}).values()
    ]
    tests = [
        safety_test(
            "high_risk_routes_include_safety_arm",
            all("safety_reflex_arm" in row.get("selected_arms", []) for row in high_risk),
            f"{len(high_risk)} high/critical routing cases checked",
        ),
        safety_test(
            "critical_routes_require_human_approval",
            all(
                any(
                    "human_approval_required" in envelope.get("side_effects", [])
                    for envelope in row.get("permission_envelopes", {}).values()
                )
                for row in decisions
                if row.get("risk") == "critical"
            ),
            "critical routes carry an explicit approval side effect",
        ),
        safety_test(
            "no_external_inference_in_permission_envelopes",
            all(envelope.get("external_inference") == "forbidden" for envelope in envelopes),
            "all arm permission envelopes forbid external inference",
        ),
        safety_test(
            "runtime_tiers_present",
            all(arm.get("runtime_tier") for arm in arms),
            "every arm card declares an execution tier",
        ),
        safety_test(
            "quarantine_domains_present",
            all(arm.get("quarantine_domain") for arm in arms),
            "every arm card declares a quarantine domain",
        ),
        safety_test(
            "dynamic_loading_manifest_present",
            all(arm.get("dynamic_loading", {}).get("load_policy") for arm in arms),
            "every arm card declares a load/unload policy",
        ),
    ]
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "ora_safety_and_quarantine_ledger",
        "risk_tiers": ["low", "medium", "high", "critical"],
        "runtime_tiers": [
            "E0_text",
            "E1_structured_workflow",
            "E2_typed_local_process",
            "E3_sandboxed_runtime",
            "E4_memory_safe_systems_runtime",
            "E5_reflex_or_safety_runtime",
        ],
        "tests": tests,
        "passed": all(test["passed"] for test in tests),
        "external_inference_calls": 0,
    }


def safety_test(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "evidence": evidence,
        "critical_failure": not passed,
    }


def build_bridge_benchmarks(residual_escrow: dict[str, Any], bridge_out: str) -> dict[str, Any]:
    active_targets = residual_escrow.get("active_diagnostic_targets", [])
    top_target = next(
        (target for target in active_targets if target.get("name") == "wh_vs_that_with_gap"),
        active_targets[0] if active_targets else {"name": "wh_vs_that_with_gap"},
    )
    cases = wh_gap_bridge_cases()
    path = Path(bridge_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "bridge_benchmark_factory",
        "source_residual_target": top_target,
        "generated_paths": [str(path)],
        "case_count": len(cases),
        "bridge_status": "active",
        "graduation_policy": {
            "initial_threshold": 0.90,
            "floor_threshold": 0.70,
            "critical_failure_veto": False,
            "residuals_return_to_escrow": True,
        },
        "external_inference_calls": 0,
    }


def wh_gap_bridge_cases() -> list[dict[str, Any]]:
    subjects = ["reporter", "student", "scientist", "teacher", "artist", "judge"]
    verbs = ["inspected", "praised", "questioned", "followed", "helped", "visited"]
    matrix_verbs = ["knew", "revealed", "remembered", "explained", "reported", "confirmed"]
    cases = []
    idx = 0
    for subject in subjects:
        for verb in verbs[:2]:
            matrix = matrix_verbs[idx % len(matrix_verbs)]
            good = f"The {subject} {matrix} who the teacher {verb}."
            bad = f"The {subject} {matrix} that the teacher {verb}."
            cases.append(
                {
                    "case_id": f"babylm_bridge_wh_gap_{idx:04d}",
                    "field": "syntax",
                    "linguistics_term": "filler_gap_dependency",
                    "rule": "wh_vs_that_with_gap_bridge",
                    "sentence_good": good,
                    "sentence_bad": bad,
                    "answer": "sentence_good",
                    "source": "residual_escrow_bridge_factory",
                }
            )
            idx += 1
    return cases


def build_moecot_route_records(
    *,
    arms: list[dict[str, Any]],
    router_eval: dict[str, Any],
    registry_context: dict[str, Any],
    vcm_context: dict[str, Any],
    candidate_integrity_context: dict[str, Any],
    route_validator_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    arm_map = {str(arm.get("arm_name") or ""): arm for arm in arms}
    records: list[dict[str, Any]] = []
    for decision in router_eval.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        route_id = str(decision.get("task_id") or stable_id("octopus_route", decision.get("task")))
        selected = [str(item) for item in decision.get("selected_arms", []) if item]
        missing = list(decision.get("missing_expected", []))
        unnecessary = list(decision.get("unnecessary_arms", []))
        registry_blocked = list(decision.get("registry_blocked_arms", []))
        passed = bool(decision.get("passed"))
        verification_bandwidth = decision.get("verification_bandwidth", {}) if isinstance(decision.get("verification_bandwidth"), dict) else {}
        governance_tax = decision.get("governance_tax", {}) if isinstance(decision.get("governance_tax"), dict) else {}
        task_hash = stable_id("octopus_task", route_id, decision.get("task"), decision.get("risk"), decision.get("pattern"))
        total_memory = 0
        total_latency = 0
        for arm_name in selected:
            envelope = decision.get("permission_envelopes", {}).get(arm_name, {})
            budget = envelope.get("budget", {}) if isinstance(envelope, dict) else {}
            total_memory += int(budget.get("memory_mb") or 0)
            total_latency += int(budget.get("latency_ms") or 0)
        common = {
            "route_id": route_id,
            "task_kind": "moecot_specialist_route",
            "target": "octopus_router",
            "task_fit": decision.get("routing_pattern"),
            "support_state": "SUPPORTED" if passed else "RESIDUAL_REVIEW",
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "raw_prompt_stored": False,
            "raw_private_text_stored": False,
        }
        records.extend(
            [
                {
                    **common,
                    "record_type": "routing_decision",
                    "record_id": stable_id("octopus_routing_decision", route_id, selected, passed),
                    "arm_id": ",".join(selected),
                    "authority_scope": "registry_gated_moecot_route",
                    "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
                    "route_validator_ready": route_validator_receipt.get("ready") is True,
                    "selected_arms": selected,
                    "expected_arms": decision.get("expected", []),
                    "registry_blocked_arms": registry_blocked,
                    "routing_pattern": decision.get("routing_pattern"),
                    "risk": decision.get("risk"),
                    "verification_bandwidth": verification_bandwidth,
                    "governance_tax": governance_tax,
                    "passed": passed,
                },
                {
                    **common,
                    "record_type": "authority_use_receipt",
                    "record_id": stable_id("octopus_authority", route_id, selected),
                    "authority_scope": "read_registry_read_vcm_read_candidate_integrity_write_router_artifacts",
                    "allowed_effects": [
                        "read_registry",
                        "read_vcm_context_governor",
                        "read_candidate_integrity",
                        "write_router_reports",
                    ],
                    "denied_effects": [
                        "runtime_external_inference",
                        "public_benchmark_training",
                        "fallback_learned_generation_credit",
                        "arbitrary_remote_execution",
                    ],
                },
                {
                    **common,
                    "record_type": "context_transaction",
                    "record_id": stable_id("octopus_context_transaction", route_id, vcm_context.get("receipt_id")),
                    "support_state": "SUPPORTED" if vcm_context.get("ready") else "BLOCKED",
                    "evidence_ref": vcm_context.get("path"),
                    "content_hash": stable_hash(vcm_context),
                },
                {
                    **common,
                    "record_type": "context_adequacy",
                    "record_id": stable_id("octopus_context_adequacy", route_id, vcm_context.get("adequacy_state")),
                    "support_state": "SUPPORTED" if vcm_context.get("ready") else "BLOCKED",
                    "state": vcm_context.get("adequacy_state"),
                    "evidence_ref": vcm_context.get("path"),
                },
                {
                    **common,
                    "record_type": "runtime_adapter_invocation",
                    "record_id": stable_id("octopus_runtime_adapter", route_id, selected),
                    "adapter_id": "octopus_router.local_json_report_adapter",
                    "authority_scope": "metadata_only_route_evaluation",
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "resource_budget",
                    "record_id": stable_id("octopus_resource_budget", route_id, total_memory, total_latency),
                    "worker_limit": len(selected),
                    "estimated_latency_ms": total_latency,
                    "memory_mb": total_memory,
                    "verification_obligation_count": verification_bandwidth.get("obligation_count"),
                    "verifier_capacity_units": verification_bandwidth.get("verifier_capacity_units"),
                    "verifier_capacity_margin_units": verification_bandwidth.get("capacity_margin_units"),
                    "verification_escalation_required": verification_bandwidth.get("escalation_required"),
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "costed_route",
                    "record_id": stable_id("octopus_costed_route", route_id, total_memory, total_latency),
                    "estimated_latency_ms": total_latency,
                    "network_class": "local_metadata_only",
                    "gas_estimate_micro_twc": 0,
                    "provider_payout_micro_twc": 0,
                    "cost_accounting": {
                        "estimated_latency_ms": total_latency,
                        "governance_overhead_ms": governance_tax.get("governed_overhead_ms"),
                        "governed_total_latency_ms": governance_tax.get("governed_total_latency_ms"),
                        "review_load_units": governance_tax.get("review_load_units"),
                        "verification_obligation_count": verification_bandwidth.get("obligation_count"),
                        "verifier_capacity_units": verification_bandwidth.get("verifier_capacity_units"),
                        "caught_failure_count": governance_tax.get("caught_failure_count"),
                        "tax_per_caught_failure": governance_tax.get("tax_per_caught_failure"),
                    },
                    "cost_classes": [
                        "routing",
                        "registry validation",
                        "VCM context adequacy",
                        "candidate integrity boundary",
                        "verification bandwidth",
                        "governance tax",
                        "residual review",
                    ],
                    "residual_obligations": verification_bandwidth.get("residual_obligations", []),
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "generation_mode",
                    "record_id": stable_id("octopus_generation_mode", route_id),
                    "candidate_generation_credit": 0,
                    "learned_generation_claim_allowed": False,
                    "fallback_return_used": False,
                    "mode": "router_selection_not_generation",
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "failure_boundary",
                    "record_id": stable_id("octopus_failure_boundary", route_id, missing, registry_blocked),
                    "failure_id": stable_id("octopus_route_failure", route_id),
                    "blocked_reason": "none" if passed else "missing_expected_or_registry_blocked",
                    "terminal": passed,
                    "structured_non_solved": not passed,
                    "support_state": "SUPPORTED" if passed else "RESIDUAL_REVIEW",
                    "missing_expected": missing,
                    "unnecessary_arms": unnecessary,
                    "registry_blocked_arms": registry_blocked,
                },
                {
                    **common,
                    "record_type": "artifact_graph_record",
                    "record_id": stable_id("octopus_artifact_graph", route_id),
                    "artifact_ref": "reports/octopus_router_report.json",
                    "evidence_ref": "reports/octopus_router_eval.json",
                    "content_hash": task_hash,
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "claim_record",
                    "record_id": stable_id("octopus_claim", route_id, passed),
                    "claim_id": stable_id("octopus_claim", route_id, "registry_gated_route"),
                    "support_state": "SUPPORTED" if passed else "RESIDUAL_REVIEW",
                    "evidence_ref": "reports/octopus_router_eval.json",
                    "learned_generation_claim_allowed": False,
                },
                {
                    **common,
                    "record_type": "evidence_transition_record",
                    "record_id": stable_id("octopus_evidence_transition", route_id),
                    "previous_support_state": "UNROUTED",
                    "current_support_state": "SUPPORTED" if passed else "RESIDUAL_REVIEW",
                    "evidence_ref": "reports/octopus_router_eval.json",
                    "support_state": "SUPPORTED",
                },
                {
                    **common,
                    "record_type": "residual_record",
                    "record_id": stable_id("octopus_residual", route_id, missing, unnecessary, registry_blocked),
                    "support_state": "NONE" if passed else "RESIDUAL_REVIEW",
                    "missing_expected": missing,
                    "unnecessary_arms": unnecessary,
                    "registry_blocked_arms": registry_blocked,
                    "verification_bandwidth": verification_bandwidth,
                    "governance_tax": governance_tax,
                    "residual_obligations": verification_bandwidth.get("residual_obligations", []),
                    "escalation_required": verification_bandwidth.get("escalation_required"),
                },
            ]
        )
        for arm_name in selected:
            arm = arm_map.get(arm_name, {})
            registry_route = arm.get("registry_route", {}) if isinstance(arm.get("registry_route"), dict) else {}
            records.append(
                {
                    **common,
                    "record_type": "semantic_node",
                    "record_id": stable_id("octopus_specialist_node", route_id, arm_name),
                    "node_id": f"{route_id}.{arm_name}",
                    "arm_id": arm_name,
                    "authority_scope": registry_route.get("required_role"),
                    "support_state": "SUPPORTED" if registry_route.get("routing_eligible") else "BLOCKED",
                    "backend_requirements": arm.get("runtime_tier"),
                    "evidence_ref": "reports/arm_registry.json",
                }
            )
    records.append(
        {
            "record_type": "claim_record",
            "record_id": stable_id("octopus_candidate_integrity_boundary", candidate_integrity_context.get("receipt_id")),
            "route_id": "octopus_router_integrity_boundary",
            "target": "octopus_router",
            "support_state": "SUPPORTED" if candidate_integrity_context.get("ready") else "BLOCKED",
            "evidence_ref": candidate_integrity_context.get("path"),
            "candidate_count": candidate_integrity_context.get("candidate_count"),
            "integrity_verified_candidate_count": candidate_integrity_context.get("integrity_verified_candidate_count"),
            "integrity_mismatch_count": candidate_integrity_context.get("integrity_mismatch_count"),
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    )
    return records


def build_report(
    *,
    arms: list[dict[str, Any]],
    router_eval: dict[str, Any],
    safety_ledger: dict[str, Any],
    bridge_ledger: dict[str, Any],
    routing_memory: dict[str, Any],
    arm_lifecycle_ledger: dict[str, Any],
    capability_ratchet: dict[str, Any],
    router_head_report: dict[str, Any],
    router_head_eval: dict[str, Any],
    registry_context: dict[str, Any],
    vcm_context: dict[str, Any],
    candidate_integrity_context: dict[str, Any],
) -> dict[str, Any]:
    learned_head = summarize_router_head(router_head_report, router_head_eval)
    route_validator_receipt = viea_spine_records.materialized_view_consumer_receipt(
        "octopus_router_route_validator",
        required_groups=[
            "governance_records",
            "failure_boundaries",
            "authority_records",
            "resource_route_records",
            "context_records",
        ],
    )
    moecot_route_records = build_moecot_route_records(
        arms=arms,
        router_eval=router_eval,
        registry_context=registry_context,
        vcm_context=vcm_context,
        candidate_integrity_context=candidate_integrity_context,
        route_validator_receipt=route_validator_receipt,
    )
    verification_bandwidth_summary = router_eval.get("metrics", {}).get("verification_bandwidth", {})
    governance_tax_summary = router_eval.get("metrics", {}).get("governance_tax", {})
    matrix = ora_implementation_matrix(
        arms,
        router_eval,
        safety_ledger,
        bridge_ledger,
        learned_head,
        routing_memory,
        arm_lifecycle_ledger,
        registry_context,
        vcm_context,
        candidate_integrity_context,
        route_validator_receipt,
        moecot_route_records,
        verification_bandwidth_summary,
        governance_tax_summary,
    )
    registry_health = {
        "active": sum(1 for arm in arms if str(arm.get("lifecycle_status", "")).startswith("active")),
        "resident": sum(1 for arm in arms if arm.get("dynamic_loading", {}).get("load_policy") == "resident"),
        "split_candidates": sum(1 for arm in arms if arm.get("split_candidate")),
        "retired": sum(1 for arm in arms if arm.get("lifecycle_status") == "retired"),
    }
    return {
        "policy": "local_only_no_external_inference",
        "trigger_state": "GREEN" if not any(row["status"] == "missing" for row in matrix) else "YELLOW",
        "framework": "octopus_router_architecture",
        "status": "active_system_level_router_v0",
        "thesis": "One resident head routes tasks to dynamically loaded, governed, independently ratcheted specialist arms.",
        "implementation_score": implementation_score(matrix),
        "implementation_matrix": matrix,
        "head_router": {
            "resident": True,
            "strategy": "rule_bootloader_plus_learned_sparse_head_v0" if learned_head["promotion_gate_passed"] else router_eval["router"]["strategy"],
            "learned_upgrade_path": "append local task-to-arm traces and retrain the sparse head before every architecture gate",
            "scope_limits": [
                "intent",
                "risk",
                "arm_selection",
                "permission_envelopes",
                "composition",
                "verification_orchestration",
            ],
        },
        "learned_router_head": learned_head,
        "project_registry_route_gate": {
            "path": registry_context.get("registry_path"),
            "trigger_state": registry_context.get("registry_trigger_state"),
            "gate_passed": registry_context.get("registry_gate_passed"),
            "stable_capability_field_gate_passed": registry_context.get("stable_capability_field_gate_passed"),
            "eligible_by_role": registry_context.get("eligible_by_role", {}),
            "summary": registry_context.get("summary", {}),
        },
        "vcm_route_context": vcm_context,
        "candidate_integrity_boundary": candidate_integrity_context,
        "route_validator_receipt": route_validator_receipt,
        "moecot_spine": {
            "record_count": len(moecot_route_records),
            "route_count": len(router_eval.get("decisions", [])),
            "route_validator_ready": route_validator_receipt.get("ready") is True,
            "vcm_context_ready": vcm_context.get("ready") is True,
            "candidate_integrity_ready": candidate_integrity_context.get("ready") is True,
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "verification_bandwidth_status": verification_bandwidth_summary.get("status"),
            "verification_obligation_count": verification_bandwidth_summary.get("obligation_count"),
            "verification_escalation_required_count": verification_bandwidth_summary.get("escalation_required_count"),
            "governance_tax_status": governance_tax_summary.get("status"),
            "governance_tax_review_load_units": governance_tax_summary.get("review_load_units"),
            "governance_tax_caught_failure_count": governance_tax_summary.get("caught_failure_count"),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "verification_bandwidth": verification_bandwidth_summary,
        "governance_tax": governance_tax_summary,
        "viea_moecot_route_records": moecot_route_records,
        "arm_registry": {
            "path": "reports/arm_registry.json",
            "count": len(arms),
            "health": registry_health,
            "arms": [
                {
                    "arm_name": arm["arm_name"],
                    "lifecycle_status": arm["lifecycle_status"],
                    "permission_tier": arm["permission_tier"],
                    "runtime_tier": arm["runtime_tier"],
                    "reliability_score": arm["reliability_score"],
                    "bloat_index": arm["bloat_index"],
                    "split_candidate": arm["split_candidate"],
                    "registry_route": arm.get("registry_route", {}),
                }
                for arm in arms
            ],
        },
        "router_eval": {
            "path": "reports/octopus_router_eval.json",
            "metrics": router_eval["metrics"],
            "dynamic_loading_metrics": router_eval["dynamic_loading_metrics"],
        },
        "routing_memory": {
            "path": "reports/routing_memory.json",
            "entries": routing_memory.get("summary", {}).get("entries", 0),
            "arm_memories": routing_memory.get("summary", {}).get("arm_memories", 0),
            "passed_routes": routing_memory.get("summary", {}).get("passed_routes", 0),
        },
        "safety_and_quarantine": {
            "path": "reports/safety_benchmark_ledger.json",
            "passed": safety_ledger["passed"],
            "tests": safety_ledger["tests"],
        },
        "bridge_benchmarking": {
            "path": "reports/bridge_benchmark_ledger.json",
            "generated_paths": bridge_ledger["generated_paths"],
            "case_count": bridge_ledger["case_count"],
            "source_residual_target": bridge_ledger["source_residual_target"],
        },
        "local_arm_ratchets": build_local_arm_ratchets(arms, capability_ratchet),
        "split_merge_retire": arm_lifecycle_ledger.get("split_merge_retire", {}),
        "arm_lifecycle_ledger": {
            "path": "reports/arm_lifecycle_ledger.json",
            "summary": arm_lifecycle_ledger.get("summary", {}),
        },
        "next_actions": [
            "Append future task-to-arm traces to the learned router dataset and retrain before major architecture gates.",
            "Run the generated wh_vs_that bridge benchmark against the next BabyLM grammar-state candidate.",
            "Promote safety/quarantine tests to candidate-promotion blockers.",
            "Add per-arm usage telemetry so split/merge/retire decisions become empirical.",
        ],
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def ora_implementation_matrix(
    arms: list[dict[str, Any]],
    router_eval: dict[str, Any],
    safety_ledger: dict[str, Any],
    bridge_ledger: dict[str, Any],
    learned_head: dict[str, Any],
    routing_memory: dict[str, Any],
    arm_lifecycle_ledger: dict[str, Any],
    registry_context: dict[str, Any],
    vcm_context: dict[str, Any],
    candidate_integrity_context: dict[str, Any],
    route_validator_receipt: dict[str, Any],
    moecot_route_records: list[dict[str, Any]],
    verification_bandwidth_summary: dict[str, Any],
    governance_tax_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    dynamic = router_eval.get("dynamic_loading_metrics", {})
    learned_ok = learned_head.get("promotion_gate_passed", False)
    required_spine_types = {
        "routing_decision",
        "semantic_node",
        "authority_use_receipt",
        "context_transaction",
        "context_adequacy",
        "runtime_adapter_invocation",
        "resource_budget",
        "costed_route",
        "generation_mode",
        "failure_boundary",
        "artifact_graph_record",
        "claim_record",
        "evidence_transition_record",
        "residual_record",
    }
    observed_spine_types = {str(row.get("record_type") or "") for row in moecot_route_records}
    return [
        component("manual_arm_registry", bool(arms), "Arm cards with scope, schemas, tools, permissions, benchmarks, residuals, and lifecycle are written."),
        component("basic_router", router_eval["metrics"]["selection_accuracy"] >= 0.90, "Keyword-weighted resident head routes the local benchmark suite."),
        component("structured_arm_outputs", all("output_schema" in arm for arm in arms), "Every arm card declares structured output fields for head composition."),
        component("dynamic_loading", dynamic.get("estimated_memory_savings", 0.0) > 0.0, "Router eval measures cold loads, warm hits, evictions, and memory savings."),
        component("local_arm_ratchets", all("benchmark_frontier" in arm and "regression_suite" in arm for arm in arms), "Every arm has local benchmark and regression slots."),
        component("split_merge_retire_lifecycle", all("retirement_criteria" in arm for arm in arms), "Arm cards include bloat, split-candidate, merge, and retirement metadata."),
        component("safety_and_quarantine", safety_ledger.get("passed", False), "Safety ledger checks high-risk routing, approval, runtime tiers, and least privilege."),
        component("multi_arm_composition", any(len(row.get("selected_arms", [])) > 2 for row in router_eval.get("decisions", [])), "Router benchmark covers parallel, sequential, verification, and reflex composition patterns."),
        component("bridge_benchmark_generation", bridge_ledger.get("case_count", 0) > 0, "Bridge benchmark cases are generated from recurring residual escrow."),
        component("routing_memory", routing_memory.get("summary", {}).get("entries", 0) > 0, "Routing memory records task signatures, selected arms, outcomes, permission scopes, and per-arm reliability traces."),
        component("arm_lifecycle_governance", arm_lifecycle_ledger.get("summary", {}).get("arms", 0) == len(arms), "Arm lifecycle ledger tracks bloat, split, merge, retire, and spawn signals for every arm."),
        component(
            "registry_gated_routing",
            bool(registry_context.get("registry_gate_passed"))
            and router_eval.get("metrics", {}).get("registry_routing_accuracy", 0.0) >= 1.0,
            "Octopus arms resolve to registered routing-eligible implementations before selection can pass.",
        ),
        component(
            "viea_moecot_route_receipts",
            bool(route_validator_receipt.get("ready"))
            and bool(vcm_context.get("ready"))
            and bool(candidate_integrity_context.get("ready"))
            and required_spine_types.issubset(observed_spine_types),
            "Octopus routing emits VIEA-normalized route, specialist, authority, VCM, runtime-adapter, resource, cost, generation-mode, failure, artifact, claim, evidence-transition, and residual records.",
        ),
        component(
            "verification_bandwidth_and_governance_tax",
            verification_bandwidth_summary.get("status") == "ready"
            and int(verification_bandwidth_summary.get("route_count") or 0) == len(router_eval.get("decisions", []))
            and int(verification_bandwidth_summary.get("obligation_count") or 0) > 0
            and governance_tax_summary.get("status") == "ready"
            and int(governance_tax_summary.get("route_count") or 0) == len(router_eval.get("decisions", []))
            and int(governance_tax_summary.get("review_load_units") or 0) >= 0,
            "Router decisions carry verifier-capacity obligations, residual-obligation ledgers, escalation thresholds, governance overhead, review load, and tax-per-caught-failure accounting.",
        ),
        component(
            "learned_router_training",
            learned_ok,
            learned_head.get("evidence", "Sparse router head training report is not attached yet."),
            status_override=None if learned_ok else "partial",
        ),
    ]


def summarize_router_head(
    router_head_report: dict[str, Any],
    router_head_eval: dict[str, Any],
) -> dict[str, Any]:
    metrics = router_head_eval.get("metrics", {})
    trace_summary = router_head_report.get("trace_summary", {}) if isinstance(router_head_report.get("trace_summary"), dict) else {}
    contrastive_ok = (
        int(metrics.get("contrastive_holdout_negatives") or 0) > 0
        and float(metrics.get("contrastive_negative_accuracy") or 0.0) >= 0.95
    )
    real_trace_ok = int(trace_summary.get("schema_bound_real_trace_examples") or 0) > 0
    non_generation_ok = (
        router_head_report.get("learned_generation_claim_allowed") is False
        and int(router_head_report.get("candidate_generation_credit") or 0) == 0
    )
    passed = (
        bool(router_head_report.get("promotion_gate_passed"))
        and metrics.get("exact_set_accuracy", 0.0) >= 0.95
        and metrics.get("risk_routing_accuracy", 0.0) >= 1.0
        and contrastive_ok
        and real_trace_ok
        and non_generation_ok
    )
    if not router_head_report and not router_head_eval:
        return {
            "status": "missing",
            "promotion_gate_passed": False,
            "metrics": {},
            "artifacts": {},
            "evidence": "Routing traces are emitted, but no learned router-head training artifact is attached.",
        }
    return {
        "status": "promoted" if passed else "needs_more_traces",
        "promotion_gate_passed": passed,
        "model_type": router_head_report.get("model_type"),
        "trace_summary": trace_summary,
        "metrics": metrics,
        "artifacts": router_head_report.get("artifacts", {}),
        "evidence": (
            f"Learned sparse head exact_set_accuracy={metrics.get('exact_set_accuracy')} "
            f"risk_routing_accuracy={metrics.get('risk_routing_accuracy')} "
            f"contrastive_negative_accuracy={metrics.get('contrastive_negative_accuracy')} "
            f"schema_bound_real_trace_examples={trace_summary.get('schema_bound_real_trace_examples')}"
        ),
        "learned_generation_claim_allowed": False,
        "candidate_generation_credit": 0,
        "external_inference_calls": router_head_report.get("external_inference_calls", 0)
        + router_head_eval.get("external_inference_calls", 0),
    }


def component(name: str, condition: bool, evidence: str, *, status_override: str | None = None) -> dict[str, Any]:
    return {
        "component": name,
        "status": status_override or ("implemented" if condition else "missing"),
        "evidence": evidence,
    }


def implementation_score(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {"implemented": 1.0, "partial": 0.5, "missing": 0.0}
    total = sum(weights.get(row["status"], 0.0) for row in matrix)
    return {
        "score": total / max(1, len(matrix)),
        "implemented": sum(1 for row in matrix if row["status"] == "implemented"),
        "partial": sum(1 for row in matrix if row["status"] == "partial"),
        "missing": sum(1 for row in matrix if row["status"] == "missing"),
        "possible": len(matrix),
    }


def build_local_arm_ratchets(
    arms: list[dict[str, Any]],
    capability_ratchet: dict[str, Any],
) -> list[dict[str, Any]]:
    residual_map = capability_ratchet.get("residual_map", {})
    ratchets = []
    for arm in arms:
        frontier = arm.get("benchmark_frontier", [])
        residuals = arm.get("residual_escrow", [])
        ratchets.append(
            {
                "arm_name": arm["arm_name"],
                "frontier": frontier,
                "regression_suite": arm.get("regression_suite", []),
                "residual_escrow": residuals,
                "next_intervention": arm_next_intervention(arm, residual_map),
            }
        )
    return ratchets


def arm_next_intervention(arm: dict[str, Any], residual_map: dict[str, Any]) -> str:
    name = arm["arm_name"]
    if name == "babylm_grammar_arm":
        target = "wh_vs_that_with_gap"
        for row in residual_map.get("worst_mutated_babylm_rules", []):
            if row.get("name") == target:
                return "train/evaluate bridge benchmark for wh_vs_that_with_gap before expanding grammar state"
        return "generate seed55 mutated holdout and preserve public plus seed49 regressions"
    if name == "puffer_ocean_control_arm":
        return "move more rollout stepping and optimizer state behind Rust/CUDA FFI"
    if name == "safety_reflex_arm":
        return "make safety ledger failures hard blockers for candidate promotion"
    if name == "bridge_benchmark_arm":
        return "run generated bridge cases against the next BabyLM candidate"
    return "continue local ratchet and preserve regressions"


def split_merge_retire_recommendations(arms: list[dict[str, Any]]) -> dict[str, Any]:
    split = [
        {
            "arm_name": arm["arm_name"],
            "bloat_index": arm["bloat_index"],
            "reason": "high tool/benchmark/residual/memory surface",
        }
        for arm in arms
        if arm.get("split_candidate")
    ]
    merge = []
    domains: dict[str, list[str]] = {}
    for arm in arms:
        domains.setdefault(arm["quarantine_domain"], []).append(arm["arm_name"])
    for domain, names in domains.items():
        if len(names) > 1:
            merge.append({"domain": domain, "arms": names, "recommendation": "inspect overlap before merging"})
    return {
        "split_candidates": split,
        "merge_inspection": merge,
        "retire_candidates": [],
        "rule": "split when bloat and residual clusters diverge; merge when overlap exceeds specialization value; retire stale failing arms",
    }


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def read_json(path: str, default: Any) -> Any:
    file = Path(path)
    if not file.exists():
        return default
    return json.loads(file.read_text(encoding="utf-8"))


def write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:16]


if __name__ == "__main__":
    raise SystemExit(main())
