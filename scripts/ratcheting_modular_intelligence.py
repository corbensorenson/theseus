"""Ratcheting Modular Intelligence conformance audit for SymLiquid.

RMI is the unified layer above the benchmark ratchet, loop closure, ORA router,
routing memory, arm lifecycle governance, and safety gate. This script verifies
that those pieces exist as local artifacts before heavier training proceeds.
It does not call external inference providers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-treadmill", default="reports/benchmark_treadmill_status.json")
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--model-ledger", default="reports/model_ledger.json")
    parser.add_argument("--tool-registry", default="reports/tool_registry.json")
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument("--public-comparator-ledger", default="reports/public_comparator_ledger.json")
    parser.add_argument("--capability-ratchet", default="reports/capability_ratchet_report.json")
    parser.add_argument("--rgs", default="reports/ratcheting_generative_system_report.json")
    parser.add_argument("--octopus-router", default="reports/octopus_router_report.json")
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--routing-memory", default="reports/routing_memory.json")
    parser.add_argument("--arm-lifecycle", default="reports/arm_lifecycle_ledger.json")
    parser.add_argument("--router-head", default="reports/octopus_router_head_report.json")
    parser.add_argument("--router-head-eval", default="reports/octopus_router_head_eval.json")
    parser.add_argument("--safety-ledger", default="reports/safety_benchmark_ledger.json")
    parser.add_argument("--bridge-ledger", default="reports/bridge_benchmark_ledger.json")
    parser.add_argument("--event-log", default="reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json")
    parser.add_argument("--out", default="reports/ratcheting_modular_intelligence_report.json")
    args = parser.parse_args()

    artifacts = {
        "benchmark_treadmill": read_json(args.benchmark_treadmill, {}),
        "benchmark_ledger": read_json(args.benchmark_ledger, []),
        "model_ledger": read_json(args.model_ledger, {}),
        "tool_registry": read_json(args.tool_registry, {}),
        "residual_escrow": read_json(args.residual_escrow, {}),
        "public_comparator_ledger": read_json(args.public_comparator_ledger, {}),
        "capability_ratchet": read_json(args.capability_ratchet, {}),
        "rgs": read_json(args.rgs, {}),
        "octopus_router": read_json(args.octopus_router, {}),
        "arm_registry": read_json(args.arm_registry, {}),
        "routing_memory": read_json(args.routing_memory, {}),
        "arm_lifecycle": read_json(args.arm_lifecycle, {}),
        "router_head": read_json(args.router_head, {}),
        "router_head_eval": read_json(args.router_head_eval, {}),
        "safety_ledger": read_json(args.safety_ledger, {}),
        "bridge_ledger": read_json(args.bridge_ledger, {}),
        "event_log": read_json(args.event_log, {}),
    }
    report = build_report(args, artifacts)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["implementation_score"]["missing"] == 0 else 2


def build_report(args: argparse.Namespace, artifacts: dict[str, Any]) -> dict[str, Any]:
    matrix = implementation_matrix(artifacts)
    arms = artifacts["arm_registry"].get("arms", [])
    benchmark_ledger = artifacts["benchmark_ledger"]
    tool_registry = artifacts["tool_registry"]
    residual_escrow = artifacts["residual_escrow"]
    routing_memory = artifacts["routing_memory"]
    arm_lifecycle = artifacts["arm_lifecycle"]
    compact_generative_records = build_compact_generative_records(artifacts, matrix)
    return {
        "policy": "local_only_no_external_inference",
        "framework": "ratcheting_modular_intelligence",
        "status": "active_rmi_audit_v0",
        "thesis": "Turn pressure into verified modular structure, then ratchet that structure toward harder capabilities.",
        "implementation_score": implementation_score(matrix),
        "implementation_matrix": matrix,
        "formal_system_state": {
            "H_t": "head_router + learned sparse router head",
            "A_t": [arm.get("arm_name") for arm in arms],
            "R_t": {
                "rule_router": "reports/octopus_router_eval.json",
                "learned_router": args.router_head,
                "routing_memory": args.routing_memory,
            },
            "M_t": {
                "global": "project goals and calibration policy",
                "arm_local": "reports/arm_registry.json plus reports/routing_memory.json",
                "shared_task": "permission envelopes inside router eval",
                "safety": args.safety_ledger,
                "residual": args.residual_escrow,
            },
            "T_t": args.tool_registry,
            "B_t": args.benchmark_ledger,
            "G_t": "regression entries in reports/benchmark_ledger.json",
            "E_t": args.residual_escrow,
            "V_t": [args.safety_ledger, args.bridge_ledger, "reports/architecture_gate_report.json"],
        },
        "five_pillars": five_pillars(artifacts),
        "compact_generative_records": compact_generative_records,
        "routing_and_modularity": {
            "arm_count": len(arms),
            "routing_memory_entries": routing_memory.get("summary", {}).get("entries", 0),
            "learned_router_metrics": artifacts["router_head_eval"].get("metrics", {}),
            "arm_lifecycle_summary": arm_lifecycle.get("summary", {}),
        },
        "benchmark_curriculum": {
            "regression_count": sum(1 for row in benchmark_ledger if row.get("lifecycle") == "regression"),
            "public_comparators": len(artifacts["public_comparator_ledger"].get("comparators", [])),
            "residual_escrow_summary": residual_escrow.get("summary", {}),
            "threshold_policy": artifacts["benchmark_treadmill"].get("curriculum_policy", {}),
        },
        "procedural_memory": {
            "tool_count": len(tool_registry.get("tools", [])),
            "registry_health": tool_registry.get("registry_health", {}),
            "active_tools": [
                tool.get("tool_name")
                for tool in tool_registry.get("tools", [])
                if tool.get("lifecycle") == "active"
            ],
        },
        "safety_runtime": {
            "safety_passed": artifacts["safety_ledger"].get("passed", False),
            "runtime_tiers": sorted({arm.get("runtime_tier") for arm in arms if arm.get("runtime_tier")}),
            "critical_veto_policy": "time-decayed thresholds never relax critical-failure vetoes",
        },
        "next_actions": next_actions(artifacts, matrix),
        "artifacts": {
            "benchmark_treadmill": args.benchmark_treadmill,
            "benchmark_ledger": args.benchmark_ledger,
            "model_ledger": args.model_ledger,
            "tool_registry": args.tool_registry,
            "residual_escrow": args.residual_escrow,
            "capability_ratchet": args.capability_ratchet,
            "rgs": args.rgs,
            "octopus_router": args.octopus_router,
            "arm_registry": args.arm_registry,
            "routing_memory": args.routing_memory,
            "arm_lifecycle": args.arm_lifecycle,
            "router_head": args.router_head,
            "router_head_eval": args.router_head_eval,
            "safety_ledger": args.safety_ledger,
            "bridge_ledger": args.bridge_ledger,
            "event_log": args.event_log,
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def implementation_matrix(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_ledger = artifacts["benchmark_ledger"]
    tool_registry = artifacts["tool_registry"]
    residual_escrow = artifacts["residual_escrow"]
    octopus = artifacts["octopus_router"]
    arms = artifacts["arm_registry"].get("arms", [])
    routing_memory = artifacts["routing_memory"]
    arm_lifecycle = artifacts["arm_lifecycle"]
    router_head_eval = artifacts["router_head_eval"]
    safety = artifacts["safety_ledger"]
    bridge = artifacts["bridge_ledger"]
    event_log = artifacts["event_log"]
    return [
        component(
            "compact_generative_structure",
            bool(artifacts["model_ledger"]) and bool(arms) and bool(tool_registry.get("tools")),
            "Model ledger, arm cards, tool cards, schemas, and verifier surfaces are present.",
        ),
        component(
            "active_compression",
            residual_escrow.get("summary", {}).get("cluster_count", 0) > 0
            and routing_memory.get("summary", {}).get("entries", 0) > 0,
            "Experience is compressed into residual escrow, routing memory, ledgers, and tool registry.",
        ),
        component(
            "cognitive_loop_closure",
            len(tool_registry.get("tools", [])) >= 14,
            "Repeated local workflows are represented as verified procedural tool cards.",
        ),
        component(
            "benchmark_ratcheting",
            bool(benchmark_ledger)
            and len(artifacts["public_comparator_ledger"].get("comparators", [])) > 0
            and residual_escrow.get("summary", {}).get("case_count", 0) > 0,
            "Benchmark ledger, public calibration, residual escrow, and time-decayed threshold policy exist.",
        ),
        component(
            "octopus_routing",
            score(octopus) >= 1.0 and len(arms) >= 1,
            "ORA has a resident head, arm registry, dynamic loading metrics, and local arm ratchets.",
        ),
        component(
            "learned_router_head",
            router_head_eval.get("metrics", {}).get("exact_set_accuracy", 0.0) >= 0.95
            and router_head_eval.get("metrics", {}).get("risk_routing_accuracy", 0.0) >= 1.0,
            "Learned sparse router head passes holdout and risk-routing gates.",
        ),
        component(
            "routing_memory",
            routing_memory.get("summary", {}).get("entries", 0) > 0
            and routing_memory.get("summary", {}).get("arm_memories", 0) >= len(arms),
            "Routing memory stores task signatures, selected arms, outcomes, and permission summaries.",
        ),
        component(
            "arm_lifecycle_governance",
            arm_lifecycle.get("summary", {}).get("arms", 0) == len(arms),
            "Arm lifecycle ledger tracks split, merge, retire, and spawn rules for each arm.",
        ),
        component(
            "safety_runtime_tiers",
            safety.get("passed", False)
            and any(arm.get("runtime_tier") == "E5_reflex_or_safety_runtime" for arm in arms),
            "Safety ledger passes and at least one reflex/failsafe runtime arm exists.",
        ),
        component(
            "bridge_benchmark_protocol",
            bridge.get("case_count", 0) > 0,
            "Bridge benchmark exists for recurring residual escrow pressure.",
        ),
        component(
            "high_bandwidth_embodied_logging",
            event_log.get("summary", {}).get("event_count", 0) > 0,
            "Embodied rollout logging includes raw, event, semantic, skill, and residual streams.",
        ),
        component(
            "external_inference_forbidden",
            external_inference_ok(artifacts),
            "All RMI artifacts report zero external inference calls or violations.",
        ),
    ]


def five_pillars(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "compact_generative_structure": {
            "model_ledger": bool(artifacts["model_ledger"]),
            "arms": len(artifacts["arm_registry"].get("arms", [])),
            "tools": len(artifacts["tool_registry"].get("tools", [])),
        },
        "active_compression": {
            "residual_clusters": artifacts["residual_escrow"].get("summary", {}).get("cluster_count", 0),
            "routing_memory_entries": artifacts["routing_memory"].get("summary", {}).get("entries", 0),
            "benchmark_ledger_entries": len(artifacts["benchmark_ledger"]),
        },
        "cognitive_loop_closure": {
            "tool_registry_health": artifacts["tool_registry"].get("registry_health", {}),
            "execution_modes": artifacts["capability_ratchet"].get("procedural_ratchet", {}).get("execution_modes", {}),
        },
        "benchmark_ratcheting": {
            "regressions": sum(1 for row in artifacts["benchmark_ledger"] if row.get("lifecycle") == "regression"),
            "public_comparators": len(artifacts["public_comparator_ledger"].get("comparators", [])),
            "bridge_cases": artifacts["bridge_ledger"].get("case_count", 0),
        },
        "octopus_routing": {
            "ora_score": score(artifacts["octopus_router"]),
            "arm_lifecycle": artifacts["arm_lifecycle"].get("summary", {}),
            "learned_router": artifacts["router_head_eval"].get("metrics", {}),
        },
    }


def build_compact_generative_records(artifacts: dict[str, Any], matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    implemented = sum(1 for row in matrix if row.get("status") == "implemented")
    possible = max(1, len(matrix))
    residual_summary = artifacts["residual_escrow"].get("summary", {})
    routing_summary = artifacts["routing_memory"].get("summary", {})
    arm_count = len(artifacts["arm_registry"].get("arms", []))
    tool_count = len(artifacts["tool_registry"].get("tools", []))
    promotion_state = "verified_for_scope" if implemented == possible else "candidate"
    return [
        {
            "record_type": "compact_generative_record",
            "record_id": stable_id("compact_generative_record", "rmi", implemented, possible, arm_count, tool_count),
            "system_id": "theseus-rmi-compact-generative-structure",
            "target_system": "Project Theseus RMI control plane, arm registry, tool registry, residual escrow, routing memory, and benchmark ratchet.",
            "compact_seed": "model ledger, arm cards, tool cards, benchmark ledger, residual escrow, routing memory, and safety/bridge ledgers",
            "rule_system": "Pressure becomes residuals; repeated residuals become tools or arm routing evidence; promotion requires verifier and evidence-ledger support.",
            "memory_state": "local reports only; no raw private prompt text and no external inference tokens are served at runtime",
            "generation_status": "generated",
            "residual_channel": [
                f"residual_clusters={residual_summary.get('cluster_count', 0)}",
                f"routing_memory_entries={routing_summary.get('entries', 0)}",
                "public transfer remains calibration-only and cannot be trained on",
            ],
            "correction_mechanism": "downgrade unsupported claims, preserve residual escrow, reroute through verifier/tool/assistant gates, and require candidate integrity before promotion.",
            "verification_contract": "RMI audit, architecture gate, candidate-integrity audit, public calibration registry, and VIEA materialized record gate must agree before capability promotion.",
            "verification_status": "passed" if implemented == possible else "residual",
            "verifier_independence": "The compact generative structure cannot certify itself; independent gates and report-evidence store receipts materialize support and defeaters.",
            "governance_interface": "AGENTS.md no-cheat rules, roadmap matrix, project registry, and source-sync steward decisions define authority boundaries.",
            "authority_boundary": "Architecture and routing evidence only; no learned-generation, public benchmark, compression-ratio, or ASI capability claim is implied.",
            "use_envelope": [
                "RMI architecture audit",
                "roadmap source synchronization",
                "assistant/operator governance orientation",
                "promotion preflight context",
            ],
            "burden_ledger": [
                "generation burden remains on neural candidate generator",
                "verification burden remains on independent gates",
                "source-sync burden remains on roadmap crosswalk/steward decisions",
                "public transfer burden remains unresolved until governed calibration evidence improves",
            ],
            "cost_accounting": [
                f"arm_count={arm_count}",
                f"tool_count={tool_count}",
                f"implemented_components={implemented}/{possible}",
            ],
            "generative_leverage": "A compact set of ledgers and contracts can regenerate the current RMI operating map without copying full report payloads into runtime context.",
            "hidden_complexity_risks": [
                "private gates can look strong while public transfer remains weak",
                "report volume can obscure unresolved generator quality walls",
                "router/tool evidence can be misread as learned-generation capability if not fenced",
            ],
            "fallback_path": "Use full reports, gate outputs, and preserved artifacts when exact evidence, replay, or promotion support is needed.",
            "fallback_status": "available",
            "residual_burden_status": "bounded" if implemented == possible else "dominant",
            "promotion_state": promotion_state,
            "promotion_blockers": [] if implemented == possible else [row["component"] for row in matrix if row["status"] != "implemented"],
            "retirement_condition": "Retire or narrow this compact record if it stops matching the live registry, VIEA view, candidate-integrity audit, or public-calibration registry.",
            "support_state_effect": "eligible_for_bounded_evidence_review" if promotion_state == "verified_for_scope" else "record_shape_only",
            "source_refs": [
                "AGENTS.md",
                "configs/roadmap_implementation_matrix.json",
                "configs/project_manifest_registry.json",
                "configs/viea_spine_record_contracts.json",
            ],
            "evidence_refs": [
                "reports/ratcheting_modular_intelligence_report.json",
                "reports/capability_ratchet_report.json",
                "reports/tool_registry.json",
                "reports/residual_escrow.json",
                "reports/viea_spine_materialized_view.json",
            ],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "non_claims": [
                "not learned-generation evidence",
                "not proof that SymLiquid beats a matched transformer",
                "not a public benchmark score",
                "not a compression benchmark",
            ],
        }
    ]


def component(name: str, condition: bool, evidence: str) -> dict[str, Any]:
    return {
        "component": name,
        "status": "implemented" if condition else "missing",
        "evidence": evidence,
    }


def implementation_score(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    implemented = sum(1 for row in matrix if row["status"] == "implemented")
    missing = sum(1 for row in matrix if row["status"] == "missing")
    possible = max(1, len(matrix))
    return {
        "score": implemented / possible,
        "implemented": implemented,
        "partial": 0,
        "missing": missing,
        "possible": len(matrix),
    }


def score(report: dict[str, Any]) -> float:
    return float(report.get("implementation_score", {}).get("score", 0.0))


def external_inference_ok(payload: Any) -> bool:
    violations: list[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("external_inference_calls", 0) not in (0, None):
                violations.append(value.get("external_inference_calls"))
            if value.get("external_inference_violations"):
                violations.append(value.get("external_inference_violations"))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return not violations


def stable_id(*parts: Any) -> str:
    import hashlib

    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def next_actions(artifacts: dict[str, Any], matrix: list[dict[str, Any]]) -> list[str]:
    missing = [row["component"] for row in matrix if row["status"] != "implemented"]
    if missing:
        return [f"Implement RMI component: {name}" for name in missing]
    treadmill_actions = artifacts["benchmark_treadmill"].get("next_commands", [])
    return [
        "Keep RMI as a promotion gate before heavy local training.",
        "Append real task-to-arm traces into routing memory before retraining the router head.",
        "Use arm lifecycle ledger bloat/merge/retire signals before expanding the specialist set.",
        *treadmill_actions[:2],
    ]


def read_json(path: str, default: Any) -> Any:
    file = Path(path)
    if not file.exists():
        return default
    return json.loads(file.read_text(encoding="utf-8"))


def write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
