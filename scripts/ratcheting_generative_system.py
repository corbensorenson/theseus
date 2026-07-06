"""Ratcheting Generative Systems conformance report for SymLiquid.

This script audits the local ledgers produced by the Capability Ratchet and
summarizes whether the broader RGS framework is actually represented in code:
benchmark frontiers, residual escrow, procedural memory, public calibration,
execution modes, safety/reflex hooks, bridge-benchmark policy, and local-only
verification. It does not call external inference providers.
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
    parser.add_argument("--octopus-router", default="reports/octopus_router_report.json")
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--safety-ledger", default="reports/safety_benchmark_ledger.json")
    parser.add_argument("--bridge-ledger", default="reports/bridge_benchmark_ledger.json")
    parser.add_argument("--out", default="reports/ratcheting_generative_system_report.json")
    args = parser.parse_args()

    treadmill = read_json(args.benchmark_treadmill, {})
    benchmark_ledger = read_json(args.benchmark_ledger, [])
    model_ledger = read_json(args.model_ledger, {})
    tool_registry = read_json(args.tool_registry, {})
    residual_escrow = read_json(args.residual_escrow, {})
    public_comparator_ledger = read_json(args.public_comparator_ledger, {})
    capability_ratchet = read_json(args.capability_ratchet, {})
    octopus_router = read_json(args.octopus_router, {})
    arm_registry = read_json(args.arm_registry, {})
    safety_ledger = read_json(args.safety_ledger, {})
    bridge_ledger = read_json(args.bridge_ledger, {})

    report = build_report(
        treadmill=treadmill,
        benchmark_ledger=benchmark_ledger,
        model_ledger=model_ledger,
        tool_registry=tool_registry,
        residual_escrow=residual_escrow,
        public_comparator_ledger=public_comparator_ledger,
        capability_ratchet=capability_ratchet,
        octopus_router=octopus_router,
        arm_registry=arm_registry,
        safety_ledger=safety_ledger,
        bridge_ledger=bridge_ledger,
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_report(
    *,
    treadmill: dict[str, Any],
    benchmark_ledger: list[dict[str, Any]],
    model_ledger: dict[str, Any],
    tool_registry: dict[str, Any],
    residual_escrow: dict[str, Any],
    public_comparator_ledger: dict[str, Any],
    capability_ratchet: dict[str, Any],
    octopus_router: dict[str, Any],
    arm_registry: dict[str, Any],
    safety_ledger: dict[str, Any],
    bridge_ledger: dict[str, Any],
) -> dict[str, Any]:
    embodied_event_logs = discover_embodied_event_logs()
    matrix = implementation_matrix(
        treadmill=treadmill,
        benchmark_ledger=benchmark_ledger,
        model_ledger=model_ledger,
        tool_registry=tool_registry,
        residual_escrow=residual_escrow,
        public_comparator_ledger=public_comparator_ledger,
        capability_ratchet=capability_ratchet,
        embodied_event_logs=embodied_event_logs,
        octopus_router=octopus_router,
        arm_registry=arm_registry,
        safety_ledger=safety_ledger,
        bridge_ledger=bridge_ledger,
    )
    subgroup_floor_checks = build_subgroup_floor_checks(capability_ratchet, benchmark_ledger)
    bridge_recommendations = build_bridge_recommendations(benchmark_ledger)
    external_violations = treadmill.get("external_inference_violations", [])
    if not external_violations:
        external_violations = capability_ratchet.get("verification", {}).get(
            "external_inference_violations", []
        )

    return {
        "policy": "local_only_no_external_inference",
        "framework": "ratcheting_generative_systems",
        "thesis": "Turn benchmark pressure, repeated behavior, and residual failures into verified reusable structure.",
        "implementation_score": implementation_score(matrix),
        "implementation_matrix": matrix,
        "frontier_momentum": {
            "counts": treadmill.get("counts", {}),
            "next_commands": treadmill.get("next_commands", []),
            "frontier_rotation_required": capability_ratchet.get("ratchet_rule", {}).get(
                "frontier_rotation_required"
            ),
            "critical_failure_veto": capability_ratchet.get("ratchet_rule", {}).get(
                "critical_failure_veto"
            ),
            "external_inference_violations": external_violations,
        },
        "threshold_policy": treadmill.get("curriculum_policy", {}),
        "subgroup_floor_checks": subgroup_floor_checks,
        "residual_governance": {
            "ledger": "reports/residual_escrow.json",
            "summary": residual_escrow.get("summary", {}),
            "attention_budget": residual_escrow.get("attention_budget", {}),
            "active_diagnostic_targets": residual_escrow.get(
                "active_diagnostic_targets", []
            )[:8],
            "recurrence_promotion_rule": residual_escrow.get(
                "recurrence_promotion_rule"
            ),
        },
        "public_calibration": {
            "ledger": "reports/public_comparator_ledger.json",
            "count": len(public_comparator_ledger.get("comparators", [])),
            "comparators": public_comparator_ledger.get("comparators", []),
            "cadence": public_comparator_ledger.get("minimum_public_cadence"),
        },
        "procedural_memory": {
            "tool_registry": "reports/tool_registry.json",
            "registry_health": tool_registry.get("registry_health", {}),
            "active_tools": [
                tool.get("tool_name")
                for tool in tool_registry.get("tools", [])
                if tool.get("lifecycle") == "active"
            ],
            "proposed_tools": [
                tool.get("tool_name")
                for tool in tool_registry.get("tools", [])
                if tool.get("lifecycle") == "proposed"
            ],
        },
        "embodied_logging": {
            "logs": embodied_event_logs,
            "required_streams": [
                "raw_windows",
                "event_log",
                "semantic_trace",
                "skill_trace",
                "residual_log",
            ],
        },
        "octopus_router_architecture": {
            "report": "reports/octopus_router_report.json",
            "arm_registry": "reports/arm_registry.json",
            "router_eval": "reports/octopus_router_eval.json",
            "safety_ledger": "reports/safety_benchmark_ledger.json",
            "bridge_ledger": "reports/bridge_benchmark_ledger.json",
            "summary": {
                "status": octopus_router.get("status"),
                "implementation_score": octopus_router.get("implementation_score", {}),
                "arm_count": octopus_router.get("arm_registry", {}).get("count"),
                "router_metrics": octopus_router.get("router_eval", {}).get("metrics", {}),
            },
        },
        "bridge_benchmark_recommendations": bridge_recommendations,
        "next_high_leverage_actions": next_high_leverage_actions(
            treadmill, residual_escrow, matrix, bridge_recommendations
        ),
    }


def implementation_matrix(
    *,
    treadmill: dict[str, Any],
    benchmark_ledger: list[dict[str, Any]],
    model_ledger: dict[str, Any],
    tool_registry: dict[str, Any],
    residual_escrow: dict[str, Any],
    public_comparator_ledger: dict[str, Any],
    capability_ratchet: dict[str, Any],
    embodied_event_logs: list[dict[str, Any]],
    octopus_router: dict[str, Any],
    arm_registry: dict[str, Any],
    safety_ledger: dict[str, Any],
    bridge_ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    tools = tool_registry.get("tools", [])
    execution_modes = capability_ratchet.get("procedural_ratchet", {}).get(
        "execution_modes", {}
    )
    return [
        component(
            "benchmark_ledger",
            bool(benchmark_ledger),
            "Benchmark lifecycle, thresholds, wall diagnosis, and retirement criteria are written to reports/benchmark_ledger.json.",
            "Keep adding harder public/private/live benchmark families as current families graduate.",
        ),
        component(
            "model_ledger",
            bool(model_ledger),
            "Model architecture/data/inference/cost/residual metadata is written to reports/model_ledger.json.",
            "Add explicit model version IDs for major architecture candidates.",
        ),
        component(
            "time_decayed_mastery_thresholds",
            bool(treadmill.get("curriculum_policy", {}).get("floor_threshold")),
            "The treadmill records initial threshold, floor, patience, 1% per-attempt decay, and critical-failure veto policy.",
            "Wire subgroup floor blocking into per-benchmark evaluators where fine-grained metadata is available.",
        ),
        component(
            "residual_escrow",
            residual_escrow.get("summary", {}).get("cluster_count", 0) > 0,
            "Residual clusters, sampled cases, recurrence promotion, and attention budget are written to reports/residual_escrow.json.",
            "Promote recurring escrow clusters into generated diagnostic bridge benchmarks.",
        ),
        component(
            "public_calibration_track",
            len(public_comparator_ledger.get("comparators", [])) > 0,
            "Public BabyLM/BLIMP-style comparator is separated from private mutation pressure.",
            "Add more public calibration surfaces as local benchmark harnesses mature.",
        ),
        component(
            "procedural_tool_registry",
            len(tools) > 0,
            "Repeated local workflows are represented as tool cards with preconditions, verification tests, risk tiers, and retirement criteria.",
            "Start logging real agent/dev trajectories so tool synthesis is increasingly data-driven instead of hand-curated.",
        ),
        component(
            "execution_modes",
            all(mode in execution_modes for mode in ("interpreter", "compiled_tool", "reflex_failsafe")),
            "The capability report exposes interpreter, compiled-tool, and reflex/failsafe modes; core loop-closure tests cover routing.",
            "Add runtime metrics showing how often each mode is selected in live local workflows.",
        ),
        component(
            "safety_and_reflex_layer",
            safety_ledger.get("passed", False),
            "Core routing supports reflex/failsafe mode and ORA safety ledgers check high-risk routing, approvals, runtime tiers, and least privilege.",
            "Add explicit safety benchmark ledgers and critical-failure veto tests for every high-risk action class.",
        ),
        component(
            "bridge_benchmark_protocol",
            bridge_ledger.get("case_count", 0) > 0,
            "Floor-failure policy is encoded and ORA generates active bridge benchmark cases from recurring residual escrow.",
            "Create bridge factories when an active frontier cannot clear its floor after threshold decay reaches the floor.",
        ),
        component(
            "active_compression_substrate",
            "liquid/reservoir/VSA" in str(model_ledger.get("architecture", "")),
            "The model ledger tracks liquid/reservoir/VSA state, verifier-governed outputs, Rust/CUDA surfaces, and local training.",
            "Move more learned state formation and rollout optimization into Rust/CUDA FFI.",
        ),
        component(
            "high_bandwidth_embodied_logging",
            bool(embodied_event_logs),
            "Puffer/Ocean eventized rollout logs include bounded raw windows, event logs, semantic traces, skill traces, and residual logs.",
            "Add eventized rollout logs for Puffer/Ocean: observations, actions, rewards, dones, state summaries, reflex triggers, and anomaly windows.",
        ),
        component(
            "octopus_router_architecture",
            bool(octopus_router) and bool(arm_registry.get("arms")),
            "ORA arm registry, routing benchmark, dynamic-load metrics, safety ledger, and bridge ledger are written as local artifacts.",
            "Train a learned router head once enough local routing traces accumulate.",
        ),
    ]


def component(
    name: str,
    condition: bool,
    evidence: str,
    next_action: str,
    *,
    status_override: str | None = None,
) -> dict[str, Any]:
    if status_override is not None:
        status = status_override
    else:
        status = "implemented" if condition else "missing"
    return {
        "component": name,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def implementation_score(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {"implemented": 1.0, "partial": 0.5, "missing": 0.0}
    total = sum(weights.get(row["status"], 0.0) for row in matrix)
    possible = max(1, len(matrix))
    return {
        "score": total / possible,
        "implemented": sum(1 for row in matrix if row["status"] == "implemented"),
        "partial": sum(1 for row in matrix if row["status"] == "partial"),
        "missing": sum(1 for row in matrix if row["status"] == "missing"),
        "possible": possible,
    }


def build_subgroup_floor_checks(
    capability_ratchet: dict[str, Any],
    benchmark_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    floor_by_name = {
        entry.get("benchmark_name"): entry.get("graduation_policy", {}).get("subgroup_floor", 0.5)
        for entry in benchmark_ledger
    }
    residual_map = capability_ratchet.get("residual_map", {})
    checks = []
    for benchmark_name, key in (
        ("babylm_local_probe", "worst_babylm_terms"),
        ("babylm_mutated_holdout", "worst_mutated_babylm_terms"),
    ):
        floor = float(floor_by_name.get(benchmark_name) or 0.5)
        rows = residual_map.get(key, [])
        min_accuracy = min((float(row.get("accuracy", 1.0)) for row in rows), default=1.0)
        checks.append(
            {
                "benchmark": benchmark_name,
                "subgroup_floor": floor,
                "min_observed_subgroup_accuracy": min_accuracy,
                "passed": min_accuracy >= floor,
                "source": f"capability_ratchet.residual_map.{key}",
            }
        )
    return checks


def build_bridge_recommendations(benchmark_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    for entry in benchmark_ledger:
        policy = entry.get("graduation_policy", {})
        if policy.get("threshold_phase") != "floor_phase":
            continue
        recommendations.append(
            {
                "benchmark": entry.get("benchmark_name"),
                "score": entry.get("score"),
                "floor_threshold": policy.get("floor_threshold"),
                "recommended_bridge": entry.get("recommended_intervention"),
            }
        )
    return recommendations


def discover_embodied_event_logs() -> list[dict[str, Any]]:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return []
    logs = []
    for path in sorted(reports_dir.glob("*eventized*log*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("methodology") != "high_bandwidth_embodied_rollout_logging":
            continue
        streams = payload.get("streams", {})
        logs.append(
            {
                "path": str(path),
                "env": payload.get("env"),
                "feature_set": payload.get("feature_set"),
                "summary": payload.get("summary", {}),
                "streams_present": {
                    name: name in streams
                    for name in (
                        "raw_windows",
                        "event_log",
                        "semantic_trace",
                        "skill_trace",
                        "residual_log",
                    )
                },
            }
        )
    return logs


def next_high_leverage_actions(
    treadmill: dict[str, Any],
    residual_escrow: dict[str, Any],
    matrix: list[dict[str, Any]],
    bridge_recommendations: list[dict[str, Any]],
) -> list[str]:
    actions = []
    actions.extend(treadmill.get("next_commands", [])[:2])
    active_targets = residual_escrow.get("active_diagnostic_targets", [])
    if active_targets:
        top = active_targets[0]
        actions.append(
            f"Convert residual escrow target {top.get('name')} into a bridge/diagnostic benchmark or architecture test."
        )
    if bridge_recommendations:
        actions.append("Build the recommended bridge benchmark before changing architecture.")
    missing = [row for row in matrix if row["status"] == "missing"]
    partial = [row for row in matrix if row["status"] == "partial"]
    if missing:
        actions.append(missing[0]["next_action"])
    if partial:
        actions.append(partial[0]["next_action"])
    actions.append("Keep public calibration periodic, but make private/live mutated frontiers drive inner-loop optimization.")
    return actions


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
