"""Pre-training architecture gate for SymLiquid.

The gate checks whether the local ratchet, ORA router, safety/quarantine,
public calibration, residual escrow, and learned router head are coherent
enough to justify heavier training. It is a local-only verifier and does not
call external inference providers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-ratchet", default="reports/capability_ratchet_report.json")
    parser.add_argument("--rgs", default="reports/ratcheting_generative_system_report.json")
    parser.add_argument("--rmi", default="reports/ratcheting_modular_intelligence_report.json")
    parser.add_argument("--octopus-router", default="reports/octopus_router_report.json")
    parser.add_argument("--router-head", default="reports/octopus_router_head_report.json")
    parser.add_argument("--router-head-eval", default="reports/octopus_router_head_eval.json")
    parser.add_argument("--router-eval", default="reports/octopus_router_eval.json")
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--public-comparator-ledger", default="reports/public_comparator_ledger.json")
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument("--tool-registry", default="reports/tool_registry.json")
    parser.add_argument("--routing-memory", default="reports/routing_memory.json")
    parser.add_argument("--arm-lifecycle", default="reports/arm_lifecycle_ledger.json")
    parser.add_argument("--safety-ledger", default="reports/safety_benchmark_ledger.json")
    parser.add_argument("--bridge-ledger", default="reports/bridge_benchmark_ledger.json")
    parser.add_argument("--out", default="reports/architecture_gate_report.json")
    args = parser.parse_args()

    artifacts = {
        "capability_ratchet": read_json(args.capability_ratchet, {}),
        "rgs": read_json(args.rgs, {}),
        "rmi": read_json(args.rmi, {}),
        "octopus_router": read_json(args.octopus_router, {}),
        "router_head": read_json(args.router_head, {}),
        "router_head_eval": read_json(args.router_head_eval, {}),
        "router_eval": read_json(args.router_eval, {}),
        "benchmark_ledger": read_json(args.benchmark_ledger, []),
        "public_comparator_ledger": read_json(args.public_comparator_ledger, {}),
        "residual_escrow": read_json(args.residual_escrow, {}),
        "tool_registry": read_json(args.tool_registry, {}),
        "routing_memory": read_json(args.routing_memory, {}),
        "arm_lifecycle": read_json(args.arm_lifecycle, {}),
        "safety_ledger": read_json(args.safety_ledger, {}),
        "bridge_ledger": read_json(args.bridge_ledger, {}),
    }

    report = build_report(args, artifacts)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_heavy_training"] else 2


def build_report(args: argparse.Namespace, artifacts: dict[str, Any]) -> dict[str, Any]:
    rgs_score = score(artifacts["rgs"])
    rmi_score = score(artifacts["rmi"])
    ora_score = score(artifacts["octopus_router"])
    router_metrics = artifacts["router_eval"].get("metrics", {})
    head_metrics = artifacts["router_head_eval"].get("metrics", {})
    benchmark_ledger = artifacts["benchmark_ledger"]
    regressions = [
        row
        for row in benchmark_ledger
        if row.get("lifecycle") == "regression"
    ]
    comparators = artifacts["public_comparator_ledger"].get("comparators", [])
    residual_summary = artifacts["residual_escrow"].get("summary", {})
    tools = artifacts["tool_registry"].get("tools", [])
    routing_memory = artifacts["routing_memory"]
    arm_lifecycle = artifacts["arm_lifecycle"]
    bridge = artifacts["bridge_ledger"]
    safety = artifacts["safety_ledger"]

    checks = [
        gate("rgs_complete", rgs_score >= 1.0, f"RGS implementation score={rgs_score:.3f}"),
        gate("rmi_complete", rmi_score >= 1.0, f"RMI implementation score={rmi_score:.3f}"),
        gate("ora_complete", ora_score >= 1.0, f"ORA implementation score={ora_score:.3f}"),
        gate(
            "rule_router_eval_passed",
            router_metrics.get("selection_accuracy", 0.0) >= 1.0
            and router_metrics.get("risk_routing_accuracy", 0.0) >= 1.0,
            f"selection={router_metrics.get('selection_accuracy')} risk={router_metrics.get('risk_routing_accuracy')}",
        ),
        gate(
            "learned_router_head_promoted",
            artifacts["router_head"].get("promotion_gate_passed", False)
            and head_metrics.get("exact_set_accuracy", 0.0) >= 0.95
            and head_metrics.get("risk_routing_accuracy", 0.0) >= 1.0,
            f"exact={head_metrics.get('exact_set_accuracy')} risk={head_metrics.get('risk_routing_accuracy')}",
        ),
        gate("safety_ledger_passed", safety.get("passed", False), safety_summary(safety)),
        gate("regression_suite_present", len(regressions) >= 1, f"regression_count={len(regressions)}"),
        gate("public_calibration_present", len(comparators) >= 1, f"public_comparators={len(comparators)}"),
        gate(
            "residual_escrow_present",
            residual_summary.get("cluster_count", 0) > 0,
            f"clusters={residual_summary.get('cluster_count')} cases={residual_summary.get('case_count')}",
        ),
        gate(
            "bridge_benchmark_present",
            bridge.get("case_count", 0) > 0,
            f"bridge_cases={bridge.get('case_count')} paths={bridge.get('generated_paths')}",
        ),
        gate(
            "procedural_tools_registered",
            len(tools) >= 15,
            f"tool_count={len(tools)} active={artifacts['tool_registry'].get('registry_health', {}).get('active')}",
        ),
        gate(
            "routing_memory_present",
            routing_memory.get("summary", {}).get("entries", 0) > 0,
            f"entries={routing_memory.get('summary', {}).get('entries')} arm_memories={routing_memory.get('summary', {}).get('arm_memories')}",
        ),
        gate(
            "arm_lifecycle_governed",
            arm_lifecycle.get("summary", {}).get("arms", 0) > 0,
            f"arms={arm_lifecycle.get('summary', {}).get('arms')} split_candidates={arm_lifecycle.get('summary', {}).get('split_candidates')}",
        ),
        gate(
            "external_inference_zero",
            external_inference_ok(artifacts),
            "all architecture artifacts report zero external inference calls/violations",
        ),
    ]
    ready = all(row["passed"] for row in checks)
    return {
        "policy": "local_only_no_external_inference",
        "framework": "symliquid_architecture_gate",
        "status": "ready_for_heavy_training" if ready else "blocked",
        "ready_for_heavy_training": ready,
        "gate_count": len(checks),
        "passed_count": sum(1 for row in checks if row["passed"]),
        "checks": checks,
        "artifacts": {
            "capability_ratchet": args.capability_ratchet,
            "rgs": args.rgs,
            "rmi": args.rmi,
            "octopus_router": args.octopus_router,
            "router_eval": args.router_eval,
            "router_head": args.router_head,
            "router_head_eval": args.router_head_eval,
            "safety_ledger": args.safety_ledger,
            "bridge_ledger": args.bridge_ledger,
            "tool_registry": args.tool_registry,
            "routing_memory": args.routing_memory,
            "arm_lifecycle": args.arm_lifecycle,
        },
        "training_policy": {
            "heavy_training_allowed": ready,
            "rule": "Do not start heavy training unless this gate is green; failed gates become ratchet residuals.",
            "next_frontier": "BabyLM grammar state and Rust/CUDA rollout training only after architecture setup stays green.",
        },
        "next_actions": next_actions(checks),
        "external_inference_calls": 0,
    }


def gate(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "evidence": evidence,
    }


def score(report: dict[str, Any]) -> float:
    return float(report.get("implementation_score", {}).get("score", 0.0))


def safety_summary(safety: dict[str, Any]) -> str:
    tests = safety.get("tests", [])
    passed = sum(1 for row in tests if row.get("passed"))
    return f"passed={safety.get('passed')} tests={passed}/{len(tests)}"


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


def next_actions(checks: list[dict[str, Any]]) -> list[str]:
    failed = [row["gate"] for row in checks if not row["passed"]]
    if not failed:
        return [
            "Proceed to heavier local training only through the ratchet workflow.",
            "Re-run this gate after every architecture change, benchmark frontier update, or arm split/merge.",
        ]
    return [
        f"Resolve gate: {name}"
        for name in failed
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
