"""Run the local Capability Ratchet maintenance workflow.

This is the compiled development loop for SymLiquid:
benchmark ledger -> residual map -> benchmark factory -> model ledger ->
tool registry -> next intervention.

The script does not call external inference providers. It orchestrates local
Python/Rust reports already produced by SymLiquid.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    parser.add_argument(
        "--public-babylm-report",
        default="reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json",
    )
    parser.add_argument("--public-babylm-eval", default="data/babylm_blimp_filtered_eval.jsonl")
    parser.add_argument(
        "--mutated-babylm-report",
        default=None,
    )
    parser.add_argument("--mutated-babylm-eval", default=None)
    parser.add_argument("--mutated-count", type=int, default=4800)
    parser.add_argument("--mutated-seed", type=int, default=None)
    parser.add_argument("--out", default="reports/capability_ratchet_run.json")
    parser.add_argument("--workflow-trace-out", default="reports/workflow_routing_traces.jsonl")
    parser.add_argument("--skip-holdout-generation", action="store_true")
    parser.add_argument("--force-holdout-generation", action="store_true")
    args = parser.parse_args()

    latest_seed, latest_report, latest_eval = resolve_latest_mutated_babylm(args.reports)
    if args.mutated_babylm_report is None:
        args.mutated_babylm_report = latest_report or (
            "reports/babylm_mutated_holdout_seed49_stateful_grammar_state_frontier.json"
        )
    if args.mutated_babylm_eval is None:
        args.mutated_babylm_eval = latest_eval or "data/babylm_mutated_holdout_seed49.jsonl"
    if args.mutated_seed is None:
        args.mutated_seed = latest_seed or 49

    commands: list[list[str]] = [
        [
            sys.executable,
            "scripts/benchmark_treadmill.py",
            "--reports",
            args.reports,
            "--out",
            "reports/benchmark_treadmill_status.json",
            "--benchmark-ledger-out",
            "reports/benchmark_ledger.json",
            "--model-ledger-out",
            "reports/model_ledger.json",
            "--public-comparator-ledger-out",
            "reports/public_comparator_ledger.json",
        ]
    ]

    if Path(args.public_babylm_report).exists() and Path(args.public_babylm_eval).exists():
        commands.append(
            [
                sys.executable,
                "scripts/analyze_babylm_residuals.py",
                "--report",
                args.public_babylm_report,
                "--eval-input",
                args.public_babylm_eval,
                "--out",
                "reports/babylm_residual_analysis.json",
            ]
        )

    should_generate_holdout = (
        not args.skip_holdout_generation
        and (args.force_holdout_generation or not Path(args.mutated_babylm_eval).exists())
    )
    if should_generate_holdout:
        commands.append(
            [
                sys.executable,
                "scripts/generate_babylm_mutated_holdout.py",
                "--residual-analysis",
                "reports/babylm_residual_analysis.json",
                "--count",
                str(args.mutated_count),
                "--seed",
                str(args.mutated_seed),
                "--out",
                args.mutated_babylm_eval,
                "--report-out",
                f"reports/babylm_mutated_holdout_seed{args.mutated_seed}_factory.json",
            ]
        )

    if Path(args.mutated_babylm_report).exists() and Path(args.mutated_babylm_eval).exists():
        commands.append(
            [
                sys.executable,
                "scripts/analyze_babylm_residuals.py",
                "--report",
                args.mutated_babylm_report,
                "--eval-input",
                args.mutated_babylm_eval,
                "--out",
                "reports/babylm_mutated_residual_analysis.json",
            ]
        )

    commands.extend(
        [
            [
                sys.executable,
                "scripts/benchmark_treadmill.py",
                "--reports",
                args.reports,
                "--out",
                "reports/benchmark_treadmill_status.json",
                "--benchmark-ledger-out",
                "reports/benchmark_ledger.json",
                "--model-ledger-out",
                "reports/model_ledger.json",
                "--public-comparator-ledger-out",
                "reports/public_comparator_ledger.json",
            ],
            [
                sys.executable,
                "scripts/capability_ratchet.py",
                "--benchmark-ledger",
                "reports/benchmark_ledger.json",
                "--model-ledger",
                "reports/model_ledger.json",
                "--residual-analysis",
                "reports/babylm_residual_analysis.json",
                "--mutated-residual-analysis",
                "reports/babylm_mutated_residual_analysis.json",
                "--public-comparator-ledger",
                "reports/public_comparator_ledger.json",
                "--out",
                "reports/capability_ratchet_report.json",
                "--tool-registry-out",
                "reports/tool_registry.json",
                "--residual-escrow-out",
                "reports/residual_escrow.json",
            ],
        ]
    )

    eventized_command = eventized_rollout_command()
    if eventized_command is not None:
        commands.append(eventized_command)

    commands.append(
        [
            sys.executable,
            "scripts/octopus_router.py",
            "--benchmark-ledger",
            "reports/benchmark_ledger.json",
            "--model-ledger",
            "reports/model_ledger.json",
            "--tool-registry",
            "reports/tool_registry.json",
            "--residual-escrow",
            "reports/residual_escrow.json",
            "--capability-ratchet",
            "reports/capability_ratchet_report.json",
            "--event-log",
            "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
            "--arm-registry-out",
            "reports/arm_registry.json",
            "--router-eval-out",
            "reports/octopus_router_eval.json",
            "--routing-memory-out",
            "reports/routing_memory.json",
            "--arm-lifecycle-out",
            "reports/arm_lifecycle_ledger.json",
            "--safety-ledger-out",
            "reports/safety_benchmark_ledger.json",
            "--bridge-ledger-out",
            "reports/bridge_benchmark_ledger.json",
            "--bridge-out",
            "benchmarks/bridges/babylm_wh_gap_bridge.jsonl",
            "--out",
            "reports/octopus_router_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/train_octopus_router_head.py",
            "--router-eval",
            "reports/octopus_router_eval.json",
            "--arm-registry",
            "reports/arm_registry.json",
            "--dataset-out",
            "reports/octopus_router_trace_dataset.json",
            "--model-out",
            "reports/octopus_router_head_model.json",
            "--eval-out",
            "reports/octopus_router_head_eval.json",
            "--extra-traces",
            args.workflow_trace_out,
            "--out",
            "reports/octopus_router_head_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/octopus_router.py",
            "--benchmark-ledger",
            "reports/benchmark_ledger.json",
            "--model-ledger",
            "reports/model_ledger.json",
            "--tool-registry",
            "reports/tool_registry.json",
            "--residual-escrow",
            "reports/residual_escrow.json",
            "--capability-ratchet",
            "reports/capability_ratchet_report.json",
            "--event-log",
            "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
            "--arm-registry-out",
            "reports/arm_registry.json",
            "--router-eval-out",
            "reports/octopus_router_eval.json",
            "--routing-memory-out",
            "reports/routing_memory.json",
            "--arm-lifecycle-out",
            "reports/arm_lifecycle_ledger.json",
            "--safety-ledger-out",
            "reports/safety_benchmark_ledger.json",
            "--bridge-ledger-out",
            "reports/bridge_benchmark_ledger.json",
            "--bridge-out",
            "benchmarks/bridges/babylm_wh_gap_bridge.jsonl",
            "--router-head-report",
            "reports/octopus_router_head_report.json",
            "--router-head-eval",
            "reports/octopus_router_head_eval.json",
            "--out",
            "reports/octopus_router_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/ratcheting_generative_system.py",
            "--benchmark-treadmill",
            "reports/benchmark_treadmill_status.json",
            "--benchmark-ledger",
            "reports/benchmark_ledger.json",
            "--model-ledger",
            "reports/model_ledger.json",
            "--tool-registry",
            "reports/tool_registry.json",
            "--residual-escrow",
            "reports/residual_escrow.json",
            "--public-comparator-ledger",
            "reports/public_comparator_ledger.json",
            "--capability-ratchet",
            "reports/capability_ratchet_report.json",
            "--octopus-router",
            "reports/octopus_router_report.json",
            "--arm-registry",
            "reports/arm_registry.json",
            "--safety-ledger",
            "reports/safety_benchmark_ledger.json",
            "--bridge-ledger",
            "reports/bridge_benchmark_ledger.json",
            "--out",
            "reports/ratcheting_generative_system_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/ratcheting_modular_intelligence.py",
            "--benchmark-treadmill",
            "reports/benchmark_treadmill_status.json",
            "--benchmark-ledger",
            "reports/benchmark_ledger.json",
            "--model-ledger",
            "reports/model_ledger.json",
            "--tool-registry",
            "reports/tool_registry.json",
            "--residual-escrow",
            "reports/residual_escrow.json",
            "--public-comparator-ledger",
            "reports/public_comparator_ledger.json",
            "--capability-ratchet",
            "reports/capability_ratchet_report.json",
            "--rgs",
            "reports/ratcheting_generative_system_report.json",
            "--octopus-router",
            "reports/octopus_router_report.json",
            "--arm-registry",
            "reports/arm_registry.json",
            "--routing-memory",
            "reports/routing_memory.json",
            "--arm-lifecycle",
            "reports/arm_lifecycle_ledger.json",
            "--router-head",
            "reports/octopus_router_head_report.json",
            "--router-head-eval",
            "reports/octopus_router_head_eval.json",
            "--safety-ledger",
            "reports/safety_benchmark_ledger.json",
            "--bridge-ledger",
            "reports/bridge_benchmark_ledger.json",
            "--event-log",
            "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
            "--out",
            "reports/ratcheting_modular_intelligence_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/architecture_gate.py",
            "--capability-ratchet",
            "reports/capability_ratchet_report.json",
            "--rgs",
            "reports/ratcheting_generative_system_report.json",
            "--octopus-router",
            "reports/octopus_router_report.json",
            "--rmi",
            "reports/ratcheting_modular_intelligence_report.json",
            "--router-head",
            "reports/octopus_router_head_report.json",
            "--router-head-eval",
            "reports/octopus_router_head_eval.json",
            "--router-eval",
            "reports/octopus_router_eval.json",
            "--benchmark-ledger",
            "reports/benchmark_ledger.json",
            "--public-comparator-ledger",
            "reports/public_comparator_ledger.json",
            "--residual-escrow",
            "reports/residual_escrow.json",
            "--tool-registry",
            "reports/tool_registry.json",
            "--safety-ledger",
            "reports/safety_benchmark_ledger.json",
            "--bridge-ledger",
            "reports/bridge_benchmark_ledger.json",
            "--out",
            "reports/architecture_gate_report.json",
        ]
    )

    commands.append(
        [
            sys.executable,
            "scripts/training_preflight.py",
            "--run-split-check",
            "--run-candidate-gate",
            "--out",
            "reports/training_preflight_report.json",
        ]
    )

    run_log = []
    for command in commands:
        started = time.time()
        result = subprocess.run(command, text=True, capture_output=True)
        runtime_ms = int((time.time() - started) * 1000)
        run_log.append(
            {
                "command": " ".join(command),
                "returncode": result.returncode,
                "runtime_ms": runtime_ms,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        append_workflow_trace(
            Path(args.workflow_trace_out),
            command=command,
            returncode=result.returncode,
            runtime_ms=runtime_ms,
        )
        if result.returncode != 0:
            write_json(
                Path(args.out),
                build_report(False, run_log, "command_failed", args),
            )
            print(run_log[-1]["stderr_tail"], file=sys.stderr)
            return result.returncode

    report = build_report(True, run_log, "ok", args)
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def build_report(
    ok: bool,
    run_log: list[dict[str, Any]],
    status: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "capability_ratchet_compiled_workflow",
        "status": status,
        "ok": ok,
        "steps": len(run_log),
        "commands": run_log,
        "outputs": {
            "benchmark_treadmill": "reports/benchmark_treadmill_status.json",
            "benchmark_ledger": "reports/benchmark_ledger.json",
            "model_ledger": "reports/model_ledger.json",
            "public_comparator_ledger": "reports/public_comparator_ledger.json",
            "babylm_residual_analysis": "reports/babylm_residual_analysis.json",
            "mutated_babylm_eval": args.mutated_babylm_eval,
            "mutated_babylm_residual_analysis": "reports/babylm_mutated_residual_analysis.json",
            "residual_escrow": "reports/residual_escrow.json",
            "capability_ratchet": "reports/capability_ratchet_report.json",
            "ratcheting_generative_system": "reports/ratcheting_generative_system_report.json",
            "ratcheting_modular_intelligence": "reports/ratcheting_modular_intelligence_report.json",
            "embodied_event_log": "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
            "octopus_router": "reports/octopus_router_report.json",
            "arm_registry": "reports/arm_registry.json",
            "router_eval": "reports/octopus_router_eval.json",
            "routing_memory": "reports/routing_memory.json",
            "arm_lifecycle_ledger": "reports/arm_lifecycle_ledger.json",
            "octopus_router_trace_dataset": "reports/octopus_router_trace_dataset.json",
            "octopus_router_head_model": "reports/octopus_router_head_model.json",
            "octopus_router_head_eval": "reports/octopus_router_head_eval.json",
            "octopus_router_head_report": "reports/octopus_router_head_report.json",
            "safety_benchmark_ledger": "reports/safety_benchmark_ledger.json",
            "bridge_benchmark_ledger": "reports/bridge_benchmark_ledger.json",
            "bridge_benchmark": "benchmarks/bridges/babylm_wh_gap_bridge.jsonl",
            "tool_registry": "reports/tool_registry.json",
            "architecture_gate": "reports/architecture_gate_report.json",
            "training_preflight": "reports/training_preflight_report.json",
            "workflow_routing_traces": args.workflow_trace_out,
        },
        "external_inference_calls": 0,
    }


def append_workflow_trace(
    path: Path, *, command: list[str], returncode: int, runtime_ms: int
) -> None:
    trace = classify_workflow_trace(command)
    payload = {
        "trace_id": f"workflow_{int(time.time() * 1000)}_{abs(hash(tuple(command))) % 1000000}",
        "task": trace["task"],
        "workflow": trace["workflow"],
        "command": " ".join(command),
        "selected_arms": trace["selected_arms"],
        "expected_arms": trace["selected_arms"],
        "risk": trace["risk"],
        "routing_pattern": trace["routing_pattern"],
        "returncode": returncode,
        "success": returncode == 0,
        "runtime_ms": runtime_ms,
        "review_step_count": len(trace["selected_arms"]),
        "review_step_basis": "selected_workflow_arms",
        "maintenance_mode": maintenance_mode_from_text(" ".join(command), trace["task"]),
        "maintenance_mode_basis": "command_or_task_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "split": "train",
        "source": "capability_ratchet_workflow",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def classify_workflow_trace(command: list[str]) -> dict[str, Any]:
    command_text = " ".join(command).lower()
    if "benchmark_treadmill.py" in command_text:
        return workflow(
            "benchmark ledger update and public comparator refresh",
            ["benchmark_ratchet_arm", "public_calibration_arm"],
            "parallel",
        )
    if "analyze_babylm_residuals.py" in command_text:
        return workflow(
            "BabyLM residual analysis and escrow clustering",
            ["babylm_grammar_arm", "residual_analysis_arm"],
            "sequential",
        )
    if "generate_babylm_mutated_holdout.py" in command_text:
        return workflow(
            "mutated BabyLM holdout generation from residual targets",
            ["babylm_grammar_arm", "benchmark_factory_arm"],
            "sequential",
        )
    if "capability_ratchet.py" in command_text:
        return workflow(
            "capability ratchet ledger and tool registry update",
            ["benchmark_ratchet_arm", "loop_closure_arm", "residual_analysis_arm"],
            "parallel",
        )
    if "symliquid_puffer_adapter.py" in command_text:
        return workflow(
            "Puffer/Ocean eventized rollout trace generation",
            ["puffer_ocean_arm", "embodied_logging_arm"],
            "single",
        )
    if "octopus_router.py" in command_text:
        return workflow(
            "Octopus router evaluation, arm registry, bridge and lifecycle update",
            ["head_router", "benchmark_ratchet_arm", "safety_reflex_arm"],
            "parallel",
        )
    if "train_octopus_router_head.py" in command_text:
        return workflow(
            "learned router head training from rule and workflow traces",
            ["head_router", "routing_memory_arm"],
            "single",
        )
    if "ratcheting_generative_system.py" in command_text:
        return workflow(
            "RGS system synthesis audit",
            ["systems_architecture_arm", "benchmark_ratchet_arm"],
            "parallel",
        )
    if "ratcheting_modular_intelligence.py" in command_text:
        return workflow(
            "RMI full architecture synthesis audit",
            ["systems_architecture_arm", "head_router", "safety_reflex_arm"],
            "parallel",
        )
    if "architecture_gate.py" in command_text:
        return workflow(
            "architecture gate verification before heavy training",
            ["systems_architecture_arm", "safety_reflex_arm", "benchmark_ratchet_arm"],
            "verification",
            risk="medium",
        )
    if "training_preflight.py" in command_text:
        return workflow(
            "real training preflight with CUDA telemetry, split checks, and promotion gates",
            ["systems_architecture_arm", "benchmark_ratchet_arm", "safety_reflex_arm"],
            "verification",
            risk="medium",
        )
    return workflow("local ratchet maintenance command", ["head_router"], "single")


def workflow(
    task: str,
    arms: list[str],
    routing_pattern: str,
    *,
    risk: str = "low",
) -> dict[str, Any]:
    return {
        "task": task,
        "workflow": task,
        "selected_arms": arms,
        "routing_pattern": routing_pattern,
        "risk": risk,
    }


def maintenance_mode_from_text(*values: str) -> str:
    text = " ".join(str(value or "") for value in values).lower().replace("-", "_").replace(" ", "_")
    if "circle" in text and "seed" in text and "rebuild" in text:
        return "circle_seed_rule_rebuild"
    return "object_only"


def eventized_rollout_command() -> list[str] | None:
    artifact = Path("reports/symliquid_ocean_slot_tmaze_policy_rust_trainer_seed3.json")
    if not artifact.exists():
        return None
    return [
        sys.executable,
        "adapters/pufferlib/symliquid_puffer_adapter.py",
        "--artifact",
        str(artifact),
        "--env",
        "ocean-slot-tmaze",
        "--num-envs",
        "32",
        "--rollout-smoke-steps",
        "64",
        "--event-log-out",
        "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
        "--event-log-env-limit",
        "4",
        "--event-log-step-limit",
        "64",
        "--out",
        "reports/puffer_ocean_slot_tmaze_eventized_smoke.json",
    ]


def resolve_latest_mutated_babylm(reports_dir: str) -> tuple[int | None, str | None, str | None]:
    """Pick the latest trained mutated BabyLM seed with a matching eval file."""

    report_pattern = re.compile(
        r"babylm_mutated_holdout_seed(\d+)_stateful_grammar_state_frontier\.json$"
    )
    candidates: list[tuple[int, Path, Path]] = []
    for report in Path(reports_dir).glob("babylm_mutated_holdout_seed*_stateful_grammar_state_frontier.json"):
        match = report_pattern.match(report.name)
        if match is None:
            continue
        seed = int(match.group(1))
        eval_path = Path(f"data/babylm_mutated_holdout_seed{seed}.jsonl")
        if eval_path.exists():
            candidates.append((seed, report, eval_path))
    if not candidates:
        return None, None, None
    seed, report, eval_path = max(candidates, key=lambda item: item[0])
    return seed, str(report), str(eval_path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
