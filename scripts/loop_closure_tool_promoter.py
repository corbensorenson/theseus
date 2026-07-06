"""Promote high-value loop-closure candidates into the tool registry.

The harvester finds repeated workflows; this script turns the safest recurring
ones into explicit, verified tool cards. It only registers local deterministic
commands and never calls the teacher or external inference.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvester", default="reports/loop_closure_harvester.json")
    parser.add_argument("--registry", default="reports/tool_registry.json")
    parser.add_argument("--out", default="reports/loop_closure_tool_promoter.json")
    args = parser.parse_args()

    harvester = read_json(ROOT / args.harvester)
    registry_path = ROOT / args.registry
    registry = read_json(registry_path) or {
        "policy": "local_only_no_external_inference",
        "framework": "capability_ratchet_tool_registry",
        "registry_health": {},
        "tools": [],
    }
    before = len(registry.get("tools", []))
    existing = {str(tool.get("tool_name")) for tool in registry.get("tools", []) if isinstance(tool, dict)}
    promoted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for tool in curated_tools(harvester):
        blocked_reason = benchmark_tool_block_reason(tool)
        if blocked_reason:
            blocked.append({"tool_name": tool["tool_name"], "task_family": tool["task_family"], "blocked_reason": blocked_reason})
            continue
        if tool["tool_name"] in existing:
            continue
        registry.setdefault("tools", []).append(tool)
        existing.add(tool["tool_name"])
        promoted.append({"tool_name": tool["tool_name"], "task_family": tool["task_family"]})

    refresh_health(registry)
    write_json(registry_path, registry)
    report = {
        "policy": "sparkstream_loop_closure_tool_promoter_v0",
        "created_utc": now(),
        "registry": args.registry,
        "harvester": args.harvester,
        "before_tools": before,
        "after_tools": len(registry.get("tools", [])),
        "promoted": promoted,
        "blocked": blocked,
        "learning_integrity": {
            "benchmark_task_tool_distillation_allowed": False,
            "policy": "benchmark/eval/frontier workflows must become residuals or training data, not benchmark-answer tools",
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def curated_tools(harvester: dict[str, Any]) -> list[dict[str, Any]]:
    ready_names = {
        str(item.get("tool_name"))
        for item in harvester.get("candidates", [])
        if isinstance(item, dict) and item.get("status") == "ready_for_tool_synthesis"
    }
    tools = [
        tool_card(
            "benchmark_adapter_smoke_tool",
            "benchmark_ingestion",
            "Run local adapter-card smoke tests and classify ready, runtime-blocked, and governance-blocked sources.",
            "python scripts/benchmark_adapter_smoke.py --out reports/benchmark_adapter_smoke_status.json",
            ["factory_report", "resource_pantry", "card_id", "needs_smoke_only", "limit", "out"],
            ["scripts/benchmark_adapter_smoke.py", "reports/benchmark_adapter_factory.json"],
            ["python_compile", "smoke_report_schema", "external_inference_calls_zero"],
        ),
        tool_card(
            "candidate_bottleneck_reducer_tool",
            "benchmark_ingestion",
            "Reduce safe local setup/runtime blockers before escalating to the teacher.",
            "python scripts/candidate_bottleneck_reducer.py --fix --out reports/candidate_bottleneck_reducer.json",
            ["policy", "out", "fix"],
            ["scripts/candidate_bottleneck_reducer.py", "configs/candidate_bottleneck_policy.json"],
            ["python_compile", "candidate_flow_ready_or_manual_blockers_classified", "external_inference_calls_zero"],
        ),
        tool_card(
            "pressure_runner_tool",
            "benchmark_pressure",
            "Turn smoke-passed adapter cards into scored local frontier reports.",
            "python scripts/pressure_runner.py --card-id source_gym_pybullet_drones --frontier-family drone_rl --out reports/pressure_source_gym_pybullet_drones_seed1.json",
            ["card_id", "frontier_family", "seed", "episodes", "steps", "out"],
            ["scripts/pressure_runner.py", "benchmarks/cards"],
            ["python_compile", "treadmill_compatible_summary", "external_inference_calls_zero"],
        ),
        tool_card(
            "code_residual_forge_tool",
            "code_frontier_residual_forge",
            "Classify code benchmark residuals, write repair traces, export code transfer artifacts, and emit same-family rotation hints.",
            "python scripts/code_residual_forge.py --out reports/code_residual_forge.json --transfer-out reports/code_transfer_artifacts.json --rotation-out reports/code_frontier_rotation.json --trace-out reports/code_repair_traces.jsonl",
            ["policy", "benchmark_ledger", "curriculum", "frontier_policy", "profile_report", "out", "transfer_out", "rotation_out", "trace_out"],
            ["scripts/code_residual_forge.py", "configs/code_residual_forge_policy.json", "reports/code_residual_forge.json"],
            ["python_compile", "code_residual_clusters_nonempty", "transfer_artifacts_written", "external_inference_calls_zero"],
        ),
        tool_card(
            "synthetic_benchmark_factory_tool",
            "synthetic_benchmark_pressure",
            "Generate local synthetic benchmark mutations and cross-arm hybrid pressure cards from existing benchmark metadata and residual escrow.",
            "python scripts/synthetic_benchmark_factory.py --policy configs/synthetic_benchmark_policy.json --write-cards --out reports/synthetic_benchmark_factory.json --markdown-out reports/synthetic_benchmark_factory.md",
            ["policy", "benchmark_ledger", "residual_escrow", "arm_registry", "arm_sucker_registry", "code_transfer_artifacts", "write_cards", "out", "markdown_out"],
            ["scripts/synthetic_benchmark_factory.py", "configs/synthetic_benchmark_policy.json", "reports/synthetic_benchmark_factory.json"],
            ["python_compile", "synthetic_cases_generated", "cards_written", "public_comparator_quarantined", "external_inference_calls_zero"],
        ),
        tool_card(
            "multi_stream_code_pressure_tool",
            "multi_stream_code_pressure",
            "Generate causally verified multi-stream code traces, compare against single-stream repair, probe monitorability, and keep promotion private-pressure gated.",
            "python scripts/multi_stream_trace_factory.py --policy configs/multi_stream_policy.json --write-cards --out reports/multi_stream_trace_factory.json --markdown-out reports/multi_stream_trace_factory.md; python scripts/multi_stream_code_pressure_runner.py --card-id multistream_code_repair_pressure --seed 41 --case-manifest data/multi_stream_benchmarks/multistream_code_repair_pressure.jsonl --code-transfer-artifacts reports/code_transfer_artifacts.json --out reports/multi_stream_code_pressure_multistream_code_repair_pressure_seed41.json",
            ["policy", "case_manifest", "card_id", "seed", "code_transfer_artifacts", "out", "trace_out", "verifier_out", "single_stream_out"],
            ["scripts/multi_stream_trace_factory.py", "scripts/multi_stream_code_pressure_runner.py", "scripts/multi_stream_causal_verifier.py", "configs/multi_stream_policy.json"],
            ["python_compile", "causal_verifier_green", "apples_to_apples_case_overlap", "monitorability_probe_green", "external_inference_calls_zero"],
        ),
        tool_card(
            "local_code_repair_organism_tool",
            "code_frontier_repair",
            "Run the native code repair loop: candidate generation, sandbox tests, patch traces, residual classes, transfer consumption, and heredity delta.",
            "python scripts/local_code_repair_organism.py --card-id source_livecodebench --seed 14 --source-path D:/ProjectTheseus/resource_pantry/git/livecodebench --transfer-artifacts reports/code_transfer_artifacts.json --out reports/local_code_repair_organism_source_livecodebench_seed14.json",
            ["card_id", "seed", "source_path", "transfer_artifacts", "out", "trace_out", "artifact_out"],
            ["scripts/local_code_repair_organism.py", "reports/code_transfer_artifacts.json"],
            ["python_compile", "sandbox_patch_tests_ran", "transfer_altered_behavior", "external_inference_calls_zero"],
        ),
        tool_card(
            "self_edit_experiment_lane_tool",
            "self_edit_governance",
            "Convert residual clusters into bounded source-patch experiments with verification, rollback notes, and no teacher apply mode.",
            "python scripts/self_edit_experiment_lane.py --out reports/self_edit_experiment_lane.json",
            ["out", "bundle_out"],
            ["scripts/self_edit_experiment_lane.py", "reports/code_residual_forge.json"],
            ["python_compile", "residual_to_patch_contracts_written", "rollback_plan_written", "external_inference_calls_zero"],
        ),
        tool_card(
            "long_horizon_memory_probe_tool",
            "memory_governance",
            "Probe whether compressed Theseus traces preserve the active goal, reject stale decoys, and recover the same next action over simulated long horizons.",
            "python scripts/long_horizon_memory_probe.py --out reports/long_horizon_memory_probe.json",
            ["out", "trace_out"],
            ["scripts/long_horizon_memory_probe.py", "reports/benchmaxx_curriculum.json", "reports/frontier_policy_status.json"],
            ["python_compile", "goal_recall_passes", "decoy_rejection_passes", "external_inference_calls_zero"],
        ),
        tool_card(
            "closed_loop_workflow_tool",
            "closed_loop_workflow",
            "Refresh the benchmark/model/residual ledgers that turn completed runs into next-frontier decisions.",
            "python scripts/benchmark_treadmill.py --reports reports --out reports/benchmark_treadmill_report.json --benchmark-ledger-out reports/benchmark_ledger.json --model-ledger-out reports/model_ledger.json --public-comparator-ledger-out reports/public_comparator_ledger.json",
            ["reports", "out", "benchmark_ledger_out", "model_ledger_out", "public_comparator_ledger_out"],
            ["scripts/benchmark_treadmill.py", "reports/benchmark_ledger.json", "reports/model_ledger.json"],
            ["python_compile", "ledger_schema_valid", "external_inference_calls_zero"],
        ),
        tool_card(
            "octopus_router_refresh_tool",
            "router_refresh",
            "Refresh the Octopus Router arm routing state from benchmark, model, tool, residual, and capability ledgers.",
            "python scripts/octopus_router.py --benchmark-ledger reports/benchmark_ledger.json --model-ledger reports/model_ledger.json --tool-registry reports/tool_registry.json --residual-escrow reports/residual_escrow.json --capability-ratchet reports/capability_ratchet_report.json --out reports/octopus_router_report.json",
            ["benchmark_ledger", "model_ledger", "tool_registry", "residual_escrow", "capability_ratchet", "out"],
            ["scripts/octopus_router.py", "reports/tool_registry.json", "reports/residual_escrow.json"],
            ["python_compile", "router_report_schema", "external_inference_calls_zero"],
        ),
        tool_card(
            "training_preflight_tool",
            "training_preflight",
            "Run split checks, candidate gates, and training readiness probes before expensive pressure.",
            "python scripts/training_preflight.py --run-split-check --run-candidate-gate --out reports/training_preflight_report.json",
            ["run_smokes", "run_split_check", "run_candidate_gate", "out"],
            ["scripts/training_preflight.py", "reports/training_preflight_report.json"],
            ["python_compile", "preflight_schema_valid", "external_inference_calls_zero"],
            risk_tier="medium",
        ),
        tool_card(
            "symliquid_local_training_tool",
            "local_symliquid_training",
            "Run the local SymLiquid CUDA training lane with bounded parameters and report-only side effects.",
            "cargo run -p symliquid-cli --features cuda -- train-standalone-cuda --epochs 8 --samples-per-launch 64 --hv-dim 4096 --out reports/symliquid_standalone_cuda_train_report.json",
            ["epochs", "samples_per_launch", "hv_dim", "train_seed", "eval_seed", "out"],
            ["crates/symliquid-cli", "crates/symliquid-cuda", "reports/symliquid_standalone_cuda_train_report.json"],
            ["cargo_check", "cuda_no_fallback", "training_report_schema"],
        ),
        tool_card(
            "checkpoint_chain_repair_tool",
            "checkpoint_repair",
            "Create a fresh major checkpoint when an old minor-delta chain cannot materialize cleanly.",
            "python scripts/checkpoint_registry.py create --kind major --label sparkstream_repair --reason checkpoint_chain_repair --out reports/checkpoint_last.json",
            ["kind", "label", "reason", "profile", "status", "out"],
            ["scripts/checkpoint_registry.py", "reports/checkpoint_registry.json"],
            ["python_compile", "checkpoint_manifest_schema", "materialize_reports_error_instead_of_crash"],
            risk_tier="medium",
        ),
        tool_card(
            "transfer_eval_suite_tool",
            "transfer_evaluation",
            "Emit ASI-relevant transfer surfaces for code repair, tool use, web tasks, long context, RL control, self-debugging, and voice I/O.",
            "python scripts/transfer_eval_suite.py --out reports/transfer_eval_suite.json",
            ["out", "emit_surfaces"],
            ["scripts/transfer_eval_suite.py", "configs/transfer_eval_policy.json"],
            ["python_compile", "transfer_eval_report_schema", "external_inference_calls_zero"],
        ),
        tool_card(
            "genesis_kernel_snapshot_tool",
            "invention_artifact_governance",
            "Compile live Theseus reports into Genesis artifacts, claim ledger, critique log, primitive candidates, release manifest, artifact debt, and feedback plan.",
            "python scripts/genesis_kernel.py ingest-theseus --out reports/genesis_kernel/report.json --bundle-dir reports/genesis_kernel/latest_release",
            ["policy", "out", "bundle_dir", "report"],
            ["scripts/genesis_kernel.py", "configs/genesis_kernel_policy.json", "reports/genesis_kernel/report.json"],
            ["python_compile", "genesis_hard_release_gates_clear", "external_inference_calls_zero"],
        ),
    ]
    if "babylm_residual_analysis_tool" in ready_names:
        tools.append(
            tool_card(
                "residual_analysis_workflow_tool",
                "residual_analysis",
                "Refresh residual clusters and convert unsolved cases into escrow pressure.",
                "python scripts/analyze_babylm_residuals.py --report reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json --eval-input data/babylm_blimp_filtered_eval.jsonl --out reports/babylm_residual_analysis.json",
                ["report", "eval_input", "out", "min_cases", "limit"],
                ["scripts/analyze_babylm_residuals.py", "reports/residual_escrow.json"],
                ["python_compile", "residual_groups_nonempty", "escrow_update_available"],
            )
        )
    return tools


def tool_card(
    name: str,
    task_family: str,
    purpose: str,
    command: str,
    parameters: list[str],
    provenance: list[str],
    verification_tests: list[str],
    *,
    risk_tier: str = "low",
) -> dict[str, Any]:
    benchmark_related = is_benchmark_or_eval_text(" ".join([name, task_family, purpose, command]))
    return {
        "tool_name": name,
        "version": "0.1.0",
        "lifecycle": "active",
        "task_family": task_family,
        "purpose": purpose,
        "command": command,
        "inputs": ["local_filesystem_artifacts"],
        "outputs": ["json_report"],
        "parameters": parameters,
        "preconditions": ["workspace_present", "no_external_inference"],
        "postconditions": ["report_written", "external_inference_calls_zero_or_report_rejected"],
        "verification_tests": verification_tests,
        "verification_grade": "runtime_monitored",
        "runtime_tier": "typed_function_or_local_process",
        "latency_class": "interactive",
        "risk_tier": risk_tier,
        "allowed_side_effects": ["write_reports", "read_local_data"],
        "permissions": ["local_filesystem", "local_rust_python_execution"],
        "fallback": "interpreter_mode_manual_diagnosis",
        "fallback_mode": "interpreter",
        "provenance": provenance,
        "metrics": {
            "success_rate": None,
            "cost_savings": "avoids repeated manual workflow reconstruction",
            "failure_count": None,
        },
        "retirement_criteria": [
            "tool_output_no_longer_matches_schema",
            "better_tool_supersedes_this_workflow",
            "environment_or_benchmark_contract_changes",
        ],
        "learning_integrity": {
            "benchmark_related": benchmark_related,
            "benchmark_infrastructure_only": benchmark_related,
            "may_generate_benchmark_answers": False,
            "may_distill_benchmark_task_solutions": False,
        },
    }


def benchmark_tool_block_reason(tool: dict[str, Any]) -> str:
    name = str(tool.get("tool_name") or "")
    text = " ".join(
        [
            name,
            str(tool.get("task_family") or ""),
            str(tool.get("purpose") or ""),
            str(tool.get("command") or ""),
        ]
    )
    if not is_benchmark_or_eval_text(text):
        return ""
    allowed_infrastructure_tools = {
        "benchmark_adapter_smoke_tool",
        "candidate_bottleneck_reducer_tool",
        "pressure_runner_tool",
        "code_residual_forge_tool",
        "synthetic_benchmark_factory_tool",
        "multi_stream_code_pressure_tool",
        "local_code_repair_organism_tool",
        "closed_loop_workflow_tool",
        "training_preflight_tool",
        "octopus_router_refresh_tool",
        "residual_analysis_workflow_tool",
        "transfer_eval_suite_tool",
    }
    if name in allowed_infrastructure_tools:
        return ""
    return "benchmark_or_eval_task_tool_distillation_blocked"


def is_benchmark_or_eval_text(value: str) -> bool:
    lowered = value.lower()
    tokens = [
        "benchmark",
        "eval",
        "frontier",
        "pressure",
        "humaneval",
        "human_eval",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "mbpp",
        "swe_bench",
        "codeclash",
    ]
    return any(token in lowered for token in tokens)


def refresh_health(registry: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for tool in registry.get("tools", []):
        lifecycle = str(tool.get("lifecycle") or "unknown")
        counts[lifecycle] = counts.get(lifecycle, 0) + 1
    registry["registry_health"] = {
        "active": counts.get("active", 0),
        "proposed": counts.get("proposed", 0),
        "candidate": counts.get("candidate", 0),
        "retired": counts.get("retired", 0),
    }


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
