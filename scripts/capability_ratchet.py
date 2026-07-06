"""Capability Ratchet report builder for SymLiquid.

This script fuses the benchmark ratchet, residual map, and procedural tool
registry into one local-only artifact. It does not run model inference or call
external providers; it only reads local reports produced by the harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--model-ledger", default="reports/model_ledger.json")
    parser.add_argument("--residual-analysis", default="reports/babylm_residual_analysis.json")
    parser.add_argument(
        "--mutated-residual-analysis",
        default="reports/babylm_mutated_residual_analysis.json",
    )
    parser.add_argument(
        "--public-comparator-ledger",
        default="reports/public_comparator_ledger.json",
    )
    parser.add_argument("--out", default="reports/capability_ratchet_report.json")
    parser.add_argument("--tool-registry-out", default="reports/tool_registry.json")
    parser.add_argument("--residual-escrow-out", default="reports/residual_escrow.json")
    args = parser.parse_args()

    benchmark_ledger = read_json(args.benchmark_ledger, [])
    model_ledger = read_json(args.model_ledger, {})
    residual_analysis = read_json(args.residual_analysis, {})
    mutated_residual_analysis = read_json(args.mutated_residual_analysis, {})
    public_comparator_ledger = read_json(args.public_comparator_ledger, {})
    residual_escrow = build_residual_escrow(residual_analysis, mutated_residual_analysis)
    tool_registry = build_tool_registry(
        benchmark_ledger, model_ledger, residual_analysis, mutated_residual_analysis
    )
    report = build_capability_report(
        benchmark_ledger=benchmark_ledger,
        model_ledger=model_ledger,
        residual_analysis=residual_analysis,
        mutated_residual_analysis=mutated_residual_analysis,
        public_comparator_ledger=public_comparator_ledger,
        tool_registry=tool_registry,
        tool_registry_out=args.tool_registry_out,
        residual_escrow=residual_escrow,
        residual_escrow_out=args.residual_escrow_out,
    )

    write_json(args.tool_registry_out, tool_registry)
    write_json(args.residual_escrow_out, residual_escrow)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_capability_report(
    benchmark_ledger: list[dict[str, Any]],
    model_ledger: dict[str, Any],
    residual_analysis: dict[str, Any],
    mutated_residual_analysis: dict[str, Any],
    public_comparator_ledger: dict[str, Any],
    tool_registry: dict[str, Any],
    tool_registry_out: str,
    residual_escrow: dict[str, Any],
    residual_escrow_out: str,
) -> dict[str, Any]:
    frontier = [entry for entry in benchmark_ledger if entry.get("lifecycle") == "frontier"]
    regressions = [entry for entry in benchmark_ledger if entry.get("lifecycle") == "regression"]
    diagnostics = [entry for entry in benchmark_ledger if entry.get("lifecycle") == "diagnostic"]
    invalid = [entry for entry in benchmark_ledger if entry.get("lifecycle") == "invalid"]
    active_frontier = frontier[0] if frontier else None
    residual_targets = residual_analysis.get("recommendation", {})

    return {
        "policy": "local_only_no_external_inference",
        "framework": "capability_ratchet",
        "inherits": [
            "benchmaxxing_performance_ratchet",
            "cognitive_loop_closure",
            "compact_generative_systems",
        ],
        "ratchet_rule": {
            "frontier_gain_required": True,
            "regression_loss_forbidden": True,
            "external_inference_forbidden": True,
            "initial_mastery_threshold": 0.90,
            "ordinary_floor_threshold": 0.70,
            "threshold_decay_mode": "per_attempt_after_patience",
            "threshold_decay_rate_per_cycle": 0.01,
            "threshold_patience_cycles": 3,
            "threshold_decay_requires_stalled_effort": False,
            "perfect_mastery_not_required_for_promotion": True,
            "critical_failure_veto": True,
            "residual_escrow_required_after_graduation": True,
            "passes_current_gate": len(invalid) == 0
            and (len(frontier) >= 1 or (bool(regressions) and len(diagnostics) == 0)),
            "frontier_rotation_required": len(frontier) == 0 and bool(regressions),
        },
        "benchmark_ratchet": {
            "frontier": [entry["benchmark_name"] for entry in frontier],
            "regression_suite": [entry["benchmark_name"] for entry in regressions],
            "diagnostic_surfaces": [entry["benchmark_name"] for entry in diagnostics],
            "invalid_surfaces": [entry["benchmark_name"] for entry in invalid],
            "public_comparators": [
                entry.get("benchmark_name")
                for entry in public_comparator_ledger.get("comparators", [])
            ],
            "active_frontier": None
            if active_frontier is None
            else {
                "benchmark": active_frontier["benchmark_name"],
                "capability": active_frontier["capability_measured"],
                "score": active_frontier["score"],
                "residual": active_frontier["residual"],
                "wall_type": active_frontier["wall_type"],
            },
        },
        "procedural_ratchet": {
            "tool_registry_out": tool_registry_out,
            "active_tools": [
                tool["tool_name"]
                for tool in tool_registry["tools"]
                if tool["lifecycle"] in ("active", "probationary")
            ],
            "candidate_tools": [
                tool["tool_name"]
                for tool in tool_registry["tools"]
                if tool["lifecycle"] in ("candidate", "proposed")
            ],
            "execution_modes": {
                "interpreter": "novel, ambiguous, or unsatisfied preconditions",
                "compiled_tool": "verified repeated workflow with valid parameters",
                "reflex_failsafe": "hard-latency or safety-critical boundary",
            },
        },
        "structural_ratchet": {
            "next_wall": model_ledger.get("next_wall"),
            "diagnostic_ladder": [
                "benchmark_audit",
                "data_improvement",
                "training_improvement",
                "inference_improvement",
                "loop_closure",
                "benchmark_frontier_expansion",
                "architecture_change",
            ],
            "architecture_change_hypothesis": architecture_hypothesis(active_frontier, residual_targets),
        },
        "residual_map": {
            "model_ledger_residuals": model_ledger.get("residual_map", []),
            "babylm_targets": residual_targets,
            "worst_babylm_terms": residual_analysis.get("worst_by_linguistics_term", [])[:6],
            "worst_babylm_rules": residual_analysis.get("worst_by_rule", [])[:8],
            "mutated_babylm_present": bool(mutated_residual_analysis),
            "worst_mutated_babylm_terms": mutated_residual_analysis.get(
                "worst_by_linguistics_term", []
            )[:6],
            "worst_mutated_babylm_rules": mutated_residual_analysis.get("worst_by_rule", [])[:8],
        },
        "residual_escrow": {
            "ledger": residual_escrow_out,
            "budget": residual_escrow.get("attention_budget"),
            "cluster_count": residual_escrow.get("summary", {}).get("cluster_count", 0),
            "case_count": residual_escrow.get("summary", {}).get("case_count", 0),
            "active_diagnostic_targets": residual_escrow.get(
                "active_diagnostic_targets", []
            )[:8],
            "recurrence_rule": residual_escrow.get("recurrence_promotion_rule"),
        },
        "verification": {
            "benchmark_ledger_entries": len(benchmark_ledger),
            "model_ledger_present": bool(model_ledger),
            "residual_analysis_present": bool(residual_analysis),
            "mutated_residual_analysis_present": bool(mutated_residual_analysis),
            "public_comparator_ledger_present": bool(public_comparator_ledger),
            "public_comparator_count": len(public_comparator_ledger.get("comparators", [])),
            "tool_registry_entries": len(tool_registry["tools"]),
            "residual_escrow_present": bool(residual_escrow),
            "residual_escrow_entries": residual_escrow.get("summary", {}).get(
                "case_count", 0
            ),
            "external_inference_violations": [
                entry["benchmark_name"]
                for entry in benchmark_ledger
                if entry.get("external_inference_calls", 0) > 0
            ],
        },
        "public_comparison": {
            "ledger": "reports/public_comparator_ledger.json",
            "rule": public_comparator_ledger.get(
                "comparison_rule",
                "Run public comparators before candidate promotion.",
            ),
            "comparators": public_comparator_ledger.get("comparators", []),
        },
        "next_actions": next_actions(active_frontier, residual_targets),
    }


def build_residual_escrow(
    residual_analysis: dict[str, Any],
    mutated_residual_analysis: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        ("public_babylm", residual_analysis),
        ("mutated_babylm", mutated_residual_analysis),
    ]
    pressure_sources = collect_pressure_residual_sources(Path("reports"))
    cluster_map: dict[tuple[str, str], dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    for source_name, analysis in sources:
        if not analysis:
            continue
        for kind, key in (
            ("term", "worst_by_linguistics_term"),
            ("rule", "worst_by_rule"),
        ):
            for row in analysis.get(key, []):
                if float(row.get("residual", 0.0)) <= 0.0:
                    continue
                cluster_key = (kind, str(row.get("name", "unknown")))
                cluster = cluster_map.setdefault(
                    cluster_key,
                    {
                        "kind": kind,
                        "name": cluster_key[1],
                        "sources": [],
                        "max_residual": 0.0,
                        "total_cases": 0,
                        "total_failures": 0,
                    },
                )
                cluster["sources"].append(source_name)
                residual = float(row.get("residual", 0.0))
                cases_count = int(row.get("cases", 0))
                correct = int(row.get("correct", 0))
                cluster["max_residual"] = max(cluster["max_residual"], residual)
                cluster["total_cases"] += cases_count
                cluster["total_failures"] += max(0, cases_count - correct)
        for failure in analysis.get("failure_examples", []):
            cases.append(
                {
                    "source": source_name,
                    "case_id": failure.get("case_id"),
                    "field": failure.get("field"),
                    "linguistics_term": failure.get("linguistics_term"),
                    "rule": failure.get("rule"),
                    "expected": failure.get("expected"),
                    "output": failure.get("output"),
                    "status": "residual_escrow",
                    "reattempt_schedule": "periodic",
                    "promotion_rule": "promote_to_regression_after_consistent_future_passes",
                    "sentence_good": failure.get("sentence_good"),
                    "sentence_bad": failure.get("sentence_bad"),
                }
            )
    for source_name, report in pressure_sources:
        report_score = safe_float(get_path(report, ["summary", "score"], report.get("score"))) or 0.0
        report_family = str(report.get("frontier_family") or get_path(report, ["summary", "suite"], "pressure"))
        for residual in report.get("residuals", []):
            if not isinstance(residual, dict):
                continue
            residual_type = str(residual.get("type") or "pressure_residual")
            cluster_key = ("pressure", f"{report_family}:{residual_type}")
            cluster = cluster_map.setdefault(
                cluster_key,
                {
                    "kind": "pressure",
                    "name": cluster_key[1],
                    "sources": [],
                    "max_residual": 0.0,
                    "total_cases": 0,
                    "total_failures": 0,
                },
            )
            cluster["sources"].append(source_name)
            residual_value = max(0.0, min(1.0, 1.0 - report_score))
            cluster["max_residual"] = max(cluster["max_residual"], residual_value)
            cluster["total_cases"] += 1
            cluster["total_failures"] += 1
            cases.append(
                {
                    "source": source_name,
                    "case_id": f"{source_name}:{residual_type}",
                    "field": "pressure_runner_residual",
                    "residual_type": residual_type,
                    "detail": residual.get("detail"),
                    "status": "residual_escrow",
                    "reattempt_schedule": "periodic",
                    "promotion_rule": "promote_to_regression_after_consistent_future_passes",
                    "benchmark_family": report_family,
                    "score": report_score,
                    "report": source_name,
                }
            )

    clusters = []
    active_targets = []
    for cluster in cluster_map.values():
        unique_sources = sorted(set(cluster["sources"]))
        recurring = len(unique_sources) > 1
        severe = float(cluster["max_residual"]) >= 0.10
        status = "reactivated_diagnostic" if recurring or severe else "residual_escrow"
        frequency = "every_candidate_promotion" if status == "reactivated_diagnostic" else "spaced_periodic"
        record = {
            **cluster,
            "sources": unique_sources,
            "status": status,
            "recurring_across_benchmarks": recurring,
            "reattempt_frequency": frequency,
            "critical_failure": False,
        }
        clusters.append(record)
        if status == "reactivated_diagnostic":
            active_targets.append(
                {
                    "kind": record["kind"],
                    "name": record["name"],
                    "max_residual": record["max_residual"],
                    "sources": record["sources"],
                    "reason": "recurring_or_high_residual_cluster",
                }
            )

    clusters.sort(key=lambda row: (-row["max_residual"], row["kind"], row["name"]))
    active_targets.sort(key=lambda row: -row["max_residual"])
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "residual_escrow_active_backlog",
        "attention_budget": {
            "frontier": 0.60,
            "regression": 0.20,
            "residual_escrow": 0.10,
            "public_calibration": 0.10,
        },
        "rules": {
            "frontier_momentum": "Graduated benchmark tails enter escrow instead of blocking frontier movement.",
            "critical_failure_veto": "Safety-critical residuals must be reattempted every gate and can block graduation.",
            "recurrence_promotion": "Residual clusters recurring across public and mutated/live benchmarks become active diagnostic targets.",
            "bridge_benchmark": "If a benchmark cannot clear its floor, insert a bridge benchmark or diagnose architecture/evaluation quality.",
        },
        "recurrence_promotion_rule": "recurring_across_benchmarks or max_residual>=0.10",
        "summary": {
            "cluster_count": len(clusters),
            "case_count": len(cases),
            "active_diagnostic_target_count": len(active_targets),
        },
        "active_diagnostic_targets": active_targets,
        "clusters": clusters,
        "cases": cases,
    }


def collect_pressure_residual_sources(reports_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(reports_dir.glob("pressure_*.json"), key=lambda item: item.stat().st_mtime)[-64:]:
        report = read_json(str(path), {})
        if not isinstance(report, dict) or not report.get("residuals"):
            continue
        if int(report.get("external_inference_calls") or 0) != 0:
            continue
        rows.append((str(path).replace("\\", "/"), report))
    return rows


def build_tool_registry(
    benchmark_ledger: list[dict[str, Any]],
    model_ledger: dict[str, Any],
    residual_analysis: dict[str, Any],
    mutated_residual_analysis: dict[str, Any],
) -> dict[str, Any]:
    mutated_report = mutated_residual_analysis.get(
        "report", "reports/babylm_mutated_holdout_seed49_stateful_grammar_state_frontier.json"
    )
    mutated_eval = mutated_residual_analysis.get(
        "eval_input", "data/babylm_mutated_holdout_seed49.jsonl"
    )
    tools = [
        tool_card(
            name="benchmark_treadmill_runner",
            lifecycle="active",
            task_family="benchmark_ratchet",
            purpose="Scan local reports, classify benchmark lifecycle state, and emit ratchet next actions.",
            command="python scripts/benchmark_treadmill.py --reports reports --out reports/benchmark_treadmill_status.json --benchmark-ledger-out reports/benchmark_ledger.json --model-ledger-out reports/model_ledger.json --public-comparator-ledger-out reports/public_comparator_ledger.json",
            parameters=[
                "reports",
                "out",
                "benchmark_ledger_out",
                "model_ledger_out",
                "public_comparator_ledger_out",
            ],
            verification_tests=["python_compile", "ledger_outputs_exist", "external_inference_violation_check"],
            provenance=["scripts/benchmark_treadmill.py"],
            verification_grade="runtime_monitored",
            risk_tier="low",
        ),
        tool_card(
            name="capability_ratchet_orchestrator",
            lifecycle="active",
            task_family="capability_ratchet",
            purpose="Run the compiled local workflow that refreshes ledgers, residual maps, benchmark factories, tool registry, and next intervention.",
            command="python scripts/run_capability_ratchet.py --out reports/capability_ratchet_run.json",
            parameters=["reports", "public_babylm_report", "public_babylm_eval", "mutated_babylm_report", "mutated_babylm_eval", "out"],
            verification_tests=["python_compile", "all_required_outputs_written", "external_inference_calls_zero"],
            provenance=["scripts/run_capability_ratchet.py"],
            verification_grade="runtime_monitored",
            risk_tier="medium",
        ),
        tool_card(
            name="babylm_residual_analyzer",
            lifecycle="active",
            task_family="residual_analysis",
            purpose="Join BabyLM/BLIMP eval reports to metadata and identify residual pressure families.",
            command="python scripts/analyze_babylm_residuals.py --report reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json --eval-input data/babylm_blimp_filtered_eval.jsonl --out reports/babylm_residual_analysis.json",
            parameters=["report", "eval_input", "out", "min_cases", "limit"],
            verification_tests=["python_compile", "result_eval_row_count_match", "residual_groups_nonempty"],
            provenance=["scripts/analyze_babylm_residuals.py", "reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json"],
            verification_grade="runtime_monitored",
            risk_tier="low",
        ),
        tool_card(
            name="babylm_mutated_holdout_factory",
            lifecycle="active",
            task_family="benchmark_frontier_expansion",
            purpose="Generate local mutated BabyLM/BLIMP holdouts from residual pressure families.",
            command="python scripts/generate_babylm_mutated_holdout.py --residual-analysis reports/babylm_residual_analysis.json --count 2400 --seed 31 --out data/babylm_mutated_holdout_seed31.jsonl --report-out reports/babylm_mutated_holdout_seed31_factory.json",
            parameters=["source", "residual_analysis", "count", "seed", "out", "report_out"],
            verification_tests=["python_compile", "jsonl_schema", "source_exact_pair_exclusion", "external_inference_calls_zero"],
            provenance=["scripts/generate_babylm_mutated_holdout.py", "reports/babylm_residual_analysis.json"],
            verification_grade="synthetic_passed",
            risk_tier="low",
        ),
        tool_card(
            name="unseen_adversarial_rag_mutator",
            lifecycle="active",
            task_family="benchmark_frontier_expansion",
            purpose="Generate harder local unseen adversarial RAG holdouts without external inference.",
            command="python scripts/generate_unseen_adversarial_rag.py --count 360 --seed 29 --out benchmarks/snapshots/unseen_adversarial_rag_seed29_harder.json",
            parameters=["count", "seed", "out"],
            verification_tests=["json_suite_loads", "symliquid_local_benchmark_passes", "external_inference_calls_zero"],
            provenance=["scripts/generate_unseen_adversarial_rag.py"],
            verification_grade="adversarial_tested",
            risk_tier="low",
        ),
        tool_card(
            name="rust_ffi_puffer_rollout_trainer",
            lifecycle="active",
            task_family="local_control_training",
            purpose="Train and evaluate local Puffer/Ocean-style policies through the Rust FFI rollout loop.",
            command="python adapters/pufferlib/symliquid_puffer_adapter.py --train-discrete-policy --use-rust-ffi ...",
            parameters=["env", "iterations", "population", "elite_count", "num_envs", "train_steps", "eval_steps", "seed", "policy_out", "out"],
            verification_tests=["cargo_test_symliquid_ffi", "rollout_smoke_report", "external_inference_calls_zero"],
            provenance=["adapters/pufferlib/symliquid_puffer_adapter.py", "crates/symliquid-ffi/src/rollout.rs"],
            verification_grade="runtime_monitored",
            risk_tier="medium",
        ),
        tool_card(
            name="puffer_ocean_eventized_rollout_logger",
            lifecycle="active",
            task_family="embodied_logging",
            purpose="Generate bounded raw/event/semantic/skill/residual logs for local Puffer/Ocean rollouts.",
            command="python adapters/pufferlib/symliquid_puffer_adapter.py --artifact reports/symliquid_ocean_slot_tmaze_policy_rust_trainer_seed3.json --env ocean-slot-tmaze --num-envs 32 --rollout-smoke-steps 64 --event-log-out reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json --out reports/puffer_ocean_slot_tmaze_eventized_smoke.json",
            parameters=[
                "artifact",
                "env",
                "num_envs",
                "rollout_smoke_steps",
                "event_log_out",
                "event_log_env_limit",
                "event_log_step_limit",
                "out",
            ],
            verification_tests=[
                "event_log_schema_present",
                "raw_event_semantic_skill_residual_streams_present",
                "external_inference_calls_zero",
            ],
            provenance=[
                "adapters/pufferlib/symliquid_puffer_adapter.py",
                "reports/symliquid_ocean_slot_tmaze_policy_rust_trainer_seed3.json",
            ],
            verification_grade="runtime_monitored",
            risk_tier="low",
        ),
        tool_card(
            name="babylm_frontier_trainer",
            lifecycle="proposed",
            task_family="frontier_training",
            purpose="Train the active BabyLM/BLIMP frontier probe and compare against regression and mutated holdouts.",
            command=f"cargo run --release -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input {mutated_eval} --train-limit 53888 --eval-limit 4800 --steps 120000 --hv-dim 8192 --lr 0.08 --stateful --pairwise-contrast --balance-rules --prior-weight 1.0 --out {mutated_report}",
            parameters=["input", "eval_input", "train_limit", "eval_limit", "steps", "hv_dim", "lr", "stateful", "pairwise_contrast", "balance_rules", "prior_weight", "out"],
            verification_tests=["public_eval_score", "private_or_mutated_holdout_required", "regression_suite_preserved"],
            provenance=["reports/benchmark_treadmill_status.json", "reports/babylm_residual_analysis.json"],
            verification_grade="replay_passed",
            risk_tier="medium",
        ),
    ]

    if mutated_residual_analysis:
        tools.append(
            tool_card(
                name="babylm_mutated_residual_analyzer",
                lifecycle="active",
                task_family="residual_analysis",
                purpose="Analyze failures on the local mutated BabyLM holdout so public gains cannot hide transfer loss.",
                command=f"python scripts/analyze_babylm_residuals.py --report {mutated_report} --eval-input {mutated_eval} --out reports/babylm_mutated_residual_analysis.json",
                parameters=["report", "eval_input", "out", "min_cases", "limit"],
                verification_tests=["python_compile", "mutated_result_eval_row_count_match", "residual_groups_nonempty"],
                provenance=[
                    "scripts/analyze_babylm_residuals.py",
                    mutated_report,
                    mutated_eval,
                ],
                verification_grade="runtime_monitored",
                risk_tier="low",
            )
        )

    tools.append(
        tool_card(
            name="residual_escrow_builder",
            lifecycle="active",
            task_family="residual_escrow",
            purpose="Keep graduated benchmark tails alive as an active backlog without letting them hold the frontier hostage.",
            command="python scripts/capability_ratchet.py --benchmark-ledger reports/benchmark_ledger.json --model-ledger reports/model_ledger.json --residual-analysis reports/babylm_residual_analysis.json --mutated-residual-analysis reports/babylm_mutated_residual_analysis.json --public-comparator-ledger reports/public_comparator_ledger.json --out reports/capability_ratchet_report.json --tool-registry-out reports/tool_registry.json --residual-escrow-out reports/residual_escrow.json",
            parameters=["residual_analysis", "mutated_residual_analysis", "residual_escrow_out"],
            verification_tests=[
                "residual_clusters_written",
                "recurring_residuals_promoted_to_active_diagnostic_targets",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/capability_ratchet.py",
                "reports/babylm_residual_analysis.json",
                "reports/babylm_mutated_residual_analysis.json",
            ],
            verification_grade="runtime_monitored",
            risk_tier="low",
        )
    )

    tools.append(
        tool_card(
            name="ratcheting_generative_system_auditor",
            lifecycle="active",
            task_family="rgs_conformance",
            purpose="Audit whether SymLiquid's ledgers, tool registry, residual escrow, public calibration, and safety hooks satisfy the Ratcheting Generative Systems framework.",
            command="python scripts/ratcheting_generative_system.py --benchmark-treadmill reports/benchmark_treadmill_status.json --benchmark-ledger reports/benchmark_ledger.json --model-ledger reports/model_ledger.json --tool-registry reports/tool_registry.json --residual-escrow reports/residual_escrow.json --public-comparator-ledger reports/public_comparator_ledger.json --capability-ratchet reports/capability_ratchet_report.json --out reports/ratcheting_generative_system_report.json",
            parameters=[
                "benchmark_treadmill",
                "benchmark_ledger",
                "model_ledger",
                "tool_registry",
                "residual_escrow",
                "public_comparator_ledger",
                "capability_ratchet",
                "out",
            ],
            verification_tests=[
                "implementation_matrix_present",
                "missing_or_partial_components_have_next_actions",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/ratcheting_generative_system.py",
                "reports/benchmark_ledger.json",
                "reports/residual_escrow.json",
                "reports/tool_registry.json",
            ],
            verification_grade="runtime_monitored",
            risk_tier="low",
        )
    )

    tools.append(
        tool_card(
            name="octopus_router_architecture_builder",
            lifecycle="active",
            task_family="system_level_routing",
            purpose="Build ORA arm cards, route benchmark cases, permission envelopes, routing memory, arm lifecycle ledger, safety ledgers, bridge benchmarks, and dynamic-loading metrics.",
            command="python scripts/octopus_router.py --benchmark-ledger reports/benchmark_ledger.json --model-ledger reports/model_ledger.json --tool-registry reports/tool_registry.json --residual-escrow reports/residual_escrow.json --capability-ratchet reports/capability_ratchet_report.json --event-log reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json --out reports/octopus_router_report.json",
            parameters=[
                "benchmark_ledger",
                "model_ledger",
                "tool_registry",
                "residual_escrow",
                "capability_ratchet",
                "event_log",
                "arm_registry_out",
                "router_eval_out",
                "routing_memory_out",
                "arm_lifecycle_out",
                "safety_ledger_out",
                "bridge_ledger_out",
                "bridge_out",
                "out",
            ],
            verification_tests=[
                "arm_registry_written",
                "router_eval_selection_accuracy_gate",
                "routing_memory_written",
                "arm_lifecycle_ledger_written",
                "safety_ledger_passed",
                "bridge_benchmark_cases_written",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/octopus_router.py",
                "reports/tool_registry.json",
                "reports/residual_escrow.json",
                "reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json",
            ],
            verification_grade="runtime_monitored",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="octopus_router_head_trainer",
            lifecycle="active",
            task_family="system_level_routing",
            purpose="Train and evaluate the local sparse ORA router head from task-to-arm traces without external inference.",
            command="python scripts/train_octopus_router_head.py --router-eval reports/octopus_router_eval.json --arm-registry reports/arm_registry.json --dataset-out reports/octopus_router_trace_dataset.json --model-out reports/octopus_router_head_model.json --eval-out reports/octopus_router_head_eval.json --out reports/octopus_router_head_report.json",
            parameters=[
                "router_eval",
                "arm_registry",
                "dataset_out",
                "model_out",
                "eval_out",
                "out",
            ],
            verification_tests=[
                "python_compile",
                "holdout_exact_set_accuracy_gate",
                "risk_routing_accuracy_gate",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/train_octopus_router_head.py",
                "reports/octopus_router_eval.json",
                "reports/arm_registry.json",
            ],
            verification_grade="holdout_passed",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="architecture_gate_runner",
            lifecycle="active",
            task_family="architecture_governance",
            purpose="Block heavy training until the ratchet, RMI audit, ORA, safety, learned routing, routing memory, arm lifecycle, residual escrow, bridges, and public calibration are coherent.",
            command="python scripts/architecture_gate.py --out reports/architecture_gate_report.json",
            parameters=[
                "capability_ratchet",
                "rgs",
                "rmi",
                "octopus_router",
                "router_head",
                "router_head_eval",
                "router_eval",
                "benchmark_ledger",
                "public_comparator_ledger",
                "residual_escrow",
                "tool_registry",
                "routing_memory",
                "arm_lifecycle",
                "safety_ledger",
                "bridge_ledger",
                "out",
            ],
            verification_tests=[
                "rgs_complete",
                "rmi_complete",
                "ora_complete",
                "learned_router_head_promoted",
                "routing_memory_present",
                "arm_lifecycle_governed",
                "safety_ledger_passed",
                "public_calibration_present",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/architecture_gate.py",
                "reports/ratcheting_generative_system_report.json",
                "reports/octopus_router_report.json",
                "reports/octopus_router_head_report.json",
            ],
            verification_grade="promotion_gate",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="ratcheting_modular_intelligence_auditor",
            lifecycle="active",
            task_family="rmi_conformance",
            purpose="Audit the unified RMI spec: compact structure, active compression, loop closure, benchmark ratcheting, octopus routing, routing memory, and arm lifecycle governance.",
            command="python scripts/ratcheting_modular_intelligence.py --out reports/ratcheting_modular_intelligence_report.json",
            parameters=[
                "benchmark_treadmill",
                "benchmark_ledger",
                "model_ledger",
                "tool_registry",
                "residual_escrow",
                "public_comparator_ledger",
                "capability_ratchet",
                "rgs",
                "octopus_router",
                "arm_registry",
                "routing_memory",
                "arm_lifecycle",
                "router_head",
                "router_head_eval",
                "safety_ledger",
                "bridge_ledger",
                "event_log",
                "out",
            ],
            verification_tests=[
                "five_pillars_present",
                "routing_memory_present",
                "arm_lifecycle_governed",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/ratcheting_modular_intelligence.py",
                "reports/octopus_router_report.json",
                "reports/routing_memory.json",
                "reports/arm_lifecycle_ledger.json",
            ],
            verification_grade="framework_audit_passed",
            risk_tier="low",
        )
    )

    tools.append(
        tool_card(
            name="real_training_preflight_gate",
            lifecycle="active",
            task_family="training_governance",
            purpose="Block long training until RTX 2060 Super CUDA telemetry, release builds, leakage checks, profile gates, ablations, and candidate promotion rules are visible.",
            command="python scripts/training_preflight.py --run-split-check --run-candidate-gate --out reports/training_preflight_report.json",
            parameters=[
                "profiles",
                "ablation_matrix",
                "architecture_gate",
                "rmi",
                "split_leakage",
                "candidate_gate",
                "standalone_smoke",
                "rollout_smoke",
                "run_build_check",
                "run_smokes",
                "strict",
                "out",
            ],
            verification_tests=[
                "gpu_telemetry_present",
                "release_binary_present",
                "rollout_cuda_fast_enough",
                "seed55_frontier_present",
                "split_leakage_clean",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/training_preflight.py",
                "configs/training_profiles_rtx2060super.json",
                "configs/ablation_matrix_rtx2060super.json",
            ],
            verification_grade="promotion_gate",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="babylm_split_leakage_checker",
            lifecycle="active",
            task_family="benchmark_verification",
            purpose="Verify public, private, mutated, and bridge BabyLM/BLIMP splits do not share exact minimal pairs before candidate training.",
            command="python scripts/check_babylm_splits.py --out reports/babylm_split_leakage_report.json",
            parameters=["split", "out"],
            verification_tests=[
                "exact_pair_overlap_zero",
                "sentence_overlap_reported",
                "missing_split_reported",
            ],
            provenance=[
                "scripts/check_babylm_splits.py",
                "data/babylm_blimp_filtered_train.jsonl",
                "data/babylm_mutated_holdout_seed49.jsonl",
            ],
            verification_grade="runtime_monitored",
            risk_tier="low",
        )
    )

    tools.append(
        tool_card(
            name="candidate_promotion_gate",
            lifecycle="active",
            task_family="candidate_promotion",
            purpose="Promote a candidate only when architecture/RMI gates are green, public comparator holds, seed49 remains regression, seed55 improves/clears floor, residual escrow is active, and CUDA runtime is reported.",
            command="python scripts/candidate_promotion_gate.py --runtime-report reports/preflight_cuda_rollout_smoke.json --out reports/candidate_promotion_gate.json",
            parameters=[
                "architecture_gate",
                "rmi",
                "public_report",
                "seed49_regression",
                "seed55_frontier",
                "residual_escrow",
                "residual_baseline",
                "runtime_report",
                "max_cluster_delta",
                "max_active_diagnostic_delta",
                "max_critical_delta",
                "max_residual_delta",
                "out",
            ],
            verification_tests=[
                "architecture_gate_green",
                "rmi_score_green",
                "public_comparator_no_regression",
                "seed49_regression_holds",
                "seed55_frontier_clears_floor",
                "residual_delta_bounded",
                "runtime_cost_reported",
            ],
            provenance=[
                "scripts/candidate_promotion_gate.py",
                "reports/architecture_gate_report.json",
                "reports/ratcheting_modular_intelligence_report.json",
            ],
            verification_grade="promotion_gate",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="residual_escrow_snapshotter",
            lifecycle="active",
            task_family="candidate_promotion",
            purpose="Snapshot residual escrow before frontier or candidate runs so promotion can compare residual deltas instead of only checking escrow exists.",
            command="python scripts/snapshot_residual_escrow.py --source reports/residual_escrow.json --out reports/residual_escrow_pre_candidate_baseline.json",
            parameters=["source", "out"],
            verification_tests=[
                "baseline_file_written",
                "summary_preserved",
                "cluster_count_recorded",
            ],
            provenance=["scripts/snapshot_residual_escrow.py", "reports/residual_escrow.json"],
            verification_grade="promotion_gate_dependency",
            risk_tier="low",
        )
    )

    tools.append(
        tool_card(
            name="rtx2060super_ablation_matrix_runner",
            lifecycle="active",
            task_family="ablation_governance",
            purpose="Run matched pre-candidate ablations for CPU baseline, CUDA readout, frozen rollout state, learned state, bridge pressure, and public-plus-mutated calibration.",
            command="python scripts/run_ablation_matrix.py --out reports/ablation_matrix_rtx2060super_report.json",
            parameters=["matrix", "out", "workflow_trace_out", "timeout_seconds", "skip_existing"],
            verification_tests=[
                "all_command_ablations_return_zero",
                "runtime_profiles_summarized",
                "workflow_traces_appended",
                "external_inference_calls_zero",
            ],
            provenance=[
                "scripts/run_ablation_matrix.py",
                "configs/ablation_matrix_rtx2060super.json",
            ],
            verification_grade="matched_ablation_passed",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="rtx2060super_profile_vram_stress",
            lifecycle="active",
            task_family="runtime_verification",
            purpose="Probe inner_loop and candidate CUDA rollout profile shapes against RTX 2060 Super VRAM limits before long runs.",
            command="python scripts/profile_vram_stress.py --profile inner_loop --profile candidate --out reports/profile_vram_stress_report.json",
            parameters=["profiles", "profile", "out", "poll_seconds", "timeout_seconds"],
            verification_tests=[
                "cuda_no_fallback",
                "max_vram_under_profile_limit",
                "candidate_shape_runs",
            ],
            provenance=[
                "scripts/profile_vram_stress.py",
                "configs/training_profiles_rtx2060super.json",
            ],
            verification_grade="runtime_monitored",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="one_command_training_ratchet_profile_runner",
            lifecycle="active",
            task_family="training_governance",
            purpose="Run a configured profile as one repeatable ratchet: residual snapshot, seed55 frontier, ablations, VRAM stress, ledger refresh, promotion gate, and preflight.",
            command="python scripts/run_training_ratchet_profile.py --profile inner_loop --out reports/training_ratchet_profile_run.json",
            parameters=[
                "profile",
                "profiles",
                "out",
                "workflow_trace_out",
                "skip_ablation",
                "skip_vram_stress",
                "skip_capability_ratchet",
                "timeout_seconds",
            ],
            verification_tests=[
                "seed55_frontier_report_written",
                "ablation_matrix_report_written",
                "vram_stress_report_written",
                "candidate_gate_rerun",
                "workflow_traces_appended",
            ],
            provenance=[
                "scripts/run_training_ratchet_profile.py",
                "configs/training_profiles_rtx2060super.json",
            ],
            verification_grade="compiled_workflow",
            risk_tier="medium",
        )
    )

    tools.append(
        tool_card(
            name="msvc_developer_shell_loader",
            lifecycle="active",
            task_family="windows_native_tooling",
            purpose="Load Visual Studio's x64 MSVC environment into PowerShell before native Python/Puffer extension builds.",
            command="powershell -ExecutionPolicy Bypass -File scripts/use_msvc_dev_shell.ps1",
            parameters=["visual_studio_installation", "arch", "host_arch"],
            verification_tests=["cl_visible_after_load", "vsdevcmd_found"],
            provenance=["scripts/use_msvc_dev_shell.ps1"],
            verification_grade="environment_setup",
            risk_tier="low",
        )
    )

    saturated = [
        entry["benchmark_name"]
        for entry in benchmark_ledger
        if entry.get("lifecycle") == "regression"
    ]
    if saturated:
        tools.append(
            tool_card(
                name="regression_guard_runner",
                lifecycle="proposed",
                task_family="regression_preservation",
                purpose="Run locked regression surfaces before promoting a new SymLiquid candidate.",
                command="python scripts/benchmark_treadmill.py --reports reports --out reports/benchmark_treadmill_status.json --benchmark-ledger-out reports/benchmark_ledger.json --model-ledger-out reports/model_ledger.json --public-comparator-ledger-out reports/public_comparator_ledger.json",
                parameters=["candidate_report_set", "saturation_threshold", "broken_threshold"],
                verification_tests=[f"preserve:{name}" for name in saturated[:12]],
                provenance=["reports/benchmark_ledger.json", "reports/model_ledger.json"],
                verification_grade="replay_passed",
                risk_tier="medium",
            )
        )

    return {
        "policy": "local_only_no_external_inference",
        "framework": "capability_ratchet_tool_registry",
        "registry_health": {
            "active": sum(1 for tool in tools if tool["lifecycle"] == "active"),
            "proposed": sum(1 for tool in tools if tool["lifecycle"] == "proposed"),
            "candidate": sum(1 for tool in tools if tool["lifecycle"] == "candidate"),
            "retired": sum(1 for tool in tools if tool["lifecycle"] == "retired"),
        },
        "tools": tools,
        "source_ledgers": {
            "benchmark_families": len(benchmark_ledger),
            "model_version": model_ledger.get("model_version", "unknown"),
            "residual_analysis": residual_analysis.get("methodology", "missing"),
        },
    }


def tool_card(
    *,
    name: str,
    lifecycle: str,
    task_family: str,
    purpose: str,
    command: str,
    parameters: list[str],
    verification_tests: list[str],
    provenance: list[str],
    verification_grade: str,
    risk_tier: str,
) -> dict[str, Any]:
    return {
        "tool_name": name,
        "version": "0.1.0",
        "lifecycle": lifecycle,
        "task_family": task_family,
        "purpose": purpose,
        "command": command,
        "inputs": ["local_filesystem_artifacts"],
        "outputs": ["json_report"],
        "parameters": parameters,
        "preconditions": [
            "workspace_present",
            "local_artifacts_available",
            "no_external_inference",
        ],
        "postconditions": [
            "report_written",
            "external_inference_calls_zero_or_report_rejected",
        ],
        "verification_tests": verification_tests,
        "verification_grade": verification_grade,
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
    }


def architecture_hypothesis(
    active_frontier: dict[str, Any] | None,
    residual_targets: dict[str, Any],
) -> dict[str, Any] | None:
    if active_frontier is None:
        return None
    benchmark = active_frontier["benchmark_name"]
    if benchmark in ("babylm_local_probe", "babylm_mutated_holdout"):
        return {
            "hypothesis": "The current BabyLM wall remains because sequence-state formation is too shallow for agreement, binding, ellipsis, and argument-structure residuals.",
            "missing_mechanism": "learned liquid/reservoir/VSA grammar state with role, number, animacy, and dependency slots",
            "expected_frontier_gain": "reduce residual on high-pressure BLIMP families",
            "must_preserve_regressions": True,
            "target_rules": residual_targets.get("target_rules", []),
            "target_terms": residual_targets.get("target_terms", []),
        }
    return {
        "hypothesis": f"{benchmark} residual requires the next least-complex intervention from the diagnostic ladder.",
        "missing_mechanism": active_frontier.get("wall_type", "unknown_wall"),
        "expected_frontier_gain": active_frontier.get("recommended_intervention"),
        "must_preserve_regressions": True,
    }


def next_actions(
    active_frontier: dict[str, Any] | None,
    residual_targets: dict[str, Any],
) -> list[str]:
    if active_frontier is None:
        return [
            "All tracked surfaces are saturated; add harder local live or mutated benchmarks.",
            "Promote current benchmark suite to regression guard before architecture changes.",
        ]
    if active_frontier["benchmark_name"] == "babylm_local_probe":
        rules = ", ".join(residual_targets.get("target_rules", [])[:4])
        return [
            "Generate mutated/private BabyLM-style holdouts for the top residual families.",
            f"Target learned grammar state at rules: {rules}",
            "Train a candidate only if it improves BabyLM frontier and preserves the saturated regression suite.",
        ]
    if active_frontier["benchmark_name"] == "babylm_mutated_holdout":
        rules = ", ".join(residual_targets.get("target_rules", [])[:4])
        return [
            "Treat the mutated BabyLM holdout as the active anti-Goodhart frontier.",
            f"Train grammar-state candidates against public BLIMP, but promote only if mutated residuals improve on: {rules}",
            "Keep public BLIMP/BabyLM in the comparator ledger so progress remains apples-to-apples.",
            "After promotion, regenerate a new mutated holdout seed before further tuning.",
        ]
    return [active_frontier.get("recommended_intervention", "Run diagnostic ladder.")]


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


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
