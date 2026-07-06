"""One-command local training ratchet runner for SymLiquid profiles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_budget_planner import build_budget_report, write_json as write_budget_json


ROOT = Path(__file__).resolve().parents[1]
PROFILE_HEARTBEAT = ROOT / "reports" / "training_ratchet_profile_heartbeat.json"
SPARKSTREAM_STATUS = ROOT / "reports" / "sparkstream_status.json"
STEP_HEARTBEAT_SECONDS = 5
PRESSURE_FRONTIER_FAMILIES = {
    "minecraft_rl",
    "drone_rl",
    "coding_local_sandbox",
    "web_agent_local",
    "transfer_eval",
}


def training_data_path(*parts: str) -> str:
    root = os.environ.get("THESEUS_TRAINING_DATA_ROOT")
    base = Path(root) if root else ROOT / "data" / "training_data"
    return str(base.joinpath(*parts))


def symliquid_cli_command(subcommand: str) -> list[str]:
    release_dir = ROOT / "target" / "release"
    candidates = [release_dir / "symliquid-cli"]
    if sys.platform.startswith("win"):
        candidates.insert(0, release_dir / "symliquid-cli.exe")
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate), subcommand]
    return ["cargo", "run", "--release", "-p", "symliquid-cli", "--", subcommand]


def candidate_runtime_report() -> str:
    if sys.platform == "darwin":
        return "reports/macos_training_preflight.json"
    return "reports/preflight_cuda_rollout_smoke.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="inner_loop", choices=["smoke", "inner_loop", "candidate", "seed_sweep"])
    parser.add_argument("--profiles", default="configs/training_profiles_rtx2060super.json")
    parser.add_argument("--out", default="reports/training_ratchet_profile_run.json")
    parser.add_argument("--workflow-trace-out", default="reports/workflow_routing_traces.jsonl")
    parser.add_argument("--frontier-seed", type=int, default=55)
    parser.add_argument(
        "--frontier-family",
        default="babylm_mutated",
        choices=[
            "babylm_mutated",
            "rl_local",
            "minecraft_rl",
            "drone_rl",
            "coding_local_sandbox",
            "web_agent_local",
            "transfer_eval",
        ],
    )
    parser.add_argument("--frontier-eval", default="")
    parser.add_argument("--frontier-report", default="")
    parser.add_argument("--rl-frontier-env", default="")
    parser.add_argument("--rl-frontier-seed", type=int, default=1)
    parser.add_argument("--pressure-card-id", default="")
    parser.add_argument("--budget-mode", choices=["auto", "fixed"], default="auto")
    parser.add_argument("--budget-out", default="reports/training_budget_plan.json")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--force-frontier-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-vram-stress", action="store_true")
    parser.add_argument("--skip-capability-ratchet", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    profile_config = read_json(Path(args.profiles))
    profile = (profile_config.get("profiles") or {}).get(args.profile)
    if not profile:
        raise SystemExit(f"profile not found: {args.profile}")

    steps: list[dict[str, Any]] = []
    ok = True

    commands = []
    frontier_eval = ""
    frontier_report = ""
    rl_paths = rl_frontier_paths(args.rl_frontier_env, args.rl_frontier_seed)
    if args.profile == "smoke":
        commands.append(
            step_command(
                "preflight_smoke",
                [
                    sys.executable,
                    "scripts/training_preflight.py",
                    "--run-smokes",
                    "--run-split-check",
                    "--run-candidate-gate",
                    "--out",
                    "reports/training_preflight_report.json",
                ],
            )
        )
    else:
        commands.append(
            step_command(
                "residual_escrow_pre_candidate_snapshot",
                [
                    sys.executable,
                    "scripts/snapshot_residual_escrow.py",
                    "--source",
                    "reports/residual_escrow.json",
                    "--out",
                    "reports/residual_escrow_pre_candidate_baseline.json",
                ],
            )
        )
        commands.append(open_conversation_training_pantry_step())
        frontier_eval = args.frontier_eval or f"data/babylm_mutated_holdout_seed{args.frontier_seed}.jsonl"
        frontier_report = args.frontier_report or (
            f"reports/babylm_mutated_holdout_seed{args.frontier_seed}_stateful_grammar_state_frontier.json"
        )
        synthetic_blend = "data/synthetic/babylm_train_plus_synthetic_current.jsonl"
        if args.frontier_family == "rl_local":
            commands.append(token_superposition_efficiency_step(profile))
            commands.append(rl_frontier_train_step(profile, args.rl_frontier_env, args.rl_frontier_seed, rl_paths))
            commands.append(rl_frontier_smoke_step(profile, args.rl_frontier_env, rl_paths))
        elif args.frontier_family in {"minecraft_rl", "drone_rl", "coding_local_sandbox", "web_agent_local", "transfer_eval"}:
            budget_report = build_budget_report(
                profiles=profile_config,
                profile_name=args.profile,
                frontier_family=args.frontier_family,
                pressure_card_id=args.pressure_card_id or default_pressure_card(args.frontier_family),
                mode=args.budget_mode,
            )
            write_budget_json(ROOT / args.budget_out, budget_report)
            if args.frontier_family in {"coding_local_sandbox", "transfer_eval", "web_agent_local"}:
                commands.append(token_superposition_efficiency_step(profile))
            commands.append(
                pressure_runner_step(
                    profile,
                    args.frontier_family,
                    args.pressure_card_id,
                    args.rl_frontier_seed,
                    budget_report,
                    args.budget_out,
                )
            )
            if args.frontier_family == "coding_local_sandbox":
                code_card_id = args.pressure_card_id or default_pressure_card(args.frontier_family)
                code_max_cases = code_max_cases_per_card(profile, budget_report)
                code_candidate_manifest = student_code_candidate_manifest_path(code_card_id, args.rl_frontier_seed)
                commands.append(open_code_training_pantry_step())
                commands.append(sts_learning_forge_step())
                commands.append(cognitive_context_router_step())
                commands.append(long_horizon_programming_curriculum_step())
                commands.append(sts_native_parallel_probe_step())
                commands.append(code_residual_curriculum_step())
                commands.append(
                    student_code_candidate_generator_step(
                        code_card_id,
                        args.rl_frontier_seed,
                        max_cases_per_card=code_max_cases,
                    )
                )
                commands.append(
                    real_code_benchmark_graduation_step(
                        code_card_id,
                        args.rl_frontier_seed,
                        max_cases_per_card=code_max_cases,
                        student_candidate_manifest=code_candidate_manifest,
                        skip_student_candidate_generation=True,
                        out=student_learning_baseline_report_path(code_card_id, args.rl_frontier_seed),
                        trace_out=student_learning_baseline_trace_path(code_card_id, args.rl_frontier_seed),
                        transfer_artifact_out=student_learning_baseline_transfer_path(code_card_id, args.rl_frontier_seed),
                    )
                )
                commands.append(
                    student_learning_closure_step(
                        code_card_id,
                        args.rl_frontier_seed,
                        candidate_manifest=code_candidate_manifest,
                        trace_in=student_learning_baseline_trace_path(code_card_id, args.rl_frontier_seed),
                    )
                )
                commands.append(
                    real_code_benchmark_graduation_step(
                        code_card_id,
                        args.rl_frontier_seed,
                        max_cases_per_card=code_max_cases,
                        student_candidate_manifest=code_candidate_manifest,
                        skip_student_candidate_generation=True,
                    )
                )
                commands.append(sts_repair_ablation_step())
                commands.append(
                    code_residual_forge_step(
                        code_card_id,
                        args.rl_frontier_seed,
                    )
                )
        else:
            commands.append(token_superposition_efficiency_step(profile))
            commands.append(synthetic_data_step(profile, synthetic_blend))
            generation = frontier_generation_step(profile, args.frontier_seed, frontier_eval, args.force_frontier_generation)
            if generation is not None:
                commands.append(generation)
            commands.append(mutated_frontier_step(profile, synthetic_blend, frontier_eval, frontier_report))
        if not args.skip_ablation:
            commands.append(
                step_command(
                    "ablation_matrix",
                    [
                        sys.executable,
                        "scripts/run_ablation_matrix.py",
                        "--out",
                        "reports/ablation_matrix_rtx2060super_report.json",
                    ],
                )
            )
        if not args.skip_vram_stress:
            stress_profiles = ["inner_loop"] if args.profile == "inner_loop" else ["inner_loop", "candidate"]
            commands.append(
                step_command(
                    "profile_vram_stress",
                    [
                        sys.executable,
                        "scripts/profile_vram_stress.py",
                        *sum((["--profile", name] for name in stress_profiles), []),
                        "--out",
                        "reports/profile_vram_stress_report.json",
                    ],
                )
            )
        if not args.skip_capability_ratchet:
            commands.append(
                step_command(
                    "capability_ratchet_refresh",
                    [
                        sys.executable,
                        "scripts/run_capability_ratchet.py",
                            "--mutated-babylm-report",
                            frontier_report,
                            "--mutated-babylm-eval",
                            frontier_eval,
                            "--skip-holdout-generation",
                        "--out",
                        "reports/capability_ratchet_run.json",
                    ],
                )
            )
        runtime_report = candidate_runtime_report()
        candidate_gate_command = [
            sys.executable,
            "scripts/candidate_promotion_gate.py",
            "--runtime-report",
            runtime_report,
            "--out",
            "reports/candidate_promotion_gate.json",
        ]
        if args.frontier_family == "babylm_mutated":
            candidate_gate_command.extend(["--frontier-report", frontier_report])
        elif args.frontier_family in PRESSURE_FRONTIER_FAMILIES:
            candidate_gate_command.extend(
                [
                    "--frontier-report",
                    pressure_report_path(
                        args.frontier_family,
                        args.pressure_card_id or default_pressure_card(args.frontier_family),
                        args.rl_frontier_seed,
                    ),
                ]
            )
        training_preflight_command = [
            sys.executable,
            "scripts/training_preflight.py",
            "--run-split-check",
            "--run-candidate-gate",
            "--out",
            "reports/training_preflight_report.json",
            "--rollout-smoke",
            runtime_report,
            "--candidate-gate-profile-step-in-progress",
            "training_preflight_refresh",
        ]
        if args.frontier_family == "babylm_mutated":
            training_preflight_command.extend(["--frontier-report", frontier_report])
        elif args.frontier_family in PRESSURE_FRONTIER_FAMILIES:
            training_preflight_command.extend(
                [
                    "--frontier-report",
                    pressure_report_path(
                        args.frontier_family,
                        args.pressure_card_id or default_pressure_card(args.frontier_family),
                        args.rl_frontier_seed,
                    ),
                ]
            )
        commands.extend(
            [
                step_command(
                    "training_preflight_refresh",
                    training_preflight_command,
                ),
                step_command(
                    "candidate_promotion_gate",
                    candidate_gate_command,
                    allow_failure=True,
                ),
            ]
        )
    commands.append(genesis_kernel_snapshot_step())
    commands.append(reality_manipulator_step())
    commands.append(grammar_suckers_step())
    commands.append(deterministic_taming_stack_step())
    commands.append(architecture_guidance_loop_step(allow_teacher=args.allow_teacher))
    commands.append(teacher_budget_audit_step())
    commands.append(cell_lifecycle_step())
    commands.append(broad_transfer_matrix_step())
    commands.append(transfer_generalization_audit_step())
    commands.append(broad_code_calibration_scheduler_step())
    commands.append(student_first_evidence_audit_step())
    commands.append(learning_scoreboard_step())
    commands.append(viea_autonomy_spine_step())
    commands.append(overnight_learning_readiness_step())

    artifacts = {
        "seed55_frontier": "reports/babylm_mutated_holdout_seed55_stateful_grammar_state_frontier.json",
        "mutated_frontier": frontier_report if args.profile != "smoke" else "",
        "mutated_frontier_eval": frontier_eval if args.profile != "smoke" else "",
        "rl_frontier_train": rl_paths["train_report"] if args.profile != "smoke" and args.frontier_family == "rl_local" else "",
        "rl_frontier_smoke": rl_paths["smoke_report"] if args.profile != "smoke" and args.frontier_family == "rl_local" else "",
        "rl_frontier_policy": rl_paths["policy"] if args.profile != "smoke" and args.frontier_family == "rl_local" else "",
        "pressure_runner": pressure_report_path(args.frontier_family, args.pressure_card_id, args.rl_frontier_seed)
        if args.profile != "smoke" and args.frontier_family not in {"babylm_mutated", "rl_local"}
        else "",
        "code_residual_forge": "reports/code_residual_forge.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "code_transfer_artifacts": "reports/code_transfer_artifacts.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "code_frontier_rotation": "reports/code_frontier_rotation.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "code_repair_traces": "reports/code_repair_traces.jsonl"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "real_code_benchmark_graduation": "reports/real_code_benchmark_graduation.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_code_candidate_generator": "reports/code_lm_closure.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "local_theseus_student_code_checkpoint": "reports/student_code_lm_checkpoint.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_code_candidates": "reports/student_code_candidates.jsonl"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "sts_learning_forge": "reports/sts_learning_forge.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "sts_native_parallel_probe": "reports/sts_native_parallel_probe.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_closure": "reports/student_learning_closure.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_code_checkpoint": "reports/student_learning_code_checkpoint.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_neural_code_checkpoint": "reports/student_neural_code_checkpoint.json"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_code_candidates": "reports/student_learning_code_candidates.jsonl"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_training_examples": "reports/student_learning_training_examples.jsonl"
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_baseline_real_code": student_learning_baseline_report_path(
            args.pressure_card_id or default_pressure_card(args.frontier_family),
            args.rl_frontier_seed,
        )
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "student_learning_baseline_trace": student_learning_baseline_trace_path(
            args.pressure_card_id or default_pressure_card(args.frontier_family),
            args.rl_frontier_seed,
        )
        if args.profile != "smoke" and args.frontier_family == "coding_local_sandbox"
        else "",
        "broad_transfer_matrix": "reports/broad_transfer_matrix.json",
        "broad_transfer_matrix_markdown": "reports/broad_transfer_matrix.md",
        "transfer_generalization_audit": "reports/transfer_generalization_audit.json",
        "transfer_generalization_audit_markdown": "reports/transfer_generalization_audit.md",
        "broad_code_calibration_scheduler": "reports/broad_code_calibration_scheduler.json",
        "broad_code_calibration_scheduler_markdown": "reports/broad_code_calibration_scheduler.md",
        "ablation_matrix": "reports/ablation_matrix_rtx2060super_report.json",
        "vram_stress": "reports/profile_vram_stress_report.json",
        "capability_ratchet": "reports/capability_ratchet_run.json",
        "candidate_gate": "reports/candidate_promotion_gate.json",
        "preflight": "reports/training_preflight_report.json",
        "workflow_traces": args.workflow_trace_out,
        "synthetic_data": "reports/synthetic_data_curator.json",
        "training_budget": args.budget_out,
        "genesis_kernel": "reports/genesis_kernel/report.json",
        "genesis_release_manifest": "reports/genesis_kernel/latest_release/release_manifest.json",
        "reality_manipulator": "reports/reality_manipulator.json",
        "reality_manipulator_markdown": "reports/reality_manipulator.md",
        "reality_manipulator_world": "reports/reality_manipulator/latest_world/world.json",
        "reality_manipulator_release_manifest": "reports/reality_manipulator/latest_world/release_manifest.json",
        "open_code_training_pantry": "reports/open_code_training_pantry.json",
        "open_conversation_training_pantry": "reports/open_conversation_training_pantry.json",
        "open_conversation_training_pantry_markdown": "reports/open_conversation_training_pantry.md",
        "grammar_suckers": "reports/grammar_suckers.json",
        "grammar_suckers_markdown": "reports/grammar_suckers.md",
        "grammar_suckers_sbl_traces": "data/grammar_suckers/sbl_rule_traces.jsonl",
        "deterministic_taming_stack": "reports/deterministic_taming_stack.json",
        "architecture_guidance_loop": "reports/architecture_guidance_loop.json",
        "architecture_guided_experiments": "reports/architecture_guided_experiments.json",
        "teacher_budget_audit": "reports/teacher_budget_audit.json",
        "cell_lifecycle": "reports/cell_lifecycle.json",
        "cell_lifecycle_markdown": "reports/cell_lifecycle.md",
        "cell_lifecycle_prune_plan": "reports/cell_lifecycle_prune_plan.json",
        "code_residual_curriculum": "reports/code_residual_curriculum.json",
        "code_residual_curriculum_private_train": training_data_path(
            "residual_code_curriculum",
            "private_train",
            "residual_code_lm_tasks.jsonl",
        ),
        "sts_repair_ablation": "reports/sts_repair_ablation.json",
        "cognitive_context_router": "reports/cognitive_context_router.json",
        "cognitive_context_sts_rows": "data/sts_learning/cognitive_context_spaces_seed14.jsonl",
        "cognitive_context_merged_sts_rows": "data/sts_learning/sts_code_context_spaces_seed14.jsonl",
        "long_horizon_programming_curriculum": "reports/long_horizon_programming_curriculum.json",
        "long_horizon_programming_tasks": training_data_path(
            "long_horizon_programming",
            "private_train",
            "repo_repair_tasks.jsonl",
        ),
        "long_horizon_programming_sts": training_data_path(
            "long_horizon_programming",
            "sts",
            "repo_repair_sts_rows.jsonl",
        ),
        "student_first_evidence_audit": "reports/student_first_evidence_audit.json",
        "learning_scoreboard": "reports/learning_scoreboard.json",
        "learning_scoreboard_markdown": "reports/learning_scoreboard.md",
        "viea_autonomy_spine": "reports/viea_autonomy_spine.json",
        "feedback_action_queue": "reports/feedback_action_queue.json",
        "broad_transfer_action_queue": "reports/broad_transfer_action_queue.json",
        "repo_repair_main_curriculum": "reports/repo_repair_main_curriculum.json",
        "teacher_architect_closure": "reports/teacher_architect_closure.json",
        "symliquid_state_engine_queue": "reports/symliquid_state_engine_queue.json",
        "overnight_learning_readiness": "reports/overnight_learning_readiness.json",
        "overnight_learning_readiness_markdown": "reports/overnight_learning_readiness.md",
    }

    steps.extend(load_resumed_steps(args, artifacts) if args.resume else [])
    completed_names = {
        str(step.get("name"))
        for step in steps
        if isinstance(step, dict) and step.get("returncode") == 0
    }

    for item in commands:
        if str(item.get("name")) in completed_names:
            write_profile_report(args, ok, steps, artifacts)
            continue
        row = run_step(item, timeout=args.timeout_seconds)
        steps.append(row)
        append_workflow_trace(
            Path(args.workflow_trace_out),
            task=row["name"],
            command=" ".join(row["command"]),
            returncode=row["returncode"],
            runtime_ms=row["runtime_ms"],
        )
        if row["returncode"] != 0 and not row.get("allow_failure"):
            ok = False
        write_profile_report(args, ok, steps, artifacts)
        if row["returncode"] != 0 and not row.get("allow_failure"):
            break

    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "one_command_training_ratchet_profile",
        "created_utc": now(),
        "updated_utc": now(),
        "profile": args.profile,
        "frontier_family": args.frontier_family,
        "ok": ok,
        "steps": steps,
        "artifacts": artifacts,
        "failure": first_failure(steps),
        "recovery": recovery_plan(args, steps, artifacts),
        "external_inference_calls": 0,
    }
    write_json(Path(args.out), report)
    write_profile_final_status(args, ok, steps)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def write_profile_report(
    args: argparse.Namespace,
    ok: bool,
    steps: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    write_json(
        Path(args.out),
        {
            "policy": "local_only_no_external_inference",
            "methodology": "one_command_training_ratchet_profile",
            "updated_utc": now(),
            "profile": args.profile,
            "frontier_family": args.frontier_family,
            "ok": ok,
            "steps": steps,
            "artifacts": artifacts,
            "failure": first_failure(steps),
            "recovery": recovery_plan(args, steps, artifacts),
            "external_inference_calls": 0,
        },
    )


def write_profile_final_status(args: argparse.Namespace, ok: bool, steps: list[dict[str, Any]]) -> None:
    status = "completed" if ok else "failed"
    payload = {
        "updated_utc": now(),
        "phase": f"profile_{status}",
        "message": f"{args.profile} profile {status}; steps={len(steps)}",
        "profile": args.profile,
        "profile_step": {
            "name": (steps[-1].get("name") if steps else ""),
            "status": status,
            "returncode": (steps[-1].get("returncode") if steps else None),
        },
    }
    write_json(SPARKSTREAM_STATUS, payload)


def synthetic_data_step(profile: dict[str, Any], blend_out: str) -> dict[str, Any]:
    babylm = profile.get("babylm") or {}
    train_limit = int(babylm.get("train_limit", 50000))
    return step_command(
        "synthetic_data_curator",
        [
            sys.executable,
            "scripts/synthetic_data_curator.py",
            "--policy",
            "configs/synthetic_data_policy.json",
            "--blend-total",
            str(train_limit),
            "--blend-out",
            blend_out,
            "--out-data",
            "data/synthetic/babylm_residual_targeted_current.jsonl",
            "--out",
            "reports/synthetic_data_curator.json",
        ],
    )


def frontier_generation_step(
    profile: dict[str, Any],
    seed: int,
    frontier_eval: str,
    force: bool,
) -> dict[str, Any] | None:
    eval_path = ROOT / frontier_eval
    if eval_path.exists() and not force:
        return None
    babylm = profile.get("babylm") or {}
    count = int(babylm.get("eval_limit", 4800))
    residual_source = "reports/babylm_mutated_residual_analysis.json"
    if not (ROOT / residual_source).exists():
        residual_source = "reports/babylm_residual_analysis.json"
    return step_command(
        f"generate_mutated_frontier_seed{seed}",
        [
            sys.executable,
            "scripts/generate_babylm_mutated_holdout.py",
            "--residual-analysis",
            residual_source,
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--out",
            frontier_eval,
            "--report-out",
            f"reports/babylm_mutated_holdout_seed{seed}_factory.json",
        ],
    )


def mutated_frontier_step(
    profile: dict[str, Any],
    train_input: str,
    frontier_eval: str,
    frontier_report: str,
) -> dict[str, Any]:
    babylm = profile.get("babylm") or {}
    return step_command(
        "mutated_frontier",
        [
            *symliquid_cli_command("train-babylm-probe"),
            "--input",
            train_input,
            "--eval-input",
            frontier_eval,
            "--train-limit",
            str(babylm.get("train_limit", 50000)),
            "--eval-limit",
            str(babylm.get("eval_limit", 4800)),
            "--steps",
            str(babylm.get("steps", 1000)),
            "--hv-dim",
            str(babylm.get("hv_dim", 16384)),
            "--lr",
            str(babylm.get("lr", 0.08)),
            "--stateful",
            "--pairwise-contrast",
            "--balance-rules",
            "--prior-weight",
            str(babylm.get("prior_weight", 1.0)),
            "--out",
            frontier_report,
        ],
    )


def token_superposition_efficiency_step(profile: dict[str, Any]) -> dict[str, Any]:
    tst = profile.get("token_superposition") or {}
    return step_command(
        "token_superposition_platform_apples_to_apples",
        [
            sys.executable,
            "scripts/token_superposition_training.py",
            "--config",
            str(tst.get("config", "configs/token_superposition_training.json")),
            "--out",
            str(tst.get("out", "reports/token_superposition_training.json")),
            "--skip-if-evidence",
        ],
        timeout_seconds=int(tst.get("timeout_seconds", 1800)),
    )


def open_code_training_pantry_step() -> dict[str, Any]:
    return step_command(
        "open_code_training_pantry",
        [
            sys.executable,
            "scripts/open_code_training_pantry.py",
            "--root",
            training_data_path("open_code_pantry"),
            "--repo-config",
            "configs/open_code_training_pantry_expanded.json",
            "--out",
            "reports/open_code_training_pantry.json",
        ],
        timeout_seconds=900,
    )


def open_conversation_training_pantry_step() -> dict[str, Any]:
    return step_command(
        "open_conversation_training_pantry",
        [
            sys.executable,
            "scripts/open_conversation_training_pantry.py",
            "--config",
            "configs/open_conversation_training_pantry.json",
            "--root",
            training_data_path("open_conversation_pantry"),
            "--allow-network-fetch",
            "--out",
            "reports/open_conversation_training_pantry.json",
            "--markdown-out",
            "reports/open_conversation_training_pantry.md",
        ],
        timeout_seconds=240,
    )


def sts_learning_forge_step() -> dict[str, Any]:
    return step_command(
        "sts_learning_forge",
        [
            sys.executable,
            "scripts/sts_learning_forge.py",
            "--out",
            "reports/sts_learning_forge.json",
            "--out-data",
            "data/sts_learning/sts_code_streams_seed14.jsonl",
        ],
        timeout_seconds=120,
    )


def sts_native_parallel_probe_step() -> dict[str, Any]:
    return step_command(
        "sts_native_parallel_probe",
        [
            sys.executable,
            "scripts/sts_native_parallel_probe.py",
            "--input",
            "data/sts_learning/sts_code_context_spaces_seed14.jsonl",
            "--out",
            "reports/sts_native_parallel_probe.json",
        ],
        timeout_seconds=360,
    )


def cognitive_context_router_step() -> dict[str, Any]:
    return step_command(
        "cognitive_context_router",
        [
            sys.executable,
            "scripts/cognitive_context_router.py",
            "--policy",
            "configs/cognitive_context_policy.json",
            "--base-sts",
            "data/sts_learning/sts_code_streams_seed14.jsonl",
            "--out-data",
            "data/sts_learning/cognitive_context_spaces_seed14.jsonl",
            "--merged-out-data",
            "data/sts_learning/sts_code_context_spaces_seed14.jsonl",
            "--out",
            "reports/cognitive_context_router.json",
            "--markdown-out",
            "reports/cognitive_context_router.md",
        ],
        timeout_seconds=120,
    )


def long_horizon_programming_curriculum_step() -> dict[str, Any]:
    return step_command(
        "long_horizon_programming_curriculum",
        [
            sys.executable,
            "scripts/long_horizon_programming_curriculum.py",
            "--task-out",
            training_data_path("long_horizon_programming", "private_train", "repo_repair_tasks.jsonl"),
            "--sts-out",
            training_data_path("long_horizon_programming", "sts", "repo_repair_sts_rows.jsonl"),
            "--out",
            "reports/long_horizon_programming_curriculum.json",
            "--markdown-out",
            "reports/long_horizon_programming_curriculum.md",
        ],
        timeout_seconds=120,
    )


def code_residual_curriculum_step() -> dict[str, Any]:
    return step_command(
        "code_residual_curriculum",
        [
            sys.executable,
            "scripts/code_residual_curriculum.py",
            "--trace-in",
            "reports/real_code_benchmark_traces.jsonl",
            "--private-out",
            training_data_path("residual_code_curriculum", "private_train", "residual_code_lm_tasks.jsonl"),
            "--out",
            "reports/code_residual_curriculum.json",
            "--markdown-out",
            "reports/code_residual_curriculum.md",
        ],
        timeout_seconds=120,
    )


def genesis_kernel_snapshot_step() -> dict[str, Any]:
    return step_command(
        "genesis_kernel_snapshot",
        [
            sys.executable,
            "scripts/genesis_kernel.py",
            "ingest-theseus",
            "--out",
            "reports/genesis_kernel/report.json",
            "--bundle-dir",
            "reports/genesis_kernel/latest_release",
        ],
        timeout_seconds=120,
    )


def reality_manipulator_step() -> dict[str, Any]:
    return step_command(
        "reality_manipulator",
        [
            sys.executable,
            "scripts/reality_manipulator.py",
            "--out",
            "reports/reality_manipulator.json",
            "--markdown-out",
            "reports/reality_manipulator.md",
            "--bundle-dir",
            "reports/reality_manipulator/latest_world",
        ],
        timeout_seconds=120,
    )


def grammar_suckers_step() -> dict[str, Any]:
    return step_command(
        "grammar_suckers",
        [
            sys.executable,
            "scripts/grammar_suckers.py",
            "--config",
            "configs/grammar_suckers.json",
            "--out",
            "reports/grammar_suckers.json",
            "--markdown-out",
            "reports/grammar_suckers.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def deterministic_taming_stack_step() -> dict[str, Any]:
    return step_command(
        "deterministic_taming_stack",
        [
            sys.executable,
            "scripts/deterministic_taming_stack.py",
            "--run-cargo-check",
            "--out",
            "reports/deterministic_taming_stack.json",
            "--markdown-out",
            "reports/deterministic_taming_stack.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def architecture_guidance_loop_step(*, allow_teacher: bool = False) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/architecture_guidance_loop.py",
        "--out",
        "reports/architecture_guidance_loop.json",
        "--markdown-out",
        "reports/architecture_guidance_loop.md",
        "--experiments-out",
        "reports/architecture_guided_experiments.json",
    ]
    if allow_teacher:
        command.append("--allow-teacher")
    return step_command(
        "architecture_guidance_loop",
        command,
        timeout_seconds=1800 if allow_teacher else 120,
        allow_failure=allow_teacher,
    )


def teacher_budget_audit_step() -> dict[str, Any]:
    return step_command(
        "teacher_budget_audit",
        [
            sys.executable,
            "scripts/teacher_budget_audit.py",
            "--out",
            "reports/teacher_budget_audit.json",
            "--markdown-out",
            "reports/teacher_budget_audit.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def cell_lifecycle_step() -> dict[str, Any]:
    return step_command(
        "cell_lifecycle",
        [
            sys.executable,
            "scripts/cell_lifecycle.py",
            "--out",
            "reports/cell_lifecycle.json",
            "--markdown-out",
            "reports/cell_lifecycle.md",
            "--prune-plan-out",
            "reports/cell_lifecycle_prune_plan.json",
        ],
        timeout_seconds=120,
    )


def student_first_evidence_audit_step() -> dict[str, Any]:
    return step_command(
        "student_first_evidence_audit",
        [
            sys.executable,
            "scripts/student_first_evidence_audit.py",
            "--out",
            "reports/student_first_evidence_audit.json",
            "--markdown-out",
            "reports/student_first_evidence_audit.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def broad_transfer_matrix_step() -> dict[str, Any]:
    return step_command(
        "broad_transfer_matrix",
        [
            sys.executable,
            "scripts/broad_transfer_matrix.py",
            "--min-public-tasks",
            "32",
            "--out",
            "reports/broad_transfer_matrix.json",
            "--markdown-out",
            "reports/broad_transfer_matrix.md",
        ],
        allow_failure=True,
        timeout_seconds=180,
    )


def transfer_generalization_audit_step() -> dict[str, Any]:
    return step_command(
        "transfer_generalization_audit",
        [
            sys.executable,
            "scripts/transfer_generalization_audit.py",
            "--out",
            "reports/transfer_generalization_audit.json",
            "--markdown-out",
            "reports/transfer_generalization_audit.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def broad_code_calibration_scheduler_step() -> dict[str, Any]:
    return step_command(
        "broad_code_calibration_scheduler",
        [
            sys.executable,
            "scripts/broad_code_calibration_scheduler.py",
            "--min-public-tasks",
            "32",
            "--out",
            "reports/broad_code_calibration_scheduler.json",
            "--markdown-out",
            "reports/broad_code_calibration_scheduler.md",
        ],
        timeout_seconds=120,
    )


def learning_scoreboard_step() -> dict[str, Any]:
    return step_command(
        "learning_truth_scoreboard",
        [
            sys.executable,
            "scripts/learning_scoreboard.py",
            "--out",
            "reports/learning_scoreboard.json",
            "--markdown-out",
            "reports/learning_scoreboard.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def viea_autonomy_spine_step() -> dict[str, Any]:
    return step_command(
        "viea_autonomy_spine",
        [
            sys.executable,
            "scripts/viea_autonomy_spine.py",
            "--max-steps",
            "64",
            "--timeout-seconds",
            "900",
            "--out",
            "reports/viea_autonomy_spine.json",
            "--markdown-out",
            "reports/viea_autonomy_spine.md",
        ],
        allow_failure=True,
        timeout_seconds=1800,
    )


def overnight_learning_readiness_step() -> dict[str, Any]:
    return step_command(
        "overnight_learning_readiness",
        [
            sys.executable,
            "scripts/overnight_learning_readiness.py",
            "--out",
            "reports/overnight_learning_readiness.json",
            "--markdown-out",
            "reports/overnight_learning_readiness.md",
        ],
        allow_failure=True,
        timeout_seconds=120,
    )


def rl_frontier_paths(env: str, seed: int) -> dict[str, str]:
    safe_env = (env or "ocean-noisy-tmaze").replace("-", "_")
    stem = f"reports/rl_frontier_{safe_env}_seed{int(seed)}"
    return {
        "policy": f"{stem}_policy.json",
        "train_report": f"{stem}_train.json",
        "smoke_report": f"{stem}_smoke.json",
        "event_log": f"{stem}_events.json",
    }


def rl_frontier_train_step(
    profile: dict[str, Any], env: str, seed: int, paths: dict[str, str]
) -> dict[str, Any]:
    env = env or "ocean-noisy-tmaze"
    rollout = profile.get("puffer_ocean_rollout_cuda") or {}
    num_envs = int(rollout.get("rollout_batch", 128))
    return step_command(
        f"rl_frontier_train_{env}_seed{int(seed)}",
        [
            sys.executable,
            "adapters/pufferlib/symliquid_puffer_adapter.py",
            "--train-discrete-policy",
            "--use-rust-ffi",
            "--env",
            env,
            "--iterations",
            str(max(16, int((profile.get("rl") or {}).get("iterations", 24)))),
            "--population",
            str(max(24, int((profile.get("rl") or {}).get("population", 40)))),
            "--elite-count",
            str(max(4, int((profile.get("rl") or {}).get("elite_count", 8)))),
            "--num-envs",
            str(max(32, num_envs)),
            "--train-steps",
            str(max(256, int((profile.get("rl") or {}).get("train_steps", 512)))),
            "--eval-steps",
            str(max(512, int((profile.get("rl") or {}).get("eval_steps", 2048)))),
            "--seed",
            str(int(seed)),
            "--policy-out",
            paths["policy"],
            "--out",
            paths["train_report"],
        ],
    )


def rl_frontier_smoke_step(profile: dict[str, Any], env: str, paths: dict[str, str]) -> dict[str, Any]:
    env = env or "ocean-noisy-tmaze"
    rollout = profile.get("puffer_ocean_rollout_cuda") or {}
    num_envs = max(16, min(64, int(rollout.get("rollout_batch", 64))))
    return step_command(
        f"rl_frontier_smoke_{env}",
        [
            sys.executable,
            "adapters/pufferlib/symliquid_puffer_adapter.py",
            "--artifact",
            paths["policy"],
            "--env",
            env,
            "--use-rust-ffi",
            "--num-envs",
            str(num_envs),
            "--rollout-smoke-steps",
            str(max(128, int((profile.get("rl") or {}).get("smoke_steps", 512)))),
            "--event-log-out",
            paths["event_log"],
            "--event-log-env-limit",
            "4",
            "--event-log-step-limit",
            "96",
            "--out",
            paths["smoke_report"],
        ],
    )


def pressure_report_path(frontier_family: str, card_id: str, seed: int) -> str:
    safe_card = safe_card_id(card_id or frontier_family or "pressure")
    return f"reports/pressure_{safe_card}_seed{int(seed)}.json"


def safe_card_id(value: str) -> str:
    return (value or "card").replace("-", "_").replace("/", "_")


def code_max_cases_per_card(profile: dict[str, Any], budget_report: dict[str, Any] | None = None) -> int:
    base = profile.get("pressure_runner") or {}
    code_cfg = profile.get("code_public_calibration") if isinstance(profile.get("code_public_calibration"), dict) else {}
    planned = (budget_report or {}).get("pressure_runner") if isinstance((budget_report or {}).get("pressure_runner"), dict) else {}
    episodes = int(planned.get("episodes", base.get("episodes", 2)))
    min_cases = int(code_cfg.get("min_cases_per_card", 32))
    max_cases = int(code_cfg.get("max_cases_per_card", 64))
    cases_per_episode = int(code_cfg.get("cases_per_episode", 8))
    return max(min_cases, min(max_cases, max(1, episodes) * max(1, cases_per_episode)))


def code_lm_work_steps(max_cases_per_card: int) -> int:
    # Work-step budget is the primary duration control; wall-clock timeouts are only safety fuses.
    return max(120_000, int(max_cases_per_card) * 4_000)


def code_lm_step_timeout_seconds(max_cases_per_card: int) -> int:
    return max(7_200, min(21_600, int(max_cases_per_card) * 300))


def student_learning_baseline_report_path(card_id: str, seed: int) -> str:
    return f"reports/student_learning_baseline_{safe_card_id(card_id)}_seed{int(seed)}_real_code.json"


def student_learning_baseline_trace_path(card_id: str, seed: int) -> str:
    return f"reports/student_learning_baseline_{safe_card_id(card_id)}_seed{int(seed)}_traces.jsonl"


def student_learning_baseline_transfer_path(card_id: str, seed: int) -> str:
    return f"reports/transfer_artifacts/code/student_learning_baseline_{safe_card_id(card_id)}_seed{int(seed)}_transfer_artifact.json"


def pressure_runner_step(
    profile: dict[str, Any],
    frontier_family: str,
    card_id: str,
    seed: int,
    budget_report: dict[str, Any] | None = None,
    budget_path: str = "reports/training_budget_plan.json",
) -> dict[str, Any]:
    card_id = card_id or default_pressure_card(frontier_family)
    base = profile.get("pressure_runner") or {}
    planned = (budget_report or {}).get("pressure_runner") if isinstance((budget_report or {}).get("pressure_runner"), dict) else {}
    steps = int(planned.get("steps", base.get("steps", 96)))
    episodes = int(planned.get("episodes", base.get("episodes", 2)))
    train_iterations = int(planned.get("train_iterations", base.get("train_iterations", 4)))
    train_population = int(planned.get("train_population", base.get("train_population", 12)))
    elite_count = int(planned.get("elite_count", base.get("elite_count", 4)))
    eval_seed_count = int(planned.get("eval_seed_count", base.get("eval_seed_count", episodes)))
    min_candidate_evals = int(planned.get("min_train_candidate_evaluations", 0))
    min_train_env_steps = int(planned.get("min_train_env_steps", 0))
    return step_command(
        f"pressure_runner_{card_id}_seed{int(seed)}",
        [
            sys.executable,
            "scripts/pressure_runner.py",
            "--card-id",
            card_id,
            "--frontier-family",
            frontier_family,
            "--seed",
            str(int(seed)),
            "--episodes",
            str(max(1, episodes)),
            "--steps",
            str(max(1, steps)),
            "--train-iterations",
            str(max(1, train_iterations)),
            "--train-population",
            str(max(4, train_population)),
            "--elite-count",
            str(max(1, elite_count)),
            "--eval-seed-count",
            str(max(1, eval_seed_count)),
            "--min-train-candidate-evals",
            str(max(0, min_candidate_evals)),
            "--min-train-env-steps",
            str(max(0, min_train_env_steps)),
            "--budget-report",
            budget_path,
            "--out",
            pressure_report_path(frontier_family, card_id, seed),
        ],
        timeout_seconds=pressure_step_timeout_seconds(
            steps=steps,
            train_iterations=train_iterations,
            train_population=train_population,
            eval_seed_count=eval_seed_count,
        ),
    )


def code_residual_forge_step(card_id: str, seed: int) -> dict[str, Any]:
    return step_command(
        f"code_residual_forge_{card_id}_seed{int(seed)}",
        [
            sys.executable,
            "scripts/code_residual_forge.py",
            "--out",
            "reports/code_residual_forge.json",
            "--transfer-out",
            "reports/code_transfer_artifacts.json",
            "--rotation-out",
            "reports/code_frontier_rotation.json",
            "--trace-out",
            "reports/code_repair_traces.jsonl",
            "--active-card-id",
            card_id,
            "--active-report",
            pressure_report_path("coding_local_sandbox", card_id, seed),
        ],
    )


def student_code_candidate_generator_step(card_id: str, seed: int, *, max_cases_per_card: int = 32) -> dict[str, Any]:
    card_safe = safe_name(card_id)
    work_steps = code_lm_work_steps(max_cases_per_card)
    return step_command(
        f"code_lm_closure_{card_id}_seed{int(seed)}",
        [
            sys.executable,
            "scripts/code_lm_closure.py",
            "--public-cards",
            card_id,
            "--seed",
            str(int(seed)),
            "--private-count",
            "300",
            "--max-extra-private-train",
            "120",
            "--max-residual-private-train",
            "160",
            "--max-public-cases-per-card",
            str(max(1, int(max_cases_per_card))),
            "--hv-dim",
            "128",
            "--max-vocab",
            "128",
            "--epochs",
            "1",
            "--candidates-per-task",
            "2",
            "--max-rust-work-steps",
            str(work_steps),
            "--rust-timeout-seconds",
            "0",
            "--public-timeout-seconds",
            "0",
            "--sts-timeout-seconds",
            "0",
            "--public-task-manifest-out",
            f"reports/code_lm_public_tasks_{card_safe}_seed{int(seed)}.jsonl",
            "--public-candidate-out",
            student_code_candidate_manifest_path(card_id, seed),
            "--checkpoint-out",
            f"reports/student_code_lm_checkpoint_{card_safe}_seed{int(seed)}.json",
            "--private-candidate-out",
            f"reports/code_lm_private_candidates_{card_safe}_seed{int(seed)}.jsonl",
            "--rust-report-out",
            f"reports/code_lm_closure_rust_{card_safe}_seed{int(seed)}.json",
            "--sts-conditioning-input-out",
            f"reports/code_lm_sts_conditioning_input_{card_safe}_seed{int(seed)}.jsonl",
            "--sts-generation-out",
            f"reports/code_lm_sts_public_generations_{card_safe}_seed{int(seed)}.jsonl",
            "--sts-conditioning-checkpoint-out",
            f"reports/code_lm_sts_conditioning_checkpoint_{card_safe}_seed{int(seed)}.json",
            "--sts-conditioning-report-out",
            f"reports/code_lm_sts_conditioning_report_{card_safe}_seed{int(seed)}.json",
            "--skip-public-calibration",
            "--out",
            f"reports/code_lm_closure_{card_safe}_seed{int(seed)}.json",
        ],
        timeout_seconds=code_lm_step_timeout_seconds(max_cases_per_card),
    )


def student_code_candidate_manifest_path(card_id: str, seed: int) -> str:
    return f"reports/student_code_candidates_{safe_name(card_id)}_seed{int(seed)}.jsonl"


def real_code_benchmark_graduation_step(
    card_id: str,
    seed: int,
    *,
    max_cases_per_card: int = 8,
    student_candidate_manifest: str = "reports/student_code_candidates.jsonl",
    skip_student_candidate_generation: bool = False,
    out: str = "reports/real_code_benchmark_graduation.json",
    trace_out: str = "reports/real_code_benchmark_traces.jsonl",
    transfer_artifact_out: str = "reports/transfer_artifacts/code/real_code_benchmark_graduation_transfer_artifact.json",
) -> dict[str, Any]:
    name = f"real_code_benchmark_graduation_{card_id}_seed{int(seed)}"
    command = [
        sys.executable,
        "scripts/real_code_benchmark_graduation.py",
        "--cards",
        card_id,
        "--seed",
        str(int(seed)),
        "--max-cases-per-card",
        str(max(1, int(max_cases_per_card))),
        "--student-candidate-manifest",
        student_candidate_manifest,
        "--out",
        out,
        "--trace-out",
        trace_out,
        "--transfer-artifact-out",
        transfer_artifact_out,
    ]
    if skip_student_candidate_generation:
        name = f"real_code_benchmark_graduation_learned_{card_id}_seed{int(seed)}"
        command.append("--skip-student-candidate-generation")
    return step_command(
        name,
        command,
        timeout_seconds=max(3_600, min(10_800, int(max_cases_per_card) * 180)),
    )


def sts_repair_ablation_step() -> dict[str, Any]:
    return step_command(
        "sts_repair_ablation",
        [
            sys.executable,
            "scripts/sts_repair_ablation.py",
            "--real-code-report",
            "reports/real_code_benchmark_graduation.json",
            "--trace-in",
            "reports/real_code_benchmark_traces.jsonl",
            "--out",
            "reports/sts_repair_ablation.json",
            "--markdown-out",
            "reports/sts_repair_ablation.md",
        ],
        timeout_seconds=120,
    )


def student_learning_closure_step(
    card_id: str,
    seed: int,
    *,
    candidate_manifest: str,
    trace_in: str,
) -> dict[str, Any]:
    command = [
        *symliquid_cli_command("train-code-ranker"),
    ]
    if sys.platform.startswith(("win", "linux")):
        command.append("--use-cuda-readout")
    command.extend(
        [
            "--candidate-manifest",
            candidate_manifest,
            "--trace-in",
            trace_in,
            "--seed",
            str(int(seed)),
            "--model-out",
            "reports/student_neural_code_checkpoint.json",
            "--candidate-out",
            "reports/student_learning_code_candidates.jsonl",
            "--training-examples-out",
            "reports/student_learning_training_examples.jsonl",
            "--transfer-artifact-out",
            "reports/transfer_artifacts/code/student_neural_learning_closure_transfer_artifact.json",
            "--code-transfer-artifacts",
            "reports/code_transfer_artifacts.json",
            "--out",
            "reports/student_learning_closure.json",
        ]
    )
    return step_command(
        f"student_learning_closure_{card_id}_seed{int(seed)}",
        command,
    )


def default_pressure_card(frontier_family: str) -> str:
    defaults = {
        "drone_rl": "source_gym_pybullet_drones",
        "minecraft_rl": "source_crafter",
        "coding_local_sandbox": "source_bigcodebench",
        "web_agent_local": "source_webarena",
        "transfer_eval": "transfer_eval_suite",
    }
    return defaults.get(frontier_family, "source_gym_pybullet_drones")


def pressure_step_timeout_seconds(
    *,
    steps: int,
    train_iterations: int,
    train_population: int,
    eval_seed_count: int,
) -> int:
    """Give active pressure enough wall time without letting it run forever.

    Pressure runners can legitimately exceed the generic 30 minute step timeout
    when resource-aware scaling expands the train-before-eval budget. The
    estimate is intentionally conservative and still capped so the daemon can
    recover, report partial evidence, and rotate if a runner stalls.
    """
    candidate_evals = max(1, int(train_iterations)) * max(4, int(train_population))
    train_steps = candidate_evals * max(1, int(steps))
    eval_steps = max(1, int(eval_seed_count)) * max(1, int(steps))
    estimated = int((train_steps + eval_steps) * 0.006) + 300
    return max(1800, min(5400, estimated))


def step_command(
    name: str,
    command: list[str],
    *,
    allow_failure: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    row = {"name": name, "command": command, "allow_failure": allow_failure}
    if timeout_seconds is not None:
        row["timeout_seconds"] = int(timeout_seconds)
    return row


def run_step(step: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    step_timeout = int(step.get("timeout_seconds") or timeout)
    touch_profile_heartbeat(step, "started", started, step_timeout)
    try:
        process = subprocess.Popen(
            step["command"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = ""
        stderr = ""
        while True:
            elapsed = time.perf_counter() - started
            remaining = step_timeout - elapsed
            if remaining <= 0:
                process.kill()
                stdout, stderr = process.communicate()
                row = {
                    "name": step["name"],
                    "command": step["command"],
                    "allow_failure": step.get("allow_failure", False),
                    "returncode": 124,
                    "runtime_ms": int((time.perf_counter() - started) * 1000),
                    "timeout_seconds": step_timeout,
                    "stdout_tail": timeout_tail(stdout),
                    "stderr_tail": timeout_tail(stderr),
                    "error": "timeout",
                    "timed_out": True,
                    "recovery_hint": "rerun the same profile after reducing pressure budget or allowing a resumed pressure chunk",
                }
                touch_profile_heartbeat(step, "timeout", started, step_timeout, row=row)
                return row
            try:
                stdout, stderr = process.communicate(timeout=min(STEP_HEARTBEAT_SECONDS, remaining))
                break
            except subprocess.TimeoutExpired:
                touch_profile_heartbeat(step, "running", started, step_timeout)
                continue
        result_returncode = int(process.returncode or 0)
        row = {
            "name": step["name"],
            "command": step["command"],
            "allow_failure": step.get("allow_failure", False),
            "returncode": result_returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "timeout_seconds": step_timeout,
            "stdout_tail": timeout_tail(stdout),
            "stderr_tail": timeout_tail(stderr),
        }
        touch_profile_heartbeat(step, "completed" if result_returncode == 0 else "failed", started, step_timeout, row=row)
        return row
    except OSError as exc:
        return {
            "name": step["name"],
            "command": step["command"],
            "allow_failure": step.get("allow_failure", False),
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "timeout_seconds": step_timeout,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "error": "spawn_failed",
        }


def timeout_tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-4000:]
    return str(value)[-4000:]


def touch_profile_heartbeat(
    step: dict[str, Any],
    status: str,
    started: float,
    timeout_seconds: int,
    *,
    row: dict[str, Any] | None = None,
) -> None:
    runtime_ms = int((time.perf_counter() - started) * 1000)
    payload = {
        "policy": "project_theseus_training_profile_heartbeat_v0",
        "updated_utc": now(),
        "step": step.get("name"),
        "status": status,
        "runtime_ms": runtime_ms,
        "timeout_seconds": timeout_seconds,
        "returncode": row.get("returncode") if isinstance(row, dict) else None,
    }
    write_json(PROFILE_HEARTBEAT, payload)
    try:
        status_payload = read_json(SPARKSTREAM_STATUS)
    except (OSError, json.JSONDecodeError):
        status_payload = {}
    if isinstance(status_payload, dict):
        status_payload["updated_utc"] = payload["updated_utc"]
        status_payload["phase"] = "running_profile_step"
        status_payload["message"] = (
            f"{step.get('name')} {status}; "
            f"{runtime_ms // 1000}s elapsed of {timeout_seconds}s step budget"
        )
        status_payload["profile_step"] = {
            "name": step.get("name"),
            "status": status,
            "runtime_ms": runtime_ms,
            "timeout_seconds": timeout_seconds,
        }
        write_json(SPARKSTREAM_STATUS, status_payload)
    print(
        f"[profile] {step.get('name')} {status}; "
        f"{runtime_ms // 1000}s/{timeout_seconds}s",
        flush=True,
    )


def first_failure(steps: list[dict[str, Any]]) -> dict[str, Any]:
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("returncode", 0) != 0 and not step.get("allow_failure"):
            return {
                "name": step.get("name"),
                "returncode": step.get("returncode"),
                "error": step.get("error"),
                "timed_out": bool(step.get("timed_out")),
                "runtime_ms": step.get("runtime_ms"),
                "timeout_seconds": step.get("timeout_seconds"),
            }
    return {}


def recovery_plan(args: argparse.Namespace, steps: list[dict[str, Any]], artifacts: dict[str, str]) -> dict[str, Any]:
    failure = first_failure(steps)
    if not failure:
        return {"needed": False}
    completed_steps = [
        str(step.get("name"))
        for step in steps
        if isinstance(step, dict) and step.get("returncode") == 0
    ]
    plan = {
        "needed": True,
        "reason": "profile_step_failed",
        "failed_step": failure,
        "completed_steps": completed_steps,
        "artifacts": artifacts,
        "recommended_next_command": [
            sys.executable,
            "scripts/run_training_ratchet_profile.py",
            "--profile",
            args.profile,
            "--frontier-family",
            args.frontier_family,
            "--rl-frontier-seed",
            str(args.rl_frontier_seed),
            "--pressure-card-id",
            args.pressure_card_id,
            "--budget-mode",
            "fixed",
            "--resume",
            "--out",
            args.out,
        ],
    }
    if failure.get("timed_out"):
        plan["reason"] = "profile_step_timed_out"
        plan["smallest_safe_correction"] = (
            "keep completed artifacts, rerun the profile with fixed budget, and do not promote until "
            "candidate_profile_evidence_complete and pressure budget checks pass"
        )
    return plan


def load_resumed_steps(args: argparse.Namespace, artifacts: dict[str, str]) -> list[dict[str, Any]]:
    report_path = ROOT / args.out
    previous = read_json(report_path)
    if previous.get("profile") != args.profile:
        return []
    if previous.get("frontier_family") != args.frontier_family:
        return []
    previous_artifacts = previous.get("artifacts") if isinstance(previous.get("artifacts"), dict) else {}
    for key in ("pressure_runner", "mutated_frontier", "rl_frontier_train", "rl_frontier_smoke"):
        expected = str(artifacts.get(key) or "")
        actual = str(previous_artifacts.get(key) or "")
        if expected and actual and expected != actual:
            return []
    rows: list[dict[str, Any]] = []
    for step in previous.get("steps", []):
        if not isinstance(step, dict) or step.get("returncode") != 0:
            continue
        row = dict(step)
        row["resumed_from_previous_report"] = True
        rows.append(row)
    return rows


def append_workflow_trace(
    path: Path, *, task: str, command: str, returncode: int, runtime_ms: int
) -> None:
    payload = {
        "trace_id": f"profile_{int(time.time() * 1000)}_{abs(hash(command)) % 1000000}",
        "task": task,
        "workflow": "one-command training ratchet profile",
        "command": command,
        "selected_arms": ["head_router", "benchmark_ratchet_arm", "rust_cuda_systems_arm"],
        "expected_arms": ["head_router", "benchmark_ratchet_arm", "rust_cuda_systems_arm"],
        "risk": "medium" if "candidate" in task else "low",
        "routing_pattern": "sequential",
        "returncode": returncode,
        "success": returncode == 0,
        "runtime_ms": runtime_ms,
        "review_step_count": 3,
        "review_step_basis": "head_router_benchmark_ratchet_rust_cuda_arms",
        "maintenance_mode": maintenance_mode_from_text(command, task),
        "maintenance_mode_basis": "command_or_task_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "split": "train",
        "source": "training_ratchet_profile_runner",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maintenance_mode_from_text(*values: str) -> str:
    text = " ".join(str(value or "") for value in values).lower().replace("-", "_").replace(" ", "_")
    if "circle" in text and "seed" in text and "rebuild" in text:
        return "circle_seed_rule_rebuild"
    return "object_only"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
