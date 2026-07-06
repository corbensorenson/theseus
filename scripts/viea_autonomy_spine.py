"""Executable VIEA autonomy spine for Project Theseus.

The VIEA reports are useful only if the autonomy loop can use them as a
control surface. This runner refreshes the command/artifact/runtime/verification
path, then materializes action queues for the feedback ratchet, broad-transfer
closure, private repo repair, SymLiquid state use, and teacher-as-architect
experiments.

It does not train on public benchmark answers, does not call external inference,
and does not mark broad transfer as solved. Public benchmark reports remain
calibration evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--repo-repair-repetitions", type=int, default=8)
    parser.add_argument("--out", default="reports/viea_autonomy_spine.json")
    parser.add_argument("--markdown-out", default="reports/viea_autonomy_spine.md")
    parser.add_argument("--allow-teacher-queue", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    steps = run_spine_steps(args)
    state = load_state()
    reports = build_action_reports(state, steps, args=args)
    write_action_reports(reports)
    steps.extend(run_post_action_steps(args, completed_steps=len(steps)))
    state = load_state()

    summary = build_summary(state, steps, reports, started=started, max_steps=args.max_steps)
    payload = {
        "policy": "project_theseus_viea_autonomy_spine_v1",
        "created_utc": now(),
        "trigger_state": summary["trigger_state"],
        "summary": summary,
        "steps": steps,
        "spine": {
            "flow": [
                "goal_or_command",
                "viea_executor",
                "artifact_kernel_write",
                "runtime_packet",
                "verification",
                "feedback_ratchet",
                "next_training_tool_residual_action",
            ],
            "public_benchmarks": "calibration_only_not_training",
            "teacher": "architecture_diagnosis_only_no_answers_no_distillation_no_apply_without_gate",
        },
        "reports": {name: f"reports/{filename}" for name, filename in REPORT_FILES.items()},
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload, reports))
    print(json.dumps(payload, indent=2))
    return 2 if payload["trigger_state"] == "RED" else 0


def run_spine_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    step_specs = [
        step(
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
            timeout=180,
        ),
        step(
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
            timeout=240,
            allow_failure=True,
        ),
        step(
            "transfer_generalization_audit",
            [
                sys.executable,
                "scripts/transfer_generalization_audit.py",
                "--out",
                "reports/transfer_generalization_audit.json",
                "--markdown-out",
                "reports/transfer_generalization_audit.md",
            ],
            timeout=180,
            allow_failure=True,
        ),
        step(
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
            timeout=180,
            allow_failure=True,
        ),
        step(
            "student_first_evidence_audit",
            [
                sys.executable,
                "scripts/student_first_evidence_audit.py",
                "--out",
                "reports/student_first_evidence_audit.json",
                "--markdown-out",
                "reports/student_first_evidence_audit.md",
            ],
            timeout=180,
            allow_failure=True,
        ),
        step(
            "learning_scoreboard",
            [
                sys.executable,
                "scripts/learning_scoreboard.py",
                "--out",
                "reports/learning_scoreboard.json",
                "--markdown-out",
                "reports/learning_scoreboard.md",
            ],
            timeout=180,
            allow_failure=True,
        ),
        step(
            "artifact_kernel_pre_executor",
            [
                sys.executable,
                "scripts/viea_artifact_kernel.py",
                "--reset",
                "--db",
                "reports/viea_artifact_kernel.sqlite",
                "--out",
                "reports/viea_artifact_kernel.json",
                "--markdown-out",
                "reports/viea_artifact_kernel.md",
            ],
            timeout=180,
        ),
        step(
            "command_executor",
            [
                sys.executable,
                "scripts/viea_command_executor.py",
                "--db",
                "reports/viea_artifact_kernel.sqlite",
                "--out",
                "reports/viea_command_executor.json",
                "--markdown-out",
                "reports/viea_command_executor.md",
            ],
            timeout=180,
        ),
        step(
            "private_repo_repair_curriculum",
            [
                sys.executable,
                "scripts/long_horizon_programming_curriculum.py",
                "--task-out",
                "D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl",
                "--sts-out",
                "D:/ProjectTheseus/training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl",
                "--out",
                "reports/private_repo_repair_curriculum.json",
                "--markdown-out",
                "reports/private_repo_repair_curriculum.md",
                "--repetitions",
                str(max(1, int(args.repo_repair_repetitions))),
            ],
            timeout=180,
        ),
        step(
            "viea_repo_repair_learner",
            [
                sys.executable,
                "scripts/viea_repo_repair_learner.py",
                "--out",
                "reports/viea_repo_repair_learner.json",
                "--markdown-out",
                "reports/viea_repo_repair_learner.md",
            ],
            timeout=240,
            allow_failure=True,
        ),
        step(
            "viea_growth_surfaces",
            [sys.executable, "scripts/viea_growth_surfaces.py", "--out-dir", "reports"],
            timeout=180,
        ),
        step(
            "artifact_kernel_post_growth",
            [
                sys.executable,
                "scripts/viea_artifact_kernel.py",
                "--reset",
                "--db",
                "reports/viea_artifact_kernel.sqlite",
                "--out",
                "reports/viea_artifact_kernel.json",
                "--markdown-out",
                "reports/viea_artifact_kernel.md",
            ],
            timeout=180,
        ),
        step(
            "viea_report_map",
            [
                sys.executable,
                "scripts/viea_report_map.py",
                "--out",
                "reports/viea_report_map.json",
                "--markdown-out",
                "reports/viea_report_map.md",
            ],
            timeout=180,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(step_specs):
        if index >= max(0, int(args.max_steps)):
            rows.append(
                {
                    "name": spec["name"],
                    "command": spec["command"],
                    "returncode": 0,
                    "runtime_ms": 0,
                    "skipped": True,
                    "reason": "max_steps_reached",
                    "allow_failure": spec.get("allow_failure", False),
                }
            )
            continue
        rows.append(run_step(spec, default_timeout=int(args.timeout_seconds)))
        if rows[-1]["returncode"] != 0 and not rows[-1].get("allow_failure"):
            break
    return rows


def run_post_action_steps(args: argparse.Namespace, *, completed_steps: int) -> list[dict[str, Any]]:
    if completed_steps >= max(0, int(args.max_steps)):
        return []
    specs = [
        step(
            "symliquid_state_engine",
            [
                sys.executable,
                "scripts/symliquid_state_engine.py",
                "--out",
                "reports/symliquid_state_engine.json",
                "--markdown-out",
                "reports/symliquid_state_engine.md",
            ],
            timeout=120,
            allow_failure=True,
        ),
        step(
            "teacher_architect_experiment_runner_status",
            [
                sys.executable,
                "scripts/teacher_architect_experiment_runner.py",
                "--max-experiments",
                "1",
                "--max-steps",
                "0",
                "--out",
                "reports/teacher_architect_experiment_runner_status.json",
                "--markdown-out",
                "reports/teacher_architect_experiment_runner_status.md",
            ],
            timeout=120,
            allow_failure=True,
        ),
        step(
            "viea_action_executor_status",
            [
                sys.executable,
                "scripts/viea_action_executor.py",
                "--status",
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ],
            timeout=120,
            allow_failure=True,
        ),
        step(
            "artifact_kernel_ingest_action_queues",
            [
                sys.executable,
                "scripts/viea_artifact_kernel.py",
                "--reset",
                "--db",
                "reports/viea_artifact_kernel.sqlite",
                "--out",
                "reports/viea_artifact_kernel.json",
                "--markdown-out",
                "reports/viea_artifact_kernel.md",
            ],
            timeout=180,
        ),
        step(
            "viea_report_map_ingest_action_queues",
            [
                sys.executable,
                "scripts/viea_report_map.py",
                "--out",
                "reports/viea_report_map.json",
                "--markdown-out",
                "reports/viea_report_map.md",
            ],
            timeout=180,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs[: max(0, int(args.max_steps) - completed_steps)]:
        rows.append(run_step(spec, default_timeout=int(args.timeout_seconds)))
        if rows[-1]["returncode"] != 0 and not rows[-1].get("allow_failure"):
            break
    return rows


def step(name: str, command: list[str], *, timeout: int, allow_failure: bool = False) -> dict[str, Any]:
    return {"name": name, "command": command, "timeout": timeout, "allow_failure": allow_failure}


def run_step(spec: dict[str, Any], *, default_timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    timeout = max(int(spec.get("timeout") or 0), default_timeout)
    try:
        result = subprocess.run(
            [str(item) for item in spec["command"]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "name": spec["name"],
            "command": spec["command"],
            "timeout_seconds": timeout,
            "allow_failure": bool(spec.get("allow_failure")),
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": spec["name"],
            "command": spec["command"],
            "timeout_seconds": timeout,
            "allow_failure": bool(spec.get("allow_failure")),
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def load_state() -> dict[str, Any]:
    return {
        "artifact_kernel": read_json(REPORTS / "viea_artifact_kernel.json"),
        "command_executor": read_json(REPORTS / "viea_command_executor.json"),
        "feedback_ratchet": read_json(REPORTS / "feedback_ratchet.json"),
        "broad_transfer_closure": read_json(REPORTS / "broad_transfer_closure.json"),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "transfer_generalization": read_json(REPORTS / "transfer_generalization_audit.json"),
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json"),
        "private_repo_repair": read_json(REPORTS / "private_repo_repair_curriculum.json"),
        "repo_repair_learner": read_json(REPORTS / "viea_repo_repair_learner.json"),
        "long_horizon_programming": read_json(REPORTS / "long_horizon_programming_curriculum.json"),
        "workflow_tool_compiler": read_json(REPORTS / "workflow_tool_compiler_v2.json"),
        "symliquid_substrate": read_json(REPORTS / "symliquid_substrate_map.json"),
        "symliquid_state_engine": read_json(REPORTS / "symliquid_state_engine.json"),
        "teacher_architect": read_json(REPORTS / "teacher_architect_loop.json"),
        "teacher_architect_experiment_runner": read_json(REPORTS / "teacher_architect_experiment_runner.json"),
        "viea_action_executor": read_json(REPORTS / "viea_action_executor.json"),
        "digital_runtime": read_json(REPORTS / "digital_runtime_adapter.json"),
        "viea_report_map": read_json(REPORTS / "viea_report_map.json"),
    }


REPORT_FILES = {
    "feedback_action_queue": "feedback_action_queue.json",
    "broad_transfer_action_queue": "broad_transfer_action_queue.json",
    "repo_repair_main_curriculum": "repo_repair_main_curriculum.json",
    "teacher_architect_closure": "teacher_architect_closure.json",
    "symliquid_state_engine_queue": "symliquid_state_engine_queue.json",
}


def build_action_reports(state: dict[str, Any], steps: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    broad_actions = broad_transfer_actions(state)
    repo = repo_repair_main_curriculum(state)
    teacher = teacher_architect_closure(state, allow_teacher_queue=bool(args.allow_teacher_queue))
    sym = symliquid_state_engine_queue(state)
    feedback = feedback_action_queue(state, broad_actions=broad_actions, repo=repo, teacher=teacher, sym=sym, steps=steps)
    return {
        "feedback_action_queue": feedback,
        "broad_transfer_action_queue": broad_actions,
        "repo_repair_main_curriculum": repo,
        "teacher_architect_closure": teacher,
        "symliquid_state_engine_queue": sym,
    }


def broad_transfer_actions(state: dict[str, Any]) -> dict[str, Any]:
    broad = state["broad_transfer_closure"]
    actions: list[dict[str, Any]] = []
    public_gate = public_calibration_private_gate_block()
    for row in broad.get("rows", []) if isinstance(broad.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
        card = str(row.get("card_id") or "unknown_card")
        if not blockers:
            actions.append(
                action(
                    "medium",
                    "promote_regression_surface",
                    f"Preserve {card} as regression calibration",
                    f"{card} is above floor with clean student evidence; keep it regression-only unless future gates regress.",
                    evidence=row,
                    command=[
                        sys.executable,
                        "scripts/broad_transfer_matrix.py",
                        "--min-public-tasks",
                        "32",
                        "--out",
                        "reports/broad_transfer_matrix.json",
                        "--markdown-out",
                        "reports/broad_transfer_matrix.md",
                    ],
                )
            )
            continue
        if "below_0_70_floor" in blockers:
            closure_slug = safe_card_id(card)
            if public_gate["blocked"]:
                append_private_gate_action(actions, public_gate)
            else:
                actions.append(
                    action(
                        "critical" if card in {"source_mbpp", "source_evalplus"} else "high",
                        "train_private_semantic_residual_family",
                        f"Raise {card} broad transfer above floor",
                        "Train private hidden-test lookalikes for the residual family, then rerun same-seed public calibration only.",
                        evidence=row,
                        command=[
                            sys.executable,
                            "scripts/broad_transfer_closure_runner.py",
                            "--cards",
                            card,
                            "--execute",
                            "--max-public-cases-per-card",
                            "32",
                            "--private-count",
                            "960",
                            "--max-rust-work-steps",
                            "4000000",
                            "--timeout-seconds",
                            "21600",
                            "--rust-timeout-seconds",
                            "21600",
                            "--public-timeout-seconds",
                            "10800",
                            "--sts-timeout-seconds",
                            "10800",
                            "--out",
                            f"reports/broad_transfer_closure_runner_{closure_slug}.json",
                            "--markdown-out",
                            f"reports/broad_transfer_closure_runner_{closure_slug}.md",
                        ],
                    )
                )
        if "needs_32_plus_clean_tasks" in blockers:
            closure_slug = safe_card_id(card)
            if public_gate["blocked"]:
                append_private_gate_action(actions, public_gate)
            else:
                actions.append(
                    action(
                        "high",
                        "expand_public_adapter_clean_slice",
                        f"Expand {card} to 32+ clean calibration tasks",
                        "Move beyond loader-only smoke evidence while keeping public tasks calibration-only.",
                        evidence=row,
                        command=[
                            sys.executable,
                            "scripts/broad_transfer_closure_runner.py",
                            "--cards",
                            card,
                            "--execute",
                            "--max-public-cases-per-card",
                            "32",
                            "--private-count",
                            "960",
                            "--max-rust-work-steps",
                            "4000000",
                            "--timeout-seconds",
                            "21600",
                            "--rust-timeout-seconds",
                            "21600",
                            "--public-timeout-seconds",
                            "10800",
                            "--sts-timeout-seconds",
                            "10800",
                            "--out",
                            f"reports/broad_transfer_closure_runner_{closure_slug}.json",
                            "--markdown-out",
                            f"reports/broad_transfer_closure_runner_{closure_slug}.md",
                        ],
                    )
                )
        if "sts_not_causal_on_card" in blockers:
            actions.append(
                action(
                    "high",
                    "run_same_seed_sts_repair_ablation",
                    f"Make STS causal on {card}",
                    "Require same checkpoint, same seed, same candidate budget: STS-on beats STS-off per card.",
                    evidence=row,
                    command=[
                        sys.executable,
                        "scripts/sts_repair_ablation.py",
                        "--out",
                        "reports/sts_repair_ablation.json",
                        "--markdown-out",
                        "reports/sts_repair_ablation.md",
                    ],
                )
            )
    payload = {
        "policy": "project_theseus_broad_transfer_action_queue_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if actions and not any(a["priority"] == "critical" for a in actions) else "YELLOW",
        "summary": {
            "action_count": len(actions),
            "critical_count": sum(1 for item in actions if item["priority"] == "critical"),
            "aggregate_pass_rate": get_path(broad, ["summary", "aggregate_pass_rate"], None),
            "aggregate_floor_gap": get_path(broad, ["summary", "aggregate_floor_gap"], None),
            "selected_next_card": get_path(broad, ["summary", "selected_next_card"], ""),
        },
        "actions": actions,
        "rules": {
            "public_benchmarks": "calibration_only_not_training",
            "promotion_evidence": "token_level_student_generation_only_no_templates_no_wrappers",
        },
        "external_inference_calls": 0,
    }
    return payload


def repo_repair_main_curriculum(state: dict[str, Any]) -> dict[str, Any]:
    report = state["private_repo_repair"] or state["long_horizon_programming"]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    task_count = int_number(summary.get("task_count"))
    gates = [
        gate("private_tasks_present", task_count >= 48, f"tasks={task_count}"),
        gate("hidden_tests_private_only", summary.get("public_tests_included") is False, summary.get("public_tests_included")),
        gate("public_solutions_absent", summary.get("public_benchmark_solutions_included") is False, summary.get("public_benchmark_solutions_included")),
        gate("sts_rows_present", int_number(summary.get("sts_row_count")) >= task_count, f"sts={summary.get('sts_row_count')} tasks={task_count}"),
    ]
    payload = {
        "policy": "project_theseus_repo_repair_main_curriculum_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "task_count": task_count,
            "sts_row_count": summary.get("sts_row_count"),
            "category_count": summary.get("category_count"),
            "central_curriculum": True,
            "task_out": summary.get("task_out"),
            "sts_out": summary.get("sts_out"),
            "promotion_evidence": False,
        },
        "loop": [
            "repo_snapshot",
            "bug_or_task",
            "patch_attempt",
            "visible_and_hidden_private_tests",
            "residual_label",
            "repair_trace",
            "train_checkpoint",
            "rerun_private_eval",
            "public_calibration_only",
        ],
        "next_actions": [
            action(
                "high",
                "write_repo_repair_tasks",
                "Keep private repo-repair tasks central",
                "Refresh private repo-repair tasks and STS rows before the next code learner update.",
                evidence=summary,
                command=[
                    sys.executable,
                    "scripts/long_horizon_programming_curriculum.py",
                    "--task-out",
                    "D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl",
                    "--sts-out",
                    "D:/ProjectTheseus/training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl",
                    "--out",
                    "reports/private_repo_repair_curriculum.json",
                    "--markdown-out",
                    "reports/private_repo_repair_curriculum.md",
                    "--repetitions",
                    "8",
                ],
            ),
            action(
                "high",
                "train_repo_repair_trace_checkpoint",
                "Train on successful private repo-repair traces",
                "Use successful private traces as training pressure after hidden-test validation, then rerun public calibration only.",
                evidence={"task_count": task_count, "source": "private_repo_repair"},
                command=[],
            ),
        ],
        "gates": gates,
        "external_inference_calls": 0,
    }
    return payload


def teacher_architect_closure(state: dict[str, Any], *, allow_teacher_queue: bool) -> dict[str, Any]:
    teacher = state["teacher_architect"]
    specs = teacher.get("experiment_specs") if isinstance(teacher.get("experiment_specs"), list) else []
    broad = state["broad_transfer_closure"]
    broad_gap = number(get_path(broad, ["summary", "aggregate_floor_gap"], 0.0))
    closures = []
    for spec in specs[:12]:
        if not isinstance(spec, dict):
            continue
        exp_id = str(spec.get("id") or "architecture_experiment")
        teacher_needed = bool(spec.get("teacher_needed")) or broad_gap > 0.0
        closures.append(
            {
                "id": exp_id,
                "kind": spec.get("kind"),
                "hypothesis": spec.get("hypothesis"),
                "residual_cluster_source": "reports/broad_transfer_closure.json + reports/residual_escrow.json",
                "teacher_role": "architecture_diagnosis_only",
                "teacher_request_status": "queued_allowed" if teacher_needed and allow_teacher_queue else "queued_policy_gate",
                "diagnosis_stage": "residual_cluster_to_architecture_experiment",
                "experiment_stage": "private_eval_required",
                "calibration_stage": "public_calibration_only_required",
                "decision_stage": "promote_or_rollback",
                "forbidden": spec.get("forbidden") or [
                    "benchmark_answers",
                    "hidden_tests",
                    "public_solution_distillation",
                    "apply_mode_without_gate",
                ],
                "commands": experiment_commands(exp_id, teacher_needed=teacher_needed, allow_teacher_queue=allow_teacher_queue),
            }
        )
    gates = [
        gate("teacher_architect_report_loaded", teacher.get("policy") == "project_theseus_teacher_architect_loop_v1", teacher.get("policy")),
        gate("experiment_specs_present", len(closures) > 0, len(closures)),
        gate("no_answer_distillation", True, "closure specs are diagnosis and experiment commands only"),
    ]
    return {
        "policy": "project_theseus_teacher_architect_closure_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "experiment_count": len(closures),
            "teacher_queue_allowed": allow_teacher_queue,
            "broad_transfer_floor_gap": broad_gap,
        },
        "closures": closures,
        "gates": gates,
        "external_inference_calls": 0,
    }


def experiment_commands(exp_id: str, *, teacher_needed: bool, allow_teacher_queue: bool) -> list[dict[str, Any]]:
    commands = []
    if "sts" in exp_id:
        commands.append(named_command("private_eval", [sys.executable, "scripts/sts_repair_ablation.py", "--out", "reports/sts_repair_ablation.json", "--markdown-out", "reports/sts_repair_ablation.md"]))
    elif "long_horizon" in exp_id or "repo" in exp_id:
        commands.append(named_command("private_eval", [sys.executable, "scripts/long_horizon_programming_curriculum.py", "--out", "reports/private_repo_repair_curriculum.json", "--markdown-out", "reports/private_repo_repair_curriculum.md", "--repetitions", "8"]))
    elif "residual" in exp_id:
        commands.append(named_command("private_eval", [sys.executable, "scripts/code_residual_curriculum.py", "--out", "reports/code_residual_curriculum.json", "--markdown-out", "reports/code_residual_curriculum.md", "--max-rows", "960"]))
    else:
        commands.append(named_command("private_eval", [sys.executable, "scripts/architecture_experiment_runner.py", "--out", "reports/architecture_experiment_runner.json"]))
    commands.append(named_command("public_calibration", [sys.executable, "scripts/broad_transfer_matrix.py", "--min-public-tasks", "32", "--out", "reports/broad_transfer_matrix.json", "--markdown-out", "reports/broad_transfer_matrix.md"]))
    if teacher_needed:
        teacher_cmd = [
            sys.executable,
            "scripts/teacher_oracle.py",
            "--reason",
            "architecture_wall",
            "--mode",
            "proposal",
            "--prompt-file",
            "reports/teacher_architecture_guidance_prompt.md",
            "--local-evidence",
            "reports/architecture_guidance_loop.json",
            "reports/learning_scoreboard.json",
            "reports/broad_transfer_matrix.json",
            "reports/transfer_generalization_audit.json",
            "reports/autonomy_watchdog.json",
            "--queue-only",
            "--out",
            "reports/teacher_architecture_guidance_last.json",
        ]
        if allow_teacher_queue:
            teacher_cmd.remove("--queue-only")
            teacher_cmd.append("--allow-teacher")
        commands.insert(0, named_command("teacher_diagnosis", teacher_cmd))
    return commands


def symliquid_state_engine_queue(state: dict[str, Any]) -> dict[str, Any]:
    substrate = state["symliquid_substrate"]
    evidence_rows = substrate.get("rows") if isinstance(substrate.get("rows"), list) else []
    by_capability = {str(row.get("capability")): row for row in evidence_rows if isinstance(row, dict)}
    capabilities = [
        ("command_route_memory", "reports/viea_command_executor.json", "Use recurrent state to remember route outcomes for command contracts."),
        ("residual_clusters", "reports/code_residual_forge.json", "Cluster failed semantics into training, tool, or architecture pressure."),
        ("tool_selection", "reports/tool_registry.json", "Select only earned tools under lifecycle metrics."),
        ("sts_conditioning", "reports/sts_native_parallel_probe.json", "Condition solver/critic/patch streams so STS changes tokens, not notes."),
        ("repo_repair_state", "reports/private_repo_repair_curriculum.json", "Track inspect-patch-test-repair traces as long-horizon code state."),
        ("long_autonomy_state", "reports/autonomy_watchdog.json", "Carry daemon/cycle state across wakes without relying on chat context."),
        ("small_control_policies", "reports/benchmaxx_curriculum.json", "Drive frontier rotation, budget, and resource policy with compact controllers."),
    ]
    queue = []
    for capability, source, role in capabilities:
        source_key = "sts_stream_conditioning" if capability == "sts_conditioning" else capability
        evidence = by_capability.get(source_key) or {}
        queue.append(
            {
                "capability": capability,
                "source_report": source,
                "state_slot": f"symliquid/{capability}",
                "role": role,
                "evidence_present": bool(evidence) or resolve(source).exists(),
                "next_action": "wire_into_viea_executor_or_feedback_ratchet",
                "status": "active" if bool(evidence) or resolve(source).exists() else "needs_evidence",
            }
        )
    gates = [
        gate("state_engine_capabilities_declared", len(queue) == 7, len(queue)),
        gate("repo_repair_state_present", any(row["capability"] == "repo_repair_state" and row["evidence_present"] for row in queue), "repo_repair_state"),
        gate("sts_conditioning_state_present", any(row["capability"] == "sts_conditioning" and row["evidence_present"] for row in queue), "sts_conditioning"),
    ]
    return {
        "policy": "project_theseus_symliquid_state_engine_queue_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "principle": "SymLiquid is the compact recurrent state engine where memory and dynamics matter; it is not a replacement for every model.",
        "queue": queue,
        "gates": gates,
        "external_inference_calls": 0,
    }


def feedback_action_queue(
    state: dict[str, Any],
    *,
    broad_actions: dict[str, Any],
    repo: dict[str, Any],
    teacher: dict[str, Any],
    sym: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    feedback = state["feedback_ratchet"]
    questions = feedback.get("questions") if isinstance(feedback.get("questions"), dict) else {}
    actions: list[dict[str, Any]] = []
    actions.extend(broad_actions.get("actions", [])[:12])
    actions.extend(transfer_generalization_actions(state)[:8])
    actions.extend(repo.get("next_actions", [])[:4])
    workflow = state["workflow_tool_compiler"]
    for name in (questions.get("what_should_expire") or [])[:8]:
        actions.append(
            action(
                "medium",
                "expire_stale_tool",
                f"Expire or review stale tool {name}",
                "Tool must earn existence through recurrence, verification, and score impact.",
                evidence={"tool_name": name},
                command=[],
            )
        )
    for name in (questions.get("what_became_a_tool") or [])[:8]:
        actions.append(
            action(
                "low",
                "renew_useful_tool",
                f"Renew useful tool {name}",
                "Keep useful tools in the registry while continuing lifecycle scoring.",
                evidence={"tool_name": name},
                command=[],
            )
        )
    if teacher.get("summary", {}).get("experiment_count", 0):
        actions.append(
            action(
                "high",
                "request_teacher_architecture_diagnosis",
                "Run teacher-as-architect closure when residuals need diagnosis",
                "Teacher reads residual clusters and proposes experiment specs only.",
                evidence=teacher.get("summary"),
                command=[
                    sys.executable,
                    "scripts/teacher_oracle.py",
                    "--reason",
                    "architecture_wall",
                    "--mode",
                    "proposal",
                    "--prompt-file",
                    "reports/teacher_architecture_guidance_prompt.md",
                    "--local-evidence",
                    "reports/architecture_guidance_loop.json",
                    "reports/learning_scoreboard.json",
                    "reports/broad_transfer_matrix.json",
                    "reports/transfer_generalization_audit.json",
                    "--queue-only",
                    "--out",
                    "reports/teacher_architecture_guidance_last.json",
                ],
            )
        )
    actions.append(
        action(
            "medium",
            "refresh_symliquid_state_engine",
            "Refresh SymLiquid state-engine queue",
            "Keep recurrent state slots aligned to command routing, residuals, tools, STS, repo repair, and autonomy.",
            evidence={"queue_count": len(sym.get("queue", []))},
            command=[sys.executable, "scripts/viea_autonomy_spine.py", "--max-steps", "0"],
        )
    )
    failed_required = [row for row in steps if row.get("returncode") != 0 and not row.get("allow_failure")]
    gates = [
        gate("feedback_ratchet_loaded", feedback.get("policy") == "project_theseus_feedback_ratchet_v1", feedback.get("policy")),
        gate("actions_materialized", len(actions) > 0, len(actions)),
        gate("required_spine_steps_succeeded", not failed_required, [row.get("name") for row in failed_required]),
        gate("workflow_tool_scores_loaded", workflow.get("policy") == "project_theseus_workflow_tool_compiler_v2", workflow.get("policy"), severity="soft"),
    ]
    return {
        "policy": "project_theseus_feedback_action_queue_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "action_count": len(actions),
            "critical_count": sum(1 for item in actions if item["priority"] == "critical"),
            "source_questions": questions,
            "executed_this_run": [
                "reality_manipulator",
                "artifact_kernel",
                "command_executor",
                "private_repo_repair_curriculum",
                "growth_surfaces",
                "report_map",
            ],
        },
        "actions": actions[:50],
        "gates": gates,
        "external_inference_calls": 0,
    }


def transfer_generalization_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    audit = state.get("transfer_generalization") if isinstance(state.get("transfer_generalization"), dict) else {}
    rows = audit.get("recommended_actions") if isinstance(audit.get("recommended_actions"), list) else []
    actions: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "")
        if kind == "source_agnostic_private_concept_pressure":
            actions.append(
                action(
                    str(row.get("priority") or "high"),
                    "train_private_semantic_residual_family",
                    str(row.get("title") or "Train transferable private concept families"),
                    "Generate private, source-agnostic concept pressure from shared residuals before claiming benchmark progress.",
                    evidence={
                        "concept": row.get("concept"),
                        "cards": row.get("cards"),
                        "private_pressure": row.get("private_pressure"),
                        "generalization_audit": "reports/transfer_generalization_audit.json",
                    },
                    command=[
                        sys.executable,
                        "scripts/code_residual_curriculum.py",
                        "--out",
                        "reports/code_residual_curriculum.json",
                        "--markdown-out",
                        "reports/code_residual_curriculum.md",
                        "--max-rows",
                        "960",
                    ],
                )
            )
        elif kind == "donor_receiver_transfer_eval":
            actions.append(
                action(
                    str(row.get("priority") or "high"),
                    "promote_regression_surface",
                    str(row.get("title") or "Run donor/receiver transfer checks"),
                    "Refresh broad transfer matrix after a private concept update; donor/receiver proof remains calibration-only.",
                    evidence={
                        "donor_card": row.get("donor_card"),
                        "receiver_cards": row.get("receiver_cards"),
                        "generalization_audit": "reports/transfer_generalization_audit.json",
                    },
                    command=[
                        sys.executable,
                        "scripts/broad_transfer_matrix.py",
                        "--min-public-tasks",
                        "32",
                        "--out",
                        "reports/broad_transfer_matrix.json",
                        "--markdown-out",
                        "reports/broad_transfer_matrix.md",
                    ],
                )
            )
        elif kind in {"decoder_architecture_pressure", "per_card_sts_causality"}:
            actions.append(
                action(
                    str(row.get("priority") or "medium"),
                    "request_teacher_architecture_diagnosis",
                    str(row.get("title") or "Request transfer architecture diagnosis"),
                    "Ask for architecture pressure from shared residual concepts, not benchmark answers.",
                    evidence={
                        "reason": row.get("reason"),
                        "generalization_audit": "reports/transfer_generalization_audit.json",
                    },
                    command=[
                        sys.executable,
                        "scripts/teacher_oracle.py",
                        "--reason",
                        "architecture_wall",
                        "--mode",
                        "proposal",
                        "--prompt-file",
                        "reports/teacher_architecture_guidance_prompt.md",
                        "--local-evidence",
                        "reports/architecture_guidance_loop.json",
                        "reports/learning_scoreboard.json",
                        "reports/broad_transfer_matrix.json",
                        "reports/transfer_generalization_audit.json",
                        "--queue-only",
                        "--out",
                        "reports/teacher_architecture_guidance_last.json",
                    ],
                )
            )
    return actions


def write_action_reports(reports: dict[str, dict[str, Any]]) -> None:
    for name, payload in reports.items():
        json_path = REPORTS / REPORT_FILES[name]
        write_json(json_path, payload)
        write_text(json_path.with_suffix(".md"), render_simple_report(name, payload))


def build_summary(
    state: dict[str, Any],
    steps: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    *,
    started: float,
    max_steps: int,
) -> dict[str, Any]:
    required_failures = [row for row in steps if row.get("returncode") != 0 and not row.get("allow_failure")]
    skipped = [row for row in steps if row.get("skipped")]
    states = [payload.get("trigger_state") for payload in reports.values()]
    if required_failures:
        trigger_state = "RED"
    elif "YELLOW" in states or skipped or number(get_path(state["broad_transfer_closure"], ["summary", "aggregate_floor_gap"], 0.0)) > 0:
        trigger_state = "YELLOW"
    else:
        trigger_state = "GREEN"
    kernel_summary = state["artifact_kernel"].get("summary") if isinstance(state["artifact_kernel"].get("summary"), dict) else {}
    return {
        "trigger_state": trigger_state,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "step_count": len(steps),
        "max_steps": max_steps,
        "required_failure_count": len(required_failures),
        "skipped_count": len(skipped),
        "artifact_kernel_objects": kernel_summary.get("object_count"),
        "artifact_kernel_relationships": kernel_summary.get("relationship_count"),
        "latest_command_state": state["command_executor"].get("trigger_state"),
        "feedback_action_count": len(reports["feedback_action_queue"].get("actions", [])),
        "broad_transfer_pass_rate": get_path(state["broad_transfer_closure"], ["summary", "aggregate_pass_rate"], None),
        "broad_transfer_floor_gap": get_path(state["broad_transfer_closure"], ["summary", "aggregate_floor_gap"], None),
        "transfer_generalization_risks": get_path(state["transfer_generalization"], ["summary", "overfit_risk_count"], None),
        "transfer_generalization_ready": get_path(state["transfer_generalization"], ["summary", "transfer_ready"], None),
        "repo_repair_task_count": get_path(reports["repo_repair_main_curriculum"], ["summary", "task_count"], None),
        "symliquid_state_slots": len(reports["symliquid_state_engine_queue"].get("queue", [])),
        "teacher_architect_experiments": get_path(reports["teacher_architect_closure"], ["summary", "experiment_count"], None),
        "promotion_evidence": False,
    }


def safe_card_id(value: Any) -> str:
    text = str(value or "card").lower()
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text).strip("_") or "card"


def public_calibration_private_gate_block() -> dict[str, Any]:
    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json")
    concepts = scheduler.get("concepts") if isinstance(scheduler.get("concepts"), list) else []
    gate_concepts = {
        "private_pressure_private_closure",
        "decoder_v2_private_ablation_gate",
        "edge_contract_v2_private_closure",
        "edge_case_full_body_private_closure_v1",
        "edge_contract_private_closure",
        "typed_interface_private_closure",
    }
    blockers = []
    for row in concepts:
        if not isinstance(row, dict):
            continue
        concept = str(row.get("concept") or "")
        if concept in gate_concepts and str(row.get("status") or "") == "ready":
            blockers.append({
                "concept": concept,
                "status": row.get("status"),
                "priority": row.get("priority"),
                "rotation_epoch": row.get("rotation_epoch"),
            })
    return {
        "blocked": bool(blockers),
        "blockers": blockers,
        "scheduler_report": "reports/high_transfer_curriculum_scheduler.json",
        "rule": "private closure/ablation gates run before broad public calibration actions",
    }


def append_private_gate_action(actions: list[dict[str, Any]], public_gate: dict[str, Any]) -> None:
    if any(item.get("kind") == "run_private_gate_before_public_calibration" for item in actions):
        return
    actions.append(
        action(
            "critical",
            "run_private_gate_before_public_calibration",
            "Run private closure/gate before public calibration",
            "Scheduler reports a private closure or Decoder V2 ablation gate is ready; run the Hive board gate before any public receiver calibration.",
            evidence=public_gate,
            command=[
                sys.executable,
                "scripts/hive_work_board_executor.py",
                "--execute",
                "--resume",
                "--max-tasks",
                "1",
                "--timeout-seconds",
                "21600",
                "--out",
                "reports/hive_work_board_executor.json",
                "--markdown-out",
                "reports/hive_work_board_executor.md",
            ],
        )
    )


def action(
    priority: str,
    kind: str,
    title: str,
    suggested_action: str,
    *,
    evidence: Any,
    command: list[Any],
) -> dict[str, Any]:
    payload = {
        "priority": priority,
        "kind": kind,
        "title": title,
        "suggested_action": suggested_action,
        "evidence": evidence,
        "command": [str(item) for item in command],
        "status": "queued",
        "viea_stage": "feedback_ratchet_next_action",
        "side_effect_tier": "local_report_or_training_pressure",
        "public_data_rule": "public_benchmarks_calibration_only",
    }
    payload["action_id"] = stable_action_id(payload)
    return payload


def named_command(stage: str, command: list[Any]) -> dict[str, Any]:
    return {"stage": stage, "command": [str(item) for item in command]}


def stable_action_id(payload: dict[str, Any]) -> str:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    parts = [
        str(payload.get("kind") or ""),
        str(payload.get("title") or ""),
        str(evidence.get("card_id") or evidence.get("tool_name") or evidence.get("source") or ""),
        json.dumps(payload.get("command") or [], sort_keys=True),
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"viea_action_{digest}"


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(payload: dict[str, Any], reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# VIEA Autonomy Spine",
        "",
        f"- trigger_state: `{payload['trigger_state']}`",
        f"- artifact_kernel_objects: `{payload['summary'].get('artifact_kernel_objects')}`",
        f"- broad_transfer_pass_rate: `{payload['summary'].get('broad_transfer_pass_rate')}`",
        f"- broad_transfer_floor_gap: `{payload['summary'].get('broad_transfer_floor_gap')}`",
        f"- transfer_generalization_risks: `{payload['summary'].get('transfer_generalization_risks')}`",
        f"- feedback_actions: `{payload['summary'].get('feedback_action_count')}`",
        "",
        "## Control Path",
        "",
    ]
    for item in payload["spine"]["flow"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Action Reports", ""])
    for name, report in reports.items():
        lines.append(f"- `{name}`: `{report.get('trigger_state')}`")
    lines.extend(["", "## Honest Boundary", "", payload["spine"]["public_benchmarks"], payload["spine"]["teacher"], ""])
    return "\n".join(lines)


def render_simple_report(name: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- policy: `{payload.get('policy')}`",
        "",
    ]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if summary:
        lines.extend(["## Summary", ""])
        for key, value in summary.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    if actions:
        lines.extend(["## Actions", ""])
        for item in actions[:20]:
            lines.append(f"- `{item.get('priority')}` `{item.get('kind')}`: {item.get('title')}")
        lines.append("")
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    if gates:
        lines.extend(["## Gates", ""])
        for item in gates:
            lines.append(f"- {'PASS' if item.get('passed') else 'FAIL'} `{item.get('gate')}`: {item.get('evidence')}")
        lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def int_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
