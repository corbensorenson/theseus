#!/usr/bin/env python3
"""Bounded overnight self-improvement loop for Theseus.

This orchestrator keeps the current capability-transfer contract honest while
the machine is unattended. It runs local/private improvement and evidence
cycles, never executes public benchmark tests, and stops on hard no-cheat
violations instead of manufacturing a nearby green report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN_ROOT = REPORTS / "overnight_self_improvement_v1"
LEDGER = RUN_ROOT / "ledger.jsonl"
STATUS = RUN_ROOT / "status.json"
MORNING_REPORT = REPORTS / "overnight_self_improvement_v1_report.json"
MORNING_MARKDOWN = REPORTS / "overnight_self_improvement_v1_report.md"
STOP_FLAG = REPORTS / "overnight_self_improvement_stop.flag"


PUBLIC_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_human_eval,source_livecodebench"
PRIVATE_HELDOUT = "data/training_data/high_transfer/private_eval/post_v4_seed23_5x32_private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
PRIVATE_STS_CANDIDATES = "reports/code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl"
PRIVATE_NON_STS_CANDIDATES = "reports/code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl"
BROAD_TRAINING_SOURCES = "data/training_sources/broad_capability_curriculum_v1_training_sources.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=23)
    parser.add_argument("--max-cases-per-card", type=int, default=32)
    parser.add_argument("--max-candidates-per-task", type=int, default=4)
    parser.add_argument("--skip-solo-learning", action="store_true")
    parser.add_argument("--solo-profile", default="inner_loop")
    parser.add_argument("--solo-max-new-jobs", type=int, default=1)
    parser.add_argument("--solo-wait-seconds", type=float, default=20.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--min-battery-percent", type=float, default=35.0)
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--skip-vcm", action="store_true")
    parser.add_argument("--skip-architecture-sweep", action="store_true")
    parser.add_argument("--out", default=str(MORNING_REPORT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(MORNING_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    if not args.execute:
        report = planned_report(args)
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
        print(json.dumps(report, indent=2))
        return 0

    report = run_loop(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(compact_report(report), indent=2))
    return 0 if report.get("ok") else 2


def planned_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_overnight_self_improvement_loop_v1",
        "created_utc": now(),
        "ok": True,
        "execute": False,
        "planned_cycles": int(args.cycles),
        "planned_hours": float(args.hours),
        "sleep_seconds": int(args.sleep_seconds),
        "contract": contract(),
        "outputs": {
            "run_root": rel(RUN_ROOT),
            "ledger": rel(LEDGER),
            "status": rel(STATUS),
            "morning_report": str(args.out),
            "morning_markdown": str(args.markdown_out),
        },
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    keep_awake = start_keep_awake_assertion(args)
    try:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        started = time.time()
        deadline = started + max(1.0, float(args.hours)) * 3600.0
        cycles: list[dict[str, Any]] = []
        hard_stop: dict[str, Any] | None = None
        max_cycles = max(1, int(args.cycles))
        for index in range(max_cycles):
            if time.time() >= deadline:
                hard_stop = {"reason": "deadline_reached", "cycle_index": index}
                break
            if STOP_FLAG.exists():
                hard_stop = {"reason": "operator_stop_flag", "flag": rel(STOP_FLAG), "cycle_index": index}
                break
            cycle = run_cycle(index, args)
            cycles.append(cycle)
            append_jsonl(LEDGER, cycle)
            write_status(args, cycles, hard_stop=None)
            violation = hard_violation(cycle)
            if violation:
                hard_stop = violation
                write_status(args, cycles, hard_stop=hard_stop)
                break
            if index < max_cycles - 1 and time.time() < deadline:
                time.sleep(max(0, int(args.sleep_seconds)))
        report = final_report(args, cycles, hard_stop, started)
        write_status(args, cycles, hard_stop=hard_stop, final=report)
        return report
    finally:
        stop_keep_awake_assertion(keep_awake)


def start_keep_awake_assertion(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if not bool(getattr(args, "keep_awake", False)):
        return None
    caffeinate = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if not caffeinate:
        return None
    return subprocess.Popen(
        [caffeinate, "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def stop_keep_awake_assertion(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()


def run_cycle(index: int, args: argparse.Namespace) -> dict[str, Any]:
    cycle_started = time.perf_counter()
    seed = int(args.base_seed) + index
    cycle_dir = RUN_ROOT / f"cycle_{index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    steps = []

    solo_learning = cycle_dir / "hive_solo_learning_status.json"
    if not args.skip_solo_learning:
        solo_cmd = [
            sys.executable,
            "scripts/hive_solo_learning_loop.py",
            "sweep",
            "--execute",
            "--profile",
            str(args.solo_profile or "inner_loop"),
            "--max-new-jobs",
            str(max(1, int(args.solo_max_new_jobs))),
            "--wait-seconds",
            str(max(0.0, float(args.solo_wait_seconds))),
            "--out",
            rel(solo_learning),
        ]
        if bool(args.allow_battery):
            solo_cmd.append("--allow-battery")
        if args.min_battery_percent is not None:
            solo_cmd.extend(["--min-battery-percent", str(float(args.min_battery_percent))])
        if bool(args.keep_awake):
            solo_cmd.append("--keep-awake")
        steps.append(
            run_step(
                "solo_offline_learning_sweep",
                solo_cmd,
                cycle_dir,
                timeout_seconds=max(600, int(float(args.solo_wait_seconds) + 120)),
            )
        )

    public_shape = cycle_dir / "student_token_code_generator_public_shape.json"
    public_candidates = cycle_dir / "student_code_candidates_public_shape.jsonl"
    public_tasks = cycle_dir / "student_token_code_tasks_public_shape.jsonl"
    public_checkpoint = cycle_dir / "student_token_code_checkpoint.json"
    steps.append(
        run_step(
            "public_shape_prompt_only_candidate_generation",
            [
                sys.executable,
                "scripts/student_token_code_candidate_generator.py",
                "--cards",
                PUBLIC_CARDS,
                "--seed",
                str(seed),
                "--max-cases-per-card",
                str(max(1, int(args.max_cases_per_card))),
                "--max-candidates-per-task",
                str(max(1, int(args.max_candidates_per_task))),
                "--training-sources",
                BROAD_TRAINING_SOURCES,
                "--task-manifest-out",
                rel(public_tasks),
                "--checkpoint-out",
                rel(public_checkpoint),
                "--out",
                rel(public_candidates),
                "--report-out",
                rel(public_shape),
            ],
            cycle_dir,
            timeout_seconds=360,
        )
    )

    public_return_audit = cycle_dir / "public_shape_return_contract_audit_v1.json"
    steps.append(
        run_step(
            "public_shape_return_contract_audit",
            [
                sys.executable,
                "scripts/public_shape_return_contract_audit_v1.py",
                "--task-manifest",
                rel(public_tasks),
                "--candidate-manifest",
                rel(public_candidates),
                "--out",
                rel(public_return_audit),
                "--markdown-out",
                rel(cycle_dir / "public_shape_return_contract_audit_v1.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )

    sts_ablation = cycle_dir / "private_residual_v3_sts_ablation.json"
    steps.append(
        run_step(
            "private_sts_on_off_ablation",
            [
                sys.executable,
                "scripts/private_residual_v3_sts_ablation.py",
                "--heldout",
                PRIVATE_HELDOUT,
                "--sts-candidates",
                PRIVATE_STS_CANDIDATES,
                "--non-sts-candidates",
                PRIVATE_NON_STS_CANDIDATES,
                "--same-body-control-out",
                rel(cycle_dir / "sts_label_removed_control.jsonl"),
                "--category-shuffled-control-out",
                rel(cycle_dir / "category_shuffled_control.jsonl"),
                "--out",
                rel(sts_ablation),
                "--markdown-out",
                rel(cycle_dir / "private_residual_v3_sts_ablation.md"),
            ],
            cycle_dir,
            timeout_seconds=240,
        )
    )

    heldout_score = cycle_dir / "private_residual_repair_v3_heldout_score.json"
    steps.append(
        run_step(
            "private_heldout_adapter_off_score",
            [
                sys.executable,
                "scripts/private_residual_repair_v3_heldout_score.py",
                "--heldout",
                PRIVATE_HELDOUT,
                "--candidates",
                PRIVATE_STS_CANDIDATES,
                "--control-candidates",
                PRIVATE_NON_STS_CANDIDATES,
                "--adapter-off",
                "--out",
                rel(heldout_score),
                "--markdown-out",
                rel(cycle_dir / "private_residual_repair_v3_heldout_score.md"),
            ],
            cycle_dir,
            timeout_seconds=240,
        )
    )

    private_admissibility = cycle_dir / "private_full_body_candidate_admissibility_gate.json"
    steps.append(
        run_step(
            "private_full_body_admissibility_gate",
            [
                sys.executable,
                "scripts/private_full_body_candidate_admissibility_gate_v1.py",
                "--heldout",
                PRIVATE_HELDOUT,
                "--candidates",
                PRIVATE_STS_CANDIDATES,
                "--control-candidates",
                PRIVATE_NON_STS_CANDIDATES,
                "--out",
                rel(private_admissibility),
                "--markdown-out",
                rel(cycle_dir / "private_full_body_candidate_admissibility_gate.md"),
            ],
            cycle_dir,
            timeout_seconds=240,
        )
    )

    private_residual_gate = cycle_dir / "private_residual_repair_v3_gate.json"
    steps.append(
        run_step(
            "private_residual_repair_v3_gate",
            [
                sys.executable,
                "scripts/private_residual_repair_v3_gate.py",
                "--private-heldout",
                rel(heldout_score),
                "--decoder-gate",
                "reports/code_lm_train_once_fanout.json",
                "--sts-ablation",
                rel(sts_ablation),
                "--maturity-audit",
                "reports/maturity_integrity_audit.json",
                "--out",
                rel(private_residual_gate),
                "--markdown-out",
                rel(cycle_dir / "private_residual_repair_v3_gate.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )

    vcm_report = cycle_dir / "vcm_context_recovery_benchmark.json"
    if not args.skip_vcm:
        steps.append(
            run_step(
                "private_vcm_context_recovery",
                [
                    sys.executable,
                    "scripts/vcm_context_recovery_benchmark.py",
                    "--out",
                    rel(vcm_report),
                    "--markdown-out",
                    rel(cycle_dir / "vcm_context_recovery_benchmark.md"),
                    "--residuals-out",
                    rel(cycle_dir / "vcm_context_recovery_residuals.jsonl"),
                    "--ablation-out",
                    rel(cycle_dir / "vcm_on_off_ablation.json"),
                    "--ablation-markdown-out",
                    rel(cycle_dir / "vcm_on_off_ablation.md"),
                ],
                cycle_dir,
                timeout_seconds=180,
            )
        )

    mlx_report = cycle_dir / "macos_mlx_structural_action_smoke.json"
    if not args.skip_mlx and platform.machine() == "arm64":
        steps.append(
            run_step(
                "macos_mlx_structural_action_smoke",
                [
                    sys.executable,
                    "scripts/macos_mlx_structural_action_smoke.py",
                    "--execute",
                    "--out",
                    rel(mlx_report),
                    "--markdown-out",
                    rel(cycle_dir / "macos_mlx_structural_action_smoke.md"),
                    "--candidate-manifest-out",
                    rel(cycle_dir / "macos_mlx_structural_action_smoke_candidates.jsonl"),
                ],
                cycle_dir,
                timeout_seconds=240,
            )
        )
    else:
        write_json(
            mlx_report,
            {
                "trigger_state": "SKIPPED",
                "summary": {"reason": "skip_mlx_requested_or_not_arm64", "machine": platform.machine()},
                "external_inference_calls": 0,
            },
        )

    metal_report = cycle_dir / "macos_metal_production_route_readiness.json"
    steps.append(
        run_step(
            "macos_metal_production_route_readiness",
            [
                sys.executable,
                "scripts/macos_metal_production_route_readiness.py",
                "--out",
                rel(metal_report),
                "--markdown-out",
                rel(cycle_dir / "macos_metal_production_route_readiness.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )

    if not args.skip_architecture_sweep:
        steps.append(
            run_step(
                "matched_architecture_sweep_refresh_cached",
                [
                    sys.executable,
                    "scripts/neural_seed_architecture_sweep.py",
                    "--execute",
                    "--out",
                    rel(cycle_dir / "neural_seed_architecture_sweep.json"),
                    "--markdown-out",
                    rel(cycle_dir / "neural_seed_architecture_sweep.md"),
                ],
                cycle_dir,
                timeout_seconds=900,
            )
        )

    maturity = cycle_dir / "maturity_integrity_audit.json"
    steps.append(
        run_step(
            "maturity_integrity_audit",
            [
                sys.executable,
                "scripts/maturity_integrity_audit.py",
                "--out",
                rel(maturity),
                "--markdown-out",
                rel(cycle_dir / "maturity_integrity_audit.md"),
            ],
            cycle_dir,
            timeout_seconds=240,
        )
    )

    candidate_promotion = cycle_dir / "candidate_promotion_gate.json"
    steps.append(
        run_step(
            "candidate_promotion_gate_expected_blocking_ok",
            [
                sys.executable,
                "scripts/candidate_promotion_gate.py",
                "--maturity-integrity-audit",
                rel(maturity),
                "--out",
                rel(candidate_promotion),
            ],
            cycle_dir,
            timeout_seconds=180,
            allowed_returncodes={0, 1},
        )
    )

    closure = cycle_dir / "capability_transfer_closure_v1.json"
    steps.append(
        run_step(
            "capability_transfer_closure_packet",
            [
                sys.executable,
                "scripts/capability_transfer_closure_v1.py",
                "--private-admissibility",
                rel(private_admissibility),
                "--private-residual-gate",
                rel(private_residual_gate),
                "--private-heldout",
                rel(heldout_score),
                "--public-shape-generator",
                rel(public_shape),
                "--public-shape-return-audit",
                rel(public_return_audit),
                "--mac-mlx",
                rel(mlx_report),
                "--mac-metal",
                rel(metal_report),
                "--candidate-promotion",
                rel(candidate_promotion),
                "--out",
                rel(closure),
                "--markdown-out",
                rel(cycle_dir / "capability_transfer_closure_v1.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )

    cycle_report = summarize_cycle(index, seed, cycle_dir, steps, cycle_started)
    write_json(cycle_dir / "cycle_report.json", cycle_report)
    write_text(cycle_dir / "cycle_report.md", render_cycle_markdown(cycle_report))
    return cycle_report


def run_step(
    label: str,
    command: list[str],
    cycle_dir: Path,
    *,
    timeout_seconds: int,
    allowed_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    allowed = allowed_returncodes or {0}
    started = time.perf_counter()
    log_out = cycle_dir / f"{label}.stdout.txt"
    log_err = cycle_dir / f"{label}.stderr.txt"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        log_out.write_text(stdout[-20000:], encoding="utf-8")
        log_err.write_text(stderr[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "returncode": result.returncode,
            "ok": result.returncode in allowed,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(log_out),
            "stderr_log": rel(log_err),
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
    except subprocess.TimeoutExpired as exc:
        log_out.write_text((exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "", encoding="utf-8")
        log_err.write_text(str(exc)[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "returncode": 124,
            "ok": False,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(log_out),
            "stderr_log": rel(log_err),
            "stdout_tail": "",
            "stderr_tail": str(exc)[-800:],
        }


def summarize_cycle(index: int, seed: int, cycle_dir: Path, steps: list[dict[str, Any]], started: float) -> dict[str, Any]:
    closure = read_json(cycle_dir / "capability_transfer_closure_v1.json", {})
    closure_summary = object_field(closure, "summary")
    public_coverage = object_field(closure_summary, "public_candidate_coverage")
    no_cheat = object_field(closure_summary, "no_cheat")
    vcm = read_json(cycle_dir / "vcm_context_recovery_benchmark.json", {})
    vcm_summary = object_field(vcm, "summary")
    solo = read_json(cycle_dir / "hive_solo_learning_status.json", {})
    solo_always_active = object_field(solo, "always_active")
    solo_always_summary = object_field(solo_always_active, "summary")
    solo_ledger = object_field(solo, "ledger")
    solo_safety = object_field(solo, "safety")
    private_gate = read_json(cycle_dir / "private_residual_repair_v3_gate.json", {})
    private_summary = object_field(private_gate, "summary")
    mlx = read_json(cycle_dir / "macos_mlx_structural_action_smoke.json", {})
    mlx_summary = object_field(mlx, "summary")
    failed_steps = [step for step in steps if not step.get("ok")]
    return {
        "policy": "project_theseus_overnight_self_improvement_cycle_v1",
        "created_utc": now(),
        "cycle_index": index,
        "seed": seed,
        "cycle_dir": rel(cycle_dir),
        "ok": not failed_steps,
        "failed_steps": [{"label": row["label"], "returncode": row["returncode"]} for row in failed_steps],
        "summary": {
            "private_semantic_ready": closure_summary.get("private_semantic_ready"),
            "private_selected_pass_rate": closure_summary.get("private_selected_pass_rate"),
            "private_no_admissible_task_rate": closure_summary.get("private_no_admissible_task_rate"),
            "private_learned_candidate_pass_rate": closure_summary.get("private_learned_candidate_pass_rate"),
            "public_candidate_coverage_ready": closure_summary.get("public_candidate_coverage_ready"),
            "public_shape_task_count": public_coverage.get("task_count"),
            "public_shape_candidate_count": public_coverage.get("candidate_count"),
            "strict_public_promotion_candidate_count": closure_summary.get("strict_public_promotion_candidate_count"),
            "public_shape_ready_training_sources": public_coverage.get("ready_training_sources"),
            "public_shape_training_rows_used": public_coverage.get("training_rows_used"),
            "public_shape_return_contract_audit_state": public_coverage.get("return_contract_audit_state"),
            "public_shape_return_contract_selected_rate": public_coverage.get("return_contract_selected_shape_compatible_task_rate"),
            "public_shape_return_contract_any_rate": public_coverage.get("return_contract_any_shape_compatible_task_rate"),
            "sts_selected_delta": closure_summary.get("sts_matched_selected_delta"),
            "sts_oracle_delta": closure_summary.get("sts_matched_oracle_delta"),
            "vcm_private_state": vcm.get("trigger_state"),
            "vcm_private_accuracy": vcm_summary.get("vcm_answer_accuracy"),
            "vcm_private_best_baseline_accuracy": vcm_summary.get("best_baseline_answer_accuracy"),
            "solo_learning_state": solo_always_active.get("state"),
            "solo_learning_planned_actions": solo_always_summary.get("planned_actions"),
            "solo_learning_executed_actions": solo_always_summary.get("executed_actions"),
            "solo_learning_new_event_count": solo_ledger.get("new_event_count"),
            "solo_learning_failed_count": solo_ledger.get("failed_count"),
            "solo_learning_promotion_count": solo_ledger.get("promotion_count"),
            "solo_learning_teacher_used": solo_safety.get("teacher_used"),
            "solo_learning_external_inference_calls": solo_safety.get("external_inference_calls"),
            "mlx_state": mlx.get("trigger_state"),
            "mlx_used": mlx_summary.get("mlx_used"),
            "mlx_verifier_pass_rate": mlx_summary.get("verifier_pass_rate"),
            "fallback_return_count": no_cheat.get("fallback_return_count"),
            "template_like_count": no_cheat.get("template_like_count"),
            "public_leakage_count": no_cheat.get("public_leakage_count"),
            "external_inference_calls": no_cheat.get("external_inference_calls"),
            "private_residual_gate_state": private_gate.get("trigger_state"),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "closure_recommendation": closure.get("recommendation"),
        "steps": steps,
        "hard_violation": merge_violations(hard_violation_from_summary(closure_summary), hard_violation_from_solo(solo)),
    }


def hard_violation(cycle: dict[str, Any]) -> dict[str, Any] | None:
    if cycle.get("hard_violation"):
        return {"reason": "hard_no_cheat_violation", "cycle_index": cycle.get("cycle_index"), "detail": cycle.get("hard_violation")}
    if cycle.get("failed_steps"):
        return {"reason": "step_failed", "cycle_index": cycle.get("cycle_index"), "failed_steps": cycle.get("failed_steps")}
    return None


def hard_violation_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    no_cheat = object_field(summary, "no_cheat")
    public_coverage = object_field(summary, "public_candidate_coverage")
    violations = {}
    for key in [
        "fallback_return_count",
        "template_like_count",
        "public_leakage_count",
        "external_inference_calls",
        "public_training_rows",
        "teacher_used_count",
    ]:
        if int_number(no_cheat.get(key)) != 0:
            violations[key] = no_cheat.get(key)
    if public_coverage.get("public_tests_visible_to_generator") is not False:
        violations["public_tests_visible_to_generator"] = public_coverage.get("public_tests_visible_to_generator")
    if public_coverage.get("canonical_solution_seen_by_solver") is not False:
        violations["canonical_solution_seen_by_solver"] = public_coverage.get("canonical_solution_seen_by_solver")
    return violations or None


def hard_violation_from_solo(solo: dict[str, Any]) -> dict[str, Any] | None:
    if not solo:
        return None
    safety = object_field(solo, "safety")
    violations = {}
    if bool(safety.get("teacher_used")):
        violations["solo_teacher_used"] = safety.get("teacher_used")
    if int_number(safety.get("external_inference_calls")) != 0:
        violations["solo_external_inference_calls"] = safety.get("external_inference_calls")
    if str(safety.get("remote_peer_queueing") or "") not in {"disabled_in_solo_offline_commands", "disabled"}:
        violations["solo_remote_peer_queueing"] = safety.get("remote_peer_queueing")
    return violations or None


def merge_violations(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged or None


def final_report(args: argparse.Namespace, cycles: list[dict[str, Any]], hard_stop: dict[str, Any] | None, started_epoch: float) -> dict[str, Any]:
    best = best_cycle(cycles)
    latest = cycles[-1] if cycles else {}
    latest_summary = object_field(latest, "summary")
    return {
        "policy": "project_theseus_overnight_self_improvement_loop_v1",
        "created_utc": now(),
        "ok": hard_stop is None or hard_stop.get("reason") == "deadline_reached",
        "execute": True,
        "contract": contract(),
        "started_utc": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started_epoch, 3),
        "cycles_requested": int(args.cycles),
        "cycles_completed": len(cycles),
        "hard_stop": hard_stop,
        "ledger": rel(LEDGER),
        "status": rel(STATUS),
        "best_cycle": best,
        "latest_summary": latest_summary,
        "improvement_summary": improvement_summary(cycles),
        "next_exact_gate": next_exact_gate(latest),
        "cycles": cycles,
    }


def best_cycle(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    if not cycles:
        return {}
    return max(
        cycles,
        key=lambda row: (
            number(get_path(row, ["summary", "private_selected_pass_rate"], 0.0)),
            int_number(get_path(row, ["summary", "strict_public_promotion_candidate_count"], 0)),
            number(get_path(row, ["summary", "vcm_private_accuracy"], 0.0)),
            int_number(get_path(row, ["summary", "solo_learning_new_event_count"], 0)),
            -int_number(get_path(row, ["summary", "fallback_return_count"], 0)),
        ),
    )


def improvement_summary(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    if not cycles:
        return {}
    first = object_field(cycles[0], "summary")
    last = object_field(cycles[-1], "summary")
    keys = [
        "private_selected_pass_rate",
        "private_no_admissible_task_rate",
        "strict_public_promotion_candidate_count",
        "sts_selected_delta",
        "vcm_private_accuracy",
        "mlx_verifier_pass_rate",
        "solo_learning_new_event_count",
        "solo_learning_executed_actions",
    ]
    return {
        key: {
            "first": first.get(key),
            "last": last.get(key),
            "delta": round(number(last.get(key)) - number(first.get(key)), 6),
        }
        for key in keys
    }


def next_exact_gate(latest: dict[str, Any]) -> str:
    summary = object_field(latest, "summary")
    if summary.get("fallback_return_count") or summary.get("template_like_count") or summary.get("public_leakage_count"):
        return "quarantine_candidate_path_and_fix_no_cheat_violation"
    if summary.get("public_candidate_coverage_ready") is not True:
        return "repair_prompt_only_public_shaped_candidate_coverage_without_public_tests"
    if summary.get("private_semantic_ready") is not True:
        return "repair_private_semantic_transfer_or_no_admissible_gate"
    if int_number(summary.get("solo_learning_executed_actions")) <= 0:
        return "repair_local_solo_learning_queueing_so_overnight_loop_does_real_work"
    return "public_calibration_remains_locked; next work is a deliberate one-shot calibration decision or private semantic quality expansion"


def write_status(args: argparse.Namespace, cycles: list[dict[str, Any]], *, hard_stop: dict[str, Any] | None, final: dict[str, Any] | None = None) -> None:
    payload = {
        "policy": "project_theseus_overnight_self_improvement_status_v1",
        "created_utc": now(),
        "running": final is None and hard_stop is None,
        "cycles_requested": int(args.cycles),
        "cycles_completed": len(cycles),
        "hard_stop": hard_stop,
        "latest_cycle": cycles[-1] if cycles else {},
        "final_report": rel(resolve(str(args.out))) if final is not None else "",
    }
    write_json(STATUS, payload)


def contract() -> dict[str, Any]:
    return {
        "public_calibration": "locked; do not execute public benchmark tests",
        "public_training_data": "forbidden",
        "public_prompt_use": "generation_input_only_for_prompt_only_candidate_manifest",
        "fallback_returns": "forbidden",
        "templates_loop_closure": "forbidden_as_capability_credit",
        "diagnostic_adapters": "do_not_count_as_learned_capability",
        "external_inference": "zero unless governed teacher distillation explicitly records it",
        "remote_execution": "not used",
    }


def render_cycle_markdown(cycle: dict[str, Any]) -> str:
    summary = object_field(cycle, "summary")
    lines = [
        f"# Overnight Self-Improvement Cycle {cycle.get('cycle_index')}",
        "",
        f"- ok: {cycle.get('ok')}",
        f"- private selected pass rate: {summary.get('private_selected_pass_rate')}",
        f"- public-shaped coverage ready: {summary.get('public_candidate_coverage_ready')}",
        f"- strict public-promotion candidates: {summary.get('strict_public_promotion_candidate_count')}",
        f"- STS selected delta: {summary.get('sts_selected_delta')}",
        f"- VCM private state/accuracy: {summary.get('vcm_private_state')} / {summary.get('vcm_private_accuracy')}",
        f"- solo learning state/actions/events: {summary.get('solo_learning_state')} / {summary.get('solo_learning_executed_actions')} / {summary.get('solo_learning_new_event_count')}",
        f"- MLX state/pass rate: {summary.get('mlx_state')} / {summary.get('mlx_verifier_pass_rate')}",
        f"- fallback/template/leak/external: {summary.get('fallback_return_count')} / {summary.get('template_like_count')} / {summary.get('public_leakage_count')} / {summary.get('external_inference_calls')}",
    ]
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    latest = object_field(report, "latest_summary")
    lines = [
        "# Overnight Self-Improvement Loop v1",
        "",
        f"- ok: {report.get('ok')}",
        f"- cycles completed: {report.get('cycles_completed')} / {report.get('cycles_requested')}",
        f"- runtime seconds: {report.get('runtime_seconds')}",
        f"- hard stop: {report.get('hard_stop')}",
        f"- latest private selected pass rate: {latest.get('private_selected_pass_rate')}",
        f"- latest public-shaped coverage ready: {latest.get('public_candidate_coverage_ready')}",
        f"- latest strict public-promotion candidates: {latest.get('strict_public_promotion_candidate_count')}",
        f"- latest STS selected delta: {latest.get('sts_selected_delta')}",
        f"- latest VCM private accuracy: {latest.get('vcm_private_accuracy')}",
        f"- latest solo learning actions/events: {latest.get('solo_learning_executed_actions')} / {latest.get('solo_learning_new_event_count')}",
        f"- latest MLX verifier pass rate: {latest.get('mlx_verifier_pass_rate')}",
        f"- next exact gate: {report.get('next_exact_gate')}",
        "",
        "## Improvements",
    ]
    for key, row in object_field(report, "improvement_summary").items():
        lines.append(f"- {key}: {row}")
    return "\n".join(lines) + "\n"


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "ok": report.get("ok"),
        "cycles_completed": report.get("cycles_completed"),
        "hard_stop": report.get("hard_stop"),
        "latest_summary": report.get("latest_summary"),
        "next_exact_gate": report.get("next_exact_gate"),
        "ledger": report.get("ledger"),
    }


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key) if isinstance(row, dict) else None
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_number(value: Any) -> int:
    return int(number(value))


def read_json(path: Path, default: Any | None = None) -> Any:
    fallback = {} if default is None else default
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
