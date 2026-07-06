"""Vacation Mode Supervisor V3 for VIEA + Theseus + SymLiquid.

This is the long-unattended operator layer. It wraps the existing unattended
supervisor, adds hard readiness gates, triages failed actions into repair
actions, consumes the durable Hive work board, checks that each cycle made
useful progress, and can perform bounded source exploration through the governed
resource pantry.

It does not grant arbitrary shell/network authority. Network exploration is
small, license-gated, and source/metadata oriented; public benchmarks remain
calibration-only and commercial game assets/ROMs remain forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "vacation_mode_supervisor_ledger.jsonl"
DEFAULT_OUT = REPORTS / "vacation_mode_supervisor.json"
DEFAULT_MARKDOWN = REPORTS / "vacation_mode_supervisor.md"
DEFAULT_TRIAGE = REPORTS / "vacation_mode_failure_triage.json"
DEFAULT_REPAIR_QUEUE = REPORTS / "vacation_mode_repair_action_queue.json"
STOP_FLAGS = [REPORTS / "sparkstream_stop.flag", REPORTS / "unattended_autonomy_stop.flag", REPORTS / "vacation_mode_stop.flag"]
PAUSE_FLAGS = [REPORTS / "sparkstream_pause.flag", REPORTS / "viea_action_executor_pause.flag", REPORTS / "vacation_mode_pause.flag"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1, help="0 means run until a stop flag is seen.")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--explore", action="store_true", help="Run bounded governed source exploration.")
    parser.add_argument("--start-services", action="store_true", help="Restart SparkStream/Hive/Dashboard if stale.")
    parser.add_argument("--max-actions-per-cycle", type=int, default=1)
    parser.add_argument("--max-steps-per-action", type=int, default=1)
    parser.add_argument("--action-timeout-seconds", type=int, default=21600)
    parser.add_argument("--teacher-timeout-seconds", type=int, default=21600)
    parser.add_argument("--min-root-free-gib", type=float, default=10.0)
    parser.add_argument("--min-spillover-free-gib", type=float, default=100.0)
    parser.add_argument("--min-vram-free-mib", type=float, default=768.0)
    parser.add_argument("--max-gpu-temp-c", type=float, default=88.0)
    parser.add_argument("--dashboard-max-age-seconds", type=int, default=300)
    parser.add_argument("--hive-max-age-seconds", type=int, default=120)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    cycles = 0
    last_cycle: dict[str, Any] = {}
    final_state = "GREEN"
    while True:
        if stop_requested():
            final_state = "YELLOW"
            break
        if args.cycles and cycles >= max(0, int(args.cycles)):
            break
        if paused():
            last_cycle = paused_cycle(cycles + 1)
            append_jsonl(LEDGER, last_cycle)
            write_snapshot(args, started, cycles, last_cycle, "YELLOW")
            sleep_or_stop(max(1, int(args.sleep_seconds)))
            continue
        cycles += 1
        last_cycle = run_cycle(cycles, args=args)
        append_jsonl(LEDGER, last_cycle)
        write_snapshot(args, started, cycles, last_cycle, cycle_trigger(last_cycle))
        if last_cycle.get("trigger_state") == "RED":
            final_state = "RED"
            break
        if args.cycles and cycles >= max(0, int(args.cycles)):
            break
        sleep_or_stop(max(1, int(args.sleep_seconds)))

    if final_state != "RED" and (stop_requested() or paused()):
        final_state = "YELLOW"
    write_snapshot(args, started, cycles, last_cycle, final_state)
    return 2 if final_state == "RED" else 0


def run_cycle(cycle: int, *, args: argparse.Namespace) -> dict[str, Any]:
    before = snapshot_metrics()
    gates = hard_gates(args)
    hard_gate_failures = [gate for gate in gates if not gate["passed"] and gate.get("severity") == "hard"]
    rows: list[dict[str, Any]] = []
    if args.start_services and service_restart_recommended(gates):
        rows.append(run_step(service_restart_step()))
        gates = hard_gates(args)
        hard_gate_failures = [gate for gate in gates if not gate["passed"] and gate.get("severity") == "hard"]
    if hard_gate_failures:
        rows.extend(repair_transient_hard_gates(hard_gate_failures))
        if rows:
            gates = hard_gates(args)
            hard_gate_failures = [gate for gate in gates if not gate["passed"] and gate.get("severity") == "hard"]
    if hard_gate_failures:
        triage = triage_failures()
        repair_queue = build_repair_queue(triage, include_exploration=False)
        write_json(DEFAULT_TRIAGE, triage)
        write_json(DEFAULT_REPAIR_QUEUE, repair_queue)
        after = snapshot_metrics()
        progress = progress_contract(before, after, triage=triage, exploration={}, rows=rows)
        return cycle_report(cycle, args, "RED", gates, rows, triage, repair_queue, progress, before, after, "hard_gate_failure")

    active_closure = active_code_lm_closure_lock()
    if active_closure:
        rows.append(
            {
                "name": "hive_work_board_executor",
                "returncode": 0,
                "runtime_ms": 0,
                "reason": "active_code_lm_closure_lock_observe_only",
                "active_closure": active_closure,
                "stdout_tail": "",
                "stderr_tail": "",
                "allow_failure": True,
            }
        )
    else:
        board_step = work_board_executor_step(args)
        rows.append(run_step(board_step))
        active_closure = active_code_lm_closure_lock()
    if active_closure:
        rows.append(
            {
                "name": "no_progress_policy",
                "returncode": 0,
                "runtime_ms": 0,
                "reason": "active_code_lm_closure_lock_skipped",
                "active_closure": active_closure,
                "actions": [],
                "allow_failure": True,
            }
        )
    else:
        rows.append(apply_no_progress_policy())
    if active_closure:
        rows.append(
            {
                "name": "unattended_autonomy_supervisor",
                "returncode": 0,
                "runtime_ms": 0,
                "reason": "active_code_lm_closure_lock_skipped",
                "active_closure": active_closure,
                "stdout_tail": "",
                "stderr_tail": "",
                "allow_failure": True,
            }
        )
    else:
        unattended_step = unattended_supervisor_step(args)
        rows.append(run_step(unattended_step))

    triage = triage_failures()
    repair_queue = build_repair_queue(triage, include_exploration=bool(args.explore))
    write_json(DEFAULT_TRIAGE, triage)
    write_json(DEFAULT_REPAIR_QUEUE, repair_queue)

    if args.execute and repair_queue.get("actions") and should_run_repair_queue(triage) and not active_closure:
        rows.append(run_step(repair_executor_step(args)))

    rows.append(run_step(utilization_sweep_step(args)))

    exploration: dict[str, Any] = {}
    if args.explore:
        exploration = run_exploration(args)
        rows.extend(exploration.get("steps", []))

    if args.allow_teacher and should_escalate_teacher(triage) and not active_closure:
        rows.append(run_step(teacher_step(args)))

    rows.append(run_step(hive_readiness_step()))
    rows.append(run_step(morning_report_step()))
    rows.append(run_step(overnight_proof_step()))
    rows.append(run_step(long_run_governor_step()))
    after = snapshot_metrics()
    progress = progress_contract(before, after, triage=triage, exploration=exploration, rows=rows)
    trigger = "GREEN" if progress["passed"] else "YELLOW"
    required_failure = [row for row in rows if int(row.get("returncode") or 0) != 0 and not row.get("allow_failure")]
    if required_failure:
        trigger = "RED"
    return cycle_report(cycle, args, trigger, gates, rows, triage, repair_queue, progress, before, after, "cycle_complete")


def repair_transient_hard_gates(hard_gate_failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failed_gate_names = {str(gate.get("gate") or "") for gate in hard_gate_failures}
    if "personality_core_green" in failed_gate_names:
        rows.append(
            run_step(
                step(
                    "refresh_personality_runtime_audit",
                    [
                        sys.executable,
                        "scripts/personality_runtime_audit.py",
                        "--refresh",
                        "--out",
                        "reports/personality_runtime_audit.json",
                    ],
                    timeout=900,
                    allow_failure=True,
                )
            )
        )
    return rows


def hard_gates(args: argparse.Namespace) -> list[dict[str, Any]]:
    resource = read_json(REPORTS / "resource_governor.json", {})
    performance = read_json(REPORTS / "performance_optimizer.json", {})
    action_executor = read_json(REPORTS / "viea_action_executor.json", {})
    transfer = read_json(REPORTS / "transfer_generalization_audit.json", {})
    teacher_budget = read_json(REPORTS / "teacher_budget_last.json", read_json(REPORTS / "teacher_budget_audit.json", {}))
    personality = read_json(REPORTS / "personality_runtime_audit.json", {})
    root_free = get_path(resource, ["current_resources", "disk", "free_gib"], None)
    spill_free = get_path(resource, ["current_resources", "spillover", "selected", "free_gib"], None)
    gpu_free = get_path(resource, ["current_resources", "gpu", "memory_free_mib"], get_path(performance, ["summary", "gpu_free_mib"], None))
    gpu_temp = get_path(resource, ["current_resources", "gpu", "temperature_c"], None)
    dashboard = http_status("http://127.0.0.1:8787/api/health", timeout=3)
    hive = http_status("http://127.0.0.1:8791/api/hive/health", timeout=3)
    remote_accelerator = remote_training_accelerator_status()
    local_vram_ok = gpu_free is None or number_or_negative(gpu_free) >= float(args.min_vram_free_mib)
    root_free_gib = number_or_negative(root_free)
    spill_free_gib = number_or_negative(spill_free)
    spillover_redirect_ok = bool(
        root_free_gib >= 5.0
        and spill_free_gib >= float(args.min_spillover_free_gib)
        and Path("D:/ProjectTheseus").exists()
    )
    root_disk_ok = root_free_gib >= float(args.min_root_free_gib) or spillover_redirect_ok
    gates = [
        gate("stop_flags_absent", not stop_requested(), [rel(path) for path in STOP_FLAGS if path.exists()], severity="hard"),
        gate("pause_flags_absent", not paused(), [rel(path) for path in PAUSE_FLAGS if path.exists()], severity="hard"),
        gate(
            "root_disk_above_floor",
            root_disk_ok,
            {
                "root_free_gib": root_free,
                "root_floor_gib": float(args.min_root_free_gib),
                "spillover_free_gib": spill_free,
                "spillover_floor_gib": float(args.min_spillover_free_gib),
                "spillover_redirect_ok": spillover_redirect_ok,
                "protected_root_floor_gib": 5.0,
            },
            severity="hard",
        ),
        gate("spillover_disk_above_floor", spill_free_gib >= float(args.min_spillover_free_gib), spill_free, severity="hard"),
        gate(
            "gpu_vram_reserve_or_remote_accelerator_available",
            local_vram_ok or bool(remote_accelerator.get("available")),
            {
                "local_gpu_free_mib": gpu_free,
                "local_min_free_mib": float(args.min_vram_free_mib),
                "local_vram_ok": local_vram_ok,
                "remote_training_accelerator": remote_accelerator,
            },
            severity="hard",
        ),
        gate("gpu_temperature_below_ceiling", gpu_temp is None or number_or_negative(gpu_temp) <= float(args.max_gpu_temp_c), gpu_temp, severity="hard"),
        gate("action_executor_not_paused", not bool(get_path(action_executor, ["summary", "paused"], False)), get_path(action_executor, ["summary", "paused"], None), severity="hard"),
        gate("public_data_guard_present", bool(get_path(transfer, ["rules", "public_benchmarks"], "")), get_path(transfer, ["rules", "public_benchmarks"], ""), severity="hard"),
        gate("personality_core_green", personality.get("trigger_state") == "GREEN", {"trigger_state": personality.get("trigger_state"), "summary": personality.get("summary")}, severity="hard"),
        gate("teacher_budget_loaded_or_teacher_disabled", bool(teacher_budget) or not args.allow_teacher, teacher_budget.get("policy") if isinstance(teacher_budget, dict) else "", severity="soft"),
        gate("dashboard_api_responsive", dashboard["ok"], dashboard, severity="soft"),
        gate("hive_api_responsive", hive["ok"], hive, severity="soft"),
    ]
    return gates


def triage_failures() -> dict[str, Any]:
    action_report = read_json(REPORTS / "viea_action_executor.json", {})
    ledger = read_jsonl(REPORTS / "viea_action_execution_ledger.jsonl")
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in ledger:
        action_id = str(row.get("action_id") or "")
        if action_id:
            latest_by_id[action_id] = row
    failures = [row for row in latest_by_id.values() if str(row.get("status") or "") == "failed"]
    classified = [classify_failed_action(row) for row in failures]
    closure_reports = []
    for path in REPORTS.glob("broad_transfer_closure_runner_source_*.json"):
        payload = read_json(path, {})
        if payload:
            closure_reports.append(classify_closure_report(path, payload))
    closure_failures = [row for row in closure_reports if row.get("trigger_state") in {"RED", "YELLOW"} and row.get("failed_gates")]
    counts = Counter(row["class"] for row in classified)
    counts.update(row["class"] for row in closure_failures)
    actions = repair_actions_from_classes(counts, classified, closure_failures)
    payload = {
        "policy": "project_theseus_vacation_failure_triage_v2",
        "created_utc": now(),
        "trigger_state": "YELLOW" if classified or closure_failures else "GREEN",
        "summary": {
            "failed_action_count": len(classified),
            "closure_failure_count": len(closure_failures),
            "class_counts": dict(counts),
            "retryable_count": sum(1 for row in classified if row.get("retryable")),
            "teacher_should_run": any(row.get("teacher_escalation") for row in classified + closure_failures),
        },
        "failed_actions": classified[-20:],
        "closure_failures": sorted(closure_failures, key=lambda row: row.get("mtime", 0), reverse=True)[:20],
        "recommended_repair_actions": actions,
        "source_report": "reports/viea_action_executor.json",
        "external_inference_calls": 0,
    }
    return payload


def classify_failed_action(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    stdout = str(payload.get("stdout_tail") or "")
    stderr = str(payload.get("stderr_tail") or "")
    reason = str(row.get("reason") or payload.get("reason") or "")
    command = [str(item) for item in payload.get("command", [])] if isinstance(payload.get("command"), list) else []
    command_text = " ".join(command).lower()
    text = "\n".join([reason, command_text, stdout, stderr]).lower()
    kind = str(payload.get("kind") or "")
    title = str(payload.get("title") or "")
    cls = "unknown_failed_action"
    evidence = reason
    retryable = False
    teacher = False
    if "timeout" in text or "returncode_124" in text:
        cls = "timeout_or_step_budget_exhausted"
        evidence = "timeout"
        retryable = True
    elif "closure_steps_completed" in text and "false" in text:
        cls = "closure_step_failed"
        evidence = "closure_steps_completed=false"
        retryable = True
    elif "broad_transfer_closure_runner.py" in command_text and "returncode_2" in text:
        cls = "closure_step_failed"
        evidence = "broad_transfer_closure_runner_returncode_2"
        retryable = True
    elif "token_level_student_candidates" in text and "false" in text:
        cls = "missing_token_level_student_candidates"
        evidence = "token_level_student_candidates=false"
        retryable = True
        teacher = True
    elif "public_task_coverage_requested" in text and "false" in text:
        cls = "missing_public_coverage_evidence"
        evidence = "public_task_coverage_requested=false"
        retryable = True
    elif "symliquid_conditioning_used" in text and "false" in text:
        cls = "missing_symliquid_conditioning"
        evidence = "symliquid_conditioning_used=false"
        teacher = True
    elif "sts_delta_measured" in text and "false" in text:
        cls = "sts_delta_not_measured"
        evidence = "sts_delta_measured=false"
        retryable = True
    elif "below_public_code_floor" in text:
        cls = "below_public_transfer_floor"
        evidence = "below_public_code_floor"
        teacher = True
    return {
        "action_id": row.get("action_id"),
        "created_utc": row.get("created_utc"),
        "kind": kind,
        "title": title,
        "reason": reason,
        "command": command,
        "class": cls,
        "evidence": evidence,
        "retryable": retryable,
        "teacher_escalation": teacher,
    }


def classify_closure_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    failed = [row for row in gates if isinstance(row, dict) and not row.get("passed")]
    failed_names = [str(row.get("gate") or "") for row in failed]
    cls = "closure_learning_wall"
    if "token_level_student_candidates" in failed_names:
        cls = "missing_token_level_student_candidates"
    elif "public_task_coverage_requested" in failed_names:
        cls = "missing_public_coverage_evidence"
    elif "symliquid_conditioning_used" in failed_names:
        cls = "missing_symliquid_conditioning"
    elif "sts_delta_measured" in failed_names:
        cls = "sts_delta_not_measured"
    elif "closure_steps_completed" in failed_names:
        cls = "closure_step_failed"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "path": rel(path),
        "mtime": mtime,
        "trigger_state": payload.get("trigger_state"),
        "class": cls,
        "failed_gates": failed_names,
        "card_ids": payload.get("requested_cards") or payload.get("cards") or [],
        "teacher_escalation": cls in {"missing_token_level_student_candidates", "missing_symliquid_conditioning", "closure_learning_wall"},
    }


def repair_actions_from_classes(
    counts: Counter[str],
    failed_actions: list[dict[str, Any]],
    closure_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    cards = cards_from_failures(failed_actions, closure_failures)
    public_gate = public_calibration_private_gate_block()
    retryable = [row for row in failed_actions if row.get("retryable") and row.get("command")]
    for row in retryable[:1]:
        actions.append(action(
            str(row.get("kind") or "train_private_semantic_residual_family"),
            f"Retry once after {row.get('class')}: {row.get('title') or row.get('action_id')}",
            "retry a previously failed approved action once before escalating",
            priority="critical",
            command=[str(item) for item in row.get("command") or []],
            evidence={"failure_class": row.get("class"), "source_action_id": row.get("action_id")},
        ))
    if counts.get("missing_token_level_student_candidates") or counts.get("closure_step_failed"):
        if public_gate["blocked"]:
            actions.append(action(
                "run_private_gate_before_public_calibration",
                "Run private closure/gate before public calibration",
                "scheduler reports a private closure or Decoder V2 ablation gate is ready; run the board gate instead of a public retry",
                priority="critical",
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
                evidence=public_gate,
            ))
        else:
            for card in cards[:2] or ["source_mbpp"]:
                actions.append(action(
                    "train_private_semantic_residual_family",
                    f"Retry closure with token-candidate evidence for {card}",
                    "retry failed closure once with step budget and long timeout",
                    priority="critical",
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
                        f"reports/broad_transfer_closure_runner_{card}_vacation_retry.json",
                        "--markdown-out",
                        f"reports/broad_transfer_closure_runner_{card}_vacation_retry.md",
                    ],
                    evidence={"failure_class": "missing_token_level_or_closure", "card_id": card},
                ))
    if counts.get("missing_public_coverage_evidence"):
        for card in cards[:2] or ["source_livecodebench"]:
            actions.append(action(
                "expand_public_adapter_clean_slice",
                f"Expand clean public calibration coverage for {card}",
                "coverage gate failed; expand or rerun clean 32-task calibration",
                priority="high",
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
                evidence={"failure_class": "missing_public_coverage", "card_id": card},
            ))
    if counts.get("missing_symliquid_conditioning") or counts.get("sts_delta_not_measured"):
        actions.append(action(
            "run_same_seed_sts_repair_ablation",
            "Rerun same-seed STS repair ablation after missing/flat STS evidence",
            "STS must condition generation and beat STS-off per card",
            priority="high",
            command=[sys.executable, "scripts/sts_repair_ablation.py", "--out", "reports/sts_repair_ablation.json", "--markdown-out", "reports/sts_repair_ablation.md"],
            evidence={"failure_class": "sts_or_symliquid_evidence_missing"},
        ))
    if any(row.get("teacher_escalation") for row in failed_actions + closure_failures):
        actions.append(action(
            "request_teacher_architecture_diagnosis",
            "Ask teacher for architecture experiment after repeated closure failure",
            "teacher should diagnose residual cluster, not provide answers",
            priority="high",
            command=[],
            evidence={"failure_classes": dict(counts)},
        ))
    return actions


def public_calibration_private_gate_block() -> dict[str, Any]:
    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
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
        "rule": "private closure/ablation gates run before any public calibration retry",
    }


def cards_from_failures(failed_actions: list[dict[str, Any]], closure_failures: list[dict[str, Any]]) -> list[str]:
    cards: list[str] = []
    for row in failed_actions:
        text = " ".join([str(row.get("title") or ""), str(row.get("evidence") or "")]).lower()
        for card in ["source_mbpp", "source_evalplus", "source_bigcodebench", "source_livecodebench"]:
            if card.lower() in text and card not in cards:
                cards.append(card)
    for row in closure_failures:
        for card in row.get("card_ids") or []:
            if isinstance(card, str) and card.startswith("source_") and card not in cards:
                cards.append(card)
        text = str(row.get("path") or "")
        for card in ["source_mbpp", "source_evalplus", "source_bigcodebench", "source_livecodebench"]:
            if card in text and card not in cards:
                cards.append(card)
    transfer = read_json(REPORTS / "transfer_generalization_audit.json", {})
    for card in get_path(transfer, ["summary", "weak_cards"], []) or []:
        if isinstance(card, str) and card not in cards:
            cards.append(card)
    return cards


def build_repair_queue(triage: dict[str, Any], *, include_exploration: bool) -> dict[str, Any]:
    actions = list(triage.get("recommended_repair_actions") or [])
    if include_exploration:
        actions.append(action(
            "refresh_symliquid_state_engine",
            "Refresh SymLiquid state before exploration and next queue selection",
            "keep compact recurrent state aligned to route memory and transfer residuals",
            priority="medium",
            command=[sys.executable, "scripts/symliquid_state_engine.py", "--out", "reports/symliquid_state_engine.json", "--markdown-out", "reports/symliquid_state_engine.md"],
            evidence={"source": "vacation_mode_exploration"},
        ))
    return {
        "policy": "project_theseus_feedback_action_queue_v1",
        "created_utc": now(),
        "source": "vacation_mode_supervisor_v2",
        "actions": actions,
        "rules": {
            "public_benchmarks": "calibration_only_not_training",
            "repair": "retry_once_then_teacher_architecture_diagnosis",
            "exploration": "license_gated_small_source_or_metadata_fetch_only",
        },
        "external_inference_calls": 0,
    }


def run_exploration(args: argparse.Namespace) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    steps.append(run_step(step(
        "resource_pantry_refresh",
        [
            sys.executable,
            "scripts/resource_pantry.py",
            "--out",
            "reports/resource_pantry.json",
            "--markdown-out",
            "reports/resource_pantry.md",
            "--max-clones",
            "2",
        ] + (["--execute"] if args.allow_network_fetch else []),
        timeout=1800,
        allow_failure=True,
    )))
    online_cmd = [
        sys.executable,
        "scripts/online_source_catalog.py",
        "--out",
        "reports/online_source_catalog_report.json",
        "--category",
        "rl_environment",
        "--category",
        "rl_benchmark",
        "--category",
        "code_benchmark",
        "--max-imports",
        "4",
    ]
    if args.allow_network_fetch:
        online_cmd.extend(["--allow-network-fetch", "--import-sources", "--refresh-metadata"])
    steps.append(run_step(step("online_source_catalog_exploration", online_cmd, timeout=1800, allow_failure=True)))
    steps.append(run_step(step(
        "rl_benchmark_registry_refresh",
        [sys.executable, "scripts/rl_benchmark_registry.py", "--out", "reports/rl_benchmark_registry.json"],
        timeout=600,
        allow_failure=True,
    )))
    pantry = read_json(REPORTS / "resource_pantry.json", {})
    catalog = read_json(REPORTS / "online_source_catalog_report.json", {})
    payload = {
        "policy": "project_theseus_vacation_exploration_v2",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(int(row.get("returncode") or 0) == 0 or row.get("allow_failure") for row in steps) else "YELLOW",
        "allow_network_fetch": bool(args.allow_network_fetch),
        "summary": {
            "steps": len(steps),
            "resource_actions": get_path(pantry, ["summary", "actions"], 0),
            "resource_present_clones": get_path(pantry, ["summary", "present_clones"], None),
            "resource_adapter_ready": get_path(pantry, ["summary", "adapter_ready"], None),
            "catalog_imports": len(catalog.get("imports") or []) if isinstance(catalog.get("imports"), list) else 0,
            "catalog_blocked": len(catalog.get("blocked") or []) if isinstance(catalog.get("blocked"), list) else 0,
        },
        "steps": steps,
        "rules": {
            "commercial_rom_downloads": "forbidden",
            "bulk_training_downloads": "forbidden",
            "license_uncertain_sources": "metadata_or_queue_only",
            "games": "open RL/source adapters only unless local licensed runtime is separately attested",
        },
        "external_inference_calls": 0,
    }
    write_json(REPORTS / "vacation_mode_exploration.json", payload)
    write_text(REPORTS / "vacation_mode_exploration.md", render_exploration_markdown(payload))
    return payload


def progress_contract(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    triage: dict[str, Any],
    exploration: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    signals = []
    if improved(after.get("broad_pass_rate"), before.get("broad_pass_rate")):
        signals.append(signal("better_public_transfer", before.get("broad_pass_rate"), after.get("broad_pass_rate")))
    if lower(after.get("transfer_risk_count"), before.get("transfer_risk_count")):
        signals.append(signal("transfer_risk_reduced", before.get("transfer_risk_count"), after.get("transfer_risk_count")))
    if increased(after.get("completed_actions"), before.get("completed_actions")):
        signals.append(signal("action_completed", before.get("completed_actions"), after.get("completed_actions")))
    if increased(after.get("board_done"), before.get("board_done")) or any(row.get("name") == "hive_work_board_executor" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("work_board_step_completed", before.get("board_done"), after.get("board_done")))
    if int(get_path(triage, ["summary", "failed_action_count"], 0) or 0) > 0 or int(get_path(triage, ["summary", "closure_failure_count"], 0) or 0) > 0:
        signals.append(signal("useful_failed_residual_diagnosis", 0, get_path(triage, ["summary", "class_counts"], {})))
    if get_path(exploration, ["summary", "resource_actions"], 0) or get_path(exploration, ["summary", "catalog_imports"], 0):
        signals.append(signal("governed_exploration_or_fetch", 0, exploration.get("summary")))
    if any(row.get("name") == "teacher_architect_experiment_runner" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("teacher_experiment_step", 0, 1))
    if any(row.get("name") == "hive_fleet_readiness" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("hive_readiness_refreshed", 0, 1))
    if any(row.get("name") == "hive_morning_report" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("morning_report_written", 0, 1))
    if any(row.get("name") == "hive_overnight_proof" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("overnight_proof_green", 0, 1))
    if any(row.get("name") == "hive_long_run_governor" and int(row.get("returncode") or 0) == 0 for row in rows):
        signals.append(signal("long_run_governor_written", 0, 1))
    if any(row.get("name") == "no_progress_family_policy" and int(row.get("returncode") or 0) == 0 and row.get("action_count") for row in rows):
        signals.append(signal("no_progress_task_family_governed", 0, 1))
    return {
        "policy": "project_theseus_vacation_progress_contract_v3",
        "passed": bool(signals),
        "signals": signals,
        "required": [
            "better_public_transfer",
            "private_residual_shrinkage",
            "new_clean_evidence",
            "useful_failed_residual_diagnosis",
            "repaired_adapter",
            "retired_stale_action",
            "teacher_experiment_step",
            "governed_exploration_or_fetch",
            "work_board_step_completed",
            "morning_report_written",
            "overnight_proof_green",
            "no_progress_task_family_governed",
        ],
        "score_semantics": "progress contract for unattended work; not promotion evidence",
    }


def snapshot_metrics() -> dict[str, Any]:
    scoreboard = read_json(REPORTS / "learning_scoreboard.json", {})
    transfer = read_json(REPORTS / "transfer_generalization_audit.json", {})
    executor = read_json(REPORTS / "viea_action_executor.json", {})
    board_executor = read_json(REPORTS / "hive_work_board_executor.json", {})
    work_board = read_json(REPORTS / "hive_work_board.json", {})
    return {
        "created_utc": now(),
        "scoreboard_state": scoreboard.get("trigger_state"),
        "broad_pass_rate": get_path(scoreboard, ["broad_transfer_matrix", "real_public_pass_rate"], None),
        "public_transfer_pass_rate": get_path(scoreboard, ["public_transfer", "real_public_task_pass_rate"], None),
        "transfer_risk_count": get_path(transfer, ["summary", "overfit_risk_count"], None),
        "above_floor_transfer_card_count": get_path(transfer, ["summary", "above_floor_transfer_card_count"], None),
        "ready_actions": get_path(executor, ["summary", "ready_action_count"], None),
        "completed_actions": get_path(executor, ["summary", "completed_total"], None),
        "failed_actions": get_path(executor, ["summary", "failed_total"], None),
        "board_ready": get_path(board_executor, ["summary", "ready_tasks"], get_path(work_board, ["summary", "ready_or_active"], None)),
        "board_done": get_path(work_board, ["summary", "done"], None),
        "board_executed": get_path(board_executor, ["summary", "executed_tasks"], None),
    }


def cycle_report(
    cycle: int,
    args: argparse.Namespace,
    trigger_state: str,
    gates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    triage: dict[str, Any],
    repair_queue: dict[str, Any],
    progress: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_vacation_mode_cycle_v3",
        "created_utc": now(),
        "cycle": cycle,
        "trigger_state": trigger_state,
        "reason": reason,
        "execute_requested": bool(args.execute),
        "teacher_allowed": bool(args.allow_teacher),
        "network_fetch_allowed": bool(args.allow_network_fetch),
        "explore_requested": bool(args.explore),
        "gates": gates,
        "steps": rows,
        "triage": compact_triage(triage),
        "repair_action_count": len(repair_queue.get("actions") or []),
        "progress_contract": progress,
        "before": before,
        "after": after,
        "external_inference_calls": 0,
    }


def compact_triage(triage: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": triage.get("trigger_state"),
        "summary": triage.get("summary"),
        "recommended_repair_actions": [
            {
                "kind": row.get("kind"),
                "priority": row.get("priority"),
                "title": row.get("title"),
            }
            for row in (triage.get("recommended_repair_actions") or [])[:8]
        ],
    }


def write_snapshot(args: argparse.Namespace, started: float, cycles: int, last_cycle: dict[str, Any], trigger_state: str) -> None:
    payload = {
        "policy": "project_theseus_vacation_mode_supervisor_v3",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "cycles_completed": cycles,
            "execute_requested": bool(args.execute),
            "teacher_allowed": bool(args.allow_teacher),
            "network_fetch_allowed": bool(args.allow_network_fetch),
            "explore_requested": bool(args.explore),
            "paused": paused(),
            "stop_requested": stop_requested(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "ledger": rel(LEDGER),
            "triage": rel(DEFAULT_TRIAGE),
            "repair_queue": rel(DEFAULT_REPAIR_QUEUE),
        },
        "last_cycle": last_cycle,
        "rules": {
            "work_board": "durable Hive work board is consumed before loose report-driven unattended work",
            "failure_triage": "failed action -> classify -> retry once via repair queue -> teacher architecture diagnosis if still stuck",
            "progress_contract": "each cycle must produce measurable progress or a useful residual diagnosis",
            "broad_transfer": "prefer transferable concepts before benchmark-name pressure",
            "exploration": "small governed source/metadata exploration only; no bulk training data or commercial game assets",
            "hive": "fleet readiness is checked; remote execution requires private hive policy and shared secret",
            "long_run_governor": "each cycle emits a 30-minute operator report covering nodes, sync, teacher use, demotions, failures, and next action",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    last = payload.get("last_cycle", {})
    progress = last.get("progress_contract") if isinstance(last.get("progress_contract"), dict) else {}
    lines = [
        "# Vacation Mode Supervisor V3",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- cycles_completed: `{summary.get('cycles_completed')}`",
        f"- execute_requested: `{summary.get('execute_requested')}`",
        f"- teacher_allowed: `{summary.get('teacher_allowed')}`",
        f"- network_fetch_allowed: `{summary.get('network_fetch_allowed')}`",
        f"- explore_requested: `{summary.get('explore_requested')}`",
        f"- last_cycle_state: `{last.get('trigger_state')}`",
        f"- progress_contract: `{progress.get('passed')}`",
        "",
        "## Progress Signals",
        "",
    ]
    for row in progress.get("signals", []) if isinstance(progress.get("signals"), list) else []:
        lines.append(f"- `{row.get('kind')}`: {row.get('before')} -> {row.get('after')}")
    if not progress.get("signals"):
        lines.append("- none")
    lines.extend(["", "## Gates", ""])
    for row in last.get("gates", []) if isinstance(last.get("gates"), list) else []:
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark} `{row.get('gate')}` ({row.get('severity')}): {row.get('evidence')}")
    lines.extend(["", "## Last Steps", ""])
    for row in last.get("steps", []) if isinstance(last.get("steps"), list) else []:
        lines.append(f"- `{row.get('returncode')}` `{row.get('name')}` runtime_ms={row.get('runtime_ms')} error={row.get('error', '')}")
    lines.append("")
    return "\n".join(lines)


def render_exploration_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Vacation Mode Exploration",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- allow_network_fetch: `{payload.get('allow_network_fetch')}`",
        f"- resource_actions: `{summary.get('resource_actions')}`",
        f"- catalog_imports: `{summary.get('catalog_imports')}`",
        f"- resource_adapter_ready: `{summary.get('resource_adapter_ready')}`",
        "",
        "## Rules",
        "",
    ]
    for key, value in (payload.get("rules") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def unattended_supervisor_step(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/unattended_autonomy_supervisor.py",
        "--cycles",
        "1",
        "--sleep-seconds",
        "1",
        "--max-actions-per-cycle",
        str(max(1, int(args.max_actions_per_cycle))),
        "--max-steps-per-action",
        str(max(1, int(args.max_steps_per_action))),
        "--action-timeout-seconds",
        str(max(60, int(args.action_timeout_seconds))),
        "--teacher-timeout-seconds",
        str(max(300, int(args.teacher_timeout_seconds))),
        "--out",
        "reports/unattended_autonomy_supervisor.json",
        "--markdown-out",
        "reports/unattended_autonomy_supervisor.md",
    ]
    if args.execute:
        command.append("--execute")
    if args.allow_teacher:
        command.append("--allow-teacher")
    return step("unattended_autonomy_supervisor", command, timeout=max(60, int(args.action_timeout_seconds) + 3900), allow_failure=True)


def work_board_executor_step(args: argparse.Namespace) -> dict[str, Any]:
    active_closure = active_code_lm_closure_lock()
    command = [
        sys.executable,
        "scripts/hive_work_board_executor.py",
        "--resume",
        "--max-tasks",
        str(max(0, int(args.max_actions_per_cycle))),
        "--max-steps",
        str(max(1, int(args.max_steps_per_action))),
        "--timeout-seconds",
        str(max(60, int(args.action_timeout_seconds))),
        "--out",
        "reports/hive_work_board_executor.json",
        "--markdown-out",
        "reports/hive_work_board_executor.md",
    ]
    if active_closure:
        command.append("--status")
        spec = step("hive_work_board_executor", command, timeout=900, allow_failure=True)
        spec["reason"] = "active_code_lm_closure_lock_observe_only"
        spec["active_closure"] = active_closure
        return spec
    if args.execute and int(args.max_actions_per_cycle) > 0:
        command.append("--execute")
    else:
        command.append("--status")
    if args.allow_teacher:
        command.append("--allow-teacher")
    return step("hive_work_board_executor", command, timeout=max(60, int(args.action_timeout_seconds) + 300), allow_failure=False)


def active_code_lm_closure_lock() -> dict[str, Any]:
    for path in sorted(REPORTS.glob("code_lm_closure*.lock")):
        payload = read_json(path, {})
        try:
            pid = int(payload.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid and pid_matches_code_lm_lock(pid):
            return {
                "lock": rel(path),
                "pid": pid,
                "created_utc": payload.get("created_utc"),
                "out": payload.get("out"),
            }
    return {}


def pid_matches_code_lm_lock(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        f"Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" | "
                        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        try:
            row = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(row, dict) or int(row.get("ProcessId") or 0) != pid:
            return False
        command = str(row.get("CommandLine") or "").lower()
        return any(needle in command for needle in ("code_lm", "symliquid-cli", "cargo", "rustc"))
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    command = (result.stdout or "").lower()
    return result.returncode == 0 and any(needle in command for needle in ("code_lm", "symliquid-cli", "cargo", "rustc"))


def morning_report_step() -> dict[str, Any]:
    return step(
        "hive_morning_report",
        [
            sys.executable,
            "scripts/hive_morning_report.py",
            "--out",
            "reports/hive_morning_report.json",
            "--markdown-out",
            "reports/hive_morning_report.md",
        ],
        timeout=120,
        allow_failure=True,
    )


def overnight_proof_step() -> dict[str, Any]:
    return step(
        "hive_overnight_proof",
        [
            sys.executable,
            "scripts/hive_overnight_proof.py",
            "--out",
            "reports/hive_overnight_proof.json",
            "--markdown-out",
            "reports/hive_overnight_proof.md",
        ],
        timeout=120,
        allow_failure=True,
    )


def long_run_governor_step() -> dict[str, Any]:
    return step(
        "hive_long_run_governor",
        [
            sys.executable,
            "scripts/hive_long_run_governor.py",
            "--out",
            "reports/hive_long_run_governor.json",
            "--markdown-out",
            "reports/hive_long_run_governor.md",
        ],
        timeout=120,
        allow_failure=True,
    )


def apply_no_progress_policy() -> dict[str, Any]:
    started = time.perf_counter()
    report = read_json(REPORTS / "hive_work_board_executor.json", {})
    results = report.get("results") if isinstance(report.get("results"), list) else []
    actions = []
    db_path = REPORTS / "hive_work_board.sqlite"
    ledger_path = REPORTS / "hive_no_progress_families.jsonl"
    for result in results:
        if not isinstance(result, dict):
            continue
        contract = result.get("improvement_contract") if isinstance(result.get("improvement_contract"), dict) else {}
        if contract.get("passed"):
            continue
        concept = str(contract.get("concept") or "")
        if not concept:
            continue
        family = f"high_transfer:{concept}" if concept else "unknown"
        previous = [
            row
            for row in read_jsonl(ledger_path)
            if str(row.get("family") or "") == family
        ]
        occurrence = len(previous) + 1
        action = "blocked" if occurrence >= 2 else "demoted"
        append_jsonl(
            ledger_path,
            {
                "created_utc": now(),
                "policy": "project_theseus_no_progress_family_policy_v1",
                "family": family,
                "task_id": result.get("task_id"),
                "action": action,
                "occurrence": occurrence,
                "residual_cluster": contract.get("residual_cluster"),
            },
        )
        if db_path.exists():
            apply_family_update(db_path, concept, action)
        actions.append({"family": family, "action": action, "occurrence": occurrence})
    return {
        "name": "no_progress_family_policy",
        "command": [],
        "timeout": 0,
        "allow_failure": True,
        "returncode": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "action_count": len(actions),
        "actions": actions,
    }


def apply_family_update(db_path: Path, concept: str, action: str) -> None:
    if not concept:
        return
    status = "blocked" if action == "blocked" else None
    priority = "low" if action == "demoted" else None
    fields = []
    values: list[Any] = []
    if status:
        fields.append("status=?")
        values.append(status)
        fields.append("blocked_reason=?")
        values.append("no_progress_contract_twice")
    if priority:
        fields.append("priority=?")
        values.append(priority)
    fields.append("updated_utc=?")
    values.append(now())
    values.append(f'%"concept": "{concept}"%')
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            f"UPDATE tasks SET {', '.join(fields)} WHERE source='high_transfer_curriculum_scheduler' AND evidence_json LIKE ?",
            values,
        )


def repair_executor_step(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/viea_action_executor.py",
        "--queue",
        rel(DEFAULT_REPAIR_QUEUE),
        "--execute",
        "--resume",
        "--max-actions",
        "1",
        "--max-steps",
        "1",
        "--timeout-seconds",
        str(max(60, int(args.action_timeout_seconds))),
        "--out",
        "reports/vacation_mode_repair_action_executor.json",
        "--markdown-out",
        "reports/vacation_mode_repair_action_executor.md",
    ]
    if args.allow_teacher:
        command.append("--allow-teacher")
    return step("vacation_mode_repair_action_executor", command, timeout=max(60, int(args.action_timeout_seconds)), allow_failure=True)


def utilization_sweep_step(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/hive_utilization_manager.py",
        "sweep",
        "--max-new-jobs",
        str(max(1, int(args.max_actions_per_cycle))),
        "--out",
        "reports/hive_utilization_manager.json",
    ]
    if args.execute:
        command.append("--execute")
    return step("hive_utilization_sweep", command, timeout=900, allow_failure=True)


def teacher_step(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/teacher_architect_experiment_runner.py",
        "--execute",
        "--max-experiments",
        "1",
        "--max-steps",
        "1",
        "--timeout-seconds",
        str(max(300, int(args.teacher_timeout_seconds))),
        "--out",
        "reports/teacher_architect_experiment_runner.json",
        "--markdown-out",
        "reports/teacher_architect_experiment_runner.md",
    ]
    if args.allow_teacher:
        command.append("--allow-teacher")
    return step("teacher_architect_experiment_runner", command, timeout=max(300, int(args.teacher_timeout_seconds)), allow_failure=True)


def hive_readiness_step() -> dict[str, Any]:
    return step(
        "hive_fleet_readiness",
        [sys.executable, "scripts/hive_fleet_readiness.py", "--out", "reports/hive_fleet_readiness.json", "--markdown-out", "reports/hive_fleet_readiness.md"],
        timeout=300,
        allow_failure=True,
    )


def service_restart_step() -> dict[str, Any]:
    if os.name == "nt":
        return step(
            "restart_local_services",
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/start_theseus_hive.ps1",
                "-Restart",
            ],
            timeout=300,
            allow_failure=True,
        )
    return step("restart_local_services", ["sh", "scripts/start_theseus_hive.sh"], timeout=300, allow_failure=True)


def service_restart_recommended(gates: list[dict[str, Any]]) -> bool:
    return any(not row.get("passed") and row.get("gate") in {"dashboard_api_responsive", "hive_api_responsive"} for row in gates)


def should_run_repair_queue(triage: dict[str, Any]) -> bool:
    return bool(triage.get("recommended_repair_actions")) and int(get_path(triage, ["summary", "failed_action_count"], 0) or 0) > 0


def should_escalate_teacher(triage: dict[str, Any]) -> bool:
    return bool(get_path(triage, ["summary", "teacher_should_run"], False))


def action(
    kind: str,
    title: str,
    suggested_action: str,
    *,
    priority: str,
    command: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "title": title,
        "suggested_action": suggested_action,
        "priority": priority,
        "command": command,
        "evidence": evidence,
        "status": "queued",
        "viea_stage": "vacation_mode_repair",
        "public_data_rule": "public_benchmarks_calibration_only",
        "side_effect_tier": "local_report_or_training_pressure",
    }
    payload["action_id"] = stable_action_id(payload)
    return payload


def stable_action_id(action: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps({
        "kind": action.get("kind"),
        "title": action.get("title"),
        "command": action.get("command"),
        "evidence": action.get("evidence"),
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"viea_action_{digest}"


def paused_cycle(cycle: int) -> dict[str, Any]:
    return {
        "policy": "project_theseus_vacation_mode_cycle_v3",
        "created_utc": now(),
        "cycle": cycle,
        "trigger_state": "YELLOW",
        "reason": "paused",
        "pause_flags": [rel(path) for path in PAUSE_FLAGS if path.exists()],
        "external_inference_calls": 0,
    }


def step(name: str, command: list[str], *, timeout: int, allow_failure: bool) -> dict[str, Any]:
    return {"name": name, "command": command, "timeout": int(timeout), "allow_failure": bool(allow_failure)}


def run_step(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            spec["command"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(60, int(spec.get("timeout") or 60)),
        )
        return {
            **spec,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **spec,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout_safety_fuse",
        }
    except OSError as exc:
        return {**spec, "returncode": 127, "runtime_ms": int((time.perf_counter() - started) * 1000), "stdout_tail": "", "stderr_tail": str(exc), "error": "launch_failed"}


def http_status(url: str, *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read(512 * 1024).decode("utf-8"))
        return {"ok": True, "url": url, "runtime_ms": int((time.perf_counter() - started) * 1000), "policy": data.get("policy"), "created_utc": data.get("created_utc")}
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return {"ok": False, "url": url, "runtime_ms": int((time.perf_counter() - started) * 1000), "error": str(exc)}


def http_json(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return {}


def remote_training_accelerator_status() -> dict[str, Any]:
    """Return a compact proof that another Hive node can absorb training work.

    Vacation Mode should not stop the entire hive just because the coordinator's
    local GPU is already busy. Local CUDA remains protected by the utilization
    manager's node-level blockers; this gate only decides whether the supervisor
    may continue running remote/CPU-safe cycles.
    """
    payloads = [
        http_json("http://127.0.0.1:8791/api/hive/operator/status", timeout=3),
        read_json(REPORTS / "hive_operator_status.json", {}),
        read_json(REPORTS / "hive_peers.json", {}),
    ]
    for payload in payloads:
        nodes = remote_nodes(payload)
        for node in nodes:
            for slot in node.get("slots") or []:
                if not isinstance(slot, dict):
                    continue
                slot_type = str(slot.get("slot_type") or "")
                task_kinds = {str(item) for item in (slot.get("task_kinds") or [])}
                if slot_type not in {"cuda", "mlx", "mlx_apple", "mlx_cuda"}:
                    continue
                if not (task_kinds & {"cuda_eval_chunk", "cuda_training_chunk", "cuda_rollout_chunk", "mlx_eval_chunk", "mlx_training_chunk", "mlx_rollout_chunk"}):
                    continue
                if slot.get("available") is False:
                    continue
                return {
                    "available": True,
                    "node_id": node.get("node_id"),
                    "node_name": node.get("node_name"),
                    "api_url": node.get("api_url"),
                    "slot_type": slot_type,
                    "task_kinds": sorted(task_kinds),
                }
    return {"available": False}


def remote_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        return []
    hive = payload.get("hive") if isinstance(payload.get("hive"), dict) else {}
    if hive:
        local_id = str(get_path(hive, ["local_node", "node_id"], ""))
        return [
            node
            for node in (hive.get("peers") or [])
            if isinstance(node, dict) and str(node.get("node_id") or "") and str(node.get("node_id") or "") != local_id
        ]
    local_id = str(get_path(payload, ["local_node", "node_id"], ""))
    return [
        node
        for node in (payload.get("peers") or [])
        if isinstance(node, dict) and str(node.get("node_id") or "") and str(node.get("node_id") or "") != local_id
    ]


def gate(name: str, passed: bool, evidence: Any, *, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def signal(kind: str, before: Any, after: Any) -> dict[str, Any]:
    return {"kind": kind, "before": before, "after": after}


def improved(after: Any, before: Any) -> bool:
    return is_number(after) and is_number(before) and float(after) > float(before) + 1e-9


def lower(after: Any, before: Any) -> bool:
    return is_number(after) and is_number(before) and float(after) < float(before) - 1e-9


def increased(after: Any, before: Any) -> bool:
    return is_number(after) and is_number(before) and float(after) > float(before)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def number_or_negative(value: Any) -> float:
    if is_number(value):
        return float(value)
    return -1.0


def stop_requested() -> bool:
    return any(path.exists() for path in STOP_FLAGS)


def paused() -> bool:
    return any(path.exists() for path in PAUSE_FLAGS)


def sleep_or_stop(seconds: int) -> None:
    deadline = time.time() + max(1, seconds)
    while time.time() < deadline and not stop_requested():
        time.sleep(min(5, max(0.0, deadline - time.time())))


def cycle_trigger(cycle: dict[str, Any]) -> str:
    return str(cycle.get("trigger_state") or "YELLOW")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
