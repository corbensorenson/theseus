#!/usr/bin/env python3
"""High-level efficiency audit for Theseus training and benchmarking paths.

This report is intentionally operational: it looks for iteration-rate killers
that can waste hours, hide active work, or keep expensive loops CPU/control
bound. It does not launch training, mutate services, or run public calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_process_guard import windows_active_code_lm_process_rows  # noqa: E402

REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "system_efficiency_audit.json"
DEFAULT_MARKDOWN = REPORTS / "system_efficiency_audit.md"
PUBLIC_CALIBRATION_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"
SOURCE_ROOTS = [
    ROOT / "scripts",
    ROOT / "crates" / "symliquid-cli" / "src",
]
SOURCE_SUFFIX_LIMITS = {
    ".py": 1800,
    ".rs": 2200,
}
HARD_SOURCE_SUFFIX_LIMITS = {
    ".py": 3200,
    ".rs": 4200,
}
IGNORED_SOURCE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "target",
    "venv",
    ".venv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    reports = collect_reports()
    processes = process_snapshot()
    current_speed_proof = build_current_speed_proof(reports)
    public_calibration_operator_lock = public_calibration_operator_lock_state()
    discovered_bottlenecks = discover_loop_bottlenecks(reports, current_speed_proof)
    runtime_bottlenecks, stale_control_debt = partition_stale_control_debt(
        discovered_bottlenecks,
        current_speed_proof,
    )
    module_health = scan_module_health(reports["attd"])
    attd_alignment = build_attd_alignment(reports["attd"], reports["attd_packets"], runtime_bottlenecks, module_health)
    cleanup_queue = build_architecture_cleanup_queue(attd_alignment, module_health)
    findings = build_findings(reports, processes, runtime_bottlenecks, module_health, attd_alignment)
    trigger_state = "RED" if any(f["severity"] == "RED" for f in findings) else (
        "YELLOW" if any(f["severity"] == "YELLOW" for f in findings) else "GREEN"
    )
    report = {
        "policy": "project_theseus_system_efficiency_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "finding_count": len(findings),
            "red_count": sum(1 for f in findings if f["severity"] == "RED"),
            "yellow_count": sum(1 for f in findings if f["severity"] == "YELLOW"),
            "active_code_lm_process_count": len(processes),
            "duplicate_service_count": int(get_path(reports["service_hygiene"], ["summary", "duplicate_service_count"], 0)),
            "loop_bottleneck_count": len(runtime_bottlenecks),
            "top_loop_bottleneck": runtime_bottlenecks[0]["id"] if runtime_bottlenecks else "",
            "stale_control_debt_count": len(stale_control_debt),
            "top_stale_control_debt": stale_control_debt[0]["id"] if stale_control_debt else "",
            "maintainability_hotspot_count": len(module_health["hotspots"]),
            "hard_maintainability_hotspot_count": len([row for row in module_health["hotspots"] if row["severity"] == "RED"]),
            "maintainability_score": module_health["summary"]["maintainability_score"],
            "attd_trigger_state": attd_alignment["summary"]["trigger_state"],
            "attd_score": attd_alignment["summary"]["attd_score"],
            "attd_packet_count": attd_alignment["summary"]["packet_count"],
            "attd_top_component": attd_alignment["summary"]["top_component"],
            "attd_runtime_overlap_count": attd_alignment["summary"]["runtime_overlap_count"],
            "architecture_cleanup_queue_count": len(cleanup_queue),
            "top_architecture_cleanup_item": cleanup_queue[0]["id"] if cleanup_queue else "",
            "current_speed_proof_ready": bool(current_speed_proof.get("ready")),
            "current_repair_public_generation_ms": current_speed_proof.get("repair_public_generation_ms"),
            "current_repair_public_ms_per_task": current_speed_proof.get("repair_public_ms_per_task"),
            "current_lazy_beam_public_generation_ms": current_speed_proof.get("lazy_beam_public_generation_ms"),
            "current_lazy_beam_public_ms_per_task": current_speed_proof.get("lazy_beam_public_ms_per_task"),
            "current_same_seed_comparator_public_generation_ms": current_speed_proof.get(
                "same_seed_comparator_public_generation_ms"
            ),
            "current_same_seed_comparator_public_ms_per_task": current_speed_proof.get(
                "same_seed_comparator_public_ms_per_task"
            ),
            "current_public_limit8_generation_ms": current_speed_proof.get("public_limit8_generation_ms"),
            "current_public_limit8_ms_per_task": current_speed_proof.get("public_limit8_ms_per_task"),
            "current_public_limit8_task_coverage": current_speed_proof.get("public_limit8_task_coverage"),
            "current_repair_no_admissible_task_rate": current_speed_proof.get("current_no_admissible_task_rate"),
            "legacy_code_lm_closure_superseded": bool(current_speed_proof.get("legacy_closure_superseded")),
            "public_calibration_operator_locked": public_calibration_operator_lock["active"],
            "public_calibration_locked": public_calibration_operator_lock["active"]
            or not bool(get_path(reports["private_public_transfer"], ["ready_for_public_calibration"], False))
            or not bool(get_path(reports["decoder_gate"], ["ready_for_public_calibration"], False)),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "findings": findings,
        "loop_bottlenecks": runtime_bottlenecks,
        "stale_control_debt": stale_control_debt,
        "current_speed_proof": current_speed_proof,
        "public_calibration_operator_lock": public_calibration_operator_lock,
        "module_health": module_health,
        "attd_alignment": attd_alignment,
        "architecture_cleanup_queue": cleanup_queue,
        "automation_policy": {
            "run_on_watchdog": True,
            "optimize_before_long_training": True,
            "run_attd_before_architecture_changes": True,
            "prioritize_attd_packets_that_overlap_runtime_bottlenecks": True,
            "public_calibration_requires_green_private_gates": True,
            "ai_maintainability_rule": "files above soft line limits become refactor candidates; hard-limit files block architecture promotion",
        },
        "big_gain_order": [
            "profile the slowest phase first using report timing and live process evidence",
            "choose refactors where ATTD maintenance packets overlap runtime bottlenecks",
            "make lint/compile/runtime/behavior verification staged and parallel before any public or private full benchmark",
            "split hard-limit source files into bounded modules with single-responsibility ownership",
            "keep duplicate heavy work impossible through process/lease detection",
            "move STS conditioning and candidate ranker prefilter into CUDA/batched resident paths",
            "make train-once checkpoint fanout the only normal Code LM path",
            "convert hot JSONL training artifacts to binary sidecars with JSON report views",
            "run public benchmarks only after private decoder and transfer gates are green",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def collect_reports() -> dict[str, Any]:
    return {
        "resource": read_json(REPORTS / "resource_aware_execution_policy.json", {}),
        "resource_governor": read_json(REPORTS / "resource_governor.json", {}),
        "performance": read_json(REPORTS / "performance_optimizer.json", {}),
        "hive_scheduler": read_json(REPORTS / "hive_scheduler.json", {}),
        "service_hygiene": read_json(REPORTS / "service_process_hygiene.json", {}),
        "train_once": read_json(REPORTS / "code_lm_train_once_fanout.json", {}),
        "real_code": read_json(REPORTS / "real_code_benchmark_graduation.json", {}),
        "code_lm_private": read_json(REPORTS / "code_lm_closure_private_pressure_private.json", {}),
        "code_lm_recovery": read_json(REPORTS / "code_lm_closure_private_pressure_private_recovery.json", {}),
        "code_lm_rust_recovery": read_json(REPORTS / "code_lm_closure_rust_private_pressure_private_recovery.json", {}),
        "attd": read_json(REPORTS / "attd_report.json", {}),
        "attd_packets": read_json(REPORTS / "attd_maintenance_packets.json", {}),
        "shard_audit": read_json(REPORTS / "code_lm_shard_strategy_audit.json", {}),
        "decoder_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {}),
        "private_public_transfer": read_json(REPORTS / "private_public_transfer_proof.json", {}),
        "broad_transfer": read_json(REPORTS / "broad_transfer_matrix.json", {}),
        "maturity": read_json(REPORTS / "maturity_integrity_audit.json", {}),
        "same_surface_repair": read_json(REPORTS / "code_lm_same_surface_repair_proof_train_once_v1.json", {}),
        "no_admissible_repair_smoke": read_json(REPORTS / "code_lm_fanout_train_once_no_admissible_repair_12.json", {}),
        "parse_music_repair_smoke": read_json(REPORTS / "code_lm_fanout_parse_music_repair_2.json", {}),
        "lazy_beam_smoke": read_latest_json("code_lm_fanout_lazy_beam_smoke*.json", {}),
        "same_seed_comparator_smoke": read_latest_json("code_lm_fanout_same_seed_sts_comparator_smoke*.json", {}),
        "train_once_current_source_smoke": read_latest_json(
            "code_lm_closure_rust_*_current_source_smoke_fanout.json",
            {},
        ),
        "public_limit8_smoke": read_latest_json_any(
            [
                "code_lm_fanout_lazy_beam_precompute_skip_smoke*.json",
                "code_lm_fanout_public_low_latency_limit8_smoke*.json",
            ],
            {},
        ),
        "private_scale_smoke": read_latest_json("code_lm_fanout_fast_path_scale_smoke*_private_only.json", {}),
    }


def build_current_speed_proof(reports: dict[str, Any]) -> dict[str, Any]:
    train_once = reports["train_once"] if isinstance(reports.get("train_once"), dict) else {}
    train_once_summary = train_once.get("summary") if isinstance(train_once.get("summary"), dict) else {}
    train_once_architecture = (
        train_once.get("architecture") if isinstance(train_once.get("architecture"), dict) else {}
    )
    same_surface = reports["same_surface_repair"] if isinstance(reports.get("same_surface_repair"), dict) else {}
    same_summary = same_surface.get("summary") if isinstance(same_surface.get("summary"), dict) else {}
    repair_smoke = reports["no_admissible_repair_smoke"] if isinstance(reports.get("no_admissible_repair_smoke"), dict) else {}
    parse_music_smoke = reports["parse_music_repair_smoke"] if isinstance(reports.get("parse_music_repair_smoke"), dict) else {}
    lazy_beam_smoke = reports["lazy_beam_smoke"] if isinstance(reports.get("lazy_beam_smoke"), dict) else {}
    same_seed_comparator_smoke = (
        reports["same_seed_comparator_smoke"]
        if isinstance(reports.get("same_seed_comparator_smoke"), dict)
        else {}
    )
    private_scale_smoke = (
        reports["private_scale_smoke"]
        if isinstance(reports.get("private_scale_smoke"), dict)
        else {}
    )
    public_limit8_smoke = (
        reports["public_limit8_smoke"]
        if isinstance(reports.get("public_limit8_smoke"), dict)
        else {}
    )
    train_once_current_source_smoke = (
        reports["train_once_current_source_smoke"]
        if isinstance(reports.get("train_once_current_source_smoke"), dict)
        else {}
    )
    private_scale_source_kind = "explicit_fast_path_scale_smoke"
    if not private_scale_smoke and train_once_current_source_smoke:
        private_scale_smoke = train_once_current_source_smoke
        private_scale_source_kind = "train_once_current_source_private_smoke"

    repair_public_ms = number(get_path(repair_smoke, ["phase_timing_ms", "public_candidate_generation_and_write"]))
    repair_public_tasks = int(number(get_path(repair_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"])))
    repair_ms_per_task = round(repair_public_ms / max(1, repair_public_tasks), 3) if repair_public_ms else 0.0
    parse_music_public_ms = number(get_path(parse_music_smoke, ["phase_timing_ms", "public_candidate_generation_and_write"]))
    lazy_beam_public_ms = number(get_path(lazy_beam_smoke, ["summary", "phase_timing_ms", "public_candidate_generation_and_write"]))
    lazy_beam_public_tasks = int(number(get_path(lazy_beam_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"])))
    lazy_beam_ms_per_task = (
        round(lazy_beam_public_ms / max(1, lazy_beam_public_tasks), 3) if lazy_beam_public_ms else 0.0
    )
    same_seed_comparator_public_ms = number(
        get_path(same_seed_comparator_smoke, ["phase_timing_ms", "public_candidate_generation_and_write"])
    )
    same_seed_comparator_public_tasks = int(
        number(get_path(same_seed_comparator_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"]))
    )
    same_seed_comparator_ms_per_task = (
        round(same_seed_comparator_public_ms / max(1, same_seed_comparator_public_tasks), 3)
        if same_seed_comparator_public_ms
        else 0.0
    )
    historical_public_ms = number(
        get_path(train_once_summary, ["phase_timing_ms", "fanout_report", "public_candidate_generation_and_write"])
    )
    historical_public_tasks = int(number(get_path(train_once_summary, ["public_candidate_manifest_diagnostics", "task_count"])))
    if not historical_public_tasks:
        historical_public_tasks = int(number(get_path(train_once_summary, ["public_candidate_count"])))
    historical_ms_per_task = round(historical_public_ms / max(1, historical_public_tasks), 3) if historical_public_ms else 0.0
    speedup_vs_historical_public = (
        round(historical_ms_per_task / max(0.001, repair_ms_per_task), 3)
        if historical_ms_per_task and repair_ms_per_task
        else 0.0
    )
    same_surface_ready = bool(
        same_surface.get("trigger_state") == "GREEN"
        and same_surface.get("ready_for_transfer_proof_receiver_surface") is True
        and same_summary.get("same_task_surface") is True
    )
    no_admissible_clean = float(number(same_summary.get("current_no_admissible_task_rate"))) == 0.0
    repair_speed_ready = (
        repair_smoke.get("trigger_state") == "GREEN"
        and repair_public_tasks > 0
        and repair_public_ms > 0
        and repair_public_ms <= 30_000
    )
    lazy_beam_speed_ready = (
        lazy_beam_smoke.get("trigger_state") == "GREEN"
        and lazy_beam_public_tasks > 0
        and lazy_beam_public_ms > 0
        and lazy_beam_public_ms <= 1_000
    )
    same_seed_comparator_speed_ready = (
        same_seed_comparator_smoke.get("trigger_state") == "GREEN"
        and same_seed_comparator_public_tasks >= 16
        and same_seed_comparator_public_ms > 0
        and same_seed_comparator_ms_per_task <= 100.0
    )
    private_scale_private_ms = number(
        get_path(private_scale_smoke, ["phase_timing_ms", "private_candidate_generation_and_write"])
        or get_path(private_scale_smoke, ["summary", "phase_timing_ms", "private_candidate_generation_and_write"])
    )
    private_scale_tasks = int(
        number(get_path(private_scale_smoke, ["summary", "candidate_task_timing_summary", "private", "task_count"]))
    )
    private_scale_public_tasks = int(
        number(get_path(private_scale_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"]))
    )
    private_scale_ms_per_task = (
        round(private_scale_private_ms / max(1, private_scale_tasks), 3)
        if private_scale_private_ms
        else 0.0
    )
    private_scale_token_candidates = int(number(get_path(private_scale_smoke, ["summary", "private_token_level_candidate_count"])))
    private_scale_no_admissible_rate = (
        round(max(0, private_scale_tasks - private_scale_token_candidates) / max(1, private_scale_tasks), 6)
        if private_scale_tasks
        else 0.0
    )
    private_scale_speed_ready = (
        private_scale_smoke.get("run_status") == "completed"
        and private_scale_tasks >= 16
        and private_scale_public_tasks == 0
        and private_scale_private_ms > 0
        and private_scale_ms_per_task <= 1_500.0
        and int(number(private_scale_smoke.get("external_inference_calls"))) == 0
    )
    public_limit8_ms = number(
        get_path(public_limit8_smoke, ["summary", "phase_timing_ms", "public_candidate_generation_and_write"])
    )
    public_limit8_tasks = int(
        number(get_path(public_limit8_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"]))
    )
    public_limit8_candidates = int(number(get_path(public_limit8_smoke, ["summary", "public_candidate_count"])))
    public_limit8_template_like = int(number(get_path(public_limit8_smoke, ["summary", "template_like_candidate_count"])))
    public_limit8_ms_per_task = (
        round(public_limit8_ms / max(1, public_limit8_tasks), 3)
        if public_limit8_ms
        else 0.0
    )
    public_limit8_task_coverage = (
        round(public_limit8_candidates / max(1, public_limit8_tasks), 3)
        if public_limit8_tasks
        else 0.0
    )
    public_limit8_speed_ready = (
        public_limit8_smoke.get("trigger_state") == "GREEN"
        and public_limit8_tasks >= 16
        and public_limit8_ms > 0
        and public_limit8_ms_per_task <= 250.0
        and public_limit8_candidates >= public_limit8_tasks
        and public_limit8_template_like == 0
        and int(number(public_limit8_smoke.get("external_inference_calls"))) == 0
    )
    current_source_public_ms = number(
        get_path(train_once_current_source_smoke, ["summary", "phase_timing_ms", "public_candidate_generation_and_write"])
    )
    current_source_private_ms = number(
        get_path(train_once_current_source_smoke, ["summary", "phase_timing_ms", "private_candidate_generation_and_write"])
    )
    current_source_public_tasks = int(
        number(get_path(train_once_current_source_smoke, ["summary", "candidate_task_timing_summary", "public", "task_count"]))
    )
    current_source_private_tasks = int(
        number(get_path(train_once_current_source_smoke, ["summary", "candidate_task_timing_summary", "private", "task_count"]))
    )
    current_source_candidates = int(number(get_path(train_once_current_source_smoke, ["summary", "public_candidate_count"]))) + int(
        number(get_path(train_once_current_source_smoke, ["summary", "private_candidate_count"]))
    )
    train_once_current_source_smoke_ready = bool(
        train_once_current_source_smoke.get("run_status") == "completed"
        and current_source_candidates > 0
        and int(number(train_once_current_source_smoke.get("external_inference_calls"))) == 0
    )
    train_once_checkpoint_fanout = bool(
        train_once_summary.get("train_once_checkpoint_fanout") is True
        or str(train_once_architecture.get("fanout") or "").startswith("candidate generation from checkpoint")
    )
    train_once_repeated_training_blocked = bool(
        train_once_summary.get("repeated_training_per_candidate_shard") is False
        or train_once_architecture.get("repeated_training_per_candidate_shard") is False
    )
    train_once_envelope_present = bool(
        train_once.get("policy") == "project_theseus_code_lm_train_once_fanout_v1"
        and train_once_checkpoint_fanout
        and train_once_repeated_training_blocked
    )
    transfer_gates_ready = bool(
        reports["decoder_gate"].get("ready_for_public_calibration") is True
        and reports["private_public_transfer"].get("ready_for_public_calibration") is True
    )
    legacy_superseded = bool(
        train_once_envelope_present
        and same_surface_ready
        and no_admissible_clean
        and repair_speed_ready
        and (lazy_beam_speed_ready or not lazy_beam_smoke)
        and (same_seed_comparator_speed_ready or not same_seed_comparator_smoke)
        and (public_limit8_speed_ready or not public_limit8_smoke)
        and (private_scale_speed_ready or not private_scale_smoke)
    )
    return {
        "policy": "project_theseus_current_code_lm_speed_proof_v1",
        "ready": bool(
            legacy_superseded
        ),
        "legacy_closure_superseded": legacy_superseded,
        "train_once_envelope_present": train_once_envelope_present,
        "train_once_checkpoint_fanout": train_once_checkpoint_fanout,
        "train_once_repeated_training_blocked": train_once_repeated_training_blocked,
        "transfer_gates_ready": transfer_gates_ready,
        "same_surface_repair_ready": same_surface_ready,
        "same_surface_task_count": same_summary.get("task_count"),
        "actual_token_task_coverage_delta": same_summary.get("actual_token_task_coverage_delta"),
        "eligible_task_coverage_delta": same_summary.get("eligible_task_coverage_delta"),
        "no_admissible_task_rate_delta": same_summary.get("no_admissible_task_rate_delta"),
        "current_no_admissible_task_rate": same_summary.get("current_no_admissible_task_rate"),
        "repair_smoke_trigger_state": repair_smoke.get("trigger_state"),
        "repair_public_generation_ms": int(repair_public_ms),
        "repair_public_task_count": repair_public_tasks,
        "repair_public_ms_per_task": repair_ms_per_task,
        "parse_music_repair_public_generation_ms": int(parse_music_public_ms),
        "lazy_beam_smoke_report": lazy_beam_smoke.get("_source_report", ""),
        "lazy_beam_speed_ready": lazy_beam_speed_ready,
        "lazy_beam_public_generation_ms": int(lazy_beam_public_ms),
        "lazy_beam_public_task_count": lazy_beam_public_tasks,
        "lazy_beam_public_ms_per_task": lazy_beam_ms_per_task,
        "lazy_beam_public_speedup_vs_previous_repair_per_task": (
            round(repair_ms_per_task / max(0.001, lazy_beam_ms_per_task), 3)
            if repair_ms_per_task and lazy_beam_ms_per_task
            else 0.0
        ),
        "same_seed_comparator_smoke_report": same_seed_comparator_smoke.get("_source_report", ""),
        "same_seed_comparator_speed_ready": same_seed_comparator_speed_ready,
        "same_seed_comparator_public_generation_ms": int(same_seed_comparator_public_ms),
        "same_seed_comparator_public_task_count": same_seed_comparator_public_tasks,
        "same_seed_comparator_public_ms_per_task": same_seed_comparator_ms_per_task,
        "same_seed_comparator_speedup_vs_previous_repair_per_task": (
            round(repair_ms_per_task / max(0.001, same_seed_comparator_ms_per_task), 3)
            if repair_ms_per_task and same_seed_comparator_ms_per_task
            else 0.0
        ),
        "public_limit8_smoke_report": public_limit8_smoke.get("_source_report", ""),
        "public_limit8_speed_ready": public_limit8_speed_ready,
        "public_limit8_generation_ms": int(public_limit8_ms),
        "public_limit8_task_count": public_limit8_tasks,
        "public_limit8_candidate_count": public_limit8_candidates,
        "public_limit8_ms_per_task": public_limit8_ms_per_task,
        "public_limit8_task_coverage": public_limit8_task_coverage,
        "public_limit8_template_like_candidate_count": public_limit8_template_like,
        "train_once_current_source_smoke_report": train_once_current_source_smoke.get("_source_report", ""),
        "train_once_current_source_smoke_ready": train_once_current_source_smoke_ready,
        "train_once_current_source_public_generation_ms": int(current_source_public_ms),
        "train_once_current_source_private_generation_ms": int(current_source_private_ms),
        "train_once_current_source_public_task_count": current_source_public_tasks,
        "train_once_current_source_private_task_count": current_source_private_tasks,
        "private_scale_smoke_report": private_scale_smoke.get("_source_report", ""),
        "private_scale_smoke_source_kind": private_scale_source_kind if private_scale_smoke else "",
        "private_scale_speed_ready": private_scale_speed_ready,
        "private_scale_private_generation_ms": int(private_scale_private_ms),
        "private_scale_private_task_count": private_scale_tasks,
        "private_scale_private_ms_per_task": private_scale_ms_per_task,
        "private_scale_public_task_count": private_scale_public_tasks,
        "private_scale_no_admissible_task_rate": private_scale_no_admissible_rate,
        "private_scale_top_timing_ms_total": get_path(
            private_scale_smoke,
            ["summary", "candidate_task_timing_summary", "private", "top_timing_ms_total"],
            {},
        ),
        "historical_train_once_public_generation_ms": int(historical_public_ms),
        "historical_train_once_public_task_count": historical_public_tasks,
        "historical_train_once_public_ms_per_task": historical_ms_per_task,
        "bounded_repair_speedup_vs_historical_public_per_task": speedup_vs_historical_public,
        "score_semantics": (
            "bounded no-admissible/interface repair speed evidence; this proves the old monolithic closure is stale "
            "as a control-state blocker, but full train-once fanout timing remains a separate optimization target"
        ),
        "external_inference_calls": 0,
    }


def build_findings(
    reports: dict[str, Any],
    processes: list[dict[str, Any]],
    runtime_bottlenecks: list[dict[str, Any]],
    module_health: dict[str, Any],
    attd_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    resource = reports["resource"]
    resource_summary = resource.get("summary") if isinstance(resource.get("summary"), dict) else {}
    budget = resource.get("recommended_code_lm_budget") if isinstance(resource.get("recommended_code_lm_budget"), dict) else {}
    if processes and bool(budget.get("start_new_train_once_fanout") or budget.get("start_new_code_closure")):
        findings.append(issue(
            "RED",
            "duplicate_code_lm_launch_risk",
            "Resource policy would allow another Code LM worker while one is already active.",
            "Block new Code LM work whenever train-once/fanout, closure, STS, cargo, or rustc workers are alive.",
        ))
    if bool(resource_summary.get("code_lm_cpu_bound_hot_path_active")):
        findings.append(issue(
            "YELLOW",
            "cpu_bound_code_lm_subphase",
            f"Active stage={resource_summary.get('active_code_lm_stage')} gpu_util={resource_summary.get('gpu_utilization_percent')}%",
            "Move this subphase to CUDA/batched kernels or keep it short and explicitly reported as CPU-control work.",
        ))
    if bool(resource_summary.get("code_lm_gpu_low_utilization_bottleneck")):
        findings.append(issue(
            "YELLOW",
            "low_gpu_utilization_during_code_lm",
            str(resource_summary.get("code_lm_gpu_low_utilization_reason")),
            "Do not hide low utilization behind stale heartbeats; profile and port the active stage.",
        ))
    duplicate_artifacts = resource_summary.get("duplicate_code_lm_artifact_targets")
    duplicate_artifact_count = int(number(resource_summary.get("duplicate_code_lm_artifact_target_count")))
    if duplicate_artifact_count > 0:
        findings.append(issue(
            "RED",
            "duplicate_code_lm_artifact_targets",
            f"duplicate_target_groups={duplicate_artifact_count} examples={duplicate_artifacts}",
            "Prevent two launchers from writing the same Code LM checkpoint/candidate/report artifacts; this is wasted compute and corrupts timing evidence.",
        ))

    attd_summary = attd_alignment.get("summary") if isinstance(attd_alignment.get("summary"), dict) else {}
    attd_state = str(attd_summary.get("trigger_state") or "MISSING").upper()
    if attd_state in {"RED", "YELLOW"}:
        findings.append(issue(
            "RED" if attd_state == "RED" else "YELLOW",
            "attd_technical_debt_pressure",
            (
                f"attd_state={attd_state} score={attd_summary.get('attd_score')} "
                f"top_component={attd_summary.get('top_component')} packets={attd_summary.get('packet_count')}"
            ),
            "Use ATTD maintenance packets as architecture-work inputs; do not add capacity on top of rising assembly debt.",
        ))
    if int(number(attd_summary.get("runtime_overlap_count"))) > 0:
        top = attd_alignment.get("runtime_overlaps", [{}])[0]
        findings.append(issue(
            "YELLOW",
            "attd_runtime_bottleneck_overlap",
            f"{top.get('bottleneck_id')} overlaps {top.get('packet_id')} scope={top.get('overlap_scope')}",
            "Prioritize this refactor because it should reduce both wall-clock time and autonomous-change risk.",
        ))

    for bottleneck in runtime_bottlenecks[:6]:
        findings.append(issue(
            str(bottleneck["severity"]),
            str(bottleneck["id"]),
            str(bottleneck["evidence"]),
            str(bottleneck["recommended_action"]),
        ))

    hard_hotspots = [row for row in module_health["hotspots"] if row["severity"] == "RED"]
    soft_hotspots = [row for row in module_health["hotspots"] if row["severity"] == "YELLOW"]
    if hard_hotspots:
        top = hard_hotspots[0]
        findings.append(issue(
            "RED",
            "hard_ai_maintainability_hotspot",
            f"{top['path']} has {top['line_count']} lines; hard_limit={top['hard_limit']}",
            "Split hard-limit source files before adding more behavior; preserve public interfaces and move coherent helpers into owned modules.",
        ))
    elif soft_hotspots:
        top = soft_hotspots[0]
        findings.append(issue(
            "YELLOW",
            "ai_maintainability_hotspot",
            f"{top['path']} has {top['line_count']} lines; soft_limit={top['soft_limit']}",
            "Queue modularization before the next architecture patch touches this file.",
        ))

    duplicates = int(get_path(reports["service_hygiene"], ["summary", "duplicate_service_count"], 0))
    if duplicates:
        guard_count = int(get_path(reports["service_hygiene"], ["summary", "duplicate_launch_guard_count"], 0))
        guard_expected = int(get_path(reports["service_hygiene"], ["summary", "duplicate_launch_guard_expected_count"], 0))
        if guard_expected and guard_count >= guard_expected:
            action = (
                "Current duplicate processes remain live; future dashboard singleton starts are guarded. "
                "Only dedupe live services with explicit operator approval or during a planned restart."
            )
        else:
            action = "Future launchers should discover existing service processes by command line before starting a new one."
        findings.append(issue(
            "YELLOW",
            "duplicate_long_running_services",
            f"duplicate_service_count={duplicates} guarded_launch_paths={guard_count}/{guard_expected}",
            action,
        ))

    shard = reports["shard_audit"]
    if bool(get_path(shard, ["summary", "repeats_training"], False)):
        findings.append(issue(
            "RED",
            "repeated_training_shards",
            "Shard audit says shards repeat training.",
            "Use train-once checkpoint fanout; keep shards only as crash recovery/diagnostic evidence.",
        ))
    if str(reports["train_once"].get("trigger_state") or "").upper() in {"MISSING", "RED"}:
        findings.append(issue(
            "RED",
            "train_once_fanout_missing",
            "Preferred Code LM execution envelope is not available.",
            "Repair train-once checkpoint fanout before any long Code LM run.",
        ))

    decoder_ready = bool(reports["decoder_gate"].get("ready_for_public_calibration"))
    transfer_ready = bool(reports["private_public_transfer"].get("ready_for_public_calibration"))
    if not decoder_ready or not transfer_ready:
        findings.append(issue(
            "YELLOW",
            "public_calibration_locked",
            f"decoder_ready={decoder_ready} transfer_ready={transfer_ready}",
            "Keep public benchmarks calibration-only and spend cycles on private gate/fanout quality.",
        ))

    broad = reports["broad_transfer"]
    pass_rate = number(get_path(broad, ["summary", "pass_rate"], get_path(broad, ["pass_rate"], 0.0)))
    if pass_rate and pass_rate < 0.70:
        findings.append(issue(
            "YELLOW",
            "broad_public_transfer_below_floor",
            f"pass_rate={pass_rate}",
            "Treat this as a transfer-proof bottleneck, not a reason to rerun public calibration loops.",
        ))

    c_free = number(resource_summary.get("c_free_gb"))
    d_free = number(resource_summary.get("d_free_gb"))
    if c_free and c_free < 50.0 and d_free > 200.0:
        findings.append(issue(
            "YELLOW",
            "root_disk_near_iteration_floor",
            f"C_free_gb={c_free} D_free_gb={d_free}",
            "Keep large runtime/training artifacts on D: and leave C: for source, reports, and small manifests.",
        ))
    return findings


def discover_loop_bottlenecks(reports: dict[str, Any], current_speed_proof: dict[str, Any]) -> list[dict[str, Any]]:
    bottlenecks: list[dict[str, Any]] = []
    add_train_once_bottlenecks(bottlenecks, reports["train_once"], current_speed_proof)
    add_real_code_bottlenecks(bottlenecks, reports["real_code"])
    legacy_superseded = bool(current_speed_proof.get("legacy_closure_superseded"))
    add_code_lm_closure_bottlenecks(bottlenecks, reports["code_lm_private"], "code_lm_private", legacy_superseded)
    add_code_lm_closure_bottlenecks(bottlenecks, reports["code_lm_recovery"], "code_lm_recovery", legacy_superseded)
    add_code_lm_closure_bottlenecks(bottlenecks, reports["code_lm_rust_recovery"], "code_lm_rust_recovery", legacy_superseded)
    add_performance_bottlenecks(bottlenecks, reports["performance"], reports)
    bottlenecks.sort(key=lambda row: (-float(row["impact_score"]), str(row["id"])))
    return bottlenecks


def partition_stale_control_debt(
    bottlenecks: list[dict[str, Any]],
    current_speed_proof: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep old slow reports visible without treating them as current loop work."""
    has_current_path_evidence = bool(
        current_speed_proof.get("legacy_closure_superseded")
        or current_speed_proof.get("train_once_current_source_smoke_ready")
        or current_speed_proof.get("private_scale_speed_ready")
    )
    if not has_current_path_evidence:
        return bottlenecks, []

    active: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    stale_ids = {
        "code_lm_private_wall_clock_too_large",
        "code_lm_recovery_wall_clock_too_large",
        "code_lm_rust_recovery_wall_clock_too_large",
    }
    if bool(current_speed_proof.get("train_once_current_source_smoke_ready")):
        stale_ids.add("train_once_fanout_stale_current_source")
    for row in bottlenecks:
        row_id = str(row.get("id") or "")
        is_historical_train_once = row_id.startswith("historical_slow_train_once_")
        is_superseded_closure = row_id in stale_ids
        if is_historical_train_once or is_superseded_closure:
            debt = dict(row)
            debt["classification"] = "stale_control_debt"
            debt["superseded_by_current_speed_proof"] = True
            debt["recommended_action"] = (
                "Keep as diagnostic history; do not select this as overnight work unless a fresh current-path "
                "scale smoke regresses or routing falls back to the old closure."
            )
            stale.append(debt)
        else:
            active.append(row)
    stale.sort(key=lambda row: (-float(row["impact_score"]), str(row["id"])))
    active.sort(key=lambda row: (-float(row["impact_score"]), str(row["id"])))
    return active, stale


def add_train_once_bottlenecks(
    out: list[dict[str, Any]],
    report: dict[str, Any],
    current_speed_proof: dict[str, Any],
) -> None:
    if not isinstance(report, dict) or not report:
        out.append(loop_issue(
            "YELLOW",
            "train_once_fanout_report_missing",
            0.60,
            "reports/code_lm_train_once_fanout.json is missing or empty",
            "Keep train-once fanout as the normal Code LM path and refuse repeated-training shards.",
            "code_lm_train_once_fanout",
        ))
        return
    control = report.get("control_signal_contract")
    if not isinstance(control, dict):
        control = get_path(report, ["summary", "control_signal_contract"], {})
    consumers = control.get("consumers") if isinstance(control, dict) and isinstance(control.get("consumers"), list) else []
    if len(consumers) < 3:
        out.append(loop_issue(
            "YELLOW",
            "train_once_report_missing_control_consumers",
            0.35,
            f"consumer_count={len(consumers)}",
            "Every report that drives autonomy must name consumers and measured effect fields, or be explicitly diagnostic-only.",
            "code_lm_train_once_fanout",
        ))
    verification = report.get("staged_verification_contract")
    if not isinstance(verification, list):
        verification = get_path(report, ["summary", "staged_verification_contract"], [])
    verification_stages = [str(row.get("stage") or "") for row in verification if isinstance(row, dict)]
    required_stages = {"lint_parse", "compile_or_import", "cheap_behavior", "sandbox_full_tests"}
    if not required_stages.issubset(set(verification_stages)):
        out.append(loop_issue(
            "YELLOW",
            "train_once_missing_staged_verification_contract",
            0.45,
            f"stages={verification_stages}",
            "Declare fail-fast lint/parse, compile/import, cheap behavior, and full sandbox stages before expensive verification.",
            "code_lm_train_once_fanout",
        ))
    phase_ledger = report.get("phase_ledger_summary")
    if not isinstance(phase_ledger, dict):
        phase_ledger = get_path(report, ["summary", "phase_ledger_summary"], {})
    if str(report.get("run_status") or "").lower() in {"completed", "running"} and not int(number(phase_ledger.get("event_count"))):
        out.append(loop_issue(
            "YELLOW",
            "train_once_phase_ledger_missing",
            0.50,
            f"run_status={report.get('run_status')} phase={report.get('current_phase')}",
            "Append phase ledger events for build, checkpoint, fanout, ranking, and verification so slow loops are optimizer targets.",
            "code_lm_train_once_fanout",
        ))
    for row in phase_ledger.get("slow_phases", []) if isinstance(phase_ledger.get("slow_phases"), list) else []:
        if isinstance(row, dict):
            out.append(loop_issue(
                "YELLOW",
                f"train_once_slow_ledger_phase_{safe_id(row.get('phase'))}",
                0.65,
                f"phase={row.get('phase')} elapsed={row.get('elapsed_seconds')}s target={row.get('target_max_seconds')}s",
                "Split or port this phase before another long closure run; phase timing is control evidence, not training evidence.",
                "code_lm_train_once_fanout",
            ))
    current_phase = str(report.get("current_phase") or "")
    run_status = str(report.get("run_status") or report.get("trigger_state") or "")
    if run_status == "stale_artifacts_need_fanout_refresh":
        freshness = report.get("artifact_freshness") if isinstance(report.get("artifact_freshness"), dict) else {}
        if bool(current_speed_proof.get("train_once_current_source_smoke_ready")):
            evidence = (
                f"canonical_fresh={freshness.get('fresh')} "
                f"current_source_smoke={current_speed_proof.get('train_once_current_source_smoke_report')} "
                f"public_ms={current_speed_proof.get('train_once_current_source_public_generation_ms')} "
                f"private_ms={current_speed_proof.get('train_once_current_source_private_generation_ms')}"
            )
            action = (
                "Treat canonical stale fanout as diagnostic debt; do not spend a full refresh unless the bounded "
                "current-source smoke regresses or decoder/transfer gates explicitly need fresh full manifests."
            )
        else:
            evidence = (
                f"phase={current_phase} release_current={freshness.get('release_binary_current')} "
                f"fresh={freshness.get('fresh')}"
            )
            action = (
                "Reuse the checkpoint through the bounded current-source fanout smoke first; only run full fanout "
                "with --full-fanout-refresh after an explicit budget choice."
            )
        out.append(loop_issue(
            "YELLOW",
            "train_once_fanout_stale_current_source",
            0.70,
            evidence,
            action,
            "code_lm_train_once_fanout",
        ))
    if run_status.upper() == "RUNNING" and current_phase:
        heartbeat = active_phase_heartbeat_status(report, current_phase)
        if not bool(heartbeat.get("fresh")):
            out.append(loop_issue(
                "YELLOW",
                "active_train_once_phase_needs_timing",
                0.45,
                f"train_once is running phase={current_phase} heartbeat={heartbeat.get('status')} age={heartbeat.get('age_seconds')}",
                "Require phase heartbeat/progress and compare elapsed time against prior phase baselines.",
                "code_lm_train_once_fanout",
            ))
    current_fast_path_proven = bool(current_speed_proof.get("ready"))
    current_fast_path_evidence = bool(
        current_fast_path_proven
        or current_speed_proof.get("train_once_current_source_smoke_ready")
        or current_speed_proof.get("private_scale_speed_ready")
    )
    speed_context = (
        f" current_speed_proof_ready={current_fast_path_proven} "
        f"current_source_smoke={current_speed_proof.get('train_once_current_source_smoke_report')} "
        f"current_source_private_ms={current_speed_proof.get('train_once_current_source_private_generation_ms')} "
        f"current_source_private_tasks={current_speed_proof.get('train_once_current_source_private_task_count')} "
        f"lazy_public_ms_per_task={current_speed_proof.get('lazy_beam_public_ms_per_task')} "
        f"same_seed_public_ms_per_task={current_speed_proof.get('same_seed_comparator_public_ms_per_task')}"
        if current_fast_path_evidence
        else ""
    )
    stale_timing_issue_count = 0
    phase_timing = get_path(report, ["summary", "phase_timing_ms"], {})
    if isinstance(phase_timing, dict):
        for phase, value in flattened_numbers(phase_timing).items():
            if not is_runtime_timing_path(phase):
                continue
            ms = float(value)
            if ms >= 120_000:
                issue_id = f"slow_train_once_phase_{safe_id(phase)}"
                impact = min(1.0, ms / 900_000)
                evidence = f"{phase} took {int(ms)}ms"
                action = "Split this phase into measured subphases and move candidate/ranker work into resident batched paths."
                if current_fast_path_evidence:
                    stale_timing_issue_count += 1
                    issue_id = f"historical_{issue_id}"
                    impact = min(0.42, impact)
                    evidence = f"{evidence};{speed_context}"
                    action = (
                        "Treat this as stale control debt: require a current bounded scale-smoke before broad training, "
                        "and reopen as active only if the fresh fast path regresses."
                    )
                out.append(loop_issue(
                    "YELLOW",
                    issue_id,
                    impact,
                    evidence,
                    action,
                    "code_lm_train_once_fanout",
                ))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for scope, key in (
        ("private", "private_candidate_manifest_diagnostics"),
        ("public", "public_candidate_manifest_diagnostics"),
    ):
        timing = get_path(summary, [key, "candidate_task_timing_summary"], {})
        if not isinstance(timing, dict):
            continue
        top_totals = timing.get("top_timing_ms_total")
        if not isinstance(top_totals, dict):
            continue
        timing_max = timing.get("timing_ms_max") if isinstance(timing.get("timing_ms_max"), dict) else {}
        emitted = 0
        for phase, value in sorted(top_totals.items(), key=lambda item: -number(item[1])):
            if not is_runtime_timing_path(str(phase)):
                continue
            total_ms = float(number(value))
            if total_ms < 120_000:
                continue
            max_ms = int(number(timing_max.get(phase)))
            task_count = int(number(timing.get("task_count")))
            issue_id = f"slow_train_once_candidate_task_phase_{scope}_{safe_id(phase)}"
            impact = min(1.0, total_ms / 900_000)
            evidence = f"{scope} {phase} total={int(total_ms)}ms max_task={max_ms}ms tasks={task_count}"
            action = "Attack this branch inside fanout before another full closure; it is deduplicated per-task timing from emitted candidate manifests."
            if current_fast_path_evidence:
                stale_timing_issue_count += 1
                issue_id = f"historical_{issue_id}"
                impact = min(0.42, impact)
                evidence = f"{evidence};{speed_context}"
                action = (
                    "Treat this emitted manifest timing as stale debt: validate the current lazy-beam/same-seed path "
                    "with a bounded scale-smoke before broad training instead of optimizing the old trace."
                )
            out.append(loop_issue(
                "YELLOW",
                issue_id,
                impact,
                evidence,
                action,
                "code_lm_train_once_fanout",
            ))
            emitted += 1
            if emitted >= 2:
                break
    if current_fast_path_evidence and stale_timing_issue_count:
        if current_speed_proof.get("private_scale_speed_ready") is True:
            private_ms_per_task = float(number(current_speed_proof.get("private_scale_private_ms_per_task")))
            private_task_count = int(number(current_speed_proof.get("private_scale_private_task_count")))
            no_admissible_rate = float(number(current_speed_proof.get("private_scale_no_admissible_task_rate")))
            if private_ms_per_task >= 500.0:
                out.append(loop_issue(
                    "YELLOW",
                    "current_private_scale_candidate_generation_cost",
                    min(0.62, private_ms_per_task / 1500.0),
                    (
                        f"current private-only scale smoke tasks={private_task_count} "
                        f"ms_per_task={private_ms_per_task} no_admissible_rate={no_admissible_rate}"
                    ),
                    (
                        "Optimize the current fast path by batching/caching state-sequence and SymLiquid branches; "
                        "CPU should only do final contract checks."
                    ),
                    "code_lm_train_once_fanout",
                ))
            top_timing = current_speed_proof.get("private_scale_top_timing_ms_total")
            if isinstance(top_timing, dict):
                for phase, total in sorted(top_timing.items(), key=lambda item: -number(item[1]))[:2]:
                    total_ms = float(number(total))
                    per_task = round(total_ms / max(1, private_task_count), 3)
                    if total_ms >= 5_000:
                        if str(phase) == "candidate_expression_generation_ms" and private_ms_per_task <= 50.0:
                            continue
                        out.append(loop_issue(
                            "YELLOW",
                            f"current_private_scale_{safe_id(str(phase))}",
                            min(0.58, per_task / 1200.0),
                            f"{phase} total={int(total_ms)}ms per_task={per_task}",
                            (
                                "Make this branch a first-class optimizer target: precompute shared task features, "
                                "reuse decoder cache across same-category tasks, and batch state/SymLiquid expansions."
                            ),
                            "code_lm_train_once_fanout",
                        ))
        else:
            out.append(loop_issue(
                "YELLOW",
                "current_fast_path_needs_bounded_scale_smoke",
                0.58,
                (
                    "historical train-once fanout timings are stale; "
                    f"stale_timing_issue_count={stale_timing_issue_count};{speed_context}"
                ),
                (
                    "Run a bounded private-first current fast-path scale smoke before broad training; "
                    "only reopen the old fanout bottleneck if fresh same-source timing regresses."
                ),
                "code_lm_train_once_fanout",
            ))


def active_phase_heartbeat_status(report: dict[str, Any], current_phase: str) -> dict[str, Any]:
    path_text = str(report.get("active_phase_heartbeat") or "")
    slug = str(report.get("slug") or "private_pressure_private_recovery_train_once_fanout_v1")
    if not path_text:
        phase_to_name = {
            "train_once_checkpoint": "checkpoint",
            "checkpoint_fanout_candidate_generation": "fanout",
            "checkpoint_fanout_current_source_smoke": "current_source_smoke",
        }
        name = phase_to_name.get(current_phase)
        if name:
            path_text = f"reports/code_lm_train_once_fanout_{slug}_{name}.phase_heartbeat.json"
    if not path_text:
        return {"status": "missing_path", "fresh": False, "age_seconds": None}
    path = resolve(path_text)
    if not path.exists():
        return {"status": "missing_file", "fresh": False, "path": rel(path), "age_seconds": None}
    age = int(max(0.0, time.time() - path.stat().st_mtime))
    data = read_json(path, {})
    if not isinstance(data, dict):
        return {"status": "unreadable", "fresh": False, "path": rel(path), "age_seconds": age}
    status = str(data.get("status") or "")
    phase = str(data.get("phase") or "")
    return {
        "status": status or "unknown",
        "fresh": age <= 180 and phase == current_phase and status in {"running", "completed", "failed", "timed_out"},
        "path": rel(path),
        "age_seconds": age,
        "phase": phase,
        "elapsed_seconds": data.get("elapsed_seconds"),
        "progress_ratio": data.get("progress_ratio"),
        "returncode": data.get("returncode"),
        "timed_out": data.get("timed_out"),
    }


def add_real_code_bottlenecks(out: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        return
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    total_cases = int(number(summary.get("total_case_count")))
    runtime_ms = float(number(report.get("runtime_ms") or summary.get("runtime_ms")))
    workers = int(number(summary.get("verification_workers")))
    stage_contract = report.get("verification_stage_contract")
    if not isinstance(stage_contract, list):
        stage_contract = []
    stage_names = {str(row.get("stage") or "") for row in stage_contract if isinstance(row, dict)}
    required_stage_names = {
        "lint_parse",
        "beautiful_code_lint",
        "candidate_compile",
        "sandbox_runtime_load",
        "intended_behavior",
    }
    if total_cases and not required_stage_names.issubset(stage_names):
        out.append(loop_issue(
            "YELLOW",
            "real_code_missing_staged_verification_contract",
            0.45,
            f"stages={sorted(stage_names)}",
            "Benchmark reports must expose lint, quality, compile, runtime-load, and behavior stages as a reward cascade.",
            "real_code_benchmark_graduation",
        ))
    policy = report.get("parallel_verification_policy")
    if total_cases and not isinstance(policy, dict):
        out.append(loop_issue(
            "YELLOW",
            "real_code_missing_parallel_verification_policy",
            0.35,
            "parallel_verification_policy missing",
            "Benchmark reports must declare the parallel unit and worker rule so the speed governor can distinguish stale reports from true serial execution.",
            "real_code_benchmark_graduation",
        ))
    if total_cases and runtime_ms:
        ms_per_case = runtime_ms / max(1, total_cases)
        if ms_per_case >= 500:
            out.append(loop_issue(
                "YELLOW",
                "real_code_verification_slow_per_case",
                min(1.0, ms_per_case / 5000.0),
                f"real_code runtime={int(runtime_ms)}ms cases={total_cases} ms_per_case={round(ms_per_case, 2)} workers={workers}",
                "Increase verification workers, keep staged lint/compile/runtime cascade, and cache unchanged candidate/test hashes.",
                "real_code_benchmark_graduation",
            ))
        if workers <= 1 and total_cases >= 8:
            out.append(loop_issue(
                "YELLOW",
                "real_code_verification_not_parallel",
                0.55,
                f"cases={total_cases} verification_workers={workers}",
                "Run public/local verification with bounded parallel workers; CPU should judge only top candidates.",
                "real_code_benchmark_graduation",
            ))
    cascade = summary.get("verification_cascade_summary") if isinstance(summary.get("verification_cascade_summary"), dict) else {}
    skipped = int(number(cascade.get("sandbox_skipped_before_runtime_count")))
    attempts = int(number(cascade.get("candidate_attempt_count")))
    if attempts and skipped / max(1, attempts) > 0.50:
        out.append(loop_issue(
            "YELLOW",
            "candidate_failures_stop_before_runtime",
            0.50,
            f"{skipped}/{attempts} candidates never reached runtime",
            "Feed lint/compile rejection families into private candidate generation before spending sandbox or public calibration cycles.",
            "real_code_benchmark_graduation",
        ))


def add_code_lm_closure_bottlenecks(
    out: list[dict[str, Any]],
    report: dict[str, Any],
    source: str,
    legacy_superseded: bool = False,
) -> None:
    if not isinstance(report, dict) or not report:
        return
    runtime_ms = float(number(report.get("runtime_ms")))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    private_eval = report.get("private_eval") if isinstance(report.get("private_eval"), dict) else {}
    task_count = int(number(private_eval.get("eval_task_count")))
    if runtime_ms >= 30 * 60 * 1000:
        severity = "RED" if runtime_ms >= 3 * 60 * 60 * 1000 else "YELLOW"
        impact = min(1.0, runtime_ms / (6 * 60 * 60 * 1000))
        evidence = f"{source} runtime={round(runtime_ms / 60000, 2)} minutes eval_tasks={task_count}"
        action = (
            "Break report timing into train/readout/STS/fanout/verification subphases; never let an opaque closure consume an overnight loop."
        )
        if legacy_superseded:
            severity = "YELLOW"
            impact = min(0.30, impact)
            evidence = (
                f"{evidence}; historical old closure path is superseded by GREEN train-once fanout, "
                "same-surface repair proof, decoder gate, and transfer proof"
            )
            action = (
                "Keep this as historical debt and block new routing through the old monolithic closure; optimize the current train-once fanout phases instead."
            )
        out.append(loop_issue(
            severity,
            f"{source}_wall_clock_too_large",
            impact,
            evidence,
            action,
            source,
        ))
    verification = private_eval.get("private_verification") if isinstance(private_eval.get("private_verification"), dict) else {}
    if verification:
        compile_rate = float(number(verification.get("compile_pass_rate")))
        runtime_rate = float(number(verification.get("runtime_load_rate")))
        behavior_rate = float(number(verification.get("intended_behavior_pass_rate")))
        if compile_rate < 0.70:
            out.append(loop_issue(
                "YELLOW",
                f"{source}_compile_reward_wall",
                0.55,
                f"compile_pass_rate={compile_rate}",
                "Train on lint/compile residuals before runtime tests; failing candidates should not reach sandbox.",
                source,
            ))
        elif runtime_rate < 0.70:
            out.append(loop_issue(
                "YELLOW",
                f"{source}_runtime_load_wall",
                0.50,
                f"runtime_load_rate={runtime_rate}",
                "Target import/name/type-load residuals before semantic correctness pressure.",
                source,
            ))
        elif behavior_rate < 0.70:
            out.append(loop_issue(
                "YELLOW",
                f"{source}_semantic_behavior_wall",
                0.45,
                f"intended_behavior_pass_rate={behavior_rate}",
                "Only after compile/runtime are healthy, train semantic residual families and ranker reward.",
                source,
            ))
    estimated_public_sandbox_steps = int(number(get_path(summary, ["step_plan", "estimated_public_sandbox_steps"], 0)))
    if estimated_public_sandbox_steps > 5000:
        out.append(loop_issue(
            "YELLOW",
            f"{source}_sandbox_step_budget_high",
            min(0.80, estimated_public_sandbox_steps / 20000.0),
            f"estimated_public_sandbox_steps={estimated_public_sandbox_steps}",
            "Use GPU/ranker prefilter and staged cascade so CPU sandbox only sees the top slice.",
            source,
        ))


def add_performance_bottlenecks(out: list[dict[str, Any]], report: dict[str, Any], reports: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        return
    for row in report.get("bottlenecks", []) if isinstance(report.get("bottlenecks"), list) else []:
        if not isinstance(row, dict):
            continue
        if is_stale_performance_bottleneck(row, report, reports):
            continue
        severity = "RED" if str(row.get("severity")).upper() == "RED" else "YELLOW"
        out.append(loop_issue(
            severity,
            f"performance_optimizer_{safe_id(row.get('id'))}",
            0.40 if severity == "YELLOW" else 0.70,
            str(row.get("detail") or row.get("evidence") or row.get("id")),
            "Clear performance_optimizer bottleneck before launching long training.",
            "performance_optimizer",
        ))


def is_stale_performance_bottleneck(row: dict[str, Any], performance: dict[str, Any], reports: dict[str, Any]) -> bool:
    row_id = str(row.get("id") or "")
    perf_time = parse_utc(performance.get("created_utc"))
    resource_time = parse_utc(get_path(reports.get("resource_governor"), ["created_utc"], ""))
    scheduler_time = parse_utc(get_path(reports.get("hive_scheduler"), ["created_utc"], ""))
    if row_id == "resource_governor_throttled":
        current_can_run = get_path(reports.get("resource_governor"), ["decision", "can_run_requested_profile"], None)
        if current_can_run is True and resource_time and (perf_time is None or perf_time < resource_time):
            return True
    if row_id == "incomplete_worker_chunk_plan":
        current_chunks = int(number(get_path(reports.get("hive_scheduler"), ["summary", "real_worker_chunks"], 0)))
        if current_chunks >= 3 and scheduler_time and (perf_time is None or perf_time < scheduler_time):
            return True
    return False


def scan_module_health(attd_report: dict[str, Any] | None = None) -> dict[str, Any]:
    role_map = attd_role_map(attd_report or {})
    metrics: list[dict[str, Any]] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIX_LIMITS:
                continue
            if any(part in IGNORED_SOURCE_DIRS for part in path.parts):
                continue
            rel_path = rel(path)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            line_count = text.count("\n") + 1
            function_count = count_defs(text, path.suffix)
            metrics.append({
                "path": rel_path,
                "attd_role": role_map.get(rel_path, ""),
                "suffix": path.suffix,
                "line_count": line_count,
                "function_count": function_count,
                "bytes": path.stat().st_size,
                "soft_limit": SOURCE_SUFFIX_LIMITS[path.suffix],
                "hard_limit": HARD_SOURCE_SUFFIX_LIMITS[path.suffix],
            })
    hotspots = []
    for item in metrics:
        severity = ""
        if item["line_count"] > item["hard_limit"]:
            severity = "RED"
        elif item["line_count"] > item["soft_limit"]:
            severity = "YELLOW"
        if severity:
            hotspots.append({
                **item,
                "severity": severity,
                "reason": "source_file_exceeds_ai_maintainability_line_limit",
                "recommended_action": modularization_action(item),
            })
    hotspots.sort(key=lambda row: (0 if row["severity"] == "RED" else 1, -int(row["line_count"]), row["path"]))
    by_suffix = Counter(item["suffix"] for item in metrics)
    hard_count = sum(1 for row in hotspots if row["severity"] == "RED")
    soft_count = sum(1 for row in hotspots if row["severity"] == "YELLOW")
    score = max(0.0, 1.0 - hard_count * 0.08 - soft_count * 0.025)
    return {
        "policy": "project_theseus_ai_maintainability_audit_v1",
        "summary": {
            "source_file_count": len(metrics),
            "source_file_counts_by_suffix": dict(sorted(by_suffix.items())),
            "hotspot_count": len(hotspots),
            "hard_hotspot_count": hard_count,
            "soft_hotspot_count": soft_count,
            "maintainability_score": round(score, 4),
            "largest_file_lines": max((item["line_count"] for item in metrics), default=0),
        },
        "largest_files": sorted(metrics, key=lambda row: (-int(row["line_count"]), row["path"]))[:20],
        "hotspots": hotspots[:30],
        "architecture_rule": "AI-maintained systems need bounded modules with narrow ownership; large files are iteration-rate bottlenecks even when tests pass.",
    }


def count_defs(text: str, suffix: str) -> int:
    if suffix == ".py":
        return sum(1 for line in text.splitlines() if line.lstrip().startswith(("def ", "class ")))
    if suffix == ".rs":
        return sum(1 for line in text.splitlines() if line.lstrip().startswith(("fn ", "pub fn ", "struct ", "pub struct ", "enum ", "pub enum ")))
    return 0


def modularization_action(item: dict[str, Any]) -> str:
    path = str(item["path"])
    if "code_lm_closure" in path:
        return "Split Code LM closure into train/readout, STS conditioning, candidate generation, verification, reporting, and artifact IO modules."
    if path.startswith("scripts/"):
        return "Move command orchestration, report shaping, and domain logic into separate small modules; keep the script as a thin CLI."
    return "Split by ownership boundary and preserve public APIs with focused smoke tests."


def build_attd_alignment(
    attd_report: dict[str, Any],
    packets_report: dict[str, Any],
    runtime_bottlenecks: list[dict[str, Any]],
    module_health: dict[str, Any],
) -> dict[str, Any]:
    components = attd_report.get("components") if isinstance(attd_report.get("components"), dict) else {}
    top_component = ""
    if components:
        top_component = max(components.items(), key=lambda item: float(number(item[1])))[0]
    packets = [
        row
        for row in packets_report.get("packets", [])
        if isinstance(row, dict)
    ] if isinstance(packets_report.get("packets"), list) else []
    runtime_overlaps = find_attd_runtime_overlaps(packets, runtime_bottlenecks, module_health)
    hotspots = module_health.get("hotspots") if isinstance(module_health.get("hotspots"), list) else []
    attd_hotspot_paths = {
        str(path)
        for item in packets
        for path in item.get("scope", [])
        if str(path)
    }
    strict_only_hotspots = [
        row
        for row in hotspots
        if row.get("path") not in attd_hotspot_paths and row.get("severity") == "RED"
    ]
    governance = attd_report.get("governance") if isinstance(attd_report.get("governance"), dict) else {}
    return {
        "policy": "project_theseus_attd_efficiency_alignment_v1",
        "summary": {
            "trigger_state": str(attd_report.get("trigger_state") or "MISSING"),
            "attd_score": number(attd_report.get("attd_score")),
            "top_component": top_component,
            "packet_count": len(packets),
            "runtime_overlap_count": len(runtime_overlaps),
            "strict_ai_hotspots_not_in_attd_packet_scope": len(strict_only_hotspots),
            "allows_long_autonomy": governance.get("allows_long_autonomy"),
            "requires_maintenance_packets": governance.get("requires_maintenance_packets"),
        },
        "components": components,
        "hard_caps": attd_report.get("hard_caps") if isinstance(attd_report.get("hard_caps"), dict) else {},
        "maintenance_packets": packets[:8],
        "runtime_overlaps": runtime_overlaps,
        "strict_ai_hotspots_not_in_attd_packet_scope": strict_only_hotspots[:8],
        "interpretation": (
            "ATTD explains whether speed patches are making the system harder to safely extend. "
            "The optimizer should prefer work that lowers both runtime bottlenecks and ATTD assembly burden."
        ),
    }


def build_architecture_cleanup_queue(attd_alignment: dict[str, Any], module_health: dict[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(attd_alignment.get("runtime_overlaps", []) if isinstance(attd_alignment.get("runtime_overlaps"), list) else []):
        if not isinstance(row, dict):
            continue
        queue.append(
            {
                "id": f"runtime_attd_overlap_{safe_id(row.get('bottleneck_id'))}_{index + 1}",
                "priority": float(number(row.get("combined_priority"))),
                "kind": "runtime_bottleneck_modularization",
                "bottleneck_id": row.get("bottleneck_id"),
                "packet_id": row.get("packet_id"),
                "component": row.get("packet_component"),
                "scope": row.get("overlap_scope") if isinstance(row.get("overlap_scope"), list) else [],
                "bounded_action": row.get("recommended_action"),
                "verification": [
                    "python -m py_compile for touched Python wrappers",
                    "cargo build --release --bin symliquid-cli for touched Rust hot paths",
                    "python scripts/system_efficiency_audit.py --out reports/system_efficiency_audit.json --markdown-out reports/system_efficiency_audit.md",
                    "python scripts/asi_wall_breaker_governor.py --out reports/asi_wall_breaker_governor.json --markdown-out reports/asi_wall_breaker_governor.md",
                ],
                "promotion_semantics": "architecture_cleanup_only_not_capability_promotion",
            }
        )
    for index, row in enumerate(module_health.get("hotspots", []) if isinstance(module_health.get("hotspots"), list) else []):
        if not isinstance(row, dict) or row.get("severity") != "RED":
            continue
        already_scoped = any(str(row.get("path")) in item.get("scope", []) for item in queue)
        if already_scoped:
            continue
        queue.append(
            {
                "id": f"hard_module_split_{safe_id(row.get('path'))}",
                "priority": 0.75 - min(index, 20) * 0.01,
                "kind": "hard_maintainability_split",
                "bottleneck_id": "hard_ai_maintainability_hotspot",
                "packet_id": "",
                "component": row.get("attd_role") or "unassigned",
                "scope": [row.get("path")],
                "bounded_action": row.get("recommended_action"),
                "verification": [
                    "Run the narrow compile or py_compile check for the touched file.",
                    "Refresh ATTD/system efficiency reports and require the hotspot count to fall or preserve behavior.",
                ],
                "promotion_semantics": "architecture_cleanup_only_not_capability_promotion",
            }
        )
    queue.sort(key=lambda item: (-float(number(item.get("priority"))), str(item.get("id"))))
    return queue[:20]


def find_attd_runtime_overlaps(
    packets: list[dict[str, Any]],
    runtime_bottlenecks: list[dict[str, Any]],
    module_health: dict[str, Any],
) -> list[dict[str, Any]]:
    module_paths = {str(row.get("path")) for row in module_health.get("hotspots", []) if row.get("path")}
    source_hints = {
        "code_lm_private": ["code_lm_closure", "symliquid-cli/src/code_lm_closure"],
        "code_lm_recovery": ["code_lm_closure", "symliquid-cli/src/code_lm_closure"],
        "code_lm_rust_recovery": ["code_lm_closure", "symliquid-cli/src/code_lm_closure"],
        "real_code_benchmark_graduation": ["real_code_benchmark_graduation.py"],
        "code_lm_train_once_fanout": ["code_lm_train_once_fanout.py", "code_lm_closure.py"],
        "performance_optimizer": ["performance_optimizer.py", "resource_aware_execution_policy.py"],
    }
    overlaps: list[dict[str, Any]] = []
    for bottleneck in runtime_bottlenecks:
        hints = source_hints.get(str(bottleneck.get("source")), [str(bottleneck.get("source") or "")])
        for packet in packets:
            scope = [str(item) for item in packet.get("scope", []) if str(item)]
            overlap_scope = [
                item
                for item in scope
                if any(hint and hint in item for hint in hints)
                or item in module_paths and any(hint and hint in item for hint in hints)
            ]
            if not overlap_scope:
                continue
            overlaps.append({
                "bottleneck_id": bottleneck.get("id"),
                "bottleneck_source": bottleneck.get("source"),
                "bottleneck_impact_score": bottleneck.get("impact_score"),
                "packet_id": packet.get("packet_id"),
                "packet_component": packet.get("component"),
                "packet_score": packet.get("score"),
                "overlap_scope": overlap_scope[:5],
                "combined_priority": round(float(number(bottleneck.get("impact_score"))) + float(number(packet.get("score"))), 4),
                "recommended_action": packet.get("bounded_action"),
            })
    overlaps.sort(key=lambda row: (-float(number(row.get("combined_priority"))), str(row.get("packet_id"))))
    return overlaps


def attd_role_map(attd_report: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    roles = get_path(attd_report, ["roles", "roles"], [])
    if not isinstance(roles, list):
        return out
    for role in roles:
        if not isinstance(role, dict):
            continue
        role_id = str(role.get("role") or "")
        for item in role.get("largest_files", []) if isinstance(role.get("largest_files"), list) else []:
            if isinstance(item, dict) and item.get("path"):
                out[str(item["path"])] = role_id
    return out


def loop_issue(
    severity: str,
    issue_id: str,
    impact_score: float,
    evidence: str,
    recommended_action: str,
    source: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "id": issue_id,
        "impact_score": round(float(impact_score), 4),
        "evidence": evidence,
        "recommended_action": recommended_action,
        "source": source,
    }


def flattened_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flattened_numbers(item, name))
    else:
        num = number(value)
        if num:
            out[prefix or "value"] = num
    return out


def is_runtime_timing_path(path: str) -> bool:
    lower = str(path or "").lower()
    if not lower:
        return False
    non_timing_terms = (
        "work_step",
        "work_steps",
        "max_work",
        "estimated_work",
        "budget",
        "count",
        "rows",
        "epoch",
        "vocab",
        "dim",
        "active",
        "within",
        "policy",
        "semantic",
    )
    if any(term in lower for term in non_timing_terms):
        return False
    if lower.startswith("fanout_report.") or lower.startswith("ledger_elapsed_ms."):
        return True
    return lower.endswith(("_ms", ".runtime_ms", ".elapsed_ms", ".duration_ms", ".wall_ms"))


def process_snapshot() -> list[dict[str, Any]]:
    return windows_active_code_lm_process_rows()


def issue(severity: str, issue_id: str, evidence: str, action: str) -> dict[str, str]:
    return {"severity": severity, "id": issue_id, "evidence": evidence, "big_gain_action": action}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# System Efficiency Audit",
        "",
        f"- Status: `{report['trigger_state']}`",
        f"- Findings: `{report['summary']['finding_count']}`",
        f"- Active Code LM processes: `{report['summary']['active_code_lm_process_count']}`",
        f"- Duplicate services: `{report['summary']['duplicate_service_count']}`",
        f"- Loop bottlenecks: `{report['summary']['loop_bottleneck_count']}`",
        f"- Top loop bottleneck: `{report['summary']['top_loop_bottleneck'] or 'none'}`",
        f"- Stale control debt: `{report['summary'].get('stale_control_debt_count', 0)}`",
        f"- AI maintainability score: `{report['summary']['maintainability_score']}`",
        f"- Hard maintainability hotspots: `{report['summary']['hard_maintainability_hotspot_count']}`",
        f"- ATTD: `{report['summary']['attd_trigger_state']}` score `{report['summary']['attd_score']}` top `{report['summary']['attd_top_component'] or 'none'}`",
        f"- ATTD/runtime overlaps: `{report['summary']['attd_runtime_overlap_count']}`",
        f"- Architecture cleanup queue: `{report['summary'].get('architecture_cleanup_queue_count')}`",
        f"- Public calibration locked: `{report['summary']['public_calibration_locked']}`",
        "",
        "## Findings",
        "",
    ]
    for finding in report["findings"]:
        lines.extend([
            f"- `{finding['severity']}` `{finding['id']}`: {finding['evidence']}",
            f"  - Big-gain action: {finding['big_gain_action']}",
        ])
    lines.extend(["", "## Loop Bottlenecks", ""])
    for item in report.get("loop_bottlenecks", [])[:8]:
        lines.extend([
            f"- `{item['severity']}` `{item['id']}` impact `{item['impact_score']}` from `{item['source']}`",
            f"  - Evidence: {item['evidence']}",
            f"  - Action: {item['recommended_action']}",
        ])
    if not report.get("loop_bottlenecks"):
        lines.append("- none")
    lines.extend(["", "## Stale Control Debt", ""])
    for item in report.get("stale_control_debt", [])[:8]:
        lines.extend([
            f"- `{item['severity']}` `{item['id']}` impact `{item['impact_score']}` from `{item['source']}`",
            f"  - Evidence: {item['evidence']}",
            f"  - Action: {item['recommended_action']}",
        ])
    if not report.get("stale_control_debt"):
        lines.append("- none")
    lines.extend(["", "## ATTD Alignment", ""])
    attd = report.get("attd_alignment") if isinstance(report.get("attd_alignment"), dict) else {}
    attd_summary = attd.get("summary") if isinstance(attd.get("summary"), dict) else {}
    lines.extend([
        f"- Trigger: `{attd_summary.get('trigger_state')}`",
        f"- Score: `{attd_summary.get('attd_score')}`",
        f"- Top component: `{attd_summary.get('top_component') or 'none'}`",
        f"- Maintenance packets: `{attd_summary.get('packet_count')}`",
        f"- Runtime overlaps: `{attd_summary.get('runtime_overlap_count')}`",
    ])
    for item in attd.get("runtime_overlaps", [])[:6]:
        lines.extend([
            f"- `{item.get('bottleneck_id')}` + `{item.get('packet_id')}` priority `{item.get('combined_priority')}`",
            f"  - Scope: {', '.join(item.get('overlap_scope', []))}",
            f"  - Action: {item.get('recommended_action')}",
        ])
    lines.extend(["", "## Architecture Cleanup Queue", ""])
    for item in report.get("architecture_cleanup_queue", [])[:10]:
        lines.extend([
            f"- `{item.get('id')}` priority `{item.get('priority')}`",
            f"  - Scope: {', '.join(str(x) for x in (item.get('scope') or []))}",
            f"  - Action: {item.get('bounded_action')}",
        ])
    if not report.get("architecture_cleanup_queue"):
        lines.append("- none")
    lines.extend(["", "## Maintainability Hotspots", ""])
    for item in get_path(report, ["module_health", "hotspots"], [])[:10]:
        lines.extend([
            f"- `{item['severity']}` `{item['path']}` lines `{item['line_count']}` functions `{item['function_count']}`",
            f"  - Action: {item['recommended_action']}",
        ])
    if not get_path(report, ["module_health", "hotspots"], []):
        lines.append("- none")
    lines.extend(["", "## Big-Gain Order", ""])
    for item in report["big_gain_order"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def public_calibration_operator_lock_state() -> dict[str, Any]:
    active = PUBLIC_CALIBRATION_OPERATOR_LOCK.exists()
    reason = ""
    if active:
        try:
            reason = PUBLIC_CALIBRATION_OPERATOR_LOCK.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "operator lock file exists but could not be read"
    return {
        "active": active,
        "path": str(PUBLIC_CALIBRATION_OPERATOR_LOCK.relative_to(ROOT)).replace("\\", "/"),
        "reason": reason,
    }


def read_latest_json(pattern: str, default: Any) -> Any:
    paths = [path for path in REPORTS.glob(pattern) if path.is_file()]
    if not paths:
        return default
    path = max(paths, key=lambda item: item.stat().st_mtime)
    payload = read_json(path, default)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["_source_report"] = rel(path)
    return payload


def read_latest_json_any(patterns: list[str], default: Any) -> Any:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in REPORTS.glob(pattern) if path.is_file())
    if not paths:
        return default
    path = max(paths, key=lambda item: item.stat().st_mtime)
    payload = read_json(path, default)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["_source_report"] = rel(path)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path_obj).replace("\\", "/")


def safe_id(value: Any) -> str:
    text = str(value or "unknown").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
