#!/usr/bin/env python3
"""Chunked/resumable Code LM recovery.

This is the execution-envelope repair for the recovery path: instead of asking
one Rust/SymLiquid process to finish all private/public candidate generation,
run deterministic private-eval/public-task subshards with unique artifacts,
merge only completed shards, and keep partial shards diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_process_guard import windows_code_lm_process_rows  # noqa: E402
from code_lm_private_rows import high_transfer_private_rows_string  # noqa: E402

REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "code_lm_chunked_recovery.json"
DEFAULT_MARKDOWN = REPORTS / "code_lm_chunked_recovery.md"
PRIVATE_ROWS = high_transfer_private_rows_string()


def release_binary_path() -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return ROOT / "target" / "release" / name


def cuda_readout_requested_by_launcher() -> bool:
    raw = os.environ.get("THESEUS_CODE_LM_CUDA_READOUT", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return sys.platform.startswith("win")


def release_binary_backend() -> str:
    if cuda_readout_requested_by_launcher():
        return "cuda_release_readout"
    if sys.platform == "darwin":
        return "macos_native_cpu_readout"
    return "native_cpu_readout"


def release_build_command() -> list[str]:
    command = ["cargo", "build", "--release", "-p", "symliquid-cli"]
    if cuda_readout_requested_by_launcher():
        command.extend(["--features", "cuda"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slug", default="private_pressure_private_recovery_chunked_v4")
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--max-shards-per-run", type=int, default=1)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run remaining shards in one smooth recovery session, checking resources between shards.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=0,
        help="Optional wall-clock cap for a continuous recovery session. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--wait-for-active-worker",
        action="store_true",
        help="Wait for an already-running Code LM worker to finish, then continue recovery.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=0,
        help="Maximum time to wait for an active worker before deferring. 0 means wait indefinitely.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    out = resolve(args.out)
    markdown = resolve(args.markdown_out)
    shard_count = max(1, int(args.shard_count))
    state: dict[str, Any] = {
        "policy": "project_theseus_code_lm_chunked_recovery_v1",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": bool(args.execute),
        "slug": args.slug,
        "shard_count": shard_count,
        "max_shards_per_run": max(1, int(args.max_shards_per_run)),
        "continuous": bool(args.continuous),
        "max_wall_seconds": max(0, int(args.max_wall_seconds)),
        "wait_for_active_worker": bool(args.wait_for_active_worker),
        "wait_timeout_seconds": max(0, int(args.wait_timeout_seconds)),
        "rules": {
            "public_calibration": "never run by this script",
            "public_benchmark_training": "forbidden",
            "resume": "completed shard reports are skipped; failed shards remain diagnostic-only",
            "subshards": "smaller public/private eval subshards preserve useful evidence without losing the whole recovery attempt",
            "continuous": "continuous mode runs remaining shards in one session while checking resources between shards",
            "architecture_status": "current shards are a crash-recovery envelope, not the target train-once/distribute-many hive training substrate",
        },
        "shards": [],
        "external_inference_calls": 0,
    }
    write_progress(out, markdown, state)
    if not args.execute:
        print(json.dumps(state, indent=2))
        return 0

    resource = run_resource_policy()
    budget = object_field(resource, "recommended_code_lm_budget")
    state["resource_policy"] = {
        "profile": get_path(resource, ["summary", "profile"], ""),
        "start_new_code_closure": bool(budget.get("start_new_code_closure")),
        "start_new_chunked_code_closure": bool(budget.get("start_new_chunked_code_closure")),
        "reason": budget.get("reason"),
    }
    state["shard_strategy_assessment"] = shard_strategy_assessment(args.slug, shard_count)
    if active_code_workers():
        if args.wait_for_active_worker:
            wait_started = time.perf_counter()
            while active_code_workers():
                elapsed = int(time.perf_counter() - wait_started)
                if args.wait_timeout_seconds and elapsed >= args.wait_timeout_seconds:
                    state["trigger_state"] = "YELLOW"
                    state["current_phase"] = "deferred_existing_code_worker_wait_timeout"
                    state["waited_seconds"] = elapsed
                    state["next_actions"] = ["Existing Code LM worker did not clear before wait timeout; retry continuous recovery later."]
                    write_progress(out, markdown, state)
                    print(json.dumps(state, indent=2))
                    return 0
                state["trigger_state"] = "YELLOW"
                state["current_phase"] = "waiting_existing_code_worker"
                state["waited_seconds"] = elapsed
                state["next_actions"] = ["Waiting for the active Code LM worker to finish before continuing chunked recovery."]
                write_progress(out, markdown, state)
                time.sleep(15)
            resource = run_resource_policy()
            budget = object_field(resource, "recommended_code_lm_budget")
            state["resource_policy"] = {
                "profile": get_path(resource, ["summary", "profile"], ""),
                "start_new_code_closure": bool(budget.get("start_new_code_closure")),
                "start_new_chunked_code_closure": bool(budget.get("start_new_chunked_code_closure")),
                "reason": budget.get("reason"),
            }
        else:
            state["trigger_state"] = "YELLOW"
            state["current_phase"] = "deferred_existing_code_worker"
            state["next_actions"] = ["A Code LM/SymLiquid worker is already active; do not stack chunked recovery."]
            write_progress(out, markdown, state)
            print(json.dumps(state, indent=2))
            return 0
    if not resource_allows_code_work(budget):
        state["trigger_state"] = "YELLOW"
        state["current_phase"] = "deferred_resource_policy"
        state["next_actions"] = ["Resource policy deferred Code LM work; retry when the machine is clear."]
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 0

    build = ensure_release_cuda_binary()
    state["release_cuda_binary"] = build
    if not build.get("ready"):
        state["trigger_state"] = "RED"
        state["current_phase"] = "release_cuda_binary_not_ready"
        state["next_actions"] = [
            f"Build {rel(release_binary_path())} explicitly before Code LM recovery; do not fall back to cargo run in the hot path."
        ]
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 2

    ran = 0
    env = chunk_env()
    max_shards_this_run = shard_count if args.continuous else max(1, int(args.max_shards_per_run))
    state["effective_max_shards_this_run"] = max_shards_this_run
    for index in range(shard_count):
        shard = shard_status(args.slug, index, shard_count)
        if shard["completed"]:
            state["shards"].append(shard)
            continue
        if ran >= max_shards_this_run:
            state["shards"].append(shard)
            continue
        if args.max_wall_seconds and (time.perf_counter() - started) >= args.max_wall_seconds:
            state["trigger_state"] = "YELLOW"
            state["current_phase"] = "deferred_wall_clock_budget"
            state["next_actions"] = ["Continuous recovery hit its wall-clock budget; resume later without losing completed shards."]
            state["completed_shards"] = sum(1 for row in state["shards"] if row.get("completed"))
            write_progress(out, markdown, state)
            break
        if ran:
            resource = run_resource_policy()
            budget = object_field(resource, "recommended_code_lm_budget")
            state["resource_policy"] = {
                "profile": get_path(resource, ["summary", "profile"], ""),
                "start_new_code_closure": bool(budget.get("start_new_code_closure")),
                "start_new_chunked_code_closure": bool(budget.get("start_new_chunked_code_closure")),
                "reason": budget.get("reason"),
            }
            if not resource_allows_code_work(budget):
                state["trigger_state"] = "YELLOW"
                state["current_phase"] = "deferred_resource_policy_between_shards"
                state["next_actions"] = ["Resource policy paused continuous recovery between shards; resume later."]
                state["completed_shards"] = sum(1 for row in state["shards"] if row.get("completed"))
                write_progress(out, markdown, state)
                break
        disk_projection = shard_disk_projection(args.slug, shard_count)
        state["shard_disk_projection"] = disk_projection
        if disk_projection["free_mb"] < 2048:
            state["trigger_state"] = "YELLOW"
            state["current_phase"] = "deferred_low_report_drive_space"
            state["next_actions"] = [
                "Report drive free space is below 2 GB; do not start another shard until artifacts are compacted or moved."
            ]
            state["completed_shards"] = sum(1 for row in state["shards"] if row.get("completed"))
            write_progress(out, markdown, state)
            break
        lease = claim_shard_lease(args.slug, index, shard_count)
        if not lease.get("claimed"):
            shard["hive_lease"] = lease
            state["shards"].append(shard)
            continue
        budget_summary = shard_budget_summary(budget)
        command = shard_command(args.slug, index, shard_count, budget)
        state["trigger_state"] = "RUNNING"
        state["current_phase"] = "running_chunked_subshard"
        state["completed_shards"] = sum(1 for row in state["shards"] if row.get("completed"))
        state["active_shard"] = {
            "index": index,
            "shard_slug": shard_slug(args.slug, index, shard_count),
            "shard_count": shard_count,
            "budget": budget_summary,
            "closure_report": rel(REPORTS / f"code_lm_closure_{shard_slug(args.slug, index, shard_count)}.json"),
            "rust_heartbeat": rel(REPORTS / f"code_lm_closure_rust_{shard_slug(args.slug, index, shard_count)}.heartbeat.json"),
            "hive_lease": lease,
        }
        write_progress(out, markdown, state)
        phase = run_command(
            command,
            env=env,
            timeout_seconds=bounded_budget_int(budget, "chunk_rust_timeout_seconds", 900, 900) + 600,
            log_path=REPORTS / f"code_lm_chunked_recovery_shard_{index:02d}.log",
        )
        ran += 1
        shard = shard_status(args.slug, index, shard_count)
        shard["phase"] = phase
        finish_shard_lease(lease, phase, shard)
        if phase["returncode"] != 0:
            score_partial_shard(args.slug, index, shard_count)
            shard["partial_score"] = rel(shard_partial_score_path(args.slug, index, shard_count))
        state["shards"].append(shard)
        write_progress(out, markdown, state)
        if phase["returncode"] != 0:
            break

    merge = merge_completed_shards(args.slug, shard_count)
    state["merge"] = merge
    state["completed_shards"] = merge["completed_shards"]
    state["current_phase"] = "merged_completed_shards"
    all_complete = merge["completed_shards"] == shard_count and shard_count > 0
    if all_complete:
        ablation = run_command(
            receiver_ablation_command(args.slug, shard_count),
            env=os.environ.copy(),
            timeout_seconds=1800,
            log_path=REPORTS / "private_type_shape_receiver_ablation.chunked_recovery.log",
        )
        gate = run_command(
            [
                sys.executable,
                "scripts/decoder_v2_private_ablation_gate.py",
                "--closure-report",
                f"reports/code_lm_closure_{args.slug}_merged.json",
                "--closure-report",
                "reports/code_lm_closure_private_pressure_private.json",
            ],
            env=os.environ.copy(),
            timeout_seconds=1800,
            log_path=REPORTS / "decoder_v2_private_ablation_gate.chunked_recovery.log",
        )
        proof = run_command(
            [sys.executable, "scripts/private_public_transfer_proof.py"],
            env=os.environ.copy(),
            timeout_seconds=900,
            log_path=REPORTS / "private_public_transfer_proof.chunked_recovery.log",
        )
        sts_ablation = run_command(
            [sys.executable, "scripts/sts_causal_decoder_ablation.py"],
            env=os.environ.copy(),
            timeout_seconds=900,
            log_path=REPORTS / "sts_causal_decoder_ablation.chunked_recovery.log",
        )
        symliquid_state = run_command(
            [sys.executable, "scripts/symliquid_state_engine.py"],
            env=os.environ.copy(),
            timeout_seconds=900,
            log_path=REPORTS / "symliquid_state_engine.chunked_recovery.log",
        )
        agent_lane = run_command(
            [sys.executable, "scripts/agent_lane_transfer_gate.py"],
            env=os.environ.copy(),
            timeout_seconds=900,
            log_path=REPORTS / "agent_lane_transfer_gate.chunked_recovery.log",
        )
        maturity = run_command(
            [sys.executable, "scripts/maturity_integrity_audit.py"],
            env=os.environ.copy(),
            timeout_seconds=900,
            log_path=REPORTS / "maturity_integrity_audit.chunked_recovery.log",
        )
        state["post_merge_phases"] = {
            "private_type_shape_receiver_ablation": ablation,
            "decoder_v2_private_ablation_gate": gate,
            "private_public_transfer_proof": proof,
            "sts_causal_decoder_ablation": sts_ablation,
            "symliquid_state_engine": symliquid_state,
            "agent_lane_transfer_gate": agent_lane,
            "maturity_integrity_audit": maturity,
        }
    ready = bool(get_path(read_json(REPORTS / "private_public_transfer_proof.json", {}), ["summary", "ready_for_public_calibration"], False))
    state["ready_for_public_calibration"] = ready
    state["trigger_state"] = "GREEN" if all_complete and ready else ("YELLOW" if merge["completed_shards"] else "RED")
    state["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    state["next_actions"] = next_actions(all_complete, ready, merge, ran)
    write_progress(out, markdown, state)
    print(json.dumps(state, indent=2))
    return 0 if state["trigger_state"] != "RED" else 2


def resource_allows_code_work(budget: dict[str, Any]) -> bool:
    return bool(budget.get("start_new_code_closure") or budget.get("start_new_chunked_code_closure"))


def shard_command(slug: str, index: int, count: int, budget: dict[str, Any]) -> list[str]:
    shard = shard_slug(slug, index, count)
    summary = shard_budget_summary(budget)
    command = [
        sys.executable,
        "scripts/code_lm_closure.py",
        "--skip-public-calibration",
        "--public-cards",
        "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
        "--seed",
        "23",
        "--max-public-cases-per-card",
        "32",
        "--private-count",
        str(summary["private_count"]),
        "--epochs",
        str(summary["epochs"]),
        "--candidates-per-task",
        str(summary["candidates_per_task"]),
        "--disable-extra-private-train",
        "--disable-residual-private-train",
        "--disable-repo-repair-private-train",
        "--high-transfer-private-train-jsonl",
        PRIVATE_ROWS,
        "--max-high-transfer-private-train",
        str(summary["max_high_transfer_private_train"]),
        "--max-rust-work-steps",
        str(summary["max_rust_work_steps"]),
        "--rust-timeout-seconds",
        str(summary["rust_timeout_seconds"]),
        "--sts-timeout-seconds",
        str(summary["sts_timeout_seconds"]),
        "--private-eval-shard-count",
        str(count),
        "--private-eval-shard-index",
        str(index),
        "--public-task-shard-count",
        str(count),
        "--public-task-shard-index",
        str(index),
        "--private-curriculum-out",
        f"data/private_code_curriculum/code_lm_closure_{shard}.jsonl",
        "--public-task-manifest-out",
        f"reports/code_lm_public_tasks_{shard}.jsonl",
        "--checkpoint-out",
        f"reports/student_code_lm_checkpoint_{shard}.json",
        "--private-candidate-out",
        f"reports/code_lm_private_candidates_{shard}.jsonl",
        "--public-candidate-out",
        f"reports/student_code_candidates_{shard}.jsonl",
        "--rust-report-out",
        f"reports/code_lm_closure_rust_{shard}.json",
        "--public-report-out",
        f"reports/real_code_benchmark_graduation_{shard}_skipped.json",
        "--public-trace-out",
        f"reports/real_code_benchmark_traces_{shard}_skipped.jsonl",
        "--out",
        f"reports/code_lm_closure_{shard}.json",
        "--sts-conditioning-input-out",
        f"reports/code_lm_sts_conditioning_input_{shard}.jsonl",
        "--sts-generation-out",
        f"reports/code_lm_sts_public_generations_{shard}.jsonl",
        "--sts-conditioning-checkpoint-out",
        f"reports/code_lm_sts_conditioning_checkpoint_{shard}.json",
        "--sts-conditioning-report-out",
        f"reports/code_lm_sts_conditioning_report_{shard}.json",
        "--sts-decoder-control-policy-jsonl",
        "reports/sts_decoder_control_rows.jsonl",
        "--lock-path",
        f"reports/code_lm_closure_{shard}.lock",
        "--typed-edge-exec-receiver-v1",
        "--edge-obligation-decode-gate-v1",
        "--private-type-shape-receiver-veto-v1",
        "--edge-obligation-report-out",
        f"reports/edge_obligation_decode_gate_v1_{shard}.json",
        "--edge-obligation-markdown-out",
        f"reports/edge_obligation_decode_gate_v1_{shard}.md",
    ]
    if cuda_readout_requested_by_launcher():
        command.insert(command.index("--rust-timeout-seconds"), "--use-cuda-readout")
    return command


def ensure_release_cuda_binary() -> dict[str, Any]:
    exe = release_binary_path()
    sources = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "code_ranker.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    if cuda_readout_requested_by_launcher():
        sources.extend(
            [
                ROOT / "crates" / "symliquid-cuda" / "src" / "readout_cuda.rs",
                ROOT / "crates" / "symliquid-cuda" / "kernels" / "readout_kernels.cu",
            ]
        )
    sources.extend(sorted((ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure").glob("*.rs")))
    newest_source_mtime = max((path.stat().st_mtime for path in sources if path.exists()), default=0.0)
    exe_mtime = exe.stat().st_mtime if exe.exists() else 0.0
    stale = (not exe.exists()) or exe_mtime < newest_source_mtime
    status: dict[str, Any] = {
        "path": rel(exe),
        "exists": exe.exists(),
        "stale_relative_to_sources": stale,
        "release_binary_backend": release_binary_backend(),
        "cuda_readout_requested_by_launcher": cuda_readout_requested_by_launcher(),
        "cuda_feature_build_enforced": cuda_readout_requested_by_launcher(),
        "newest_source_mtime": newest_source_mtime,
        "exe_mtime": exe_mtime,
        "cargo_run_hot_path_allowed": False,
    }
    if stale:
        started = time.perf_counter()
        phase = run_command(
            release_build_command(),
            env=os.environ.copy(),
            timeout_seconds=1800,
            log_path=REPORTS / "code_lm_chunked_recovery_release_build.log",
        )
        status["build"] = phase
        status["build_runtime_ms"] = int((time.perf_counter() - started) * 1000)
    status["ready"] = exe.exists() and (exe.stat().st_mtime if exe.exists() else 0.0) >= newest_source_mtime
    return status


def shard_budget_summary(budget: dict[str, Any]) -> dict[str, int]:
    return {
        "private_count": bounded_budget_int(budget, "chunk_private_count", 24, 24),
        "epochs": bounded_budget_int(budget, "chunk_epochs", 1, 1),
        "candidates_per_task": bounded_budget_int(budget, "chunk_candidates_per_task", 2, 2),
        "max_high_transfer_private_train": bounded_budget_int(budget, "chunk_max_high_transfer_private_train", 192, 192),
        "max_rust_work_steps": bounded_budget_int(budget, "chunk_max_rust_work_steps", 120_000, 120_000),
        "rust_timeout_seconds": bounded_budget_int(budget, "chunk_rust_timeout_seconds", 900, 900),
        "sts_timeout_seconds": bounded_budget_int(budget, "chunk_sts_timeout_seconds", 600, 600),
    }


def bounded_budget_int(budget: dict[str, Any], key: str, default: int, maximum: int) -> int:
    try:
        value = int(budget.get(key) or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def shard_status(slug: str, index: int, count: int) -> dict[str, Any]:
    shard = shard_slug(slug, index, count)
    closure_path = REPORTS / f"code_lm_closure_{shard}.json"
    rust_path = REPORTS / f"code_lm_closure_rust_{shard}.json"
    closure = read_json(closure_path, {})
    rust = read_json(rust_path, {})
    completed = closure.get("run_status") == "completed" or (
        rust.get("run_status") == "completed"
        and bool(rust.get("private_candidate_manifest"))
        and bool(rust.get("public_candidate_manifest"))
    )
    return {
        "index": index,
        "shard_slug": shard,
        "completed": completed,
        "closure_report": rel(closure_path),
        "closure_run_status": closure.get("run_status"),
        "rust_report": rel(rust_path),
        "rust_run_status": rust.get("run_status"),
        "private_candidate_rows": count_jsonl_rows(resolve(first_string([rust.get("private_candidate_manifest"), closure.get("private_candidate_manifest")]))),
        "public_candidate_rows": count_jsonl_rows(resolve(first_string([rust.get("public_candidate_manifest"), closure.get("public_candidate_manifest")]))),
    }


def merge_completed_shards(slug: str, count: int) -> dict[str, Any]:
    private_out = REPORTS / f"code_lm_private_candidates_{slug}_merged.jsonl"
    public_out = REPORTS / f"student_code_candidates_{slug}_merged.jsonl"
    closure_out = REPORTS / f"code_lm_closure_{slug}_merged.json"
    rust_out = REPORTS / f"code_lm_closure_rust_{slug}_merged.json"
    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    last_checkpoint = ""
    for index in range(count):
        status = shard_status(slug, index, count)
        if not status["completed"]:
            continue
        shard = shard_slug(slug, index, count)
        closure = read_json(REPORTS / f"code_lm_closure_{shard}.json", {})
        rust = read_json(REPORTS / f"code_lm_closure_rust_{shard}.json", {})
        last_checkpoint = first_string([rust.get("checkpoint"), closure.get("checkpoint"), last_checkpoint])
        for row in read_jsonl(resolve(first_string([rust.get("private_candidate_manifest"), closure.get("private_candidate_manifest")]))):
            row["source_chunk_shard_index"] = index
            row["source_chunk_shard_count"] = count
            private_rows.append(row)
        for row in read_jsonl(resolve(first_string([rust.get("public_candidate_manifest"), closure.get("public_candidate_manifest")]))):
            row["source_chunk_shard_index"] = index
            row["source_chunk_shard_count"] = count
            public_rows.append(row)
        completed.append(status)
    write_jsonl(private_out, private_rows)
    write_jsonl(public_out, public_rows)
    all_complete = len(completed) == count and count > 0
    rust_report = {
        "policy": "project_theseus_code_lm_closure_rust_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW" if all_complete else "RED",
        "run_status": "completed" if all_complete else "diagnostic_partial_chunk_merge",
        "checkpoint": last_checkpoint,
        "private_candidate_manifest": rel(private_out),
        "public_candidate_manifest": rel(public_out),
        "summary": {
            "chunked_recovery": True,
            "completed_shards": len(completed),
            "shard_count": count,
            "private_candidate_count": len(private_rows),
            "public_candidate_count": len(public_rows),
            "public_task_count": len({str(row.get("task_id") or "") for row in public_rows if row.get("task_id")}),
            "full_body_token_candidate_count": sum(1 for row in public_rows if truthy(row.get("full_body_token_candidate"))),
            "token_level_code_generation_learned": any(truthy(row.get("token_level_code_generation_learned")) for row in public_rows),
            "score_semantics": "completed shard merge only; public calibration remains gated elsewhere",
        },
        "external_inference_calls": 0,
    }
    write_json(rust_out, rust_report)
    closure_report = {
        "policy": "project_theseus_code_lm_closure_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW" if all_complete else "RED",
        "run_status": "completed" if all_complete else "diagnostic_partial_chunk_merge",
        "diagnostic_only": not all_complete,
        "progress_stage": "chunked_recovery_merged",
        "checkpoint": last_checkpoint,
        "private_candidate_manifest": rel(private_out),
        "public_candidate_manifest": rel(public_out),
        "rust_report": rel(rust_out),
        "summary": rust_report["summary"],
        "chunked_shards": completed,
        "external_inference_calls": 0,
    }
    write_json(closure_out, closure_report)
    return {
        "completed_shards": len(completed),
        "shard_count": count,
        "all_complete": all_complete,
        "merged_closure_report": rel(closure_out),
        "merged_rust_report": rel(rust_out),
        "merged_private_candidate_manifest": rel(private_out),
        "merged_public_candidate_manifest": rel(public_out),
        "private_candidate_rows": len(private_rows),
        "public_candidate_rows": len(public_rows),
    }


def receiver_ablation_command(slug: str, shard_count: int) -> list[str]:
    command = [
        sys.executable,
        "scripts/private_type_shape_receiver_ablation.py",
        "--candidate-manifest",
        f"reports/code_lm_private_candidates_{slug}_merged.jsonl",
        "--out",
        "reports/private_type_shape_receiver_ablation_chunked_recovery.json",
        "--markdown-out",
        "reports/private_type_shape_receiver_ablation_chunked_recovery.md",
        "--max-tasks",
        "192",
    ]
    for index in range(shard_count):
        command.extend([
            "--task-source",
            f"data/private_code_curriculum/code_lm_closure_{shard_slug(slug, index, shard_count)}.jsonl",
        ])
    return command


def score_partial_shard(slug: str, index: int, count: int) -> None:
    shard = shard_slug(slug, index, count)
    run_command(
        [
            sys.executable,
            "scripts/code_lm_partial_artifact_scorer.py",
            "--closure-report",
            f"reports/code_lm_closure_{shard}.json",
            "--rust-report",
            f"reports/code_lm_closure_rust_{shard}.json",
            "--heartbeat",
            f"reports/code_lm_closure_rust_{shard}.heartbeat.json",
            "--out",
            str(shard_partial_score_path(slug, index, count).relative_to(ROOT)),
            "--markdown-out",
            f"reports/code_lm_partial_artifact_score_{shard}.md",
        ],
        env=os.environ.copy(),
        timeout_seconds=900,
        log_path=REPORTS / f"code_lm_partial_artifact_score_{shard}.log",
    )


def shard_strategy_assessment(slug: str, shard_count: int) -> dict[str, Any]:
    projection = shard_disk_projection(slug, shard_count)
    completed = 0
    repeated_training = False
    for index in range(shard_count):
        status = shard_status(slug, index, shard_count)
        if status.get("completed"):
            completed += 1
        closure = read_json(REPORTS / f"code_lm_closure_{shard_slug(slug, index, shard_count)}.json", {})
        command = get_path(closure, ["phase", "command"], [])
        command_text = " ".join(command) if isinstance(command, list) else ""
        if "--epochs" in command_text and "--checkpoint-out" in command_text:
            repeated_training = True
    return {
        "policy": "project_theseus_code_lm_shard_strategy_assessment_v1",
        "current_role": "crash_recovery_and_candidate_eval_sharding",
        "target_role": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
        "completed_shards": completed,
        "shard_count": shard_count,
        "repeated_training_per_shard_detected": repeated_training,
        "acceptable_now": True,
        "promotion_credit_allowed": False,
        "reason": (
            "Sharding is acceptable as a recovery envelope, but it should not be treated as a mature "
            "distributed training architecture while readout/STS training repeats per shard."
        ),
        "disk_projection": projection,
        "next_architecture_step": "split Code LM into train/readout checkpoint phase and checkpoint fanout candidate-generation shards",
    }


def shard_disk_projection(slug: str, shard_count: int) -> dict[str, Any]:
    artifact_files = shard_artifact_files(slug)
    total_bytes = sum(path.stat().st_size for path in artifact_files)
    completed_sizes: list[int] = []
    for index in range(shard_count):
        status = shard_status(slug, index, shard_count)
        if not status.get("completed"):
            continue
        shard = shard_slug(slug, index, shard_count)
        completed_sizes.append(sum(path.stat().st_size for path in artifact_files if shard in path.name))
    avg_completed = int(sum(completed_sizes) / len(completed_sizes)) if completed_sizes else 0
    projected = max(total_bytes, avg_completed * shard_count)
    free = shutil.disk_usage(REPORTS).free
    return {
        "artifact_file_count": len(artifact_files),
        "current_mb": round(total_bytes / 1024**2, 3),
        "projected_full_run_mb": round(projected / 1024**2, 3),
        "free_mb": round(free / 1024**2, 3),
        "completed_shard_avg_mb": round(avg_completed / 1024**2, 3),
        "low_space_block_threshold_mb": 2048,
        "within_space_budget": free >= 2 * 1024**3 and projected < max(2 * 1024**3, free // 2),
    }


def shard_artifact_files(slug: str) -> list[Path]:
    files: list[Path] = []
    for root in [REPORTS, ROOT / "data" / "private_code_curriculum"]:
        if root.exists():
            files.extend(path for path in root.glob(f"*{slug}*") if path.is_file())
    return files


def claim_shard_lease(slug: str, index: int, count: int) -> dict[str, Any]:
    shard = shard_slug(slug, index, count)
    lease_dir = REPORTS / "code_lm_chunked_recovery_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease_path = lease_dir / f"{shard}.json"
    now_unix = time.time()
    payload = {
        "policy": "project_theseus_code_lm_chunked_recovery_hive_lease_v1",
        "status": "claimed",
        "created_utc": now(),
        "created_unix": now_unix,
        "updated_utc": now(),
        "updated_unix": now_unix,
        "slug": slug,
        "shard_slug": shard,
        "shard_index": index,
        "shard_count": count,
        "machine": socket.gethostname(),
        "pid": os.getpid(),
        "lease_semantics": "best-effort shared-filesystem shard claim to prevent duplicate hive workers; completed reports remain authoritative",
    }
    try:
        with lease_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {"claimed": True, "path": rel(lease_path), "status": "claimed", "machine": payload["machine"], "pid": payload["pid"]}
    except FileExistsError:
        existing = read_json(lease_path, {})
        age = max(0.0, now_unix - float(existing.get("updated_unix") or existing.get("created_unix") or lease_path.stat().st_mtime))
        completed = shard_status(slug, index, count).get("completed")
        return {
            "claimed": False,
            "path": rel(lease_path),
            "status": existing.get("status") or "existing",
            "reason": "completed_shard" if completed else "existing_hive_lease",
            "age_seconds": int(age),
            "machine": existing.get("machine"),
            "pid": existing.get("pid"),
        }


def finish_shard_lease(lease: dict[str, Any], phase: dict[str, Any], shard: dict[str, Any]) -> None:
    if not lease.get("claimed"):
        return
    path = resolve(str(lease.get("path") or ""))
    payload = read_json(path, {})
    if not payload:
        return
    payload["status"] = "completed" if phase.get("returncode") == 0 and shard.get("completed") else "failed"
    payload["updated_utc"] = now()
    payload["updated_unix"] = time.time()
    payload["returncode"] = phase.get("returncode")
    payload["timed_out"] = phase.get("timed_out")
    payload["closure_report"] = shard.get("closure_report")
    payload["rust_report"] = shard.get("rust_report")
    write_json(path, payload)


def run_resource_policy() -> dict[str, Any]:
    subprocess.run([sys.executable, "scripts/resource_aware_execution_policy.py"], cwd=ROOT, text=True, capture_output=True, timeout=120)
    return read_json(REPORTS / "resource_aware_execution_policy.json", {})


def run_command(command: list[str], *, env: dict[str, str], timeout_seconds: int, log_path: Path) -> dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(command, 124, stdout=exc.stdout or "", stderr=(exc.stderr or "") + f"\nTimed out after {timeout_seconds}s")
        timed_out = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""), encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "timed_out": timed_out,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_utc": now(),
        "elapsed_seconds": int(time.time() - started),
        "log_path": rel(log_path),
    }


def active_code_workers() -> bool:
    current_pid = os.getpid()
    return any(
        int(row.get("pid") or 0) != current_pid
        for row in windows_code_lm_process_rows("code_lm_chunked_recovery.py|code_lm_closure.py|train-code-lm-closure|symliquid-cli")
    )


def chunk_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_enabled = cuda_readout_requested_by_launcher()
    env.update(
        {
            "THESEUS_CODE_LM_CUDA_READOUT": "1" if cuda_enabled else "0",
            "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "24",
            "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
            "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
            "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
            "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
            "THESEUS_USE_CUDA_RANKER": "1" if cuda_enabled else "0",
            "THESEUS_USE_CUDA_STS_RETRIEVAL": "1" if cuda_enabled else "0",
        }
    )
    return env


def next_actions(all_complete: bool, ready: bool, merge: dict[str, Any], ran: int) -> list[str]:
    if ready:
        return ["Decoder and transfer proof are ready; allow at most one bounded public calibration."]
    if all_complete:
        return ["Chunked closure completed; inspect decoder gate and private type/shape ablation before public calibration."]
    if ran:
        return ["Continue chunked recovery on the next safe wake; do not restart the monolithic closure."]
    return ["No shard ran; wait for resources or clear active code workers before continuing chunked recovery."]


def shard_slug(slug: str, index: int, count: int) -> str:
    return f"{slug}_shard{index:02d}of{count:02d}"


def shard_partial_score_path(slug: str, index: int, count: int) -> Path:
    return REPORTS / f"code_lm_partial_artifact_score_{shard_slug(slug, index, count)}.json"


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_progress(path: Path, markdown: Path, state: dict[str, Any]) -> None:
    write_json(path, state)
    lines = [
        "# Code LM Chunked Recovery",
        "",
        f"- Status: **{state.get('trigger_state')}**",
        f"- Completed shards: `{state.get('completed_shards', 0)}/{state.get('shard_count')}`",
        f"- Ready for public calibration: `{state.get('ready_for_public_calibration', False)}`",
        "",
    ]
    for action in state.get("next_actions", []):
        lines.append(f"- {action}")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def object_field(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def first_string(values: list[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
