#!/usr/bin/env python3
"""Unattended private post-v4 generalization autopilot.

This runner is the private-only bridge between a green learned-shadow proof and
the still-locked public calibration surface. It regenerates the post-v4 shadow
curriculum, builds private-safe STS streams, runs STS-on and same-seed STS-off
fanout, scores the heldout set, runs the learned-only gate, and writes a queue
for the next private autonomous step.

It must not run public calibration or export public prompts, tests, solutions,
traces, score labels, or task ids into training rows.
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
TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "post_v4_private_shadow_transfer_v1_code_lm_tasks.jsonl"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "post_v4_private_shadow_transfer_v1_heldout_code_lm_tasks.jsonl"
CURRICULUM_REPORT = REPORTS / "post_v4_private_shadow_transfer_v1.json"
CURRICULUM_MD = REPORTS / "post_v4_private_shadow_transfer_v1.md"
STS_STREAMS = REPORTS / "post_v4_private_shadow_transfer_v1_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "post_v4_private_shadow_transfer_v1_private_safe_sts_streams.json"
EMPTY_PUBLIC = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_public.jsonl"
EMPTY_STS = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_sts_streams.jsonl"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_post_v4_private_shadow_transfer_v1_heldout_smoke160.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_post_v4_private_shadow_transfer_v1_unused.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_closure_rust_post_v4_private_shadow_transfer_v1_heldout_smoke160_fanout.json"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_post_v4_private_shadow_transfer_v1_heldout_smoke160_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_post_v4_private_shadow_transfer_v1_sts_off_unused.jsonl"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_closure_rust_post_v4_private_shadow_transfer_v1_heldout_smoke160_sts_off_fanout.json"
SCORE_REPORT = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_score.json"
SCORE_MD = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_score.md"
LEARNED_ONLY_CANDIDATES = REPORTS / "code_lm_private_candidates_post_v4_private_shadow_transfer_v1_heldout_smoke160_learned_only.jsonl"
LEARNED_ONLY_SCORE = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_learned_only_score.json"
LEARNED_ONLY_SCORE_MD = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_learned_only_score.md"
LEARNED_GATE = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_learned_distillation_gate.json"
LEARNED_GATE_MD = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_learned_distillation_gate.md"
AUTOPILOT_REPORT = REPORTS / "post_v4_generalization_autopilot_v1.json"
AUTOPILOT_MD = REPORTS / "post_v4_generalization_autopilot_v1.md"
AUTOPILOT_LEDGER = REPORTS / "post_v4_generalization_autopilot_v1_ledger.jsonl"
AUTOPILOT_HEARTBEAT = REPORTS / "post_v4_generalization_autopilot_v1_heartbeat.json"
AUTOPILOT_QUEUE = REPORTS / "post_v4_generalization_autopilot_v1_queue.jsonl"
SCALING_PROFILE = REPORTS / "post_v4_generalization_autopilot_v1_scaling_profile.json"
SCALING_PROFILE_MD = REPORTS / "post_v4_generalization_autopilot_v1_scaling_profile.md"
PRECOMPUTE_ABLATION = REPORTS / "post_v4_fanout_precompute_ablation_v1.json"
PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
POST_DISTILLATION_READINESS = REPORTS / "post_distillation_public_transfer_readiness_v1.json"
POST_V4_PUBLIC_RESULT = REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json"
POST_V4_PUBLIC_TRACES = REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl"
POST_V4_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl"
OPERATOR_EXECUTE = REPORTS / "operator_bounded_public_calibration_execute.json"
OPERATOR_APPROVAL = REPORTS / "public_calibration_operator_approval_post_v4_seed23_5x32.json"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")
PRIVATE_SCALE_CAP = 2400
POST_V4_PUBLIC_ARTIFACTS = [
    POST_V4_PUBLIC_RESULT,
    POST_V4_PUBLIC_TRACES,
    POST_V4_PUBLIC_CANDIDATES,
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--train-rows", type=int, default=1200)
    parser.add_argument("--heldout-rows", type=int, default=160)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--curriculum-seed", type=int, default=67)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--private-eval-limit", type=int, default=0)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--checkpoint-in", default=rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--skip-curriculum", action="store_true")
    parser.add_argument("--skip-fanout", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-learned-gate", action="store_true")
    parser.add_argument("--refresh-queue-only", action="store_true")
    parser.add_argument("--archive-current-only", action="store_true")
    parser.add_argument("--profile-current-only", action="store_true")
    parser.add_argument("--out", default=rel(AUTOPILOT_REPORT))
    parser.add_argument("--markdown-out", default=rel(AUTOPILOT_MD))
    parser.add_argument("--queue-out", default=rel(AUTOPILOT_QUEUE))
    parser.add_argument("--scaling-profile-out", default=rel(SCALING_PROFILE))
    parser.add_argument("--scaling-profile-markdown-out", default=rel(SCALING_PROFILE_MD))
    args = parser.parse_args()

    if args.refresh_queue_only:
        report = read_json(resolve(args.out), {})
        if not isinstance(report, dict) or not report:
            print(json.dumps({"trigger_state": "RED", "blocker": "autopilot_report_missing", "report": args.out}, indent=2))
            return 2
        rows = queue_rows(report, args)
        write_jsonl(resolve(args.queue_out), rows)
        print(json.dumps({"trigger_state": "GREEN", "queue": args.queue_out, "row_count": len(rows), "rows": rows}, indent=2, sort_keys=True))
        return 0
    if args.archive_current_only:
        report = read_json(resolve(args.out), {})
        if not isinstance(report, dict) or not report:
            print(json.dumps({"trigger_state": "RED", "blocker": "autopilot_report_missing", "report": args.out}, indent=2))
            return 2
        report["archived_artifacts"] = archive_plan(report)
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
        archive_result = archive_run_artifacts(report)
        report["archived_artifacts"] = archive_result
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
        archive_canonical_outputs(archive_result)
        print(json.dumps({"trigger_state": "GREEN" if not archive_result.get("copy_failures") else "YELLOW", "archived_artifacts": archive_result}, indent=2, sort_keys=True))
        return 0 if not archive_result.get("copy_failures") else 1
    if args.profile_current_only:
        report = read_json(resolve(args.out), {})
        if not isinstance(report, dict) or not report:
            print(json.dumps({"trigger_state": "RED", "blocker": "autopilot_report_missing", "report": args.out}, indent=2))
            return 2
        profile = build_scaling_profile(report)
        summary = object_field(report, "summary")
        summary["scale_efficiency"] = object_field(profile, "summary").get("latest_scale_efficiency")
        report["summary"] = summary
        write_json(resolve(args.out), report)
        write_json(resolve(args.scaling_profile_out), profile)
        write_text(resolve(args.scaling_profile_markdown_out), render_scaling_profile_markdown(profile))
        write_jsonl(resolve(args.queue_out), queue_rows(report, args))
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0 if profile["trigger_state"] in {"GREEN", "YELLOW"} else 2

    run_id = f"post_v4_autopilot_{int(time.time())}"
    started = time.time()
    phases: list[dict[str, Any]] = []
    append_event(run_id, "run", "start", {"execute": bool(args.execute), "args": vars(args)})
    write_heartbeat(run_id, "preflight", "running", started, args, phases, {})

    preflight = preflight_report(args)
    phases.append(phase_record("preflight", 0, preflight, started, time.time()))
    append_event(run_id, "preflight", "finish", preflight)
    blocker: dict[str, Any] = {}
    completion = "dry_run_ready"

    if not preflight["ready"]:
        completion = "precise_blocker"
        blocker = {"kind": "preflight", "detail": preflight["blockers"][0] if preflight["blockers"] else "unknown preflight blocker"}
    elif args.execute:
        completion, blocker = execute_phases(run_id, args, phases, started)

    report = build_report(run_id, args, started, phases, preflight, completion, blocker)
    if args.execute:
        report["archived_artifacts"] = archive_plan(report)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), queue_rows(report, args))
    if args.execute:
        archive_result = archive_run_artifacts(report)
        report["archived_artifacts"] = archive_result
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
    write_heartbeat(run_id, "complete", str(report["trigger_state"]).lower(), started, args, phases, blocker)
    append_event(run_id, "run", "finish", {"trigger_state": report["trigger_state"], "completion": completion, "blocker": blocker})
    if args.execute:
        archive_canonical_outputs(report["archived_artifacts"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    packet = read_json(READINESS_PACKET, {})
    readiness = read_json(POST_DISTILLATION_READINESS, {})
    disk = shutil.disk_usage(ROOT)
    free_gb = disk.free / (1024**3)
    battery = battery_state()
    checkpoint = resolve(args.checkpoint_in)
    post_v4_public_state = post_v4_public_artifact_state()
    gates = [
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE)),
        gate("checkpoint_present", checkpoint.exists(), rel(checkpoint)),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("public_calibration_disallowed", packet.get("public_calibration_allowed") is False, packet.get("public_calibration_allowed")),
        gate("readiness_packet_green", packet.get("trigger_state") == "GREEN", packet.get("trigger_state")),
        gate("public_transfer_still_blocked", readiness.get("trigger_state") == "YELLOW", readiness.get("trigger_state")),
        gate("post_v4_public_artifacts_approved_or_absent", post_v4_public_state["allowed"], post_v4_public_state),
        gate("free_disk_ge_min", free_gb >= float(args.min_free_gb), {"free_gb": round(free_gb, 3), "min_free_gb": float(args.min_free_gb)}),
        gate("battery_allowed_or_ac_power", bool(args.allow_battery or not battery.get("on_battery")), battery),
    ]
    blockers = [row for row in gates if not row["passed"]]
    return {
        "ready": not blockers,
        "blockers": blockers,
        "gates": gates,
        "free_gb": round(free_gb, 3),
        "battery": battery,
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def execute_phases(
    run_id: str,
    args: argparse.Namespace,
    phases: list[dict[str, Any]],
    started: float,
) -> tuple[str, dict[str, Any]]:
    ensure_sidecars()
    commands = phase_commands(args)
    deadline = started + max(0.1, float(args.max_hours)) * 3600.0
    for name, cmd, env in commands:
        if time.time() > deadline:
            return "precise_blocker", {"kind": "time_budget_exhausted", "detail": f"stopped before {name}"}
        write_heartbeat(run_id, name, "running", started, args, phases, {})
        append_event(run_id, name, "start", {"cmd": cmd, "env": sorted(env)})
        phase_start = time.time()
        result = run_command(cmd, env=env, timeout=max(60, int(deadline - time.time())))
        phases.append(phase_record(name, result["returncode"], result, phase_start, time.time()))
        append_event(run_id, name, "finish", result)
        if result["returncode"] != 0:
            return "precise_blocker", {"kind": "phase_failed", "phase": name, "returncode": result["returncode"], "stderr_tail": result["stderr_tail"]}
    learned_gate = read_json(LEARNED_GATE, {})
    learned_summary = object_field(learned_gate, "summary")
    if learned_gate.get("trigger_state") == "GREEN" and int(learned_summary.get("prototype_pass_count") or 0) == 0:
        return "post_v4_private_learned_shadow_ready", {}
    return "precise_blocker", {
        "kind": "learned_gate_not_green",
        "trigger_state": learned_gate.get("trigger_state"),
        "summary": learned_summary,
    }


def phase_commands(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, str]]]:
    commands: list[tuple[str, list[str], dict[str, str]]] = []
    if not args.skip_curriculum:
        commands.append(
            (
                "generate_private_shadow_curriculum",
                [
                    sys.executable,
                    "scripts/post_v4_private_shadow_transfer_v1.py",
                    "--train-rows",
                    str(max(1200, int(args.train_rows))),
                    "--heldout-rows",
                    str(max(160, int(args.heldout_rows))),
                    "--seed",
                    str(int(args.curriculum_seed)),
                    "--private-train-out",
                    rel(TRAIN),
                    "--private-heldout-out",
                    rel(HELDOUT),
                    "--out",
                    rel(CURRICULUM_REPORT),
                    "--markdown-out",
                    rel(CURRICULUM_MD),
                ],
                {},
            )
        )
    if not args.skip_fanout:
        commands.append(
            (
                "build_private_safe_sts_streams",
                [
                    sys.executable,
                    "scripts/private_task_sts_streams.py",
                    "--tasks",
                    rel(HELDOUT),
                    "--out",
                    rel(STS_STREAMS),
                    "--report-out",
                    rel(STS_STREAMS_REPORT),
                    "--task-limit",
                    str(max(0, int(args.private_eval_limit))),
                ],
                {},
            )
        )
        commands.append(
            (
                "fanout_sts_on",
                fanout_command(args, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, STS_STREAMS),
                fanout_env(sts_on=True),
            )
        )
    if not args.skip_control:
        commands.append(
            (
                "fanout_sts_off_control",
                fanout_command(args, CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, EMPTY_STS),
                fanout_env(sts_on=False),
            )
        )
    if not args.skip_score:
        commands.append(
            (
                "score_shadow_heldout",
                [
                    sys.executable,
                    "scripts/broad_private_generalization_score_v1.py",
                    "--heldout",
                    rel(HELDOUT),
                    "--candidates",
                    rel(PRIVATE_CANDIDATES),
                    "--control-candidates",
                    rel(CONTROL_PRIVATE_CANDIDATES),
                    "--timeout-seconds",
                    str(max(1, int(args.score_timeout_seconds))),
                    "--task-limit",
                    str(max(0, int(args.private_eval_limit))),
                    "--min-heldout-rows",
                    str(max(160, int(args.heldout_rows))),
                    "--out",
                    rel(SCORE_REPORT),
                    "--markdown-out",
                    rel(SCORE_MD),
                ],
                {},
            )
        )
    if not args.skip_learned_gate:
        commands.append(
            (
                "learned_only_distillation_gate",
                [
                    sys.executable,
                    "scripts/broad_private_learned_distillation_gate_v1.py",
                    "--heldout",
                    rel(HELDOUT),
                    "--candidates",
                    rel(PRIVATE_CANDIDATES),
                    "--control-candidates",
                    rel(CONTROL_PRIVATE_CANDIDATES),
                    "--score",
                    rel(SCORE_REPORT),
                    "--private-train",
                    rel(TRAIN),
                    "--learned-only-candidates-out",
                    rel(LEARNED_ONLY_CANDIDATES),
                    "--learned-only-score-out",
                    rel(LEARNED_ONLY_SCORE),
                    "--learned-only-score-markdown-out",
                    rel(LEARNED_ONLY_SCORE_MD),
                    "--timeout-seconds",
                    str(max(1, int(args.score_timeout_seconds))),
                    "--task-limit",
                    str(max(0, int(args.private_eval_limit))),
                    "--min-heldout-rows",
                    str(max(160, int(args.heldout_rows))),
                    "--out",
                    rel(LEARNED_GATE),
                    "--markdown-out",
                    rel(LEARNED_GATE_MD),
                ],
                {},
            )
        )
    return commands


def fanout_command(args: argparse.Namespace, private_out: Path, public_out: Path, report_out: Path, sts_streams: Path) -> list[str]:
    return [
        rel(RELEASE),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(HELDOUT),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC),
        "--checkpoint-in",
        rel(resolve(args.checkpoint_in)),
        "--seed",
        str(int(args.seed)),
        "--candidates-per-task",
        str(max(1, int(args.candidates_per_task))),
        "--private-candidate-out",
        rel(private_out),
        "--public-candidate-out",
        rel(public_out),
        "--report-out",
        rel(report_out),
        "--public-task-limit",
        "0",
        "--private-eval-limit",
        str(max(0, int(args.private_eval_limit))),
        "--sts-streams",
        rel(sts_streams),
    ]


def fanout_env(*, sts_on: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
    env.update(beam_precompute_policy_env_from_ablation())
    if not sts_on:
        env["THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY"] = "1"
    return env


def build_report(
    run_id: str,
    args: argparse.Namespace,
    started: float,
    phases: list[dict[str, Any]],
    preflight: dict[str, Any],
    completion: str,
    blocker: dict[str, Any],
) -> dict[str, Any]:
    curriculum = read_json(CURRICULUM_REPORT, {})
    streams = read_json(STS_STREAMS_REPORT, {})
    fanout = read_json(FANOUT_REPORT, {})
    control_fanout = read_json(CONTROL_FANOUT_REPORT, {})
    score = read_json(SCORE_REPORT, {})
    learned_gate = read_json(LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned_gate, "summary")
    curriculum_summary = object_field(curriculum, "summary")
    fanout_summary = object_field(fanout, "summary")
    control_summary = object_field(control_fanout, "summary")
    hard_blocker = completion == "precise_blocker" and bool(blocker)
    green = completion == "post_v4_private_learned_shadow_ready"
    trigger_state = "GREEN" if green else ("RED" if hard_blocker and not preflight.get("ready") else "YELLOW")
    heldout_count = int(first_number(curriculum_summary.get("private_heldout_row_count"), score_summary.get("heldout_task_count"), 0))
    next_heldout = next_private_scale(heldout_count)
    phase_efficiency = scale_efficiency_from_phases(phases, heldout_count, next_heldout)
    post_v4_public_state = post_v4_public_artifact_state()
    gates = [
        gate("preflight_ready", preflight.get("ready") is True, preflight.get("blockers")),
        gate("curriculum_green", curriculum.get("trigger_state") == "GREEN", curriculum.get("trigger_state")),
        gate("sts_streams_green", streams.get("trigger_state") == "GREEN", streams.get("trigger_state")),
        gate("fanout_sts_on_private_only", public_candidate_rows(PUBLIC_CANDIDATES) == 0, {"public_candidate_rows": public_candidate_rows(PUBLIC_CANDIDATES)}),
        gate("fanout_sts_off_control_private_only", public_candidate_rows(CONTROL_PUBLIC_CANDIDATES) == 0, {"public_candidate_rows": public_candidate_rows(CONTROL_PUBLIC_CANDIDATES)}),
        gate("score_green", score.get("trigger_state") == "GREEN", score.get("trigger_state")),
        gate("learned_gate_green", learned_gate.get("trigger_state") == "GREEN", learned_gate.get("trigger_state")),
        gate("learned_prototype_pass_zero", int(learned_summary.get("prototype_pass_count") or 0) == 0, learned_summary.get("prototype_pass_count")),
        gate("public_lock_still_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("post_v4_public_artifacts_approved_or_absent", post_v4_public_state["allowed"], post_v4_public_state),
        gate("external_inference_zero", external_inference_zero(curriculum, streams, score, learned_gate), 0),
    ]
    return {
        "policy": "project_theseus_post_v4_generalization_autopilot_v1",
        "created_utc": now(),
        "run_id": run_id,
        "trigger_state": trigger_state,
        "public_calibration_allowed": False,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "inputs": {
            "execute": bool(args.execute),
            "seed": int(args.seed),
            "curriculum_seed": int(args.curriculum_seed),
            "train_rows": max(1200, int(args.train_rows)),
            "heldout_rows": max(160, int(args.heldout_rows)),
            "candidates_per_task": max(1, int(args.candidates_per_task)),
            "private_eval_limit": max(0, int(args.private_eval_limit)),
            "checkpoint_in": rel(resolve(args.checkpoint_in)),
            "public_tests_used": False,
            "public_solutions_used": False,
            "post_v4_public_artifact_state": post_v4_public_state,
        },
        "summary": {
            "completion_evidence_status": completion,
            "elapsed_seconds": round(time.time() - started, 3),
            "phase_count": len(phases),
            "private_train_row_count": curriculum_summary.get("private_train_row_count"),
            "private_heldout_row_count": curriculum_summary.get("private_heldout_row_count"),
            "sts_stream_task_count": object_field(streams, "summary").get("sts_stream_task_count"),
            "sts_on_candidate_count": fanout_summary.get("private_candidate_count"),
            "sts_on_conditioned_task_count": fanout_summary.get("sts_stream_conditioned_private_task_count"),
            "sts_on_decoder_control_policy_count": fanout_summary.get("sts_decoder_control_policy_applied_task_count"),
            "sts_off_candidate_count": control_summary.get("private_candidate_count"),
            "sts_off_conditioned_task_count": control_summary.get("sts_stream_conditioned_private_task_count"),
            "sts_off_decoder_control_policy_count": control_summary.get("sts_decoder_control_policy_applied_task_count"),
            "pass_count": score_summary.get("pass_count"),
            "pass_rate": score_summary.get("pass_rate"),
            "control_pass_count": score_summary.get("control_pass_count"),
            "control_pass_rate": score_summary.get("control_pass_rate"),
            "sts_delta": score_summary.get("sts_delta"),
            "sts_regressions": score_summary.get("sts_regressions"),
            "learned_only_pass_count": learned_summary.get("learned_only_pass_count"),
            "learned_only_pass_rate": learned_summary.get("learned_only_pass_rate"),
            "learned_token_pass_count": learned_summary.get("learned_token_pass_count"),
            "prototype_pass_count": learned_summary.get("prototype_pass_count"),
            "public_score_unchanged": public_score_summary(),
            "scale_efficiency": phase_efficiency,
            "next_wall": "broad public/general transfer remains unproven; do not unlock public calibration from private shadow evidence alone",
            "external_inference_calls": 0,
        },
        "preflight": preflight,
        "gates": gates,
        "blocker": blocker,
        "phases": phases,
        "artifacts": artifacts(),
        "next_actions": next_actions(trigger_state, completion, learned_summary, score_summary, heldout_count),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def queue_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = object_field(report, "summary")
    green = report.get("trigger_state") == "GREEN"
    current_heldout = int(summary.get("private_heldout_row_count") or object_field(report, "inputs").get("heldout_rows") or args.heldout_rows)
    next_heldout = next_private_scale(current_heldout)
    next_train = max(2400, next_heldout * 5)
    efficiency = scale_efficiency_from_report(report, current_heldout, next_heldout)
    can_scale = next_heldout > current_heldout
    optimize_before_scale = bool(green and can_scale and efficiency.get("optimization_recommended"))
    beam_policy = beam_precompute_policy_from_ablation()
    rows = []
    if optimize_before_scale and not beam_policy.get("ready"):
        rows.append(
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "private_fanout_precompute_ablation",
                "priority": 10,
                "public_calibration_allowed": False,
                "command": [
                    sys.executable,
                    "scripts/post_v4_fanout_precompute_ablation_v1.py",
                    "--task-limit",
                    "64",
                    "--candidates-per-task",
                    str(max(2, int(args.candidates_per_task))),
                ],
                "rationale": (
                    "run a bounded private-only A/B probe for batched beam precompute before "
                    "larger private scale runs"
                ),
                "requires_operator_public_unlock": False,
            }
        )
    if optimize_before_scale and beam_policy.get("ready"):
        rows.append(
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "private_beam_precompute_policy_ready",
                "priority": 10,
                "public_calibration_allowed": False,
                "rationale": "green private 64-task ablation allows the next private runtime probe to disable batched beam precompute",
                "env": beam_policy.get("env", {}),
                "evidence": beam_policy,
                "requires_operator_public_unlock": False,
            }
        )
    if optimize_before_scale:
        rows.append(
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "private_fanout_runtime_profile",
                "priority": 15,
                "public_calibration_allowed": False,
                "command": [
                    sys.executable,
                    "scripts/post_v4_generalization_autopilot_v1.py",
                    "--profile-current-only",
                ],
                "rationale": (
                    "profile measured fanout scaling before another private scale run; "
                    f"estimated next fanout seconds={efficiency.get('estimated_next_fanout_seconds')}"
                ),
                "requires_operator_public_unlock": False,
            }
        )
    if green and not can_scale:
        rows.append(
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "private_shadow_scale_cap_reached",
                "priority": 10,
                "public_calibration_allowed": False,
                "rationale": (
                    f"private learned-shadow proof is already at the configured {PRIVATE_SCALE_CAP} heldout-row cap; "
                    "refresh readiness instead of repeating the same scale run"
                ),
                "scale_efficiency": efficiency,
                "requires_operator_public_unlock": False,
            }
        )
    elif can_scale:
        rows.append(
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "private_shadow_scale_refresh",
                "priority": 30 if optimize_before_scale else (10 if green else 30),
                "public_calibration_allowed": False,
                "command": [
                    sys.executable,
                    "scripts/post_v4_generalization_autopilot_v1.py",
                    "--execute",
                    "--train-rows",
                    str(next_train),
                    "--heldout-rows",
                    str(next_heldout),
                    "--candidates-per-task",
                    str(max(2, int(args.candidates_per_task))),
                    "--allow-battery",
                ],
                "rationale": f"scale the private learned-shadow proof from {current_heldout} to {next_heldout} heldout rows before considering any new public calibration request",
                "env": beam_policy.get("env", {}) if beam_policy.get("ready") else {},
                "precompute_policy": beam_policy,
                "scale_efficiency": efficiency,
                "requires_operator_public_unlock": False,
            }
        )
    rows.extend([
        {
            "policy": "project_theseus_private_autopilot_queue_item_v1",
            "queue": "post_v4_generalization_autopilot_v1",
            "kind": "refresh_public_readiness_no_execute",
            "priority": 20,
            "public_calibration_allowed": False,
            "command": [sys.executable, "scripts/post_distillation_public_transfer_readiness_v1.py"],
            "rationale": "reconcile private learned evidence with the still-spent public score without running public calibration",
            "requires_operator_public_unlock": False,
        },
        {
            "policy": "project_theseus_private_autopilot_queue_item_v1",
            "queue": "post_v4_generalization_autopilot_v1",
            "kind": "bounded_public_operator_review_only",
            "priority": 90,
            "public_calibration_allowed": False,
            "command": [sys.executable, "scripts/public_calibration_readiness_packet.py"],
            "rationale": "produce operator review packet only; do not execute public calibration while the lock remains active",
            "requires_operator_public_unlock": True,
        },
    ])
    if not green:
        rows.insert(
            0,
            {
                "policy": "project_theseus_private_autopilot_queue_item_v1",
                "queue": "post_v4_generalization_autopilot_v1",
                "kind": "repair_failed_private_shadow_gate",
                "priority": 5,
                "public_calibration_allowed": False,
                "rationale": f"repair private shadow blocker before scale refresh: {summary.get('completion_evidence_status')}",
                "requires_operator_public_unlock": False,
            },
        )
    return sorted(rows, key=lambda row: (int(row.get("priority") or 999), str(row.get("kind") or "")))


def next_private_scale(current_heldout: int) -> int:
    if current_heldout < 480:
        return 480
    if current_heldout < 960:
        return 960
    if current_heldout < 1440:
        return 1440
    return min(PRIVATE_SCALE_CAP, current_heldout + 480)


def scale_efficiency_from_phases(phases: list[Any], current_heldout: int, next_heldout: int) -> dict[str, Any]:
    phase_rows = [phase for phase in phases if isinstance(phase, dict)]
    timings = {str(row.get("phase") or ""): float(row.get("elapsed_seconds") or 0.0) for row in phase_rows}
    sts_on = timings.get("fanout_sts_on", 0.0)
    sts_off = timings.get("fanout_sts_off_control", 0.0)
    fanout_total = sts_on + sts_off
    per_task = fanout_total / current_heldout if current_heldout > 0 else 0.0
    estimated_next = per_task * next_heldout if next_heldout > 0 else 0.0
    reasons = []
    if fanout_total >= 1200.0:
        reasons.append("current_fanout_ge_1200s")
    if estimated_next >= 1800.0:
        reasons.append("estimated_next_fanout_ge_1800s")
    if sts_on >= 900.0:
        reasons.append("sts_on_fanout_ge_900s")
    return {
        "current_heldout_rows": current_heldout,
        "next_heldout_rows": next_heldout,
        "scale_cap_heldout_rows": PRIVATE_SCALE_CAP,
        "scale_cap_reached": current_heldout >= PRIVATE_SCALE_CAP,
        "can_scale_next": next_heldout > current_heldout,
        "sts_on_fanout_seconds": round(sts_on, 3),
        "sts_off_control_seconds": round(sts_off, 3),
        "fanout_total_seconds": round(fanout_total, 3),
        "fanout_seconds_per_task": round(per_task, 6),
        "estimated_next_fanout_seconds": round(estimated_next, 3),
        "optimization_recommended": bool(reasons),
        "optimization_reasons": reasons,
        "thresholds": {
            "current_fanout_seconds": 1200.0,
            "estimated_next_fanout_seconds": 1800.0,
            "sts_on_fanout_seconds": 900.0,
        },
    }


def scale_efficiency_from_report(report: dict[str, Any], current_heldout: int, next_heldout: int) -> dict[str, Any]:
    phases = report.get("phases") if isinstance(report.get("phases"), list) else []
    phase_efficiency = scale_efficiency_from_phases(phases, current_heldout, next_heldout)
    if has_measured_fanout(phase_efficiency):
        return phase_efficiency
    summary_efficiency = object_field(object_field(report, "summary"), "scale_efficiency")
    if has_measured_fanout(summary_efficiency):
        return summary_efficiency
    return phase_efficiency


def has_measured_fanout(efficiency: dict[str, Any]) -> bool:
    return float(efficiency.get("fanout_total_seconds") or 0.0) > 0.0


def build_scaling_profile(current_report: dict[str, Any]) -> dict[str, Any]:
    runs = collect_scale_runs(current_report)
    latest = runs[-1] if runs else {}
    latest_scale = int(latest.get("heldout_rows") or 0)
    next_scale = next_private_scale(latest_scale)
    scale_cap_reached = latest_scale >= PRIVATE_SCALE_CAP
    latest_efficiency = object_field(latest, "scale_efficiency")
    if not latest_efficiency:
        latest_efficiency = scale_efficiency_from_phases([], latest_scale, next_scale)
    hotspot_profile = fanout_hotspot_profile()
    post_v4_public_state = post_v4_public_artifact_state()
    gates = [
        gate("scale_runs_present", len(runs) >= 1, {"run_count": len(runs)}),
        gate("latest_run_green", latest.get("trigger_state") == "GREEN", latest.get("trigger_state")),
        gate("latest_prototype_zero", int(latest.get("prototype_pass_count") or 0) == 0, latest.get("prototype_pass_count")),
        gate("latest_public_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("post_v4_public_artifacts_approved_or_absent", post_v4_public_state["allowed"], post_v4_public_state),
        gate("archive_manifests_have_no_copy_failures", all(int(row.get("archive_copy_failure_count") or 0) == 0 for row in runs), {
            "archive_run_count": sum(1 for row in runs if row.get("archive_manifest")),
        }),
        gate("fanout_timings_present", has_measured_fanout(latest_efficiency), latest_efficiency),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in runs), 0),
    ]
    hard_failures = [row for row in gates if not row["passed"] and row["gate"] in {"scale_runs_present", "latest_public_lock_active", "post_v4_public_artifacts_approved_or_absent", "external_inference_zero"}]
    trigger_state = "RED" if hard_failures else "GREEN"
    return {
        "policy": "project_theseus_post_v4_generalization_autopilot_scaling_profile_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "scale_run_count": len(runs),
            "latest_heldout_rows": latest_scale,
            "next_heldout_rows": next_scale,
            "scale_cap_heldout_rows": PRIVATE_SCALE_CAP,
            "scale_cap_reached": scale_cap_reached,
            "latest_scale_efficiency": latest_efficiency,
            "recommendation": (
                "private_shadow_scale_cap_reached_refresh_readiness"
                if scale_cap_reached
                else ("profile_or_optimize_fanout_before_next_scale" if latest_efficiency.get("optimization_recommended") else "scale_next_private_shadow")
            ),
            "public_calibration_allowed": False,
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "public_score_unchanged": public_score_summary(),
            "external_inference_calls": 0,
        },
        "runs": runs,
        "fanout_hotspot_profile": hotspot_profile,
        "gates": gates,
        "next_actions": scaling_profile_next_actions(latest_efficiency, hotspot_profile, scale_cap_reached=scale_cap_reached),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def collect_scale_runs(current_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archive_root = REPORTS / "post_v4_generalization_autopilot_v1_archive"
    for manifest_path in sorted(archive_root.glob("scale*/archive_manifest.json")):
        manifest = read_json(manifest_path, {})
        archive_dir = resolve(str(manifest.get("archive_dir") or manifest_path.parent))
        archived_report = read_json(archive_dir / AUTOPILOT_REPORT.name, {})
        if isinstance(archived_report, dict) and archived_report:
            rows.append(scale_run_row(archived_report, manifest=manifest, source=manifest_path))
    if isinstance(current_report, dict) and current_report:
        current = scale_run_row(current_report, manifest=object_field(current_report, "archived_artifacts"), source=AUTOPILOT_REPORT)
        key = (current.get("heldout_rows"), current.get("run_id"))
        if key not in {(row.get("heldout_rows"), row.get("run_id")) for row in rows}:
            rows.append(current)
    return sorted(rows, key=lambda row: (int(row.get("heldout_rows") or 0), str(row.get("run_id") or "")))


def scale_run_row(report: dict[str, Any], *, manifest: dict[str, Any], source: Path) -> dict[str, Any]:
    summary = object_field(report, "summary")
    heldout = int(first_number(summary.get("private_heldout_row_count"), object_field(report, "inputs").get("heldout_rows"), manifest.get("scale_heldout_rows"), 0))
    next_scale = next_private_scale(heldout)
    efficiency = scale_efficiency_from_report(report, heldout, next_scale)
    return {
        "run_id": report.get("run_id"),
        "source": rel(source),
        "trigger_state": report.get("trigger_state"),
        "heldout_rows": heldout,
        "train_rows": summary.get("private_train_row_count"),
        "pass_count": summary.get("pass_count"),
        "pass_rate": summary.get("pass_rate"),
        "control_pass_count": summary.get("control_pass_count"),
        "learned_only_pass_count": summary.get("learned_only_pass_count"),
        "prototype_pass_count": summary.get("prototype_pass_count"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "sts_on_candidate_count": summary.get("sts_on_candidate_count"),
        "sts_off_candidate_count": summary.get("sts_off_candidate_count"),
        "external_inference_calls": summary.get("external_inference_calls", report.get("external_inference_calls", 0)),
        "scale_efficiency": efficiency,
        "archive_manifest": rel(resolve(str(manifest.get("archive_manifest") or ""))) if manifest else "",
        "archive_copy_failure_count": manifest.get("copy_failure_count", 0) if manifest else 0,
    }


def fanout_hotspot_profile() -> dict[str, Any]:
    sts_on = fanout_report_hotspots("sts_on", FANOUT_REPORT)
    sts_off = fanout_report_hotspots("sts_off_control", CONTROL_FANOUT_REPORT)
    sts_on_ms = int(first_number(sts_on.get("private_candidate_generation_and_write_ms"), 0))
    sts_off_ms = int(first_number(sts_off.get("private_candidate_generation_and_write_ms"), 0))
    shared_beam_ms = max(
        int(first_number(object_field(sts_on, "runtime_breakdown").get("shared_precompute_wall_ms"), 0)),
        int(first_number(object_field(sts_off, "runtime_breakdown").get("shared_precompute_wall_ms"), 0)),
        int(first_number(object_field(sts_on, "shared_timing_ms").get("batched_beam_cache_precompute_shared_ms"), 0)),
        int(first_number(object_field(sts_off, "shared_timing_ms").get("batched_beam_cache_precompute_shared_ms"), 0)),
    )
    delta_ms = max(0, sts_on_ms - sts_off_ms)
    actions = []
    if shared_beam_ms >= 300_000:
        actions.append(
            "persist or reuse batched beam precompute across the STS-on and STS-off same-heldout passes before scaling again"
        )
    if sts_on_ms >= 900_000:
        actions.append(
            "split private_candidate_expansion into STS-conditioned decoder control, candidate expression generation, and rank/sort wall timers"
        )
    if delta_ms >= 300_000:
        actions.append(
            "profile the STS-conditioned decoder path first; STS-on private generation is materially slower than same-seed STS-off control"
        )
    if not actions:
        actions.append("keep collecting fanout timing summaries before the next private scale run")
    return {
        "policy": "project_theseus_post_v4_fanout_hotspot_profile_v1",
        "score_semantics": "runtime_profile_only_not_capability_evidence",
        "sts_on": sts_on,
        "sts_off_control": sts_off,
        "summary": {
            "sts_on_private_candidate_generation_ms": sts_on_ms,
            "sts_off_private_candidate_generation_ms": sts_off_ms,
            "sts_on_minus_sts_off_candidate_generation_ms": delta_ms,
            "shared_beam_precompute_ms": shared_beam_ms,
            "dominant_wall_phase": "private_candidate_expansion",
            "beam_precompute_policy": beam_precompute_policy_from_ablation(),
        },
        "recommended_actions": actions,
    }


def beam_precompute_policy_env_from_ablation() -> dict[str, str]:
    policy = beam_precompute_policy_from_ablation()
    env = policy.get("env") if policy.get("ready") else {}
    return dict(env) if isinstance(env, dict) else {}


def beam_precompute_policy_from_ablation() -> dict[str, Any]:
    report = read_json(PRECOMPUTE_ABLATION, {})
    summary = object_field(report, "summary")
    inputs = object_field(report, "inputs")
    recommendation = str(summary.get("recommendation") or "")
    ready = bool(
        report.get("trigger_state") == "GREEN"
        and recommendation == "prefer_beam_precompute_off_for_next_private_scale_probe"
        and int(first_number(inputs.get("task_limit"), 0)) >= 64
        and float(first_number(summary.get("beam_precompute_off_pass_rate"), 0.0))
        >= float(first_number(summary.get("default_pass_rate"), 1.0))
        and float(first_number(summary.get("runtime_delta_rate"), 0.0)) >= 0.10
        and PUBLIC_LOCK.exists()
        and not POST_V4_PUBLIC_RESULT.exists()
        and not POST_V4_PUBLIC_CANDIDATES.exists()
    )
    return {
        "policy": "project_theseus_batched_beam_precompute_policy_from_private_ablation_v1",
        "ready": ready,
        "source": rel(PRECOMPUTE_ABLATION),
        "recommendation": recommendation,
        "task_limit": int(first_number(inputs.get("task_limit"), 0)),
        "default_pass_rate": summary.get("default_pass_rate"),
        "beam_precompute_off_pass_rate": summary.get("beam_precompute_off_pass_rate"),
        "runtime_delta_rate": summary.get("runtime_delta_rate"),
        "env": {"THESEUS_CODE_LM_BATCHED_BEAM_CACHE": "0"} if ready else {},
        "score_semantics": "private_runtime_policy_probe_not_public_promotion_evidence",
    }


def fanout_report_hotspots(label: str, path: Path) -> dict[str, Any]:
    report = read_json(path, {})
    summary = object_field(report, "summary")
    phase_timing = object_field(summary, "phase_timing_ms")
    if not phase_timing:
        phase_timing = object_field(report, "phase_timing_ms")
    candidate_summary = object_field(object_field(summary, "candidate_task_timing_summary"), "private")
    return {
        "label": label,
        "report": rel(path),
        "trigger_state": report.get("trigger_state"),
        "runtime_breakdown": object_field(object_field(summary, "candidate_fanout_runtime_breakdown"), "private"),
        "runtime_ms": int(first_number(report.get("runtime_ms"), 0)),
        "private_candidate_generation_and_write_ms": int(first_number(phase_timing.get("private_candidate_generation_and_write"), 0)),
        "private_candidate_expansion_ms": int(first_number(phase_timing.get("private_candidate_expansion"), 0)),
        "checkpoint_model_load_ms": int(first_number(phase_timing.get("checkpoint_model_load"), 0)),
        "private_artifact_write_ms": int(first_number(phase_timing.get("private_artifact_write"), 0)),
        "task_count": int(first_number(candidate_summary.get("task_count"), 0)),
        "task_elapsed_ms_total": int(first_number(candidate_summary.get("elapsed_ms_total"), 0)),
        "task_elapsed_ms_max": int(first_number(candidate_summary.get("elapsed_ms_max"), 0)),
        "fanout_worker_count": int(first_number(candidate_summary.get("fanout_worker_count"), 0)),
        "shared_timing_ms": top_numeric_items(object_field(candidate_summary, "shared_timing_ms"), 8),
        "top_timing_ms_total": top_numeric_items(object_field(candidate_summary, "top_timing_ms_total"), 12),
    }


def top_numeric_items(value: dict[str, Any], limit: int) -> dict[str, int]:
    items = []
    for key, item in value.items():
        number = int(first_number(item, 0))
        if number > 0:
            items.append((str(key), number))
    return dict(sorted(items, key=lambda row: (-row[1], row[0]))[:limit])


def scaling_profile_next_actions(efficiency: dict[str, Any], hotspot_profile: dict[str, Any], *, scale_cap_reached: bool = False) -> list[str]:
    if scale_cap_reached:
        return [
            "refresh public readiness without executing calibration",
            "produce an operator-review packet for exactly one bounded public calibration decision",
            "keep public calibration locked unless the operator explicitly approves the guarded run",
        ]
    if efficiency.get("optimization_recommended"):
        return [
            "profile the Rust fanout path before the next private scale run",
            *list(hotspot_profile.get("recommended_actions") or [])[:2],
            "keep public calibration locked until private evidence is reconciled with a bounded operator review",
        ]
    return [
        "run the next private shadow scale refresh from the queue",
        "keep archiving scale-specific artifacts before canonical files are overwritten",
        "keep public calibration locked",
    ]


def render_scaling_profile_markdown(profile: dict[str, Any]) -> str:
    summary = object_field(profile, "summary")
    efficiency = object_field(summary, "latest_scale_efficiency")
    hotspot_profile = object_field(profile, "fanout_hotspot_profile")
    hotspot_summary = object_field(hotspot_profile, "summary")
    lines = [
        "# Post-v4 Generalization Autopilot Scaling Profile",
        "",
        f"- Trigger state: `{profile.get('trigger_state')}`",
        f"- Latest heldout rows: `{summary.get('latest_heldout_rows')}`",
        f"- Next heldout rows: `{summary.get('next_heldout_rows')}`",
        f"- Recommendation: `{summary.get('recommendation')}`",
        f"- Latest fanout total seconds: `{efficiency.get('fanout_total_seconds')}`",
        f"- Estimated next fanout seconds: `{efficiency.get('estimated_next_fanout_seconds')}`",
        f"- Optimization recommended: `{efficiency.get('optimization_recommended')}`",
        f"- Dominant wall phase: `{hotspot_summary.get('dominant_wall_phase')}`",
        f"- STS-on private generation ms: `{hotspot_summary.get('sts_on_private_candidate_generation_ms')}`",
        f"- STS-off private generation ms: `{hotspot_summary.get('sts_off_private_candidate_generation_ms')}`",
        f"- Shared beam precompute ms: `{hotspot_summary.get('shared_beam_precompute_ms')}`",
        "",
        "## Scale Runs",
    ]
    for row in profile.get("runs", []):
        if not isinstance(row, dict):
            continue
        row_efficiency = object_field(row, "scale_efficiency")
        lines.append(
            f"- `{row.get('heldout_rows')}` heldout: pass `{row.get('pass_count')}`, "
            f"learned `{row.get('learned_only_pass_count')}`, fanout `{row_efficiency.get('fanout_total_seconds')}`s"
        )
    lines.extend(["", "## Runtime Hotspots"])
    for action in hotspot_profile.get("recommended_actions", []):
        lines.append(f"- {action}")
    lines.extend(["", "## Next Actions"])
    for action in profile.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def next_actions(
    trigger_state: str,
    completion: str,
    learned_summary: dict[str, Any],
    score_summary: dict[str, Any],
    heldout_count: int = 0,
) -> list[str]:
    if completion == "dry_run_ready":
        return [
            "rerun with --execute to refresh the private shadow curriculum, fanout, score, learned-only gate, ledger, heartbeat, and queue",
            "keep public calibration locked; dry-run readiness is not permission to spend a public surface",
        ]
    if trigger_state == "GREEN":
        if heldout_count >= PRIVATE_SCALE_CAP:
            return [
                "refresh public readiness without executing calibration",
                "produce an operator-review packet for exactly one bounded public calibration decision",
                "keep the public calibration lock active unless the operator explicitly approves the guarded run",
                "use residual family coverage to generate teacher-request packets with provenance, not teacher-applied answers",
            ]
        return [
            "scale the private post-v4 shadow lane to a larger heldout before asking for any public review",
            "feed the queue artifact into the always-active private training loop",
            "keep the public calibration lock active; this private win does not change the 34/160 public score",
            "use residual family coverage to generate teacher-request packets with provenance, not teacher-applied answers",
        ]
    if int(learned_summary.get("prototype_pass_count") or 0) > 0:
        return ["demote prototype-dependent candidates and rerun the learned-only gate before any scale refresh"]
    if float(score_summary.get("pass_rate") or 0.0) < 0.70:
        return ["repair private shadow candidate generation before interpreting broad transfer"]
    return ["inspect the precise blocker in the autopilot report, then rerun this private-only autopilot"]


def artifacts() -> dict[str, str]:
    return {
        "curriculum": rel(CURRICULUM_REPORT),
        "sts_streams": rel(STS_STREAMS),
        "fanout": rel(FANOUT_REPORT),
        "control_fanout": rel(CONTROL_FANOUT_REPORT),
        "score": rel(SCORE_REPORT),
        "learned_gate": rel(LEARNED_GATE),
        "queue": rel(AUTOPILOT_QUEUE),
        "ledger": rel(AUTOPILOT_LEDGER),
        "heartbeat": rel(AUTOPILOT_HEARTBEAT),
    }


def archive_plan(report: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(report, "summary")
    run_id = str(report.get("run_id") or f"run_{int(time.time())}")
    heldout = int(summary.get("private_heldout_row_count") or object_field(report, "inputs").get("heldout_rows") or 0)
    archive_dir = REPORTS / "post_v4_generalization_autopilot_v1_archive" / f"scale{heldout}_{run_id}"
    sources = {
        "private_train": TRAIN,
        "private_heldout": HELDOUT,
        "curriculum_report": CURRICULUM_REPORT,
        "curriculum_markdown": CURRICULUM_MD,
        "sts_streams": STS_STREAMS,
        "sts_streams_report": STS_STREAMS_REPORT,
        "sts_on_candidates": PRIVATE_CANDIDATES,
        "sts_on_public_candidates": PUBLIC_CANDIDATES,
        "sts_on_fanout": FANOUT_REPORT,
        "sts_off_candidates": CONTROL_PRIVATE_CANDIDATES,
        "sts_off_public_candidates": CONTROL_PUBLIC_CANDIDATES,
        "sts_off_fanout": CONTROL_FANOUT_REPORT,
        "score_report": SCORE_REPORT,
        "score_markdown": SCORE_MD,
        "learned_only_candidates": LEARNED_ONLY_CANDIDATES,
        "learned_only_score": LEARNED_ONLY_SCORE,
        "learned_only_score_markdown": LEARNED_ONLY_SCORE_MD,
        "learned_gate": LEARNED_GATE,
        "learned_gate_markdown": LEARNED_GATE_MD,
        "autopilot_report": AUTOPILOT_REPORT,
        "autopilot_markdown": AUTOPILOT_MD,
        "autopilot_queue": AUTOPILOT_QUEUE,
        "autopilot_heartbeat": AUTOPILOT_HEARTBEAT,
        "autopilot_ledger": AUTOPILOT_LEDGER,
    }
    files = []
    for name, source in sources.items():
        files.append(
            {
                "name": name,
                "source": rel(source),
                "archive": rel(archive_dir / source.name),
                "exists": source.exists(),
                "bytes": source.stat().st_size if source.exists() else 0,
            }
        )
    return {
        "policy": "project_theseus_post_v4_generalization_autopilot_archive_v1",
        "archive_dir": rel(archive_dir),
        "archive_manifest": rel(archive_dir / "archive_manifest.json"),
        "scale_heldout_rows": heldout,
        "run_id": run_id,
        "files": files,
    }


def archive_run_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    plan = object_field(report, "archived_artifacts") or archive_plan(report)
    archive_dir = resolve(str(plan.get("archive_dir") or ""))
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    failures = []
    for item in plan.get("files", []):
        if not isinstance(item, dict):
            continue
        source = resolve(str(item.get("source") or ""))
        dest = resolve(str(item.get("archive") or ""))
        if not source.exists():
            failures.append({"name": item.get("name"), "source": rel(source), "reason": "source_missing"})
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(
                {
                    "name": item.get("name"),
                    "source": rel(source),
                    "archive": rel(dest),
                    "bytes": dest.stat().st_size,
                }
            )
        except Exception as exc:  # pragma: no cover - operational fuse
            failures.append({"name": item.get("name"), "source": rel(source), "archive": rel(dest), "reason": f"{type(exc).__name__}: {exc}"})
    result = dict(plan)
    result["copied_files"] = copied
    result["copy_failures"] = failures
    result["copy_failure_count"] = len(failures)
    result["copied_file_count"] = len(copied)
    result["created_utc"] = now()
    write_json(resolve(str(result["archive_manifest"])), result)
    return result


def archive_canonical_outputs(archive_result: dict[str, Any]) -> None:
    archive_dir = resolve(str(archive_result.get("archive_dir") or ""))
    for path in (AUTOPILOT_REPORT, AUTOPILOT_MD, AUTOPILOT_QUEUE, AUTOPILOT_HEARTBEAT):
        if path.exists() and archive_dir.exists():
            shutil.copy2(path, archive_dir / path.name)
    write_json(resolve(str(archive_result["archive_manifest"])), archive_result)


def run_command(cmd: list[str], *, env: dict[str, str], timeout: int) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env.update(env)
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(1, int(timeout)),
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout_seconds": timeout,
        }


def ensure_sidecars() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    EMPTY_PUBLIC.write_text("", encoding="utf-8")
    EMPTY_STS.write_text("", encoding="utf-8")


def phase_record(name: str, returncode: int, evidence: Any, started: float, finished: float) -> dict[str, Any]:
    return {
        "phase": name,
        "returncode": int(returncode),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "elapsed_seconds": round(finished - started, 3),
        "evidence": evidence,
    }


def write_heartbeat(
    run_id: str,
    phase: str,
    state: str,
    started: float,
    args: argparse.Namespace,
    phases: list[dict[str, Any]],
    blocker: dict[str, Any],
) -> None:
    write_json(
        AUTOPILOT_HEARTBEAT,
        {
            "policy": "project_theseus_post_v4_generalization_autopilot_heartbeat_v1",
            "run_id": run_id,
            "updated_utc": now(),
            "phase": phase,
            "state": state,
            "elapsed_seconds": round(time.time() - started, 3),
            "execute": bool(args.execute),
            "phase_count": len(phases),
            "blocker": blocker,
        },
    )


def append_event(run_id: str, phase: str, event: str, payload: Any) -> None:
    AUTOPILOT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "policy": "project_theseus_post_v4_generalization_autopilot_ledger_v1",
        "run_id": run_id,
        "created_utc": now(),
        "phase": phase,
        "event": event,
        "payload": payload,
    }
    with AUTOPILOT_LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def battery_state() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "on_battery": False, "reason": "not_macos"}
    try:
        result = subprocess.run(["pmset", "-g", "batt"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except Exception as exc:  # pragma: no cover - platform fuse
        return {"available": False, "on_battery": False, "reason": f"{type(exc).__name__}: {exc}"}
    text = result.stdout.lower()
    return {
        "available": result.returncode == 0,
        "on_battery": "battery power" in text,
        "raw": result.stdout.strip()[:500],
    }


def public_score_summary() -> dict[str, Any]:
    readiness = read_json(POST_DISTILLATION_READINESS, {})
    summary = object_field(readiness, "summary")
    return {
        "public_pass_rate": summary.get("public_pass_rate"),
        "public_task_count": summary.get("public_task_count"),
        "trigger_state": readiness.get("trigger_state"),
        "operator_lock_active": readiness.get("operator_lock_active"),
        "public_calibration_allowed": readiness.get("public_calibration_allowed"),
    }


def post_v4_public_artifact_state() -> dict[str, Any]:
    present = [rel(path) for path in POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    if not present:
        return {
            "allowed": True,
            "mode": "absent",
            "present_artifacts": [],
            "approval_valid": False,
            "execute_report_valid": False,
            "required_outputs_present": False,
            "operator_lock_active": PUBLIC_LOCK.exists(),
        }
    approval = read_json(OPERATOR_APPROVAL, {})
    execute = read_json(OPERATOR_EXECUTE, {})
    execute_summary = object_field(execute, "summary")
    approval_valid = (
        approval.get("policy") == "project_theseus_public_calibration_operator_approval_v1"
        and approval.get("approved") is True
        and approval.get("proposed_slug") == "post_v4_seed23_5x32"
        and int(first_number(approval.get("max_runs"), 0)) == 1
    )
    execute_valid = (
        execute.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and execute.get("trigger_state") == "GREEN"
        and execute_summary.get("executed") is True
        and execute_summary.get("proposed_slug") == "post_v4_seed23_5x32"
        and execute_summary.get("output_exists_after") is True
        and execute_summary.get("operator_lock_present_after") is True
        and int(first_number(execute_summary.get("run_returncode"), -1)) == 0
    )
    required_outputs_present = all(path.exists() for path in (POST_V4_PUBLIC_RESULT, POST_V4_PUBLIC_TRACES, POST_V4_PUBLIC_CANDIDATES))
    allowed = approval_valid and execute_valid and required_outputs_present and PUBLIC_LOCK.exists()
    return {
        "allowed": allowed,
        "mode": "approved_spent_one_shot" if allowed else "unapproved_or_incomplete",
        "present_artifacts": present,
        "approval_valid": approval_valid,
        "execute_report_valid": execute_valid,
        "required_outputs_present": required_outputs_present,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
    }


def public_candidate_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def external_inference_zero(*reports: dict[str, Any]) -> bool:
    for report in reports:
        if int(report.get("external_inference_calls") or 0) != 0:
            return False
        summary = object_field(report, "summary")
        if int(summary.get("external_inference_calls") or 0) != 0:
            return False
    return True


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    failed = [row for row in gates if not row.get("passed")]
    lines = [
        "# Post-v4 Generalization Autopilot v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Completion: `{summary.get('completion_evidence_status')}`",
        f"- Learned-only pass rate: `{summary.get('learned_only_pass_rate')}`",
        f"- Prototype pass count: `{summary.get('prototype_pass_count')}`",
        f"- Public score unchanged: `{summary.get('public_score_unchanged')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Failed gates: `{len(failed)}`",
        "",
        "## Next Actions",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    if failed:
        lines.extend(["", "## Failed Gates"])
        for row in failed:
            lines.append(f"- `{row.get('gate')}`: `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
