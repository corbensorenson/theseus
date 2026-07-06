#!/usr/bin/env python3
"""Full private v5 ecology refresh.

The v5 ecology generator only writes rows and a suggested queue. This runner is
the evidence-producing lane: regenerate rows, build private-safe STS streams,
run STS-on and same-seed STS-off fanout, score the full heldout slice, and run
the learned-only gate. It never runs public calibration.
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

TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "private_ecology_generalization_v5_code_lm_tasks.jsonl"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl"
CURRICULUM_REPORT = REPORTS / "private_ecology_generalization_v5.json"
CURRICULUM_MD = REPORTS / "private_ecology_generalization_v5.md"
CURRICULUM_QUEUE = REPORTS / "private_ecology_generalization_v5_queue.jsonl"

STS_STREAMS = REPORTS / "private_ecology_generalization_v5_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "private_ecology_generalization_v5_private_safe_sts_streams.json"
EMPTY_PUBLIC = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_public.jsonl"
EMPTY_STS = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_sts_streams.jsonl"

PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_ecology_generalization_v5_full480.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_ecology_generalization_v5_full480_empty_public.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_private_ecology_generalization_v5_full480_fanout.json"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_ecology_generalization_v5_full480_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_ecology_generalization_v5_full480_sts_off_empty_public.jsonl"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_private_ecology_generalization_v5_full480_sts_off_fanout.json"

SCORE_REPORT = REPORTS / "private_ecology_generalization_v5_full480_score.json"
SCORE_MD = REPORTS / "private_ecology_generalization_v5_full480_score.md"
LEARNED_ONLY_CANDIDATES = REPORTS / "code_lm_private_candidates_private_ecology_generalization_v5_full480_learned_only.jsonl"
LEARNED_ONLY_SCORE = REPORTS / "private_ecology_generalization_v5_full480_learned_only_score.json"
LEARNED_ONLY_SCORE_MD = REPORTS / "private_ecology_generalization_v5_full480_learned_only_score.md"
LEARNED_GATE = REPORTS / "private_ecology_generalization_v5_full480_learned_distillation_gate.json"
LEARNED_GATE_MD = REPORTS / "private_ecology_generalization_v5_full480_learned_distillation_gate.md"

REFRESH_REPORT = REPORTS / "private_ecology_generalization_v5_refresh.json"
REFRESH_MD = REPORTS / "private_ecology_generalization_v5_refresh.md"
REFRESH_QUEUE = REPORTS / "private_ecology_generalization_v5_refresh_queue.jsonl"
REFRESH_LEDGER = REPORTS / "private_ecology_generalization_v5_refresh_ledger.jsonl"
REFRESH_HEARTBEAT = REPORTS / "private_ecology_generalization_v5_refresh_heartbeat.json"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
OPERATOR_DRY_RUN = REPORTS / "operator_bounded_public_calibration_dry_run.json"
POST_DISTILLATION_READINESS = REPORTS / "post_distillation_public_transfer_readiness_v1.json"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--train-rows", type=int, default=1200)
    parser.add_argument("--heldout-rows", type=int, default=480)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--curriculum-seed", type=int, default=59)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--private-eval-limit", type=int, default=480)
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
    parser.add_argument("--out", default=rel(REFRESH_REPORT))
    parser.add_argument("--markdown-out", default=rel(REFRESH_MD))
    parser.add_argument("--queue-out", default=rel(REFRESH_QUEUE))
    args = parser.parse_args()

    run_id = f"private_ecology_v5_refresh_{int(time.time())}"
    started = time.time()
    phases: list[dict[str, Any]] = []
    append_event(run_id, "run", "start", {"execute": bool(args.execute), "args": vars(args)})
    write_heartbeat(run_id, "preflight", "running", started, args, phases, {})

    preflight = preflight_report(args)
    phases.append(phase_record("preflight", 0, preflight, started, time.time()))
    append_event(run_id, "preflight", "finish", preflight)
    completion = "dry_run_ready"
    blocker: dict[str, Any] = {}
    if not preflight["ready"]:
        completion = "precise_blocker"
        blocker = {"kind": "preflight", "detail": preflight["blockers"][0] if preflight["blockers"] else "unknown"}
    elif args.execute:
        completion, blocker = execute_phases(run_id, args, phases, started)

    report = build_report(run_id, args, started, phases, preflight, completion, blocker)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), queue_rows(report, args))
    write_heartbeat(run_id, "complete", str(report["trigger_state"]).lower(), started, args, phases, blocker)
    append_event(run_id, "run", "finish", {"trigger_state": report["trigger_state"], "completion": completion, "blocker": blocker})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    packet = read_json(READINESS_PACKET, {})
    readiness = read_json(POST_DISTILLATION_READINESS, {})
    dry_run = read_json(OPERATOR_DRY_RUN, {})
    dry_summary = object_field(dry_run, "summary")
    disk = shutil.disk_usage(ROOT)
    free_gb = disk.free / (1024**3)
    battery = battery_state()
    checkpoint = resolve(args.checkpoint_in)
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    gates = [
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE)),
        gate("checkpoint_present", checkpoint.exists(), rel(checkpoint)),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("public_calibration_disallowed", packet.get("public_calibration_allowed") is False and readiness.get("public_calibration_allowed") is False, {
            "packet": packet.get("public_calibration_allowed"),
            "readiness": readiness.get("public_calibration_allowed"),
        }),
        gate(
            "readiness_packet_loaded_not_public_ready_required",
            READINESS_PACKET.exists() and packet.get("public_calibration_allowed") is False,
            {
                "exists": READINESS_PACKET.exists(),
                "trigger_state": packet.get("trigger_state"),
                "public_calibration_allowed": packet.get("public_calibration_allowed"),
                "why": "private v5 repair may run while public-transfer readiness remains YELLOW",
            },
        ),
        gate("operator_dry_run_not_executed", dry_summary.get("executed") is False, dry_summary.get("executed")),
        gate("forbidden_post_v4_public_artifacts_absent", not forbidden_present, forbidden_present),
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
    if learned_gate.get("trigger_state") == "GREEN" and int(first_number(learned_summary.get("prototype_pass_count"), 999)) == 0:
        return "private_ecology_v5_learned_refresh_ready", {}
    return "precise_blocker", {
        "kind": "learned_gate_not_green",
        "trigger_state": learned_gate.get("trigger_state"),
        "summary": learned_summary,
    }


def phase_commands(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, str]]]:
    commands: list[tuple[str, list[str], dict[str, str]]] = []
    eval_limit = max(1, int(args.private_eval_limit))
    min_rows = max(480, eval_limit)
    if not args.skip_curriculum:
        commands.append(
            (
                "generate_private_ecology_v5_rows",
                [
                    sys.executable,
                    "scripts/private_ecology_generalization_v5.py",
                    "--train-rows",
                    str(max(1200, int(args.train_rows))),
                    "--heldout-rows",
                    str(max(480, int(args.heldout_rows))),
                    "--seed",
                    str(int(args.curriculum_seed)),
                    "--private-train-out",
                    rel(TRAIN),
                    "--private-heldout-out",
                    rel(HELDOUT),
                    "--queue-out",
                    rel(CURRICULUM_QUEUE),
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
                    str(eval_limit),
                ],
                {},
            )
        )
        commands.append(
            (
                "fanout_sts_on",
                fanout_command(args, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, STS_STREAMS, eval_limit),
                fanout_env(sts_on=True),
            )
        )
    if not args.skip_control:
        commands.append(
            (
                "fanout_sts_off_control",
                fanout_command(args, CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, EMPTY_STS, eval_limit),
                fanout_env(sts_on=False),
            )
        )
    if not args.skip_score:
        commands.append(
            (
                "score_private_ecology_v5_full480",
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
                    str(eval_limit),
                    "--min-heldout-rows",
                    str(min_rows),
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
                    str(eval_limit),
                    "--min-heldout-rows",
                    str(min_rows),
                    "--out",
                    rel(LEARNED_GATE),
                    "--markdown-out",
                    rel(LEARNED_GATE_MD),
                ],
                {},
            )
        )
    return commands


def fanout_command(args: argparse.Namespace, private_out: Path, public_out: Path, report_out: Path, sts_streams: Path, eval_limit: int) -> list[str]:
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
        str(eval_limit),
        "--sts-streams",
        rel(sts_streams),
    ]


def fanout_env(*, sts_on: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
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
    streams_summary = object_field(streams, "summary")
    fanout_summary = object_field(fanout, "summary")
    control_summary = object_field(control_fanout, "summary")
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    external_zero = external_inference_zero(curriculum, streams, fanout, control_fanout, score, learned_gate)
    fresh = freshness_after_curriculum()
    hard_blocker = completion == "precise_blocker" and bool(blocker)
    green = completion == "private_ecology_v5_learned_refresh_ready"
    gates = [
        gate("preflight_ready", preflight.get("ready") is True, preflight.get("blockers")),
        gate("curriculum_green", curriculum.get("trigger_state") == "GREEN", curriculum.get("trigger_state")),
        gate("sts_streams_green", streams.get("trigger_state") == "GREEN", streams.get("trigger_state")),
        gate("fanout_sts_on_green", fanout.get("trigger_state") == "GREEN", fanout.get("trigger_state")),
        gate("fanout_sts_off_control_green", control_fanout.get("trigger_state") == "GREEN", control_fanout.get("trigger_state")),
        gate("fanout_sts_on_private_only", public_candidate_rows(PUBLIC_CANDIDATES) == 0, {"public_candidate_rows": public_candidate_rows(PUBLIC_CANDIDATES)}),
        gate("fanout_sts_off_control_private_only", public_candidate_rows(CONTROL_PUBLIC_CANDIDATES) == 0, {"public_candidate_rows": public_candidate_rows(CONTROL_PUBLIC_CANDIDATES)}),
        gate("score_green", score.get("trigger_state") == "GREEN", score.get("trigger_state")),
        gate("learned_gate_green", learned_gate.get("trigger_state") == "GREEN", learned_gate.get("trigger_state")),
        gate("learned_prototype_pass_zero", int(first_number(learned_summary.get("prototype_pass_count"), 999)) == 0, learned_summary.get("prototype_pass_count")),
        gate("learned_token_passes_cover_heldout", int(first_number(learned_summary.get("learned_token_pass_count"), 0)) >= max(480, int(args.private_eval_limit)), learned_summary.get("learned_token_pass_count")),
        gate("score_and_learned_fresh_after_curriculum", fresh["fresh"], fresh),
        gate("public_lock_still_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("forbidden_post_v4_public_artifacts_absent", not forbidden_present, forbidden_present),
        gate("external_inference_zero", external_zero, 0),
    ]
    failed = [row for row in gates if not row["passed"]]
    trigger_state = "GREEN" if green and not failed else ("RED" if hard_blocker and not preflight.get("ready") else "YELLOW")
    return {
        "policy": "project_theseus_private_ecology_generalization_v5_refresh",
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
            "heldout_rows": max(480, int(args.heldout_rows)),
            "private_eval_limit": max(1, int(args.private_eval_limit)),
            "candidates_per_task": max(1, int(args.candidates_per_task)),
            "checkpoint_in": rel(resolve(args.checkpoint_in)),
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "summary": {
            "completion_evidence_status": completion,
            "elapsed_seconds": round(time.time() - started, 3),
            "phase_count": len(phases),
            "private_train_row_count": curriculum_summary.get("private_train_row_count"),
            "private_heldout_row_count": curriculum_summary.get("private_heldout_row_count"),
            "sts_stream_task_count": streams_summary.get("sts_stream_task_count"),
            "sts_on_candidate_count": fanout_summary.get("private_candidate_count"),
            "sts_on_conditioned_task_count": fanout_summary.get("sts_stream_conditioned_private_task_count"),
            "sts_off_candidate_count": control_summary.get("private_candidate_count"),
            "sts_off_conditioned_task_count": control_summary.get("sts_stream_conditioned_private_task_count"),
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
            "freshness": fresh,
            "next_wall": "private v5 ecology can refresh cleanly, but broad public/general transfer is still unproven",
            "score_semantics": "private v5 ecology full-refresh evidence only; not public calibration or promotion evidence",
            "external_inference_calls": 0,
        },
        "preflight": preflight,
        "gates": gates,
        "blocker": blocker,
        "phases": phases,
        "artifacts": artifacts(),
        "queue": queue_rows_from_state(trigger_state, completion),
        "next_actions": next_actions(trigger_state, completion),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def queue_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return report.get("queue") if isinstance(report.get("queue"), list) else queue_rows_from_state(str(report.get("trigger_state")), "")


def queue_rows_from_state(trigger_state: str, completion: str) -> list[dict[str, Any]]:
    if trigger_state == "GREEN":
        return [
            queue_item("refresh_teacher_preflight_proposal_only", [sys.executable, "scripts/full_training_teacher_preflight.py", "--profile", "smoke", "--skip-autonomy-readiness"], 50),
            queue_item("refresh_generalization_governor", [sys.executable, "scripts/theseus_generalization_governor_v1.py"], 90),
        ]
    return [
        queue_item(
            "rerun_private_ecology_v5_full_refresh",
            [sys.executable, "scripts/private_ecology_generalization_v5_refresh.py", "--execute"],
            10,
            evidence={"trigger_state": trigger_state, "completion": completion},
        )
    ]


def queue_item(kind: str, command: list[str], priority: int, evidence: Any | None = None) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_ecology_v5_refresh_queue_item_v1",
        "queue": "private_ecology_generalization_v5_refresh",
        "kind": kind,
        "priority": priority,
        "status": "pending",
        "command": command,
        "public_calibration_allowed": False,
        "requires_operator_public_unlock": False,
        "safe_to_execute_without_operator_public_approval": True,
        "evidence": evidence or {},
    }


def next_actions(trigger_state: str, completion: str) -> list[str]:
    if trigger_state == "GREEN":
        return [
            "mark refresh_private_ecology_v5 completed in the private residual ratchet",
            "refresh the generalization governor while the public lock remains active",
            "continue private residual work; this does not change the 34/160 public score",
        ]
    return [
        f"resolve `{completion}` before using v5 ecology as current evidence",
        "do not run public calibration from this failure state",
    ]


def artifacts() -> dict[str, str]:
    return {
        "curriculum": rel(CURRICULUM_REPORT),
        "sts_streams": rel(STS_STREAMS_REPORT),
        "sts_on_fanout": rel(FANOUT_REPORT),
        "sts_off_fanout": rel(CONTROL_FANOUT_REPORT),
        "score": rel(SCORE_REPORT),
        "learned_gate": rel(LEARNED_GATE),
        "refresh_report": rel(REFRESH_REPORT),
        "refresh_queue": rel(REFRESH_QUEUE),
        "refresh_heartbeat": rel(REFRESH_HEARTBEAT),
        "refresh_ledger": rel(REFRESH_LEDGER),
    }


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


def freshness_after_curriculum() -> dict[str, Any]:
    paths = {
        "curriculum": CURRICULUM_REPORT,
        "sts_streams": STS_STREAMS_REPORT,
        "fanout": FANOUT_REPORT,
        "control_fanout": CONTROL_FANOUT_REPORT,
        "score": SCORE_REPORT,
        "learned_gate": LEARNED_GATE,
    }
    mtimes = {name: int(path.stat().st_mtime) if path.exists() else 0 for name, path in paths.items()}
    fresh = bool(
        mtimes["curriculum"]
        and mtimes["sts_streams"] >= mtimes["curriculum"]
        and mtimes["fanout"] >= mtimes["sts_streams"]
        and mtimes["control_fanout"] >= mtimes["sts_streams"]
        and mtimes["score"] >= max(mtimes["fanout"], mtimes["control_fanout"])
        and mtimes["learned_gate"] >= mtimes["score"]
    )
    return {"fresh": fresh, "mtimes": mtimes}


def public_candidate_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def external_inference_zero(*reports: dict[str, Any]) -> bool:
    for report in reports:
        if int(first_number(report.get("external_inference_calls"), 0)) != 0:
            return False
        summary = object_field(report, "summary")
        if int(first_number(summary.get("external_inference_calls"), 0)) != 0:
            return False
    return True


def battery_state() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "on_battery": False, "reason": "not_macos"}
    try:
        result = subprocess.run(["pmset", "-g", "batt"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except Exception as exc:
        return {"available": False, "on_battery": False, "reason": f"{type(exc).__name__}: {exc}"}
    text = result.stdout.lower()
    return {
        "available": result.returncode == 0,
        "on_battery": "battery power" in text,
        "raw": result.stdout.strip()[:500],
    }


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
        REFRESH_HEARTBEAT,
        {
            "policy": "project_theseus_private_ecology_v5_refresh_heartbeat_v1",
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
    REFRESH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "policy": "project_theseus_private_ecology_v5_refresh_ledger_v1",
        "run_id": run_id,
        "created_utc": now(),
        "phase": phase,
        "event": event,
        "payload": payload,
    }
    with REFRESH_LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


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


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
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
        "# Private Ecology Generalization v5 Refresh",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Completion: `{summary.get('completion_evidence_status')}`",
        f"- Pass rate: `{summary.get('pass_rate')}`",
        f"- Learned-only pass rate: `{summary.get('learned_only_pass_rate')}`",
        f"- Learned token passes: `{summary.get('learned_token_pass_count')}`",
        f"- Prototype pass count: `{summary.get('prototype_pass_count')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Failed gates: `{len(failed)}`",
        "",
        "## Artifacts",
    ]
    for key, value in object_field(report, "artifacts").items():
        lines.append(f"- `{key}`: `{value}`")
    if failed:
        lines.extend(["", "## Failed Gates"])
        for row in failed:
            lines.append(f"- `{row.get('gate')}`: `{row.get('evidence')}`")
    lines.append("")
    return "\n".join(lines)


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
