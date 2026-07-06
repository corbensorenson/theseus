#!/usr/bin/env python3
"""Run Broad Private Generalization Ladder v1 unattended.

This runner is designed for long local runs. It writes a heartbeat and phase
ledger before and after every major step, refuses unsafe resource states by
default, keeps public calibration locked, and ends with either GREEN evidence
or a precise private-only blocker.
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
TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
CURRICULUM_REPORT = REPORTS / "broad_private_generalization_ladder_v1.json"
CURRICULUM_MD = REPORTS / "broad_private_generalization_ladder_v1.md"
SCORE_REPORT = REPORTS / "broad_private_generalization_score_v1.json"
SCORE_MD = REPORTS / "broad_private_generalization_score_v1.md"
GATE_REPORT = REPORTS / "broad_private_generalization_gate_v1.json"
GATE_MD = REPORTS / "broad_private_generalization_gate_v1.md"
UNATTENDED_REPORT = REPORTS / "broad_private_generalization_unattended_v1.json"
UNATTENDED_MD = REPORTS / "broad_private_generalization_unattended_v1.md"
LEDGER = REPORTS / "broad_private_generalization_unattended_v1_ledger.jsonl"
HEARTBEAT = REPORTS / "broad_private_generalization_unattended_v1_heartbeat.json"
EMPTY_PUBLIC_MANIFEST = REPORTS / "code_lm_public_tasks_broad_private_generalization_ladder_v1_private_only_empty.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_generalization_ladder_v1_heldout_private_only_empty.jsonl"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_generalization_ladder_v1_heldout_fanout.json"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_generalization_ladder_v1_heldout_sts_off_private_only_empty.jsonl"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_generalization_ladder_v1_heldout_sts_off_fanout.json"
STS_STREAMS = REPORTS / "broad_private_generalization_ladder_v1_heldout_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "broad_private_generalization_ladder_v1_heldout_private_safe_sts_streams.json"
EMPTY_STS = REPORTS / "broad_private_generalization_empty_sts_streams.jsonl"
PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--skip-fanout", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--heldout-rows", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--candidates-per-task", type=int, default=2)
    parser.add_argument("--fanout-task-limit", type=int, default=0)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--caffeinate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--out", default=rel(UNATTENDED_REPORT))
    parser.add_argument("--markdown-out", default=rel(UNATTENDED_MD))
    args = parser.parse_args()

    run_id = f"bpg_v1_{int(time.time())}"
    started = time.time()
    phases: list[dict[str, Any]] = []
    completion = "planned"
    blocker: dict[str, Any] = {}
    trigger_state = "YELLOW"

    append_event(run_id, "run", "start", {"execute": bool(args.execute), "args": vars(args)})
    write_heartbeat(run_id, "preflight", "running", started, args, phases, blocker)
    preflight = preflight_report(args)
    phases.append(phase_record("preflight", 0, preflight, started, time.time()))
    append_event(run_id, "preflight", "finish", preflight)

    if not preflight["ready"]:
        completion = "precise_blocker"
        blocker = {"kind": "preflight", "detail": preflight["blockers"][0] if preflight["blockers"] else "unknown preflight blocker"}
    elif not args.execute:
        completion = "dry_run_ready"
        blocker = {"kind": "dry_run", "detail": "rerun with --execute to generate, fan out, score, and gate"}
    else:
        try:
            completion, blocker = execute_phases(run_id, args, phases, started)
        except Exception as exc:  # pragma: no cover - operational fuse
            completion = "hard_blocker"
            blocker = {"kind": "runner_exception", "detail": f"{type(exc).__name__}: {exc}"}
            append_event(run_id, "runner_exception", "error", blocker)

    score = read_json(SCORE_REPORT, {})
    gate = read_json(GATE_REPORT, {})
    if completion == "green_transfer":
        trigger_state = "GREEN"
        completion = "green_transfer"
    elif completion in {"hard_blocker"}:
        trigger_state = "RED"
    else:
        trigger_state = "YELLOW"

    report = {
        "policy": "project_theseus_broad_private_generalization_unattended_v1",
        "created_utc": now(),
        "run_id": run_id,
        "trigger_state": trigger_state,
        "inputs": {
            "execute": bool(args.execute),
            "train": bool(args.train),
            "checkpoint_in": checkpoint_default(args),
            "seed": int(args.seed),
            "candidates_per_task": int(args.candidates_per_task),
            "fanout_task_limit": int(args.fanout_task_limit),
            "max_hours": float(args.max_hours),
            "allow_battery": bool(args.allow_battery),
            "public_calibration": "locked",
        },
        "summary": {
            "completion_evidence_status": completion,
            "elapsed_seconds": round(time.time() - started, 3),
            "phase_count": len(phases),
            "score_trigger_state": score.get("trigger_state"),
            "gate_trigger_state": gate.get("trigger_state"),
            "heldout_pass_rate": get_path(score, ["summary", "pass_rate"]),
            "no_admissible_task_rate": get_path(score, ["summary", "no_admissible_task_rate"]),
            "sts_delta": get_path(score, ["summary", "sts_delta"]),
            "sts_regressions": get_path(score, ["summary", "sts_regressions"]),
            "candidate_rows": get_path(score, ["summary", "candidate_row_count"]),
            "backend_decision": backend_decision(),
        },
        "precise_blocker": blocker,
        "preflight": preflight,
        "phases": phases,
        "artifacts": artifacts(),
        "next_actions": next_actions(completion, blocker, score, gate),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if args.execute:
        final_gate = refresh_final_gate()
        if final_gate.get("trigger_state"):
            final_completion = get_path(final_gate, ["summary", "completion_evidence_status"])
            final_blocker = get_path(final_gate, ["summary", "precise_blocker"])
            report["final_gate_refresh"] = {
                "created_utc": now(),
                "trigger_state": final_gate.get("trigger_state"),
                "summary": final_gate.get("summary"),
            }
            report["summary"]["gate_trigger_state"] = final_gate.get("trigger_state")
            report["summary"]["final_gate_completion_evidence_status"] = final_completion
            if final_gate.get("trigger_state") != "GREEN":
                report["trigger_state"] = "RED" if final_gate.get("trigger_state") == "RED" else "YELLOW"
                if final_completion:
                    report["summary"]["completion_evidence_status"] = final_completion
                if isinstance(final_blocker, dict) and final_blocker:
                    report["precise_blocker"] = final_blocker
            if isinstance(final_gate.get("next_actions"), list) and final_gate["next_actions"]:
                report["next_actions"] = final_gate["next_actions"]
            write_json(resolve(args.out), report)
            write_text(resolve(args.markdown_out), render_markdown(report))
    write_heartbeat(run_id, "complete", trigger_state.lower(), started, args, phases, blocker)
    append_event(run_id, "run", "finish", {"trigger_state": trigger_state, "completion": completion, "blocker": blocker})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def refresh_final_gate() -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/broad_private_generalization_gate_v1.py",
            "--out",
            rel(GATE_REPORT),
            "--markdown-out",
            rel(GATE_MD),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    gate = read_json(GATE_REPORT, {})
    if isinstance(gate, dict):
        gate["_final_refresh_returncode"] = result.returncode
        gate["_final_refresh_stdout_tail"] = result.stdout[-1200:]
        gate["_final_refresh_stderr_tail"] = result.stderr[-1200:]
    return gate if isinstance(gate, dict) else {}


def execute_phases(run_id: str, args: argparse.Namespace, phases: list[dict[str, Any]], started: float) -> tuple[str, dict[str, Any]]:
    ensure_private_sidecars()
    commands = [
        (
            "generate_curriculum",
            [
                sys.executable,
                "scripts/broad_private_generalization_ladder_v1.py",
                "--train-rows",
                str(max(2400, int(args.train_rows))),
                "--heldout-rows",
                str(max(1000, int(args.heldout_rows))),
                "--seed",
                str(int(args.seed)),
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
    ]
    if args.train:
        commands.append(
            (
                "train_private_checkpoint",
                [
                    sys.executable,
                    "scripts/code_lm_train_once_fanout.py",
                    "--execute",
                    "--private-only",
                    "--slug",
                    "broad_private_generalization_ladder_v1",
                    "--seed",
                    str(int(args.seed)),
                    "--private-count",
                    str(max(320, min(1200, int(args.train_rows)))),
                    "--epochs",
                    "4",
                    "--candidates-per-task",
                    str(max(1, int(args.candidates_per_task))),
                    "--max-high-transfer-private-train",
                    str(max(2400, int(args.train_rows))),
                    "--skip-build",
                    "--allow-active-worker",
                ],
                {},
            )
        )
    if not args.skip_fanout:
        commands.append(("build_private_safe_sts_streams", private_safe_sts_stream_command(args), {}))
        commands.append(("fanout_sts_on", fanout_command(args, checkpoint_default(args), PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, sts_streams=STS_STREAMS), fanout_env(enabled=True)))
    if not args.skip_control:
        commands.append(("fanout_sts_off_control", fanout_command(args, checkpoint_default(args), CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, sts_streams=EMPTY_STS), fanout_env(enabled=False)))
    if not args.skip_score:
        score_command = [
            sys.executable,
            "scripts/broad_private_generalization_score_v1.py",
            "--heldout",
            rel(HELDOUT),
            "--candidates",
            rel(PRIVATE_CANDIDATES),
            "--timeout-seconds",
            str(max(1, int(args.score_timeout_seconds))),
            "--out",
            rel(SCORE_REPORT),
            "--markdown-out",
            rel(SCORE_MD),
        ]
        if not args.skip_control:
            score_command[6:6] = ["--control-candidates", rel(CONTROL_PRIVATE_CANDIDATES)]
        if int(args.fanout_task_limit) > 0:
            score_command.extend(
                [
                    "--task-limit",
                    str(int(args.fanout_task_limit)),
                    "--min-heldout-rows",
                    str(int(args.fanout_task_limit)),
                ]
            )
        commands.append(("score_broad_private_heldout", score_command, {}))
    for name, command, extra_env in commands:
        if time.time() - started > max(60.0, float(args.max_hours) * 3600.0):
            return "precise_blocker", {"kind": "time_limit", "detail": f"max-hours reached before phase {name}"}
        write_heartbeat(run_id, name, "running", started, args, phases, {})
        phase_started = time.time()
        result = run_command(name, command, args, extra_env=extra_env)
        phases.append(phase_record(name, result["returncode"], result, phase_started, time.time()))
        append_event(run_id, name, "finish", result)
        write_heartbeat(run_id, name, "finished", started, args, phases, {})
        if result["returncode"] != 0 and name not in {"score_broad_private_heldout", "gate_broad_private_generalization"}:
            return "hard_blocker", {"kind": name, "detail": result.get("stderr_tail") or result.get("stdout_tail") or "phase failed"}

    score = read_json(SCORE_REPORT, {})
    if score.get("trigger_state") == "GREEN":
        return "green_transfer", {}
    blocker = first_score_blocker(score)
    return "precise_blocker", blocker or {"kind": "unknown_private_score_blocker", "detail": "gate did not clear and no precise blocker was reported"}


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers = []
    disk = shutil.disk_usage(ROOT)
    free_gb = disk.free / (1024**3)
    if free_gb < float(args.min_free_gb):
        blockers.append(f"disk free {free_gb:.2f} GB is below required {float(args.min_free_gb):.2f} GB")
    if not PUBLIC_LOCK.exists():
        blockers.append(f"public calibration lock missing at {rel(PUBLIC_LOCK)}")
    if args.execute and not args.allow_battery and running_on_battery():
        blockers.append("Mac appears to be on battery; pass --allow-battery only if you intentionally want a long battery run")
    if args.execute and not args.skip_fanout and not release_binary().exists():
        blockers.append(f"release binary missing at {rel(release_binary())}; run cargo build --release -p symliquid-cli")
    checkpoint = Path(checkpoint_default(args))
    if args.execute and not args.train and not checkpoint.exists():
        blockers.append(f"checkpoint missing at {rel(checkpoint)}; pass --train or --checkpoint-in")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "free_gb": round(free_gb, 3),
        "public_calibration_lock": {"active": PUBLIC_LOCK.exists(), "path": rel(PUBLIC_LOCK)},
        "battery": battery_state(),
        "release_binary": {"path": rel(release_binary()), "exists": release_binary().exists()},
        "checkpoint": {"path": rel(checkpoint), "exists": checkpoint.exists()},
        "backend_decision": backend_decision(),
    }


def fanout_command(
    args: argparse.Namespace,
    checkpoint: str,
    private_out: Path,
    public_out: Path,
    report_out: Path,
    *,
    sts_streams: Path | None = None,
) -> list[str]:
    command = [
        str(release_binary()),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(HELDOUT),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC_MANIFEST),
        "--checkpoint-in",
        rel(Path(checkpoint)),
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
    ]
    if int(args.fanout_task_limit) > 0:
        command.extend(["--private-eval-limit", str(int(args.fanout_task_limit))])
    if sts_streams is not None:
        command.extend(["--sts-streams", rel(sts_streams)])
    return command


def private_safe_sts_stream_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/private_task_sts_streams.py",
        "--tasks",
        rel(HELDOUT),
        "--out",
        rel(STS_STREAMS),
        "--report-out",
        rel(STS_STREAMS_REPORT),
    ]
    if int(args.fanout_task_limit) > 0:
        command.extend(["--task-limit", str(int(args.fanout_task_limit))])
    return command


def fanout_env(*, enabled: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
    if not enabled:
        env["THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY"] = "1"
    return env


def run_command(name: str, command: list[str], args: argparse.Namespace, *, extra_env: dict[str, str]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(extra_env)
    actual = list(command)
    timeout = max(60, int(float(args.max_hours) * 3600))
    if should_caffeinate(args, name):
        actual = ["caffeinate", "-dimsu", *actual]
    started = time.time()
    try:
        completed = subprocess.run(
            actual,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": actual,
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-8000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": actual,
            "returncode": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": str(exc.stderr or "")[-8000:],
            "timed_out": True,
        }


def should_caffeinate(args: argparse.Namespace, name: str) -> bool:
    return bool(args.caffeinate and platform.system() == "Darwin" and shutil.which("caffeinate") and name not in {"preflight"})


def ensure_private_sidecars() -> None:
    EMPTY_PUBLIC_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EMPTY_PUBLIC_MANIFEST.write_text("", encoding="utf-8")
    PUBLIC_CANDIDATES.write_text("", encoding="utf-8")
    CONTROL_PUBLIC_CANDIDATES.write_text("", encoding="utf-8")
    EMPTY_STS.write_text("", encoding="utf-8")


def checkpoint_default(args: argparse.Namespace) -> str:
    if str(args.checkpoint_in or "").strip():
        return str(resolve(args.checkpoint_in))
    trained = REPORTS / "student_code_lm_checkpoint_broad_private_generalization_ladder_v1.json"
    if args.train or trained.exists():
        return str(trained)
    preferred = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
    if preferred.exists():
        return str(preferred)
    candidates = sorted(REPORTS.glob("student_code_lm_checkpoint*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else str(preferred)


def first_score_blocker(score: dict[str, Any]) -> dict[str, Any]:
    summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    if not summary:
        return {"kind": "score_missing", "detail": "score report missing or empty"}
    if int(summary.get("candidate_row_count") or 0) <= 0:
        return {"kind": "candidate_generation_missing", "detail": "candidate manifest has no rows"}
    if float(summary.get("no_admissible_task_rate") or 0.0) > 0.03:
        return {"kind": "candidate_coverage", "detail": "no-admissible rate above floor", "observed": summary.get("no_admissible_task_rate")}
    if float(summary.get("pass_rate") or 0.0) < 0.70:
        return {
            "kind": "broad_private_transfer_floor",
            "detail": "heldout pass rate below broad private floor",
            "observed": summary.get("pass_rate"),
            "weakest_families": summary.get("weakest_families"),
        }
    if float(summary.get("sts_delta") or 0.0) <= 0.0 or int(summary.get("sts_regressions") or 0) > 0:
        return {"kind": "sts_causal_control", "detail": "STS same-seed control failed", "delta": summary.get("sts_delta"), "regressions": summary.get("sts_regressions")}
    return {}


def backend_decision() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    backend = "cpu"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        backend = "macos_arm64_mlx_candidate_if_available_else_cpu"
    elif system in {"Linux", "Windows"}:
        backend = "cuda_if_requested_available_else_cpu"
    return {"system": system, "machine": machine, "selected": backend}


def running_on_battery() -> bool:
    state = battery_state()
    return bool(state.get("on_battery"))


def battery_state() -> dict[str, Any]:
    if platform.system() != "Darwin" or not shutil.which("pmset"):
        return {"known": False, "on_battery": False, "raw": ""}
    try:
        result = subprocess.run(["pmset", "-g", "batt"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=False)
    except Exception as exc:
        return {"known": False, "on_battery": False, "raw": f"{type(exc).__name__}: {exc}"}
    raw = result.stdout + result.stderr
    return {"known": True, "on_battery": "Battery Power" in raw, "raw": raw.strip()[:1000]}


def artifacts() -> dict[str, str]:
    return {
        "curriculum_report": rel(CURRICULUM_REPORT),
        "heldout": rel(HELDOUT),
        "sts_streams": rel(STS_STREAMS),
        "sts_streams_report": rel(STS_STREAMS_REPORT),
        "private_candidates": rel(PRIVATE_CANDIDATES),
        "control_private_candidates": rel(CONTROL_PRIVATE_CANDIDATES),
        "score_report": rel(SCORE_REPORT),
        "gate_report": rel(GATE_REPORT),
        "ledger": rel(LEDGER),
        "heartbeat": rel(HEARTBEAT),
    }


def next_actions(completion: str, blocker: dict[str, Any], score: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    if completion == "green_transfer":
        return ["Broad private transfer gate is GREEN; keep public calibration locked until an explicit operator-approved calibration run."]
    kind = blocker.get("kind")
    if kind == "dry_run":
        return ["Run this script with --execute for the unattended private proof."]
    if kind == "preflight":
        return [str(blocker.get("detail") or "fix preflight blocker")]
    if kind == "candidate_coverage":
        return ["Repair candidate coverage for broad private families before interpreting score floors."]
    if kind == "broad_private_transfer_floor":
        return ["Patch reusable decoder/learner paths for the weakest private families, then rerun this runner."]
    if kind == "sts_causal_control":
        return ["Fix STS same-seed causal regression before treating STS-default-on as promotion evidence."]
    if gate.get("trigger_state"):
        return [f"Review {rel(GATE_REPORT)} blockers and repair the first failed private-only gate."]
    if score.get("trigger_state"):
        return [f"Review {rel(SCORE_REPORT)} weakest families and repair privately."]
    return ["Rerun with --execute after clearing prerequisites."]


def phase_record(name: str, returncode: int, result: dict[str, Any], started: float, finished: float) -> dict[str, Any]:
    return {
        "phase": name,
        "returncode": int(returncode),
        "elapsed_seconds": round(finished - started, 3),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "timed_out": bool(result.get("timed_out")),
        "stdout_tail": result.get("stdout_tail", "")[-1200:],
        "stderr_tail": result.get("stderr_tail", "")[-2000:],
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
        HEARTBEAT,
        {
            "policy": "project_theseus_broad_private_generalization_unattended_v1_heartbeat",
            "run_id": run_id,
            "updated_utc": now(),
            "phase": phase,
            "state": state,
            "elapsed_seconds": round(time.time() - started, 3),
            "phase_count": len(phases),
            "max_hours": float(args.max_hours),
            "blocker": blocker,
        },
    )


def append_event(run_id: str, phase: str, event: str, payload: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "created_utc": now(), "phase": phase, "event": event, "payload": payload}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def get_path(value: Any, path: list[str]) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def release_binary() -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return ROOT / "target" / "release" / name


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    blocker = report.get("precise_blocker") if isinstance(report.get("precise_blocker"), dict) else {}
    lines = [
        "# Broad Private Generalization Unattended V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Completion evidence: `{summary.get('completion_evidence_status')}`",
        f"- Elapsed seconds: {summary.get('elapsed_seconds')}",
        f"- Heldout pass rate: {summary.get('heldout_pass_rate')}",
        f"- No-admissible rate: {summary.get('no_admissible_task_rate')}",
        f"- STS delta: {summary.get('sts_delta')}",
        f"- Candidate rows: {summary.get('candidate_rows')}",
        f"- Blocker: `{blocker.get('kind') or 'none'}` {blocker.get('detail') or ''}",
        "",
        "## Artifacts",
    ]
    for key, value in report.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
