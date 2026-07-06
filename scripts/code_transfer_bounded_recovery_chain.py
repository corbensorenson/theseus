#!/usr/bin/env python3
"""Run a bounded recovery chain after an oversized Code LM closure times out.

The chain is deliberately conservative:
1. score partial artifacts as diagnostic-only evidence
2. compute a resource-aware budget
3. run one small resumable subshard with unique recovery artifact paths
4. merge completed subshards and gate only after the completed recovery report exists
5. run private_public_transfer_proof only as metadata, never public benchmarks
"""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_process_guard import windows_code_lm_process_rows  # noqa: E402
from code_lm_private_rows import high_transfer_private_rows_string  # noqa: E402

REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "code_transfer_bounded_recovery_chain.json"
DEFAULT_MARKDOWN = REPORTS / "code_transfer_bounded_recovery_chain.md"
PRIVATE_ROWS = high_transfer_private_rows_string()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slug", default="private_pressure_private_recovery")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    out = resolve(args.out)
    markdown = resolve(args.markdown_out)
    state: dict[str, Any] = {
        "policy": "project_theseus_code_transfer_bounded_recovery_chain_v1",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": bool(args.execute),
        "current_phase": "planned",
        "slug": args.slug,
        "chunk_plan": {
            "mode": "small_subshards_then_gate",
            "resume_key": f"reports/code_lm_closure_{args.slug}_chunked_v4_merged.json",
            "reason": "replace one oversized closure with bounded resumable public/private subshards and unique artifacts",
        },
        "phases": [],
        "rules": {
            "public_calibration": "never run by this chain",
            "public_benchmark_training": "forbidden",
            "resource_policy": "one bounded code closure at a time; unique recovery artifacts preserve previous timeout evidence",
        },
        "external_inference_calls": 0,
    }
    write_progress(out, markdown, state)
    if not args.execute:
        print(json.dumps(state, indent=2))
        return 0

    env = os.environ.copy()
    env.update(
        {
            "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "32",
            "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
            "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
            "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
            "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
        }
    )

    existing_rust = read_json(REPORTS / f"code_lm_closure_rust_{args.slug}.json", {})
    sts_ready = sts_artifact_ready(args.slug)
    detached_workers = active_recovery_workers(args.slug)
    if detached_workers and existing_rust.get("run_status") == "in_progress":
        state["trigger_state"] = "YELLOW"
        state["current_phase"] = "bounded_private_pressure_closure_detached_worker_running"
        state["detached_workers"] = detached_workers
        state["rust_progress_stage"] = existing_rust.get("progress_stage")
        state["rust_runtime_ms"] = existing_rust.get("runtime_ms")
        state["next_actions"] = [
            "Leave the detached Rust Code LM worker running; do not score partial artifacts or start another closure until it exits.",
            "When the Rust report completes, rerun this chain so it can skip the completed chunk and proceed to the decoder/private gate.",
        ]
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 0

    state["trigger_state"] = "RUNNING"
    initial_steps = [
        ("partial_artifact_scorer", [sys.executable, "scripts/code_lm_partial_artifact_scorer.py"]),
        ("resource_aware_execution_policy", [sys.executable, "scripts/resource_aware_execution_policy.py"]),
        ("baseline_transfer_snapshot", [sys.executable, "scripts/private_public_transfer_proof.py", "--write-baseline"]),
    ]
    for name, command in initial_steps:
        phase = run_phase(name, command, env, 900, out, markdown, state)
        state["phases"].append(phase)
        if phase["returncode"] != 0 and name != "partial_artifact_scorer":
            return fail(state, out, markdown, name, phase)

    resource = read_json(REPORTS / "resource_aware_execution_policy.json", {})
    budget = object_field(resource, "recommended_code_lm_budget")
    if not budget.get("start_new_code_closure"):
        if budget.get("start_new_chunked_code_closure"):
            phase = run_phase(
                "code_lm_chunked_recovery",
                [
                    sys.executable,
                    "scripts/code_lm_chunked_recovery.py",
                    "--execute",
                    "--slug",
                    f"{args.slug}_chunked_v4",
                    "--shard-count",
                    "16",
                    "--max-shards-per-run",
                    "1",
                ],
                env,
                int(budget.get("chunk_rust_timeout_seconds") or 900) + 1800,
                out,
                markdown,
                state,
            )
            state["phases"].append(phase)
            chunked = read_json(REPORTS / "code_lm_chunked_recovery.json", {})
            state["trigger_state"] = chunked.get("trigger_state") or ("YELLOW" if phase["returncode"] == 0 else "RED")
            state["current_phase"] = "chunked_recovery"
            state["chunked_recovery"] = {
                "report": "reports/code_lm_chunked_recovery.json",
                "returncode": phase["returncode"],
                "completed_shards": chunked.get("completed_shards"),
                "shard_count": chunked.get("shard_count"),
                "ready_for_public_calibration": chunked.get("ready_for_public_calibration"),
                "next_actions": chunked.get("next_actions"),
            }
            state["next_actions"] = chunked.get("next_actions") or [
                "Continue chunked recovery; do not restart the monolithic Code LM closure."
            ]
            write_progress(out, markdown, state)
            print(json.dumps(state, indent=2))
            return 0 if phase["returncode"] == 0 else phase["returncode"]
        state["trigger_state"] = "YELLOW"
        state["current_phase"] = "deferred_by_resource_policy"
        state["resource_policy"] = budget
        state["next_actions"] = ["Do not stack another Code LM closure while a heavy code worker is active; rerun this chain after the worker exits."]
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 0

    existing_closure = read_json(REPORTS / f"code_lm_closure_{args.slug}.json", {})
    rust_completed = (
        existing_rust.get("run_status") == "completed"
        and bool(existing_rust.get("private_candidate_manifest") and existing_rust.get("public_candidate_manifest"))
        and rust_fresh_for_sts(existing_rust, args.slug, sts_ready)
    )
    if (
        existing_closure.get("run_status") == "completed"
        and existing_closure.get("trigger_state") in {"GREEN", "YELLOW"}
    ) or rust_completed:
        phase = {
            "name": "bounded_private_pressure_closure",
            "returncode": 0,
            "skipped": True,
            "reason": "completed_recovery_closure_already_present"
            if existing_closure.get("run_status") == "completed"
            else "completed_rust_recovery_chunk_without_outer_wrapper",
            "report": f"reports/code_lm_closure_{args.slug}.json",
            "rust_report": f"reports/code_lm_closure_rust_{args.slug}.json",
            "completed_utc": now(),
        }
        state["phases"].append(phase)
    else:
        closure_command = private_closure_command(args.slug, budget)
        closure_timeout = int(budget.get("rust_timeout_seconds") or 3600) + 900
        phase = run_phase("bounded_private_pressure_closure", closure_command, env, closure_timeout, out, markdown, state)
        state["phases"].append(phase)
    if phase["returncode"] != 0:
        detached_workers = active_recovery_workers(args.slug)
        if phase.get("timed_out") and detached_workers:
            state["trigger_state"] = "YELLOW"
            state["current_phase"] = "bounded_private_pressure_closure_detached_worker_running"
            state["detached_workers"] = detached_workers
            state["next_actions"] = [
                "Leave the detached Rust Code LM worker running; do not score partial artifacts or start another closure until it exits.",
                "When the Rust report completes, rerun this chain so it can skip the completed chunk and proceed to the decoder/private gate.",
            ]
            write_progress(out, markdown, state)
            print(json.dumps(state, indent=2))
            return 0
        post = run_phase(
            "partial_artifact_scorer_after_bounded_failure",
            [
                sys.executable,
                "scripts/code_lm_partial_artifact_scorer.py",
                "--closure-report",
                f"reports/code_lm_closure_{args.slug}.json",
                "--rust-report",
                f"reports/code_lm_closure_rust_{args.slug}.json",
                "--heartbeat",
                f"reports/code_lm_closure_rust_{args.slug}.heartbeat.json",
                "--out",
                f"reports/code_lm_partial_artifact_score_{args.slug}.json",
                "--markdown-out",
                f"reports/code_lm_partial_artifact_score_{args.slug}.md",
            ],
            env,
            900,
            out,
            markdown,
            state,
        )
        state["phases"].append(post)
        return fail(state, out, markdown, "bounded_private_pressure_closure", phase)

    gate_command = [
        sys.executable,
        "scripts/decoder_v2_private_ablation_gate.py",
        "--closure-report",
        f"reports/code_lm_closure_{args.slug}.json",
        "--closure-report",
        "reports/code_lm_closure_private_pressure_private.json",
    ]
    for name, command, timeout in [
        ("decoder_v2_private_ablation_gate", gate_command, 1800),
        ("private_public_transfer_proof", [sys.executable, "scripts/private_public_transfer_proof.py"], 900),
        ("sts_causal_decoder_ablation", [sys.executable, "scripts/sts_causal_decoder_ablation.py"], 900),
        ("symliquid_state_engine", [sys.executable, "scripts/symliquid_state_engine.py"], 900),
        ("agent_lane_transfer_gate", [sys.executable, "scripts/agent_lane_transfer_gate.py"], 900),
        ("maturity_integrity_audit", [sys.executable, "scripts/maturity_integrity_audit.py"], 900),
        ("asi_wall_breaker_governor", [sys.executable, "scripts/asi_wall_breaker_governor.py"], 900),
        ("a_plus_operating_scorecard", [sys.executable, "scripts/a_plus_operating_scorecard.py"], 900),
    ]:
        phase = run_phase(name, command, env, timeout, out, markdown, state)
        state["phases"].append(phase)
        if phase["returncode"] != 0:
            return fail(state, out, markdown, name, phase)

    proof = read_json(REPORTS / "private_public_transfer_proof.json", {})
    gate = read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {})
    state["trigger_state"] = "GREEN" if proof.get("ready_for_public_calibration") else "YELLOW"
    state["current_phase"] = "completed"
    state["completed_utc"] = now()
    state["ready_for_public_calibration"] = bool(proof.get("ready_for_public_calibration"))
    state["decoder_gate_ready"] = bool(gate.get("ready_for_public_calibration"))
    state["next_actions"] = (
        ["Transfer proof is GREEN; allow at most one bounded public 4-card calibration."]
        if state["ready_for_public_calibration"]
        else ["Keep public calibration locked; use the completed bounded closure/gate residuals for the next decoder fix."]
    )
    write_progress(out, markdown, state)
    print(json.dumps(state, indent=2))
    return 0


def private_closure_command(slug: str, budget: dict[str, Any]) -> list[str]:
    return [
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
        str(int(budget.get("private_count") or 320)),
        "--epochs",
        str(int(budget.get("epochs") or 4)),
        "--candidates-per-task",
        str(int(budget.get("candidates_per_task") or 8)),
        "--disable-extra-private-train",
        "--disable-residual-private-train",
        "--disable-repo-repair-private-train",
        "--high-transfer-private-train-jsonl",
        PRIVATE_ROWS,
        "--max-high-transfer-private-train",
        str(int(budget.get("max_high_transfer_private_train") or 4800)),
        "--max-rust-work-steps",
        str(int(budget.get("max_rust_work_steps") or 3_000_000)),
        "--rust-timeout-seconds",
        str(int(budget.get("rust_timeout_seconds") or 5400)),
        "--sts-timeout-seconds",
        str(int(budget.get("sts_timeout_seconds") or 1800)),
        "--private-curriculum-out",
        f"data/private_code_curriculum/code_lm_closure_{slug}.jsonl",
        "--public-task-manifest-out",
        f"reports/code_lm_public_tasks_{slug}.jsonl",
        "--checkpoint-out",
        f"reports/student_code_lm_checkpoint_{slug}.json",
        "--private-candidate-out",
        f"reports/code_lm_private_candidates_{slug}.jsonl",
        "--public-candidate-out",
        f"reports/student_code_candidates_{slug}.jsonl",
        "--rust-report-out",
        f"reports/code_lm_closure_rust_{slug}.json",
        "--public-report-out",
        f"reports/real_code_benchmark_graduation_{slug}_skipped.json",
        "--public-trace-out",
        f"reports/real_code_benchmark_traces_{slug}_skipped.jsonl",
        "--out",
        f"reports/code_lm_closure_{slug}.json",
        "--sts-conditioning-input-out",
        f"reports/code_lm_sts_conditioning_input_{slug}.jsonl",
        "--sts-generation-out",
        f"reports/code_lm_sts_public_generations_{slug}.jsonl",
        "--sts-conditioning-checkpoint-out",
        f"reports/code_lm_sts_conditioning_checkpoint_{slug}.json",
        "--sts-conditioning-report-out",
        f"reports/code_lm_sts_conditioning_report_{slug}.json",
        "--sts-decoder-control-policy-jsonl",
        "reports/sts_decoder_control_rows.jsonl",
        "--lock-path",
        f"reports/code_lm_closure_{slug}.lock",
        "--typed-edge-exec-receiver-v1",
        "--edge-obligation-decode-gate-v1",
        "--private-type-shape-receiver-veto-v1",
        "--edge-obligation-report-out",
        f"reports/edge_obligation_decode_gate_v1_{slug}.json",
        "--edge-obligation-markdown-out",
        f"reports/edge_obligation_decode_gate_v1_{slug}.md",
    ]


def run_phase(
    name: str,
    command: list[str],
    env: dict[str, str],
    timeout_seconds: int,
    out: Path,
    markdown: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    log_path = REPORTS / f"{name}.bounded_recovery.log"
    phase = {
        "name": name,
        "started_utc": now(),
        "command": command,
        "timeout_seconds": timeout_seconds,
        "log_path": rel_or_abs(log_path),
        "returncode": None,
    }
    state["current_phase"] = name
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        phase["pid"] = proc.pid
        state["active_child"] = phase
        while True:
            rc = proc.poll()
            elapsed = time.time() - started
            phase["elapsed_seconds"] = int(elapsed)
            phase["last_progress_utc"] = now()
            if rc is not None:
                phase["returncode"] = int(rc)
                phase["completed_utc"] = now()
                break
            if elapsed > timeout_seconds:
                proc.kill()
                phase["returncode"] = 124
                phase["completed_utc"] = now()
                phase["timed_out"] = True
                break
            write_progress(out, markdown, state)
            time.sleep(30)
    state.pop("active_child", None)
    return phase


def fail(state: dict[str, Any], out: Path, markdown: Path, name: str, phase: dict[str, Any]) -> int:
    state["trigger_state"] = "RED"
    state["current_phase"] = "failed"
    state["failed_phase"] = name
    state["next_actions"] = ["Do not public-calibrate; inspect bounded recovery logs and partial artifact score before another run."]
    write_progress(out, markdown, state)
    print(json.dumps(state, indent=2))
    return int(phase.get("returncode") or 1)


def write_progress(out: Path, markdown: Path, state: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = now()
    out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    active = state.get("active_child") if isinstance(state.get("active_child"), dict) else {}
    lines = [
        "# Code Transfer Bounded Recovery Chain",
        "",
        f"- Status: **{state.get('trigger_state')}**",
        f"- Current phase: `{state.get('current_phase')}`",
        f"- Slug: `{state.get('slug')}`",
        f"- Updated: `{state.get('updated_utc')}`",
    ]
    if active:
        lines.extend(
            [
                f"- Active PID: `{active.get('pid')}`",
                f"- Active elapsed seconds: `{active.get('elapsed_seconds', 0)}`",
                f"- Active log: `{active.get('log_path')}`",
            ]
        )
    lines.extend(["", "## Phases", ""])
    for phase in state.get("phases", []):
        lines.append(f"- `{phase.get('name')}` rc `{phase.get('returncode')}` elapsed `{phase.get('elapsed_seconds')}`s")
    if state.get("next_actions"):
        lines.extend(["", "## Next Actions", ""])
        for action in state["next_actions"]:
            lines.append(f"- {action}")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def object_field(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def active_recovery_workers(slug: str) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for row in windows_code_lm_process_rows("train-code-lm-closure|code_lm_closure.py"):
        name = str(row.get("name") or "")
        command_line = str(row.get("command") or "")
        if slug not in command_line:
            continue
        workers.append(
            {
                "pid": row.get("pid"),
                "parent_pid": row.get("parent_pid"),
                "name": name,
                "command_hint": "train-code-lm-closure"
                if "train-code-lm-closure" in command_line
                else "code_lm_closure.py",
            }
        )
    return workers


def sts_artifact_ready(slug: str) -> bool:
    generation = REPORTS / f"code_lm_sts_public_generations_{slug}.jsonl"
    report = read_json(REPORTS / f"code_lm_sts_conditioning_report_{slug}.json", {})
    return bool(
        generation.exists()
        and generation.stat().st_size > 0
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("run_status") != "timed_out_process_tree_killed"
        and not bool(get_nested(report, ["summary", "public_benchmark_solutions_included"], False))
    )


def rust_fresh_for_sts(rust_report: dict[str, Any], slug: str, sts_ready: bool) -> bool:
    if not sts_ready:
        return True
    generation = REPORTS / f"code_lm_sts_public_generations_{slug}.jsonl"
    rust_path = REPORTS / f"code_lm_closure_rust_{slug}.json"
    rust_used_sts = bool(get_nested(rust_report, ["summary", "input_file_status", "sts_streams", "exists"], False))
    if not rust_used_sts:
        return False
    try:
        return rust_path.stat().st_mtime >= generation.stat().st_mtime
    except OSError:
        return False


def get_nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
