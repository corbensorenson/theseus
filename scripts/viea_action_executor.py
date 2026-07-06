"""Bounded executor for VIEA feedback actions.

The feedback action queue is where VIEA stops being a report family and starts
being an operating loop. This executor reads the queue, validates every action
against a small local allowlist, runs only approved commands, records a durable
ledger, and supports pause/resume/block controls.

Public benchmark data is never used as training material here. Public benchmark
commands are allowed only when the action kind is calibration/reporting.
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
DEFAULT_QUEUE = REPORTS / "feedback_action_queue.json"
DEFAULT_OUT = REPORTS / "viea_action_executor.json"
DEFAULT_MARKDOWN = REPORTS / "viea_action_executor.md"
LEDGER_PATH = REPORTS / "viea_action_execution_ledger.jsonl"
STATE_PATH = REPORTS / "viea_action_executor_state.json"
PAUSE_FLAG = REPORTS / "viea_action_executor_pause.flag"
HOOK_LEDGER_PATH = REPORTS / "hive_tool_hook_ledger.jsonl"

PRIORITY_SCORE = {"critical": 0, "high": 10, "medium": 20, "low": 30}

CALIBRATION_KINDS = {
    "promote_regression_surface",
    "expand_public_adapter_clean_slice",
    "run_same_seed_sts_repair_ablation",
}
PRIVATE_PRESSURE_KINDS = {
    "train_private_semantic_residual_family",
    "write_repo_repair_tasks",
    "train_repo_repair_trace_checkpoint",
    "refresh_symliquid_state_engine",
    "request_teacher_architecture_diagnosis",
}
LOCAL_LEDGER_KINDS = {
    "renew_useful_tool",
    "expire_stale_tool",
}
ALLOWED_KINDS = CALIBRATION_KINDS | PRIVATE_PRESSURE_KINDS | LOCAL_LEDGER_KINDS

ALLOWED_SCRIPTS_BY_KIND = {
    "promote_regression_surface": {"broad_transfer_matrix.py"},
    "expand_public_adapter_clean_slice": {"broad_transfer_matrix.py", "broad_transfer_closure_runner.py"},
    "run_same_seed_sts_repair_ablation": {"sts_repair_ablation.py"},
    "train_private_semantic_residual_family": {"code_residual_curriculum.py", "broad_transfer_closure_runner.py"},
    "write_repo_repair_tasks": {"long_horizon_programming_curriculum.py"},
    "train_repo_repair_trace_checkpoint": {"viea_repo_repair_learner.py"},
    "refresh_symliquid_state_engine": {"symliquid_state_engine.py"},
    "request_teacher_architecture_diagnosis": {
        "teacher_architect_experiment_runner.py",
        "teacher_oracle.py",
    },
}

PUBLIC_TRAINING_FORBIDDEN_TOKENS = {
    "--public-report-out",
    "--public-trace-out",
    "--public-candidate-out",
    "--public-task-manifest-out",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--only-action-id", default="", help="Run or inspect only one stable VIEA action id.")
    parser.add_argument("--mark-blocked", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume-queue", action="store_true")
    args = parser.parse_args()

    if args.pause:
        PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FLAG.write_text("paused\n", encoding="utf-8")
    if args.resume_queue:
        PAUSE_FLAG.unlink(missing_ok=True)
    if args.mark_blocked:
        entry = ledger_entry(
            args.mark_blocked,
            "blocked",
            "manual_block",
            {"reason": args.reason or "blocked_from_dashboard_or_cli"},
        )
        append_jsonl(LEDGER_PATH, entry)

    started = time.perf_counter()
    queue = read_json(resolve(args.queue), {})
    ledger = read_jsonl(LEDGER_PATH)
    state = build_state(ledger)
    actions = prepare_actions(queue, state)
    if args.only_action_id:
        actions = [row for row in actions if str(row.get("action_id") or "") == str(args.only_action_id)]
    sym_state = read_json(REPORTS / "symliquid_state_engine.json", {})
    actions = rank_actions(actions, sym_state)
    paused = PAUSE_FLAG.exists()
    should_execute = bool(args.execute and not args.dry_run and not args.status and not paused)

    results: list[dict[str, Any]] = []
    if should_execute:
        results = execute_actions(actions, state, args=args)
        ledger = read_jsonl(LEDGER_PATH)
        state = build_state(ledger)

    report = build_report(
        queue,
        actions,
        results,
        state,
        started=started,
        args=args,
        paused=paused,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_json(STATE_PATH, state)
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0


def execute_actions(actions: list[dict[str, Any]], state: dict[str, Any], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    steps_left = max(0, int(args.max_steps))
    actions_left = max(0, int(args.max_actions))
    deadline = time.perf_counter() + max(1, int(args.timeout_seconds))
    for action in actions:
        if actions_left <= 0 or steps_left <= 0:
            break
        if time.perf_counter() >= deadline:
            break
        action_id = action["action_id"]
        if args.resume and action_id in state["completed_action_ids"]:
            continue
        if action_id in state["blocked_action_ids"]:
            continue
        validation = validate_action(action, allow_teacher=bool(args.allow_teacher))
        if not validation["allowed"]:
            result = {
                "action_id": action_id,
                "status": "blocked",
                "reason": validation["reason"],
                "kind": action.get("kind"),
                "title": action.get("title"),
            }
            append_jsonl(LEDGER_PATH, ledger_entry(action_id, "blocked", validation["reason"], result))
            results.append(result)
            actions_left -= 1
            steps_left -= 1
            continue
        command = validation.get("command") or []
        if not command:
            result = {
                "action_id": action_id,
                "status": "completed",
                "reason": "local_ledger_action_recorded",
                "kind": action.get("kind"),
                "title": action.get("title"),
            }
            append_jsonl(LEDGER_PATH, ledger_entry(action_id, "completed", "local_ledger_action_recorded", result))
            results.append(result)
            actions_left -= 1
            steps_left -= 1
            continue
        result = run_command(action, command, timeout_seconds=min(max(1, int(args.timeout_seconds)), max(1, int(deadline - time.perf_counter()))))
        append_jsonl(LEDGER_PATH, ledger_entry(action_id, result["status"], result.get("reason", ""), result))
        results.append(result)
        actions_left -= 1
        steps_left -= 1
    return results


def validate_action(action: dict[str, Any], *, allow_teacher: bool) -> dict[str, Any]:
    kind = str(action.get("kind") or "")
    if kind not in ALLOWED_KINDS:
        return {"allowed": False, "reason": f"kind_not_allowlisted:{kind}"}
    if str(action.get("public_data_rule") or "") != "public_benchmarks_calibration_only":
        return {"allowed": False, "reason": "missing_public_calibration_only_rule"}

    command = normalized_command_for_action(action, allow_teacher=allow_teacher)
    if kind in LOCAL_LEDGER_KINDS:
        return {"allowed": True, "reason": "local_ledger_only", "command": []}
    if not command:
        return {"allowed": False, "reason": "empty_command_for_executable_kind"}
    if len(command) < 2:
        return {"allowed": False, "reason": "command_too_short"}
    script = Path(command[1])
    if script.is_absolute():
        try:
            script.relative_to(ROOT / "scripts")
        except ValueError:
            return {"allowed": False, "reason": "script_outside_scripts_dir"}
        script_name = script.name
    else:
        script_name = script.name
        if not str(script).replace("\\", "/").startswith("scripts/"):
            return {"allowed": False, "reason": "script_not_under_scripts"}
    if script_name not in ALLOWED_SCRIPTS_BY_KIND.get(kind, set()):
        return {"allowed": False, "reason": f"script_not_allowed_for_kind:{script_name}"}
    joined = " ".join(command).lower()
    if any(token in joined for token in ["&&", "||", ";", "| powershell", "cmd.exe", "start-process"]):
        return {"allowed": False, "reason": "shell_construct_rejected"}
    if kind in PRIVATE_PRESSURE_KINDS and script_name != "teacher_oracle.py":
        for token in PUBLIC_TRAINING_FORBIDDEN_TOKENS:
            if token.lower() in joined:
                return {"allowed": False, "reason": f"public_training_output_arg_rejected:{token}"}
    calibration_scripts = {"broad_transfer_matrix.py", "sts_repair_ablation.py", "broad_transfer_closure_runner.py"}
    if kind in CALIBRATION_KINDS and script_name not in calibration_scripts and "calibration" not in joined:
        return {"allowed": False, "reason": "non_calibration_public_command_rejected"}
    if script_name == "teacher_oracle.py":
        if "--mode" not in command or "proposal" not in command:
            return {"allowed": False, "reason": "teacher_must_be_proposal_mode"}
        if not allow_teacher and "--queue-only" not in command:
            return {"allowed": False, "reason": "teacher_requires_queue_only_without_allow_teacher"}
        if "--apply" in command or "--apply-mode" in command:
            return {"allowed": False, "reason": "teacher_apply_mode_rejected"}
    return {"allowed": True, "reason": "allowlisted", "command": command}


def normalized_command_for_action(action: dict[str, Any], *, allow_teacher: bool) -> list[str]:
    kind = str(action.get("kind") or "")
    raw = action.get("command") if isinstance(action.get("command"), list) else []
    command = [str(item) for item in raw]
    if kind == "train_repo_repair_trace_checkpoint":
        command = [
            sys.executable,
            "scripts/viea_repo_repair_learner.py",
            "--out",
            "reports/viea_repo_repair_learner.json",
            "--markdown-out",
            "reports/viea_repo_repair_learner.md",
        ]
    elif kind == "refresh_symliquid_state_engine":
        command = [
            sys.executable,
            "scripts/symliquid_state_engine.py",
            "--out",
            "reports/symliquid_state_engine.json",
            "--markdown-out",
            "reports/symliquid_state_engine.md",
        ]
    elif kind == "request_teacher_architecture_diagnosis":
        command = [
            sys.executable,
            "scripts/teacher_architect_experiment_runner.py",
            "--execute",
            "--max-experiments",
            "1",
            "--max-steps",
            "2",
            "--out",
            "reports/teacher_architect_experiment_runner.json",
            "--markdown-out",
            "reports/teacher_architect_experiment_runner.md",
        ]
        if allow_teacher:
            command.append("--allow-teacher")
    if command and looks_like_python(command[0]):
        command[0] = sys.executable
    return command


def run_command(action: dict[str, Any], command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    hook_target = hook_target_for_action(action, command)
    record_tool_hook("before", hook_target, action, command, {"timeout_seconds": timeout_seconds})
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        ok = result.returncode == 0
        payload = {
            "action_id": action["action_id"],
            "kind": action.get("kind"),
            "title": action.get("title"),
            "status": "completed" if ok else "failed",
            "reason": "command_returned_zero" if ok else f"returncode_{result.returncode}",
            "returncode": result.returncode,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        record_tool_hook("after", hook_target, action, command, payload)
        return payload
    except subprocess.TimeoutExpired as exc:
        payload = {
            "action_id": action["action_id"],
            "kind": action.get("kind"),
            "title": action.get("title"),
            "status": "failed",
            "reason": "timeout",
            "returncode": 124,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
        }
        record_tool_hook("after", hook_target, action, command, payload)
        return payload


def prepare_actions(queue: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = queue.get("actions") if isinstance(queue.get("actions"), list) else []
    actions: list[dict[str, Any]] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["action_id"] = stable_action_id(item)
        item["completed"] = item["action_id"] in state["completed_action_ids"]
        item["blocked"] = item["action_id"] in state["blocked_action_ids"]
        validation = validate_action(item, allow_teacher=False)
        item["executor_validation"] = validation["reason"]
        item["executor_allowed"] = bool(validation["allowed"])
        actions.append(item)
    return actions


def rank_actions(actions: list[dict[str, Any]], sym_state: dict[str, Any]) -> list[dict[str, Any]]:
    weights = sym_state.get("action_kind_weights") if isinstance(sym_state.get("action_kind_weights"), dict) else {}
    def score(action: dict[str, Any]) -> tuple[float, str]:
        base = PRIORITY_SCORE.get(str(action.get("priority") or ""), 99)
        weight = float(weights.get(str(action.get("kind") or ""), 0.0) or 0.0)
        done_penalty = 1000 if action.get("completed") or action.get("blocked") else 0
        return (base - weight + done_penalty, str(action.get("title") or ""))
    return sorted(actions, key=score)


def build_state(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    completed = []
    blocked = []
    failed = []
    latest: dict[str, dict[str, Any]] = {}
    for row in ledger:
        action_id = str(row.get("action_id") or "")
        if not action_id:
            continue
        latest[action_id] = row
    for action_id, row in latest.items():
        status = effective_ledger_status(row)
        if status == "completed":
            completed.append(action_id)
        elif status == "blocked":
            blocked.append(action_id)
        elif status == "failed":
            failed.append(action_id)
    return {
        "policy": "project_theseus_viea_action_executor_state_v1",
        "updated_utc": now(),
        "completed_action_ids": sorted(completed),
        "blocked_action_ids": sorted(blocked),
        "failed_action_ids": sorted(failed),
        "ledger_entries": len(ledger),
        "paused": PAUSE_FLAG.exists(),
    }


def effective_ledger_status(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status == "failed" and failed_ledger_has_valid_report(row):
        return "completed"
    return status


def failed_ledger_has_valid_report(row: dict[str, Any]) -> bool:
    """Treat nonzero diagnostic/calibration exits as complete when evidence is valid.

    Some calibration runners return 1/2 for "below floor" or "not promoted"
    even though they emitted a fresh, valid report with no hard execution
    failure. That should block promotion, not poison executor resume state.
    """

    if str(row.get("reason") or "") not in {"returncode_1", "returncode_2"}:
        return False
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    command = payload.get("command") if isinstance(payload.get("command"), list) else []
    report_path = command_out_path(command)
    if report_path is None or not report_path.exists():
        return False
    report = read_json(report_path, {})
    policy = str(report.get("policy") or "")
    trigger = str(report.get("trigger_state") or report.get("status") or "")
    if policy != "project_theseus_broad_transfer_closure_runner_v1":
        return False
    if trigger not in {"GREEN", "YELLOW", "RED"}:
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    hard_failures = int(summary.get("hard_step_failure_count") or 0)
    external_calls = int(report.get("external_inference_calls") or summary.get("external_inference_calls") or 0)
    if hard_failures != 0 or external_calls != 0:
        return False
    if trigger == "RED":
        return bool(summary.get("public_task_count") or summary.get("step_count") is not None)
    return True


def command_out_path(command: list[Any]) -> Path | None:
    values = [str(item) for item in command]
    for idx, item in enumerate(values):
        if item == "--out" and idx + 1 < len(values):
            return resolve(values[idx + 1])
    return None


def build_report(
    queue: dict[str, Any],
    actions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    started: float,
    args: argparse.Namespace,
    paused: bool,
) -> dict[str, Any]:
    invalid = [row for row in actions if not row.get("executor_allowed")]
    failed = [
        row
        for row in results
        if row.get("status") == "failed" and not result_has_valid_diagnostic_report(row)
    ]
    diagnostic_completed = [
        row
        for row in results
        if row.get("status") == "failed" and result_has_valid_diagnostic_report(row)
    ]
    executed = [row for row in results if row.get("status") == "completed"] + diagnostic_completed
    gates = [
        gate("queue_loaded", queue.get("policy") == "project_theseus_feedback_action_queue_v1", queue.get("policy")),
        gate("actions_have_stable_ids", all(row.get("action_id") for row in actions), len(actions)),
        gate("approved_local_commands_only", not invalid, [row.get("title") for row in invalid[:5]]),
        gate("public_data_calibration_only", all(str(row.get("public_data_rule") or "") == "public_benchmarks_calibration_only" for row in actions), "public benchmarks are never training input"),
        gate("ledger_resume_available", LEDGER_PATH.exists() or not results, str(LEDGER_PATH.relative_to(ROOT))),
    ]
    if failed:
        trigger = "RED"
    elif paused or invalid:
        trigger = "YELLOW"
    else:
        trigger = "GREEN"
    return {
        "policy": "project_theseus_viea_action_executor_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "summary": {
            "queue_action_count": len(actions),
            "ready_action_count": sum(1 for row in actions if row.get("executor_allowed") and not row.get("completed") and not row.get("blocked")),
            "executed_this_run": len(executed),
            "failed_this_run": len(failed),
            "diagnostic_completed_this_run": len(diagnostic_completed),
            "completed_total": len(state.get("completed_action_ids") or []),
            "blocked_total": len(state.get("blocked_action_ids") or []),
            "failed_total": len(state.get("failed_action_ids") or []),
            "paused": paused,
            "execute_requested": bool(args.execute),
            "resume_requested": bool(args.resume),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "actions": actions[:50],
        "results": results,
        "state": state,
        "gates": gates,
        "rules": {
            "shell": "never_interpret_queue_commands_with_shell",
            "windows_python": "python executables from queue are normalized to the current sys.executable",
            "public_benchmarks": "calibration_only_not_training",
            "teacher": "proposal_only_no_apply_mode; queue-only unless --allow-teacher",
        },
        "external_inference_calls": 0,
    }


def result_has_valid_diagnostic_report(result: dict[str, Any]) -> bool:
    """Normalize valid below-floor calibration reports during the same run.

    The resume ledger already treats these as completed via
    failed_ledger_has_valid_report(). Without this mirror check, the executor
    could emit RED for a run that produced usable calibration evidence and will
    resume cleanly on the next invocation.
    """

    return failed_ledger_has_valid_report(
        {
            "status": "failed",
            "reason": result.get("reason"),
            "payload": result,
        }
    )


def stable_action_id(action: dict[str, Any]) -> str:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    parts = [
        str(action.get("kind") or ""),
        str(action.get("title") or ""),
        str(evidence.get("card_id") or evidence.get("tool_name") or evidence.get("source") or ""),
        json.dumps(action.get("command") or [], sort_keys=True),
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"viea_action_{digest}"


def ledger_entry(action_id: str, status: str, reason: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": now(),
        "action_id": action_id,
        "status": status,
        "reason": reason,
        "payload": payload,
        "external_inference_calls": 0,
    }


def hook_target_for_action(action: dict[str, Any], command: list[str]) -> str:
    kind = str(action.get("kind") or "")
    script = Path(command[1]).name if len(command) > 1 else ""
    if kind.startswith("train_") or "training" in script or "curriculum" in script:
        return "training_launch"
    if "teacher" in kind or "teacher" in script:
        return "teacher_call"
    return "shell"


def record_tool_hook(phase: str, target: str, action: dict[str, Any], command: list[str], payload: dict[str, Any]) -> None:
    command_hash = hashlib.sha256(json.dumps(command, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    row = {
        "created_utc": now(),
        "policy": "project_theseus_hive_tool_hook_event_v1",
        "phase": phase,
        "target": target,
        "action_id": action.get("action_id"),
        "kind": action.get("kind"),
        "title": action.get("title"),
        "command_hash": command_hash,
        "guards": hook_guards_for_target(target, phase),
        "payload": {
            "status": payload.get("status"),
            "reason": payload.get("reason"),
            "timeout_seconds": payload.get("timeout_seconds"),
            "returncode": payload.get("returncode"),
        },
        "external_inference_calls": 0,
    }
    append_jsonl(HOOK_LEDGER_PATH, row)


def hook_guards_for_target(target: str, phase: str) -> list[str]:
    if target == "training_launch":
        return ["resource_governor", "public_data_leak_check"] if phase == "before" else ["checkpoint_report", "residual_router"]
    if target == "teacher_call":
        return ["teacher_budget", "architecture_only"] if phase == "before" else ["experiment_spec_capture", "no_answer_distillation_check"]
    return ["budget_gate", "permission_envelope"] if phase == "before" else ["record_replay", "route_feedback"]


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# VIEA Action Executor",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- queue_action_count: `{summary.get('queue_action_count')}`",
        f"- ready_action_count: `{summary.get('ready_action_count')}`",
        f"- executed_this_run: `{summary.get('executed_this_run')}`",
        f"- paused: `{summary.get('paused')}`",
        "",
        "## Results",
        "",
    ]
    for row in report.get("results", [])[:20]:
        lines.append(f"- `{row.get('status')}` `{row.get('kind')}` {row.get('title')} ({row.get('reason')})")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- {'PASS' if row.get('passed') else 'FAIL'} `{row.get('gate')}`: {row.get('evidence')}")
    lines.append("")
    return "\n".join(lines)


def looks_like_python(value: str) -> bool:
    lower = value.lower().replace("\\", "/")
    return lower.endswith("python.exe") or lower.endswith("/python") or lower == "python" or lower == sys.executable.lower().replace("\\", "/")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
