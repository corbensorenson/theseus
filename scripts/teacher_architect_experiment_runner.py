"""Execute teacher-as-architect experiment closures safely.

The teacher may diagnose architecture walls and propose experiments. It may not
provide public benchmark answers, hidden tests, or apply changes directly. This
runner closes the loop as bounded local stages:

residual cluster -> teacher diagnosis -> private eval -> public calibration ->
promote/rollback decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_CLOSURE = REPORTS / "teacher_architect_closure.json"
LEDGER_PATH = REPORTS / "teacher_architect_experiment_ledger.jsonl"

ALLOWED_STAGE_SCRIPTS = {
    "teacher_diagnosis": {"teacher_oracle.py"},
    "private_eval": {
        "architecture_experiment_runner.py",
        "code_residual_curriculum.py",
        "sts_repair_ablation.py",
        "long_horizon_programming_curriculum.py",
    },
    "public_calibration": {"broad_transfer_matrix.py"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", default=str(DEFAULT_CLOSURE.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/teacher_architect_experiment_runner.json")
    parser.add_argument("--markdown-out", default="reports/teacher_architect_experiment_runner.md")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    started = time.perf_counter()
    closure = read_json(resolve(args.closure), {})
    experiments = closure.get("closures") if isinstance(closure.get("closures"), list) else []
    selected = experiments[: max(0, int(args.max_experiments))]
    results = []
    if args.execute:
        results = run_experiments(selected, args=args)
    decisions = build_decisions(selected, results)
    gates = [
        gate("closure_loaded", closure.get("policy") == "project_theseus_teacher_architect_closure_v1", closure.get("policy")),
        gate("experiment_specs_present", len(experiments) > 0, len(experiments)),
        gate("proposal_only_teacher", all(not stage_forbidden(row) for row in selected), "no apply/no answers/no hidden tests"),
        gate("teacher_calls_reason_and_context_bound", all(teacher_stage_context_ok(row) for row in selected), "teacher stages carry --reason plus prompt/evidence context"),
        gate("closed_loop_stage_order_declared", all(stage_order_ok(row) for row in selected), stage_orders(selected)),
        gate("private_eval_before_public_calibration", all(private_eval_before_public(row) for row in selected), stage_orders(selected)),
        gate("rollback_or_promotion_rule_present", all(rollback_or_promotion_rule_present(row) for row in selected), [row.get("id") for row in selected]),
        gate("public_calibration_only", all(public_calibration_only(row) for row in selected), "broad_transfer_matrix only for public stage"),
    ]
    failed = [row for row in results if row.get("returncode") not in {0, None}]
    payload = {
        "policy": "project_theseus_teacher_architect_experiment_runner_v1",
        "created_utc": now(),
        "trigger_state": "RED" if failed else ("GREEN" if all(row["passed"] for row in gates) else "YELLOW"),
        "summary": {
            "available_experiments": len(experiments),
            "selected_experiments": len(selected),
            "executed_stage_count": len(results),
            "failed_stage_count": len(failed),
            "teacher_allowed": bool(args.allow_teacher),
            "execute_requested": bool(args.execute),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "promotion_evidence": False,
            "external_inference_calls": 0,
        },
        "decisions": decisions,
        "results": results,
        "gates": gates,
        "rules": {
            "teacher": "architecture diagnosis/proposal only",
            "public_benchmarks": "public calibration only, never training or teacher answers",
            "decision": "promote only when private eval and public calibration improve without regressions",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 2 if payload["trigger_state"] == "RED" else 0


def run_experiments(experiments: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    steps_left = max(0, int(args.max_steps))
    deadline = time.perf_counter() + max(1, int(args.timeout_seconds))
    for exp in experiments:
        commands = exp.get("commands") if isinstance(exp.get("commands"), list) else []
        for command_spec in commands:
            if steps_left <= 0 or time.perf_counter() >= deadline:
                return results
            if not isinstance(command_spec, dict):
                continue
            stage = str(command_spec.get("stage") or "")
            command = normalize_command(command_spec.get("command"), stage=stage, allow_teacher=bool(args.allow_teacher))
            validation = validate_stage(stage, command, allow_teacher=bool(args.allow_teacher))
            if not validation["allowed"]:
                row = {
                    "experiment_id": exp.get("id"),
                    "stage": stage,
                    "status": "blocked",
                    "returncode": None,
                    "reason": validation["reason"],
                    "command": command,
                }
                results.append(row)
                append_jsonl(LEDGER_PATH, ledger_row(row))
                steps_left -= 1
                continue
            row = run_stage(exp, stage, command, timeout_seconds=min(int(args.timeout_seconds), max(1, int(deadline - time.perf_counter()))))
            results.append(row)
            append_jsonl(LEDGER_PATH, ledger_row(row))
            steps_left -= 1
    return results


def normalize_command(raw: Any, *, stage: str, allow_teacher: bool) -> list[str]:
    command = [str(item) for item in raw] if isinstance(raw, list) else []
    if command and looks_like_python(command[0]):
        command[0] = sys.executable
    if stage == "teacher_diagnosis":
        if "--mode" not in command:
            command.extend(["--mode", "proposal"])
        if "--queue-only" not in command and not allow_teacher:
            command.append("--queue-only")
        if allow_teacher and "--queue-only" in command:
            command = [item for item in command if item != "--queue-only"]
            if "--allow-teacher" not in command:
                command.append("--allow-teacher")
    return command


def validate_stage(stage: str, command: list[str], *, allow_teacher: bool) -> dict[str, Any]:
    if stage not in ALLOWED_STAGE_SCRIPTS:
        return {"allowed": False, "reason": f"stage_not_allowed:{stage}"}
    if len(command) < 2:
        return {"allowed": False, "reason": "command_too_short"}
    script = Path(command[1]).name
    if script not in ALLOWED_STAGE_SCRIPTS[stage]:
        return {"allowed": False, "reason": f"script_not_allowed:{script}"}
    joined = " ".join(command).lower()
    if any(token in joined for token in ["&&", "||", ";", "cmd.exe", "start-process", "--apply", "--apply-mode"]):
        return {"allowed": False, "reason": "unsafe_command_token"}
    if stage == "teacher_diagnosis":
        if "proposal" not in command:
            return {"allowed": False, "reason": "teacher_must_be_proposal_mode"}
        if "--reason" not in command:
            return {"allowed": False, "reason": "teacher_reason_required"}
        if not any(token in command for token in ["--prompt", "--prompt-file", "--local-evidence"]):
            return {"allowed": False, "reason": "teacher_context_required"}
        if not allow_teacher and "--queue-only" not in command:
            return {"allowed": False, "reason": "teacher_requires_queue_only_without_allow_teacher"}
    if stage == "public_calibration" and script != "broad_transfer_matrix.py":
        return {"allowed": False, "reason": "public_stage_must_be_calibration_matrix"}
    return {"allowed": True, "reason": "allowlisted"}


def run_stage(exp: dict[str, Any], stage: str, command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        return {
            "experiment_id": exp.get("id"),
            "stage": stage,
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "reason": "command_returned_zero" if result.returncode == 0 else f"returncode_{result.returncode}",
            "command": command,
            "timeout_seconds": timeout_seconds,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "experiment_id": exp.get("id"),
            "stage": stage,
            "status": "failed",
            "returncode": 124,
            "reason": "timeout",
            "command": command,
            "timeout_seconds": timeout_seconds,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
        }


def build_decisions(experiments: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_exp: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_exp.setdefault(str(row.get("experiment_id") or ""), []).append(row)
    decisions = []
    for exp in experiments:
        exp_id = str(exp.get("id") or "")
        rows = by_exp.get(exp_id, [])
        completed = [row for row in rows if row.get("status") == "completed"]
        failed = [row for row in rows if row.get("status") == "failed"]
        completed_stages = {str(row.get("stage") or "") for row in completed}
        declared_stages = [
            str(spec.get("stage") or "")
            for spec in (exp.get("commands") if isinstance(exp.get("commands"), list) else [])
            if isinstance(spec, dict)
        ]
        needs_public = "public_calibration" in declared_stages
        if failed:
            decision = "rollback_or_continue_diagnosis"
        elif not rows:
            decision = "queued"
        elif "private_eval" not in completed_stages:
            decision = "awaiting_private_eval"
        elif needs_public and "public_calibration" not in completed_stages:
            decision = "awaiting_public_calibration_gate"
        else:
            decision = "private_loop_complete_review_for_promotion_or_rollback"
        decisions.append(
            {
                "experiment_id": exp_id,
                "hypothesis": exp.get("hypothesis"),
                "stage_count": len(rows),
                "completed_stage_count": len(completed),
                "failed_stage_count": len(failed),
                "completed_stages": sorted(completed_stages),
                "declared_stages": declared_stages,
                "decision": decision,
                "promotion_evidence": False,
            }
        )
    return decisions


def stage_forbidden(exp: dict[str, Any]) -> bool:
    commands = exp.get("commands") if isinstance(exp.get("commands"), list) else []
    text = json.dumps(commands, sort_keys=True).lower()
    return "--apply" in text or "canonical_solution" in text or "public_solution_distillation" in text


def public_calibration_only(exp: dict[str, Any]) -> bool:
    commands = exp.get("commands") if isinstance(exp.get("commands"), list) else []
    for spec in commands:
        if not isinstance(spec, dict) or spec.get("stage") != "public_calibration":
            continue
        command = spec.get("command") if isinstance(spec.get("command"), list) else []
        if len(command) < 2 or Path(str(command[1])).name != "broad_transfer_matrix.py":
            return False
    return True


def teacher_stage_context_ok(exp: dict[str, Any]) -> bool:
    commands = exp.get("commands") if isinstance(exp.get("commands"), list) else []
    for spec in commands:
        if not isinstance(spec, dict) or spec.get("stage") != "teacher_diagnosis":
            continue
        command = [str(item) for item in spec.get("command", [])] if isinstance(spec.get("command"), list) else []
        if "--reason" not in command:
            return False
        if not any(token in command for token in ["--prompt", "--prompt-file", "--local-evidence"]):
            return False
    return True


def stage_order_ok(exp: dict[str, Any]) -> bool:
    order = stage_order(exp)
    if not order:
        return True
    rank = {"teacher_diagnosis": 0, "private_eval": 1, "public_calibration": 2}
    numeric = [rank.get(stage, 99) for stage in order]
    return numeric == sorted(numeric)


def private_eval_before_public(exp: dict[str, Any]) -> bool:
    order = stage_order(exp)
    if "public_calibration" not in order:
        return True
    return "private_eval" in order and order.index("private_eval") < order.index("public_calibration")


def rollback_or_promotion_rule_present(exp: dict[str, Any]) -> bool:
    text = json.dumps(exp, sort_keys=True).lower()
    return any(token in text for token in ["rollback", "promotion_rule", "promotion gate", "promotion_gate", "demote", "revert"])


def stage_orders(experiments: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {str(row.get("id") or ""): stage_order(row) for row in experiments}


def stage_order(exp: dict[str, Any]) -> list[str]:
    commands = exp.get("commands") if isinstance(exp.get("commands"), list) else []
    return [str(spec.get("stage") or "") for spec in commands if isinstance(spec, dict) and spec.get("stage")]


def ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), **row, "external_inference_calls": 0}


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Teacher Architect Experiment Runner",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- selected_experiments: `{summary.get('selected_experiments')}`",
        f"- executed_stage_count: `{summary.get('executed_stage_count')}`",
        f"- teacher_allowed: `{summary.get('teacher_allowed')}`",
        "",
        "## Decisions",
        "",
    ]
    for row in payload.get("decisions", []):
        lines.append(f"- `{row.get('experiment_id')}` -> `{row.get('decision')}`")
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
