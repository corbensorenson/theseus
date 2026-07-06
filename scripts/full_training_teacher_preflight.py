"""Full-training sparse-teacher readiness gate.

This is intentionally separate from the offline solo loop. It proves that the
online teacher path is wired as a proposal-only advisor for full training, while
keeping Mac-specific accelerator gaps distinct from teacher/control-plane bugs.
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
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

sys.path.insert(0, str(ROOT / "scripts"))
import teacher_oracle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/teacher_policy.json")
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--out", default="reports/full_training_teacher_preflight.json")
    parser.add_argument("--markdown-out", default="reports/full_training_teacher_preflight.md")
    parser.add_argument("--require-teacher-cli", action="store_true")
    parser.add_argument("--allow-teacher-live", action="store_true")
    parser.add_argument("--require-live-teacher", action="store_true")
    parser.add_argument("--live-timeout-seconds", type=int, default=240)
    parser.add_argument("--skip-queue-smoke", action="store_true")
    parser.add_argument("--skip-autonomy-readiness", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)
    policy_path = resolve_path(args.policy)
    policy = read_json(policy_path, {})

    gates: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    command_results: dict[str, Any] = {}

    gates.extend(policy_contract_gates(policy))

    cli_status = teacher_cli_status(policy)
    gates.append(
        gate(
            "teacher_cli_available",
            bool(cli_status.get("available")) or not args.require_teacher_cli,
            "blocker" if args.require_teacher_cli else "warning",
            cli_status,
        )
    )

    handoff = static_handoff_contracts()
    for name, passed in handoff["checks"].items():
        gates.append(gate(name, passed, "blocker", handoff["details"].get(name, "")))

    external = run_report_command(
        [
            sys.executable,
            "scripts/external_inference_audit.py",
            "--no-scan-reports",
            "--out",
            "reports/external_inference_audit.json",
        ],
        timeout=240,
    )
    command_results["external_inference_audit"] = compact_command_result(external)
    external_report = read_json(REPORTS / "external_inference_audit.json", {})
    gates.append(
        gate(
            "external_inference_teacher_only",
            bool(external_report.get("ok")) and external_report.get("teacher_only_invariant") is True,
            "blocker",
            external_report.get("summary") or external.get("stderr_tail") or external.get("stdout_tail"),
        )
    )

    budget = run_report_command(
        [
            sys.executable,
            "scripts/teacher_budget_audit.py",
            "--out",
            "reports/teacher_budget_audit.json",
            "--markdown-out",
            "reports/teacher_budget_audit.md",
        ],
        timeout=180,
    )
    command_results["teacher_budget_audit"] = compact_command_result(budget)
    budget_report = read_json(REPORTS / "teacher_budget_audit.json", {})
    architecture_decision = get_path(budget_report, ["reason_decisions", "architecture_wall"], {})
    gates.append(
        gate(
            "teacher_budget_audit_available",
            bool(budget_report)
            and str(budget_report.get("policy") or "").startswith("project_theseus_teacher_budget_audit")
            and isinstance(budget_report.get("reason_decisions"), dict),
            "blocker",
            {
                "trigger_state": budget_report.get("trigger_state"),
                "architecture_wall_budget_allowed": get_path(architecture_decision, ["budget", "allowed"], None),
                "architecture_wall_evidence_allowed": get_path(architecture_decision, ["local_wall_evidence", "allowed"], None),
            },
        )
    )
    gates.append(
        gate(
            "teacher_architecture_wall_budget_allowed",
            bool(get_path(architecture_decision, ["budget", "allowed"], False)),
            "blocker",
            architecture_decision,
        )
    )
    gates.append(
        gate(
            "teacher_audit_has_local_wall_evidence",
            bool(get_path(architecture_decision, ["local_wall_evidence", "allowed"], False)),
            "warning",
            "Audit snapshot has no standing architecture-wall evidence; queue/live smokes pass explicit local evidence.",
        )
    )

    if not args.skip_queue_smoke:
        queue_result = teacher_queue_smoke(policy_path)
        command_results["teacher_queue_smoke"] = compact_command_result(queue_result)
        artifacts["teacher_queue_smoke"] = "reports/teacher_oracle_queue_smoke.json"
        gates.append(
            gate(
                "teacher_queue_smoke",
                queue_result.get("status") == "queued_not_executed",
                "blocker",
                {
                    "status": queue_result.get("status"),
                    "blocked_reason": queue_result.get("blocked_reason"),
                    "out": "reports/teacher_oracle_queue_smoke.json",
                },
            )
        )

    apply_block = teacher_apply_block_smoke(policy_path)
    command_results["teacher_apply_block_smoke"] = compact_command_result(apply_block)
    artifacts["teacher_apply_block_smoke"] = "reports/teacher_apply_block_smoke.json"
    gates.append(
        gate(
            "teacher_apply_mode_blocked",
            apply_block.get("status") == "blocked_by_teacher_policy"
            and apply_block.get("blocked_reason") == "teacher_apply_mode_forbidden",
            "blocker",
            {
                "status": apply_block.get("status"),
                "blocked_reason": apply_block.get("blocked_reason"),
                "out": "reports/teacher_apply_block_smoke.json",
            },
        )
    )

    live_teacher = {"status": "not_run", "blocked_reason": "allow_teacher_live_not_set"}
    if args.allow_teacher_live or args.require_live_teacher:
        live_teacher = teacher_live_smoke(policy, args.live_timeout_seconds)
        command_results["teacher_live_smoke"] = compact_command_result(live_teacher)
        artifacts["teacher_live_smoke"] = "reports/teacher_oracle_live_smoke.json"
        gates.append(
            gate(
                "teacher_live_smoke_completed",
                live_teacher.get("status") == "completed",
                "blocker" if args.require_live_teacher or args.allow_teacher_live else "warning",
                {
                    "status": live_teacher.get("status"),
                    "blocked_reason": live_teacher.get("blocked_reason"),
                    "runtime_ms": live_teacher.get("runtime_ms"),
                    "out": "reports/teacher_oracle_live_smoke.json",
                },
            )
        )
    else:
        gates.append(
            gate(
                "teacher_live_smoke_completed",
                False,
                "warning",
                "Run with --allow-teacher-live to make one bounded proposal-only Codex call.",
            )
        )

    if not args.skip_autonomy_readiness:
        readiness = run_report_command(
            [
                sys.executable,
                "scripts/autonomy_launch_readiness.py",
                "--profile",
                args.profile,
                "--require-teacher-cli",
                "--out",
                "reports/autonomy_launch_readiness_teacher_preflight.json",
            ],
            timeout=300,
        )
        command_results["autonomy_launch_readiness"] = compact_command_result(readiness)
        artifacts["autonomy_launch_readiness"] = "reports/autonomy_launch_readiness_teacher_preflight.json"
        readiness_report = read_json(REPORTS / "autonomy_launch_readiness_teacher_preflight.json", {})
        gates.append(
            gate(
                "autonomy_readiness_report_written",
                bool(readiness_report)
                and readiness_report.get("policy") == "sparkstream_autonomy_launch_readiness_v0",
                "warning",
                {
                    "ready_for_autonomous_training": readiness_report.get("ready_for_autonomous_training"),
                    "ready_for_teacher_enabled_run": readiness_report.get("ready_for_teacher_enabled_run"),
                    "blockers": [
                        row.get("name") or row.get("gate")
                        for row in readiness_report.get("blocker_failures", [])[:8]
                        if isinstance(row, dict)
                    ],
                },
            )
        )

    worker = worker_teacher_invariant()
    gates.append(
        gate(
            "worker_chunks_remain_teacher_free",
            bool(worker.get("ok")),
            "blocker",
            worker,
        )
    )

    blockers = [row for row in gates if row["severity"] == "blocker" and not row["passed"]]
    warnings = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    if blockers:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    else:
        trigger_state = "GREEN"

    report = {
        "policy": "project_theseus_full_training_teacher_preflight_v0",
        "created_utc": now(),
        "ok": not blockers,
        "trigger_state": trigger_state,
        "profile": args.profile,
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "teacher_cli_available": bool(cli_status.get("available")),
            "teacher_live_status": live_teacher.get("status"),
            "queue_smoke": command_results.get("teacher_queue_smoke", {}).get("status"),
            "apply_mode_blocked": apply_block.get("status") == "blocked_by_teacher_policy",
            "worker_teacher_invariant": bool(worker.get("ok")),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "teacher_policy": compact_teacher_policy(policy),
        "teacher_cli": cli_status,
        "gates": gates,
        "blocker_failures": blockers,
        "warning_failures": warnings,
        "commands": command_results,
        "artifacts": {
            **artifacts,
            "teacher_budget_audit": "reports/teacher_budget_audit.json",
            "external_inference_audit": "reports/external_inference_audit.json",
            "preflight_json": relpath(resolve_path(args.out)),
            "preflight_markdown": relpath(resolve_path(args.markdown_out)),
        },
        "next_commands": next_commands(trigger_state, args.allow_teacher_live),
        "notes": [
            "This gate proves the sparse teacher/control-plane path. It does not require live Windows/Mac peer reachability.",
            "On this Mac, CUDA-specific heavy-training blockers are reported as launch-readiness blockers, not teacher wiring failures.",
            "Teacher output remains proposal-only and is not accepted as worker training data.",
        ],
    }
    write_json(resolve_path(args.out), report)
    write_markdown(resolve_path(args.markdown_out), report)
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


def policy_contract_gates(policy: dict[str, Any]) -> list[dict[str, Any]]:
    budget = policy.get("budget") if isinstance(policy.get("budget"), dict) else {}
    return [
        gate(
            "teacher_policy_present",
            bool(policy) and policy.get("provider") == "codex_cli",
            "blocker",
            {"provider": policy.get("provider"), "model": policy.get("model")},
        ),
        gate(
            "teacher_default_mode_proposal",
            str(policy.get("default_mode") or "proposal") == "proposal",
            "blocker",
            policy.get("default_mode"),
        ),
        gate(
            "teacher_proposal_sandbox_read_only",
            str(policy.get("proposal_sandbox") or "") == "read-only",
            "blocker",
            policy.get("proposal_sandbox"),
        ),
        gate(
            "teacher_distillation_governed_apply_forbidden",
            bool(budget.get("distillation_training_enabled", False))
            and not bool(budget.get("apply_mode_enabled", False))
            and bool((budget.get("distillation_training_policy") or {}).get("allowed_only_through_gate", False)),
            "blocker",
            budget,
        ),
        gate(
            "teacher_requires_local_wall_evidence",
            bool(budget.get("requires_local_wall_evidence", True)),
            "blocker",
            budget,
        ),
        gate(
            "teacher_output_schema_exists",
            bool(policy.get("output_schema")) and (ROOT / str(policy.get("output_schema"))).exists(),
            "blocker",
            policy.get("output_schema"),
        ),
        gate(
            "teacher_allowed_reasons_configured",
            "architecture_wall" in {str(item) for item in policy.get("allowed_reasons", [])},
            "blocker",
            policy.get("allowed_reasons", []),
        ),
    ]


def static_handoff_contracts() -> dict[str, Any]:
    files = {
        "ratchet": read_text(ROOT / "scripts" / "run_training_ratchet_profile.py"),
        "autonomy": read_text(ROOT / "scripts" / "autonomy_cycle.py"),
        "guidance": read_text(ROOT / "scripts" / "architecture_guidance_loop.py"),
    }
    checks = {
        "ratchet_accepts_allow_teacher": 'parser.add_argument("--allow-teacher"' in files["ratchet"],
        "ratchet_passes_teacher_to_guidance": "architecture_guidance_loop_step(allow_teacher=args.allow_teacher)" in files["ratchet"],
        "guidance_invokes_teacher_oracle": "scripts/teacher_oracle.py" in files["guidance"]
        and "--allow-teacher" in files["guidance"],
        "guidance_enforces_proposal_only": "teacher_proposal_only" in files["guidance"],
        "autonomy_passes_teacher_to_ratchet": 'profile_command.append("--allow-teacher")' in files["autonomy"],
        "autonomy_sparse_teacher_step": "scripts/teacher_oracle.py" in files["autonomy"]
        and "teacher_needed" in files["autonomy"],
    }
    details = {name: "static source contract check" for name in checks}
    return {"checks": checks, "details": details}


def teacher_cli_status(policy: dict[str, Any]) -> dict[str, Any]:
    command = teacher_oracle.resolve_codex_command(policy) if policy else "codex"
    status = {
        "command": command,
        "available": False,
        "version": "",
        "error": "",
    }
    try:
        proc = subprocess.run(
            [command, "--version"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status["error"] = str(exc)
        return status
    status["available"] = proc.returncode == 0
    status["version"] = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0:
        status["error"] = (proc.stderr or proc.stdout)[-500:]
    return status


def teacher_queue_smoke(policy_path: Path) -> dict[str, Any]:
    out = REPORTS / "teacher_oracle_queue_smoke.json"
    unlink_if_exists(out)
    result = run_report_command(
        [
            sys.executable,
            "scripts/teacher_oracle.py",
            "--policy",
            relpath(policy_path),
            "--reason",
            "architecture_wall",
            "--mode",
            "proposal",
            "--prompt",
            "Full-training queue smoke. Queue one proposal-only architecture-wall request; do not execute.",
            "--local-evidence",
            "full_training_teacher_preflight=queue_smoke",
            "--queue-only",
            "--out",
            "reports/teacher_oracle_queue_smoke.json",
        ],
        timeout=120,
    )
    report = read_json(out, {})
    return {**result, **compact_teacher_request(report)}


def teacher_apply_block_smoke(policy_path: Path) -> dict[str, Any]:
    out = REPORTS / "teacher_apply_block_smoke.json"
    unlink_if_exists(out)
    result = run_report_command(
        [
            sys.executable,
            "scripts/teacher_oracle.py",
            "--policy",
            relpath(policy_path),
            "--reason",
            "safety_or_governance_uncertainty",
            "--mode",
            "apply",
            "--allow-teacher",
            "--prompt",
            "Attempted apply-mode smoke. This must be blocked by teacher policy.",
            "--local-evidence",
            "full_training_teacher_preflight=apply_block_smoke",
            "--out",
            "reports/teacher_apply_block_smoke.json",
        ],
        timeout=120,
    )
    report = read_json(out, {})
    return {**result, **compact_teacher_request(report)}


def teacher_live_smoke(policy: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    live_policy = dict(policy)
    live_policy["timeout_seconds"] = int(max(60, timeout_seconds))
    live_policy_path = REPORTS / "teacher_policy_live_smoke.json"
    live_out = REPORTS / "teacher_oracle_live_smoke.json"
    write_json(live_policy_path, live_policy)
    unlink_if_exists(live_out)
    prompt = (
        "Live smoke for Project Theseus full-training sparse teacher. "
        "Return only a concise proposal-only architecture diagnosis JSON matching the provided schema. "
        "Do not solve benchmark tasks, provide hidden answers, request apply mode, or create code edits. "
        "Recommend at most one reversible local experiment."
    )
    result = run_report_command(
        [
            sys.executable,
            "scripts/teacher_oracle.py",
            "--policy",
            relpath(live_policy_path),
            "--reason",
            "architecture_wall",
            "--mode",
            "proposal",
            "--allow-teacher",
            "--prompt",
            prompt,
            "--local-evidence",
            "full_training_teacher_preflight=live_smoke",
            "--out",
            "reports/teacher_oracle_live_smoke.json",
        ],
        timeout=int(max(75, timeout_seconds + 30)),
    )
    report = read_json(live_out, {})
    return {**result, **compact_teacher_request(report)}


def worker_teacher_invariant() -> dict[str, Any]:
    paths = [
        REPORTS / "hive_worker_chunk_ledger.jsonl",
        REPORTS / "hive_solo_learning_ledger.jsonl",
        REPORTS / "hive_job_ledger.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl_tail(path, 120):
            if isinstance(row, dict):
                rows.append({"path": relpath(path), **row})
    checked = 0
    violations = []
    for row in rows:
        kind = str(row.get("kind") or row.get("task_kind") or "")
        if not (
            kind.startswith("cuda_")
            or kind.startswith("mlx_")
            or kind == "training_smoke"
            or "solo" in str(row.get("path"))
        ):
            continue
        checked += 1
        external_calls = int_or_zero(row.get("external_inference_calls"))
        teacher_used = bool(row.get("teacher_used"))
        if external_calls != 0 or teacher_used:
            violations.append(
                {
                    "path": row.get("path"),
                    "created_utc": row.get("created_utc"),
                    "kind": kind,
                    "external_inference_calls": external_calls,
                    "teacher_used": teacher_used,
                }
            )
    return {
        "ok": not violations,
        "checked_worker_rows": checked,
        "violation_count": len(violations),
        "violations": violations[:12],
    }


def next_commands(trigger_state: str, live_ran: bool) -> list[str]:
    commands = []
    if not live_ran:
        commands.append("python3 scripts/theseus_cli.py train teacher-preflight --allow-teacher-live --require-live-teacher")
    if trigger_state != "RED":
        commands.extend(
            [
                "python3 scripts/autonomy_cycle.py --profile smoke --allow-teacher --out reports/autonomy_cycle_teacher_plan.json",
                "python3 scripts/run_training_ratchet_profile.py --profile smoke --allow-teacher --timeout-seconds 1800",
            ]
        )
    commands.append("python3 scripts/theseus_cli.py train status")
    return commands


def compact_teacher_policy(policy: dict[str, Any]) -> dict[str, Any]:
    budget = policy.get("budget") if isinstance(policy.get("budget"), dict) else {}
    return {
        "provider": policy.get("provider"),
        "model": policy.get("model"),
        "reasoning_effort": policy.get("reasoning_effort"),
        "default_mode": policy.get("default_mode"),
        "proposal_sandbox": policy.get("proposal_sandbox"),
        "approval_policy": policy.get("approval_policy"),
        "output_schema": policy.get("output_schema"),
        "proposal_only_no_distillation": budget.get("proposal_only_no_distillation"),
        "requires_local_wall_evidence": budget.get("requires_local_wall_evidence"),
        "allowed_reason_count": len(policy.get("allowed_reasons", []) if isinstance(policy.get("allowed_reasons"), list) else []),
    }


def compact_teacher_request(report: dict[str, Any]) -> dict[str, Any]:
    response = report.get("response_json") if isinstance(report.get("response_json"), dict) else {}
    return {
        "status": report.get("status"),
        "blocked_reason": report.get("blocked_reason"),
        "reason_for_call": report.get("reason_for_call"),
        "mode": report.get("mode"),
        "model": report.get("model"),
        "runtime_ms": report.get("runtime_ms"),
        "request_id": report.get("request_id"),
        "response": {
            "recommended_intervention": response.get("recommended_intervention"),
            "confidence": response.get("confidence"),
            "risk_level": response.get("risk_level"),
        },
    }


def compact_command_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "returncode": result.get("returncode"),
        "status": result.get("status"),
        "blocked_reason": result.get("blocked_reason"),
        "runtime_ms": result.get("runtime_ms"),
        "stdout_tail": result.get("stdout_tail", ""),
        "stderr_tail": result.get("stderr_tail", ""),
    }


def run_report_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": -1,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "error": str(exc),
            "command": command,
        }
    parsed = parse_json(proc.stdout.strip(), {})
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "command": command,
    }
    if isinstance(parsed, dict) and parsed:
        for key, value in compact_teacher_request(parsed).items():
            if value is not None:
                result[key] = value
        result.setdefault("status", parsed.get("status"))
    return result


def gate(name: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    blockers = report.get("blocker_failures", [])
    warnings = report.get("warning_failures", [])
    lines = [
        "# Full Training Teacher Preflight",
        "",
        f"- State: {report.get('trigger_state')}",
        f"- Teacher CLI: {report.get('summary', {}).get('teacher_cli_available')}",
        f"- Live teacher smoke: {report.get('summary', {}).get('teacher_live_status')}",
        f"- Blockers: {len(blockers)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Failed Gates",
    ]
    failed = blockers + warnings
    if failed:
        for row in failed:
            lines.append(f"- {row.get('severity')}: {row.get('name')} - {compact_detail(row.get('detail'))}")
    else:
        lines.append("- None")
    lines.extend(["", "## Next Commands"])
    for command in report.get("next_commands", []):
        lines.append(f"- `{command}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def compact_detail(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text[:280]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in lines:
        value = parse_json(line, None)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
