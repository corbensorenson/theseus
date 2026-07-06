"""Run a guarded teacher self-edit job on an isolated branch.

This is intentionally a harness, not a free-for-all. The teacher may apply a
small patch only when policy permits it, on a new branch, with local checks run
afterward. Failed checks leave the branch and report intact; this script never
uses destructive git reset/checkout to hide failed edits.
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
DEFAULT_POLICY = ROOT / "configs" / "self_evolution_policy.json"
sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402
import personality_context_builder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--reason", default="architecture_wall")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--out", default="reports/teacher_self_edit_last.json")
    parser.add_argument("--trace-out", default="")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    cfg = policy.get("guarded_self_edit") or {}
    report: dict[str, Any] = {
        "policy": "sparkstream_teacher_self_edit_runner_v0",
        "created_utc": now(),
        "policy_file": args.policy,
        "reason": args.reason,
        "execute": args.execute,
        "allow_teacher": args.allow_teacher,
        "status": "planned",
        "external_inference_calls": 0,
    }
    git_before = git_status()
    report["git_before"] = git_before
    personality_context = personality_context_builder.build_context(
        prompt=args.prompt or args.reason,
        task=f"teacher_self_edit:{args.reason}",
    )
    write_json(ROOT / "reports" / "personality_context_last.json", personality_context)
    report["personality_context"] = compact_personality_context(personality_context)
    license_check = license_manager.check_feature("teacher_bootstrap", write_report=True)
    report["license"] = {
        "teacher_bootstrap_allowed": bool(license_check.get("allowed")),
        "tier": get_path(license_check, ["entitlement", "tier"], None),
        "source": get_path(license_check, ["entitlement", "source"], None),
        "next_action": license_check.get("next_action"),
    }
    attd = read_json(ROOT / "reports" / "attd_report.json")
    attd_packets = read_json(ROOT / "reports" / "attd_maintenance_packets.json")
    report["attd_before"] = {
        "trigger_state": attd.get("trigger_state"),
        "attd_score": attd.get("attd_score"),
        "governance": attd.get("governance"),
        "packet_count": attd_packets.get("packet_count"),
        "packets": compact_packets(attd_packets),
    }

    if not args.execute:
        report["status"] = "plan_ready_not_executed"
        report["prompt_preview"] = build_prompt(args, policy)[:4000]
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0
    if not args.allow_teacher:
        report["status"] = "blocked_teacher_not_allowed"
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1
    if not license_check.get("allowed"):
        report["status"] = "blocked_license"
        report["blocked_reason"] = "Teacher bootstrap/self-edit requires a registered local install or a signed paid license."
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1
    git_current = git_before
    if (
        cfg.get("requires_clean_worktree", True)
        and cfg.get("auto_commit_dirty_worktree", False)
        and git_current.get("dirty")
        and not args.allow_dirty
    ):
        git_current = auto_commit_dirty_worktree(cfg, report)
    if cfg.get("requires_clean_worktree", True) and git_current.get("dirty") and not args.allow_dirty:
        report["status"] = "blocked_dirty_worktree"
        report["blocked_reason"] = (
            "Policy requires a clean worktree so teacher edits do not overwrite or mingle with user changes. "
            "Automatic checkpointing was unavailable or did not leave the worktree clean."
        )
        report["git_current"] = git_current
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1
    if attd.get("policy") != "sparkstream_attd_report_v0":
        report["status"] = "blocked_attd_missing"
        report["blocked_reason"] = "ATTD report is missing or stale; run scripts/attd_analyzer.py before teacher self-edit."
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1
    maintenance_reason = str(get_path(attd, ["governance", "teacher_self_edit_exception_reason"], "attd_maintenance"))
    attd_state = str(attd.get("trigger_state") or "MISSING")
    attd_allows = bool(get_path(attd, ["governance", "allows_teacher_self_edit"], False))
    if not attd_allows and args.reason != maintenance_reason:
        report["status"] = "blocked_attd_red"
        report["blocked_reason"] = (
            f"ATTD is {attd_state}; teacher self-edit is limited to reason={maintenance_reason} "
            "until maintenance packets reduce repo debt."
        )
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    branch_name = f"{cfg.get('branch_prefix', 'codex/self-evolution/')}{int(time.time())}"
    branch_step = run(["git", "switch", "-c", branch_name], timeout=60)
    report["branch_step"] = branch_step
    if branch_step["returncode"] != 0:
        report["status"] = "failed_create_branch"
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    prompt_path = ROOT / "reports" / "teacher_self_edit_prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(build_prompt(args, policy), encoding="utf-8")

    teacher_step = run(
        [
            sys.executable,
            "scripts/teacher_oracle.py",
            "--reason",
            args.reason,
            "--mode",
            "apply",
            "--prompt-file",
            str(prompt_path.relative_to(ROOT)),
            "--allow-teacher",
            "--out",
            "reports/teacher_self_edit_teacher.json",
        ],
        timeout=1800,
    )
    report["teacher_step"] = teacher_step
    report["external_inference_calls"] = 1 if teacher_step["returncode"] == 0 else 0
    report["teacher_result"] = read_json(ROOT / "reports" / "teacher_self_edit_teacher.json")

    checks: list[dict[str, Any]] = []
    for check in cfg.get("required_checks", []):
        if not isinstance(check, dict):
            continue
        command = [str(item) for item in check.get("command", [])]
        if not command:
            continue
        checks.append(
            {
                "name": check.get("name"),
                "allow_failure": bool(check.get("allow_failure", False)),
                **run(command, timeout=int(check.get("timeout_seconds", 120))),
            }
        )
    report["checks"] = checks
    attd_after = read_json(ROOT / "reports" / "attd_report.json")
    attd_packets_after = read_json(ROOT / "reports" / "attd_maintenance_packets.json")
    report["attd_after"] = {
        "trigger_state": attd_after.get("trigger_state"),
        "attd_score": attd_after.get("attd_score"),
        "governance": attd_after.get("governance"),
        "packet_count": attd_packets_after.get("packet_count"),
        "packets": compact_packets(attd_packets_after),
    }
    report["git_after"] = git_status()
    report["changed_files"] = changed_files()
    hard_failures = [check for check in checks if check.get("returncode") != 0 and not check.get("allow_failure")]
    if teacher_step["returncode"] != 0:
        report["status"] = "teacher_apply_failed"
    elif hard_failures:
        report["status"] = "checks_failed_branch_left_for_review"
    else:
        report["status"] = "checks_passed_branch_ready_for_review"
    report["branch"] = branch_name
    report["next_action"] = (
        "Review branch, compare benchmark/regression reports, then merge only if frontier improves without regression."
        if report["status"] == "checks_passed_branch_ready_for_review"
        else "Inspect branch and checks; do not merge until blockers are fixed."
    )
    write_json(ROOT / args.out, report)
    write_teacher_repair_trace(args, policy, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "checks_passed_branch_ready_for_review" else 1


def build_prompt(args: argparse.Namespace, policy: dict[str, Any]) -> str:
    state_files = [
        "reports/candidate_promotion_gate.json",
        "reports/benchmark_ledger.json",
        "reports/residual_escrow.json",
        "reports/architecture_experiment_governance.json",
        "reports/benchmark_adapter_factory.json",
        "reports/loop_closure_harvester.json",
        "reports/legacy_project_concept_audit.json",
        "reports/legacy_port_mechanisms.json",
        "reports/planforge_schedule.json",
        "reports/coherence_delirium_report.json",
        "reports/proxy_truth_audit.json",
        "reports/taskspell_contracts.json",
        "reports/low_rank_adapter_bank.json",
        "reports/world_adapter_job_runtime.json",
        "reports/self_evolution_governance.json",
        "reports/attd_report.json",
        "reports/attd_maintenance_packets.json",
        "reports/external_inference_audit.json",
        "reports/personality_context_last.json",
        "reports/personality_drift_eval.json",
        "reports/belief_update_governance.json",
    ]
    custom = args.prompt.strip()
    personality_context = read_json(ROOT / "reports" / "personality_context_last.json")
    reality_orientation = personality_context_builder.orientation_answer(custom or args.reason, personality_context) if personality_context else ""
    if custom:
        objective = custom
    elif args.reason == "attd_maintenance":
        attd_packets = read_json(ROOT / "reports" / "attd_maintenance_packets.json")
        max_packets = int(get_path(policy, ["attd_maintenance", "max_packets_per_patch"], 3))
        packet_lines = []
        for item in (attd_packets.get("packets") or [])[:max_packets]:
            if not isinstance(item, dict):
                continue
            packet_lines.append(
                f"- {item.get('packet_id')}: {item.get('bounded_action')} scope={item.get('scope')} verification={item.get('verification')}"
            )
        objective = (
            str(get_path(policy, ["attd_maintenance", "objective"], "Consume ATTD maintenance packets."))
            + "\n\nHighest-priority ATTD packets:\n"
            + ("\n".join(packet_lines) if packet_lines else "- No packets found; repair the ATTD analyzer/report path instead.")
            + "\n\nPrefer verified simplification: reduce file size, consolidate duplicated patterns, improve ownership boundaries, "
            "or add small local checks. Preserve behavior and avoid cosmetic churn."
        )
    else:
        objective = (
            "Implement the smallest source-level improvement that advances the current architecture wall. "
            "Prefer zero-parameter tooling, benchmark adapters, loop closure, and Rust/CUDA efficiency before "
            "adding parameters. Do not train models, download data, or change tracked generated reports."
        )
    return f"""You are the sparse teacher/architect for the local SymLiquid/RMI project.

Task reason: {args.reason}
Objective: {objective}

Hard rules:
- Make a minimal patch only inside tracked source/config/doc/test areas.
- Do not modify ROMs, games, checkpoints, target, or bulk data.
- Do not call external inference except this teacher call.
- Do not run long training.
- Do not add parameters or architecture unless the evidence proves lower-cost interventions are exhausted.
- Preserve regressions and candidate gates.
- Prefer Rust/CUDA hot-loop efficiency and verified tools over bloat.
- Use reports/personality_context_last.json as reality-orientation context: preserve truth before compliance, agency before convenience, least sufficient power, and anti-drift governance.
- If reason is attd_maintenance, consume the listed ATTD packets first and target measurable ATTD score, packet, or hotspot improvement.
Reality orientation: {reality_orientation}

Useful evidence files:
{chr(10).join(f"- {item}" for item in state_files)}

Return JSON describing changed files, hypothesis, verification, expected benchmark impact, and residual risks.
"""


def run(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "error": str(exc),
        }


def git_status() -> dict[str, Any]:
    branch = run(["git", "branch", "--show-current"], timeout=30)
    status = run(["git", "status", "--porcelain"], timeout=30)
    lines = [line for line in status.get("stdout_tail", "").splitlines() if line.strip()]
    return {
        "available": branch["returncode"] == 0 and status["returncode"] == 0,
        "branch": branch.get("stdout_tail", "").strip(),
        "dirty": bool(lines),
        "porcelain_count": len(lines),
        "porcelain_sample": lines[:30],
    }


def changed_files() -> list[str]:
    status = run(["git", "diff", "--name-only"], timeout=30)
    staged = run(["git", "diff", "--cached", "--name-only"], timeout=30)
    names = set(status.get("stdout_tail", "").splitlines()) | set(staged.get("stdout_tail", "").splitlines())
    return sorted(name for name in names if name)


def compact_personality_context(context: dict[str, Any]) -> dict[str, Any]:
    ctx = context.get("context") if isinstance(context.get("context"), dict) else {}
    return {
        "status": context.get("status"),
        "source_report": context.get("source_report"),
        "summary": context.get("summary", {}),
        "compact_core": ctx.get("compact_core", ""),
        "hard_safety_invariants": get_path(ctx, ["reality_contract", "hard_safety_invariants"], [])[:5],
        "anti_drift_rules": (ctx.get("anti_drift_rules") or [])[:5],
    }


def auto_commit_dirty_worktree(cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    message = str(cfg.get("auto_commit_message") or "Preserve workspace before guarded self-evolution")
    stage_command = ["git", "add", "-A"] if cfg.get("auto_stage_untracked", True) else ["git", "add", "-u"]
    stage_step = run(stage_command, timeout=300)
    commit_step: dict[str, Any]
    if stage_step["returncode"] == 0:
        commit_step = run(["git", "commit", "-m", message], timeout=300)
    else:
        commit_step = {
            "command": ["git", "commit", "-m", message],
            "returncode": 1,
            "runtime_ms": 0,
            "stdout_tail": "",
            "stderr_tail": "Skipped because git add failed.",
        }
    refreshed = git_status()
    report["dirty_worktree_checkpoint"] = {
        "attempted": True,
        "message": message,
        "stage_step": stage_step,
        "commit_step": commit_step,
        "git_after_checkpoint": refreshed,
    }
    return refreshed


def write_teacher_repair_trace(args: argparse.Namespace, policy: dict[str, Any], report: dict[str, Any]) -> None:
    trace_path = args.trace_out or str(
        get_path(policy, ["guarded_self_edit", "teacher_trace_path"], None)
        or get_path(policy, ["attd_maintenance", "trace_path"], "reports/teacher_self_edit_traces.jsonl")
    )
    trace = {
        "trace_id": f"teacher_repair_{int(time.time() * 1000)}",
        "created_utc": now(),
        "kind": "teacher_self_edit_trace",
        "reason": args.reason,
        "status": report.get("status"),
        "branch": report.get("branch"),
        "changed_files": report.get("changed_files", []),
        "attd_before": report.get("attd_before"),
        "attd_after": report.get("attd_after"),
        "checks": compact_checks(report.get("checks", [])),
        "teacher_response": get_path(report, ["teacher_result", "response_json"], None),
        "prompt_hash": sha256_file(ROOT / "reports" / "teacher_self_edit_prompt.md"),
        "success": report.get("status") == "checks_passed_branch_ready_for_review",
        "selected_arms": ["head_router", "teacher_architect_arm", "attd_repo_health_governance", "code_repair_verifier"],
        "risk": "medium",
        "runtime_tier": "E1",
        "distillation_use": "local_teacher_repair_trace",
        "external_inference_calls": report.get("external_inference_calls", 0),
    }
    append_jsonl(ROOT / trace_path, trace)
    workflow_path = get_path(policy, ["guarded_self_edit", "workflow_trace_path"], "")
    if workflow_path:
        append_jsonl(
            ROOT / str(workflow_path),
            {
                "trace_id": trace["trace_id"],
                "created_utc": trace["created_utc"],
                "task": f"teacher_self_edit:{args.reason}",
                "command": "python scripts/teacher_self_edit_runner.py --execute --allow-teacher",
                "workflow": "guarded_teacher_self_edit",
                "selected_arms": trace["selected_arms"],
                "success": trace["success"],
                "runtime_ms": get_path(report, ["teacher_step", "runtime_ms"], 0),
                "risk": "medium",
                "routing_pattern": "verification_routing",
                "verification": "teacher branch, local checks, ATTD before/after trace",
            },
        )


def compact_packets(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("packets") if isinstance(payload, dict) else []
    compact: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "packet_id": item.get("packet_id"),
                "component": item.get("component"),
                "priority": item.get("priority"),
                "scope": item.get("scope", [])[:5],
                "bounded_action": item.get("bounded_action"),
            }
        )
    return compact


def compact_checks(checks: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in checks or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("name"),
                "returncode": item.get("returncode"),
                "allow_failure": item.get("allow_failure"),
                "runtime_ms": item.get("runtime_ms"),
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
