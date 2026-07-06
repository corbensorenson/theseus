"""Solo/offline learning loop for one Mac Hive node.

This module wraps the existing utilization manager, training orchestrator, and
worker reports into a Mac-local control plane. It is intentionally local-only:
no teacher calls, no WAN peer queueing, and no artifact sync are required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints" / "hive_promoted"
POLICY_PATH = ROOT / "configs" / "hive_policy.json"
STATUS_PATH = REPORTS / "hive_solo_learning_status.json"
LEDGER_PATH = REPORTS / "hive_solo_learning_ledger.jsonl"
BEST_PATH = REPORTS / "hive_solo_best_by_arm.json"
OVERNIGHT_PATH = REPORTS / "hive_solo_overnight_report.json"
OVERNIGHT_MARKDOWN = REPORTS / "hive_solo_overnight_report.md"
PAUSE_FLAG = REPORTS / "hive_utilization_pause.flag"
STOP_FLAG = REPORTS / "hive_utilization_stop.flag"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_training_orchestrator as training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and audit a solo/offline Theseus learning loop on this Mac.")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Write the current solo learning status.")
    add_status_args(status)

    sweep = sub.add_parser("sweep", help="Run one local/offline always-active sweep, then update the solo ledger.")
    add_run_args(sweep)
    sweep.add_argument("--wait-seconds", type=float, default=90.0)

    loop = sub.add_parser("loop", help="Run repeated local/offline sweeps, then update the solo ledger.")
    add_run_args(loop)
    loop.add_argument("--cycles", type=int, default=1)
    loop.add_argument("--sleep-seconds", type=int, default=60)
    loop.add_argument("--wait-seconds", type=float, default=90.0)

    overnight = sub.add_parser("overnight", help="Write the solo overnight/offline learning report.")
    add_status_args(overnight)

    args = parser.parse_args()
    if args.command in {None, "status"}:
        report = solo_status(hours=float(getattr(args, "hours", 24.0)), write=True)
    elif args.command == "sweep":
        report = run_sweep(args)
    elif args.command == "loop":
        report = run_loop(args)
    elif args.command == "overnight":
        report = solo_overnight(hours=float(args.hours), write=True, markdown_out=resolve(args.markdown_out))
    else:
        parser.print_help()
        return 2
    out = str(getattr(args, "out", "") or "")
    if out:
        write_json(resolve(out), report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def add_status_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--out", default=str(STATUS_PATH.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(OVERNIGHT_MARKDOWN.relative_to(ROOT)))


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--max-new-jobs", type=int, default=1)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--min-battery-percent", type=float, default=None)
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--out", default=str(STATUS_PATH.relative_to(ROOT)))


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    execution = {}
    if bool(args.execute):
        execution = run_utilization("sweep", args)
        wait_for_recent_worker_reports(started, float(args.wait_seconds))
    status = solo_status(hours=24.0, write=True)
    return {
        **status,
        "mode": "sweep_execute" if bool(args.execute) else "sweep_plan",
        "utilization_execution": execution,
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    execution = {}
    if bool(args.execute):
        execution = run_utilization("loop", args)
        wait_for_recent_worker_reports(started, float(args.wait_seconds))
    status = solo_status(hours=24.0, write=True)
    return {
        **status,
        "mode": "loop_execute" if bool(args.execute) else "loop_plan",
        "utilization_execution": execution,
    }


def run_utilization(command: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/hive_utilization_manager.py",
        command,
        "--execute",
        "--offline",
        "--local-only",
        "--profile",
        str(args.profile or "inner_loop"),
        "--max-new-jobs",
        str(max(1, int(args.max_new_jobs or 1))),
        "--out",
        "reports/hive_utilization_manager.json",
    ]
    if command == "loop":
        cmd.extend(["--cycles", str(max(1, int(args.cycles or 1))), "--sleep-seconds", str(max(1, int(args.sleep_seconds or 60)))])
    if bool(args.allow_battery):
        cmd.append("--allow-battery")
    if args.min_battery_percent is not None:
        cmd.extend(["--min-battery-percent", str(float(args.min_battery_percent))])
    if bool(args.keep_awake):
        cmd.append("--keep-awake")
    started = time.perf_counter()
    sleep_seconds = int(getattr(args, "sleep_seconds", 0) or 0)
    cycles = max(1, int(getattr(args, "cycles", 1) or 1))
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=max(900, sleep_seconds * cycles + 900), env=theseus_runtime.runtime_env())
    parsed = parse_json(result.stdout)
    return {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "stdout_summary": compact_utilization_stdout(parsed),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def compact_utilization_stdout(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), list) else []
    training_round = {}
    for row in execution:
        if not isinstance(row, dict) or row.get("kind") != "training_round":
            continue
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        training_round = {
            "ok": result.get("ok"),
            "run_id": result.get("run_id"),
            "round_id": result.get("round_id"),
            "profile": result.get("profile"),
            "job_count": training.get_path(result, ["plan", "job_count"], None),
            "execution_count": len(result.get("execution") or []) if isinstance(result.get("execution"), list) else None,
        }
        break
    return {
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "trigger_state": report.get("trigger_state"),
        "created_utc": report.get("created_utc"),
        "summary": {
            "node_count": summary.get("node_count"),
            "planned_actions": summary.get("planned_actions"),
            "executed_actions": summary.get("executed_actions"),
            "blocked_nodes": summary.get("blocked_nodes"),
            "offline_mode": summary.get("offline_mode"),
            "local_only": summary.get("local_only"),
            "allow_battery": summary.get("allow_battery"),
        },
        "training_round": training_round,
        "external_inference_calls": report.get("external_inference_calls"),
        "safety": {
            "teacher_use": training.get_path(report, ["safety", "offline_mode", "teacher_use"], ""),
            "external_inference": training.get_path(report, ["safety", "offline_mode", "external_inference"], ""),
            "remote_peer_queueing": training.get_path(report, ["safety", "offline_mode", "remote_peer_queueing"], ""),
        },
    }


def wait_for_recent_worker_reports(started_epoch: float, wait_seconds: float) -> None:
    if wait_seconds <= 0:
        return
    deadline = time.time() + wait_seconds
    expected = expected_job_ids()
    while time.time() < deadline:
        reports = training.collect_worker_reports(max(0.0, started_epoch - 5.0))
        if not expected and reports:
            return
        report_ids = {str(row.get("job_id") or "") for row in reports}
        if expected and expected & report_ids:
            return
        time.sleep(1.5)


def expected_job_ids() -> set[str]:
    last = read_json(REPORTS / "hive_training_orchestrator.json", {})
    plan = last.get("plan") if isinstance(last.get("plan"), dict) else {}
    out = set()
    for job in plan.get("jobs") or []:
        if isinstance(job, dict):
            job_id = training.get_path(job, ["payload", "job_id"], "")
            if job_id:
                out.add(str(job_id))
    return out


def solo_status(*, hours: float = 24.0, write: bool = False) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    since = time.time() - max(0.1, hours) * 3600.0
    local = local_node(policy)
    lease_recovery = recover_local_stale_leases(policy, local)
    worker_reports = training.collect_worker_reports(since)
    ledger = update_solo_ledger(worker_reports)
    utilization = read_json(REPORTS / "hive_utilization_manager.json", {})
    train_state = read_json(REPORTS / "hive_training_state.json", {})
    training_last = read_json(REPORTS / "hive_training_orchestrator.json", {})
    overnight = read_json(OVERNIGHT_PATH, {})
    best = read_json(BEST_PATH, {})
    status = {
        "ok": True,
        "policy": "project_theseus_hive_solo_learning_status_v0",
        "created_utc": now(),
        "mode": "solo_offline_capable",
        "standalone": True,
        "local_only": True,
        "offline_capable": True,
        "node": local,
        "controls": control_state(),
        "mlx": mlx_state(local),
        "always_active": {
            "state": trigger_state(utilization),
            "last_mode": utilization.get("mode"),
            "last_created_utc": utilization.get("created_utc"),
            "summary": utilization.get("summary") if isinstance(utilization.get("summary"), dict) else {},
            "offline_contract": utilization.get("offline_contract") if isinstance(utilization.get("offline_contract"), dict) else {},
            "keep_awake": utilization.get("keep_awake") if isinstance(utilization.get("keep_awake"), dict) else {},
        },
        "training": {
            "last_run_id": training_last.get("run_id"),
            "last_round_id": training_last.get("round_id"),
            "profile": training_last.get("profile") or train_state.get("profile"),
            "last_execution_count": len(training_last.get("execution") or []) if isinstance(training_last.get("execution"), list) else 0,
            "active_arm_count": len(train_state.get("arms") or []) if isinstance(train_state.get("arms"), list) else 0,
            "blocked": train_state.get("blocked") if isinstance(train_state.get("blocked"), list) else [],
            "lease_recovery": lease_recovery,
        },
        "ledger": ledger,
        "best_by_arm": best.get("best_by_arm") if isinstance(best.get("best_by_arm"), dict) else {},
        "promotions": (best.get("promotions") if isinstance(best.get("promotions"), list) else [])[-10:],
        "worker_reports": worker_reports[-24:],
        "overnight": compact_solo_overnight(overnight),
        "safety": {
            "teacher_used": False,
            "teacher_use": "forbidden",
            "external_inference_calls": 0,
            "external_inference": "forbidden",
            "artifact_sync_required": False,
            "remote_peer_queueing": "disabled_in_solo_offline_commands",
            "task_surface": "registered_hive_worker_chunks_only",
        },
        "next_actions": next_actions(worker_reports, ledger, best, utilization, local),
    }
    if write:
        write_json(STATUS_PATH, status)
    return status


def update_solo_ledger(worker_reports: list[dict[str, Any]]) -> dict[str, Any]:
    best_doc = read_json(BEST_PATH, {})
    if not isinstance(best_doc, dict):
        best_doc = {}
    best_by_arm = best_doc.get("best_by_arm") if isinstance(best_doc.get("best_by_arm"), dict) else {}
    promotions = best_doc.get("promotions") if isinstance(best_doc.get("promotions"), list) else []
    existing_keys = ledger_keys()
    appended = 0
    events: list[dict[str, Any]] = []
    for row in sorted(worker_reports, key=lambda item: str(item.get("created_utc") or "")):
        event = ledger_event(row, best_by_arm)
        promotion = evaluate_promotion(event, best_by_arm)
        if promotion.get("promoted"):
            best_by_arm[str(event.get("arm_id") or "unknown")] = promotion.get("best_entry") or {}
            promotions.append(promotion.get("promotion") or {})
        event["promotion_decision"] = promotion.get("decision")
        event["promotion"] = compact_promotion_decision(promotion)
        key = str(event.get("event_key") or "")
        if key and key not in existing_keys:
            append_jsonl(LEDGER_PATH, event)
            existing_keys.add(key)
            appended += 1
        events.append(event)
    best_doc = {
        "ok": True,
        "policy": "project_theseus_hive_solo_best_by_arm_v0",
        "updated_utc": now(),
        "activation_rule": "promote_local_worker_artifact_only_when_heldout_score_improves",
        "best_by_arm": best_by_arm,
        "promotions": promotions[-200:],
        "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
    }
    write_json(BEST_PATH, best_doc)
    return {
        "path": str(LEDGER_PATH.relative_to(ROOT)),
        "event_count": len(events),
        "new_event_count": appended,
        "promotion_count": len([event for event in events if event.get("promotion_decision") == "promoted_local_best"]),
        "failed_count": len([event for event in events if event.get("ok") is False]),
        "recent": events[-12:],
    }


def ledger_event(row: dict[str, Any], best_by_arm: dict[str, Any]) -> dict[str, Any]:
    report_path = resolve_report_path(str(row.get("path") or ""))
    raw = read_json(report_path, {}) if report_path else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    job_id = str(row.get("job_id") or "")
    arm_id = str(row.get("arm_id") or row.get("task_kind") or "unknown")
    model_path = str(row.get("model_path") or "")
    output_artifacts = row.get("output_artifacts") if isinstance(row.get("output_artifacts"), list) else []
    candidate_artifact = model_path or str(row.get("path") or "")
    event_key = f"{job_id or row.get('path')}::{arm_id}::{candidate_artifact}"
    failure_reason = failure_reason_for(raw, row)
    return {
        "created_utc": row.get("created_utc") or now(),
        "ingested_utc": now(),
        "event_key": event_key,
        "ok": row.get("ok", True),
        "job_id": job_id,
        "arm_id": arm_id,
        "task_kind": row.get("task_kind"),
        "worker_kind": row.get("worker_kind"),
        "backend": row.get("backend"),
        "score": row.get("score"),
        "score_source": score_source(metrics),
        "metrics": metrics,
        "input_artifacts": row.get("input_artifacts") if isinstance(row.get("input_artifacts"), list) else [],
        "train_input": row.get("train_input"),
        "eval_input": row.get("eval_input"),
        "output_artifacts": output_artifacts,
        "model_path": model_path,
        "candidate_artifact": candidate_artifact,
        "source_report": str(row.get("path") or ""),
        "failure_reason": failure_reason,
        "teacher_used": bool(raw.get("teacher_used", False)) if isinstance(raw, dict) else False,
        "external_inference_calls": int(raw.get("external_inference_calls") or 0) if isinstance(raw, dict) else 0,
        "artifact_sync_used": False,
        "previous_best_score": training.get_path(best_by_arm, [arm_id, "score"], None),
    }


def evaluate_promotion(event: dict[str, Any], best_by_arm: dict[str, Any]) -> dict[str, Any]:
    arm_id = str(event.get("arm_id") or "unknown")
    if event.get("ok") is False:
        return {"decision": "rejected_worker_failed", "promoted": False, "reason": event.get("failure_reason")}
    score = event.get("score")
    if not isinstance(score, (int, float)):
        return {"decision": "rejected_no_numeric_score", "promoted": False}
    artifact = str(event.get("candidate_artifact") or "")
    if not artifact:
        return {"decision": "rejected_no_output_artifact", "promoted": False}
    previous = best_by_arm.get(arm_id) if isinstance(best_by_arm.get(arm_id), dict) else {}
    previous_score = previous.get("score")
    previous_key = str(previous.get("event_key") or "")
    if previous_key and previous_key == str(event.get("event_key") or ""):
        return {"decision": "already_active", "promoted": False}
    if isinstance(previous_score, (int, float)) and float(score) <= float(previous_score):
        return {
            "decision": "rejected_no_improvement",
            "promoted": False,
            "previous_score": previous_score,
            "candidate_score": score,
        }
    activated = activate_artifact(event, previous)
    best_entry = {
        "arm_id": arm_id,
        "job_id": event.get("job_id"),
        "event_key": event.get("event_key"),
        "score": score,
        "score_source": event.get("score_source"),
        "backend": event.get("backend"),
        "task_kind": event.get("task_kind"),
        "source_report": event.get("source_report"),
        "source_artifact": artifact,
        "activated_artifact": activated.get("activated_artifact"),
        "active_manifest": activated.get("active_manifest"),
        "rollback_manifest": activated.get("rollback_manifest"),
        "previous_score": previous_score,
        "updated_utc": now(),
    }
    return {
        "decision": "promoted_local_best",
        "promoted": True,
        "best_entry": best_entry,
        "promotion": {
            "created_utc": now(),
            "decision": "promoted_local_best",
            **best_entry,
            "activation": activated,
        },
    }


def activate_artifact(event: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    arm_id = safe_id(str(event.get("arm_id") or "unknown"))
    arm_dir = CHECKPOINTS / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    active_manifest = arm_dir / "active_manifest.json"
    rollback_manifest = ""
    if active_manifest.exists():
        rollback = arm_dir / f"active_manifest.rollback_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        shutil.copy2(active_manifest, rollback)
        rollback_manifest = str(rollback.relative_to(ROOT))
    source_artifact = resolve_report_path(str(event.get("candidate_artifact") or ""))
    activated_artifact = ""
    if source_artifact and source_artifact.exists():
        target = arm_dir / source_artifact.name
        if source_artifact.resolve() != target.resolve():
            shutil.copy2(source_artifact, target)
        activated_artifact = str(target.relative_to(ROOT))
    manifest = {
        "policy": "project_theseus_hive_solo_active_manifest_v0",
        "created_utc": now(),
        "arm_id": event.get("arm_id"),
        "job_id": event.get("job_id"),
        "score": event.get("score"),
        "score_source": event.get("score_source"),
        "backend": event.get("backend"),
        "task_kind": event.get("task_kind"),
        "source_report": event.get("source_report"),
        "source_artifact": event.get("candidate_artifact"),
        "activated_artifact": activated_artifact,
        "previous_active": previous,
        "rollback_manifest": rollback_manifest,
        "activation_rule": "new held-out score must exceed previous local best",
        "teacher_used": event.get("teacher_used"),
        "external_inference_calls": event.get("external_inference_calls"),
    }
    write_json(active_manifest, manifest)
    return {
        "active_manifest": str(active_manifest.relative_to(ROOT)),
        "rollback_manifest": rollback_manifest,
        "activated_artifact": activated_artifact,
    }


def solo_overnight(*, hours: float = 12.0, write: bool = False, markdown_out: Path = OVERNIGHT_MARKDOWN) -> dict[str, Any]:
    status = solo_status(hours=hours, write=True)
    ledger = status.get("ledger") if isinstance(status.get("ledger"), dict) else {}
    events = ledger.get("recent") if isinstance(ledger.get("recent"), list) else []
    improved = [row for row in events if row.get("promotion_decision") == "promoted_local_best"]
    failed = [row for row in events if row.get("ok") is False]
    rejected = [row for row in events if str(row.get("promotion_decision") or "").startswith("rejected")]
    report = {
        "ok": True,
        "policy": "project_theseus_hive_solo_overnight_report_v0",
        "created_utc": now(),
        "window_hours": hours,
        "window_start_utc": datetime.fromtimestamp(time.time() - max(0.1, hours) * 3600.0, tz=timezone.utc).isoformat(),
        "summary": {
            "worker_event_count": ledger.get("event_count", 0),
            "new_event_count": ledger.get("new_event_count", 0),
            "improved_count": len(improved),
            "failed_count": len(failed),
            "rejected_count": len(rejected),
            "promoted_arm_count": len(status.get("best_by_arm") or {}) if isinstance(status.get("best_by_arm"), dict) else 0,
            "stale_lease_count": training.get_path(status, ["training", "lease_recovery", "recovered_count"], 0),
        },
        "what_ran": events,
        "what_improved": improved,
        "what_failed": failed,
        "what_was_promoted": status.get("promotions") if isinstance(status.get("promotions"), list) else [],
        "best_by_arm": status.get("best_by_arm") if isinstance(status.get("best_by_arm"), dict) else {},
        "controls": status.get("controls"),
        "mlx": status.get("mlx"),
        "next_actions": next_actions(
            status.get("worker_reports") if isinstance(status.get("worker_reports"), list) else [],
            ledger,
            {"best_by_arm": status.get("best_by_arm") if isinstance(status.get("best_by_arm"), dict) else {}},
            read_json(REPORTS / "hive_utilization_manager.json", {}),
            status.get("node") if isinstance(status.get("node"), dict) else {},
        ),
    }
    if write:
        write_json(OVERNIGHT_PATH, report)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_overnight_markdown(report), encoding="utf-8")
    return report


def render_overnight_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Solo Mac Learning Report",
        "",
        f"- Window: {report.get('window_hours')} hours since `{report.get('window_start_utc')}`",
        f"- Worker events: {summary.get('worker_event_count', 0)}",
        f"- Improved/promoted: {summary.get('improved_count', 0)}",
        f"- Failed: {summary.get('failed_count', 0)}",
        f"- Stale leases recovered: {summary.get('stale_lease_count', 0)}",
        "",
        "## Promotions",
        "",
    ]
    promotions = report.get("what_was_promoted") if isinstance(report.get("what_was_promoted"), list) else []
    if promotions:
        for row in promotions[-12:]:
            lines.append(f"- `{row.get('arm_id')}` score `{row.get('score')}` via `{row.get('backend')}` -> `{row.get('activated_artifact') or row.get('active_manifest')}`")
    else:
        lines.append("- No local promotions in this window.")
    lines.extend(["", "## Failures", ""])
    failures = report.get("what_failed") if isinstance(report.get("what_failed"), list) else []
    if failures:
        for row in failures[:20]:
            lines.append(f"- `{row.get('job_id') or row.get('task_kind')}`: {row.get('failure_reason') or 'failed'}")
    else:
        lines.append("- No local worker failures found.")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def local_node(policy: dict[str, Any]) -> dict[str, Any]:
    view = training.operator_view(policy)
    visible = training.visible_nodes(view)
    local = next((node for node in visible if node.get("is_local")), {})
    if not local:
        status = read_json(REPORTS / "hive_status.json", {})
        local = training.peer_from_status(status) if status else {}
        if local:
            local["is_local"] = True
    if local:
        local_url = training.local_api_url(policy, local)
        check, attempts = training.fetch_json_with_attempts(local_url + "/api/hive/status", headers={}, timeouts=[1.0, 2.0])
        local = {**local, "api_url": local_url, "reachability": {"ok": bool(check and check.get("ok") is not False), "attempts": attempts}}
        if check and check.get("ok") is not False:
            local.update(
                {
                    "node_id": check.get("node_id") or local.get("node_id"),
                    "node_name": check.get("node_name") or local.get("node_name"),
                    "capabilities": check.get("capabilities") or local.get("capabilities") or [],
                    "resources": check.get("resources") or local.get("resources") or {},
                    "slots": check.get("slots") or local.get("slots") or [],
                    "tasks": check.get("tasks") or local.get("tasks") or {},
                }
            )
    return local


def recover_local_stale_leases(policy: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    nodes = [local] if local else []
    try:
        return training.recover_stale_leases(policy, nodes)
    except Exception as exc:  # noqa: BLE001 - status must remain readable.
        return {"ok": False, "error": "stale_lease_recovery_failed", "message": str(exc)}


def mlx_state(local: dict[str, Any]) -> dict[str, Any]:
    caps = {str(cap.get("id") or "") for cap in local.get("capabilities") or [] if isinstance(cap, dict)}
    resources = local.get("resources") if isinstance(local.get("resources"), dict) else {}
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    return {
        "available": bool({"mlx_apple", "apple_mlx", "mlx_cuda"} & caps) or bool(mlx.get("available")),
        "capability_ids": sorted(caps & {"mlx_apple", "apple_mlx", "mlx_cuda"}),
        "backend_ids": mlx.get("backend_ids") if isinstance(mlx.get("backend_ids"), list) else [],
        "module": mlx.get("module"),
        "status": mlx.get("status") or ("ready" if {"mlx_apple", "apple_mlx", "mlx_cuda"} & caps else "missing"),
    }


def control_state() -> dict[str, Any]:
    return {
        "paused": PAUSE_FLAG.exists(),
        "stopped": STOP_FLAG.exists(),
        "pause_flag": str(PAUSE_FLAG.relative_to(ROOT)),
        "stop_flag": str(STOP_FLAG.relative_to(ROOT)),
        "operator_api": "/api/hive/operator/utilization",
        "actions": ["sweep", "pause", "resume", "stop", "clear_stop"],
    }


def trigger_state(utilization: dict[str, Any]) -> str:
    controls = control_state()
    if controls["stopped"]:
        return "RED"
    if controls["paused"]:
        return "YELLOW"
    return str(utilization.get("trigger_state") or training.get_path(utilization, ["summary", "trigger_state"], "") or "UNKNOWN")


def next_actions(
    worker_reports: list[dict[str, Any]],
    ledger: dict[str, Any],
    best: dict[str, Any],
    utilization: dict[str, Any],
    local: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    controls = control_state()
    if controls["stopped"]:
        actions.append("Clear the utilization stop flag before starting an overnight solo loop.")
    if controls["paused"]:
        actions.append("Resume utilization before expecting always-active training to queue new work.")
    if not mlx_state(local).get("available"):
        actions.append("Install or fix MLX before expecting Apple Silicon training chunks to run.")
    if not worker_reports:
        actions.append("Run `theseus solo sweep --execute --allow-battery --max-new-jobs 1` to prove one local worker chunk.")
    if worker_reports and not (best.get("best_by_arm") if isinstance(best.get("best_by_arm"), dict) else {}):
        actions.append("Inspect worker scores; no local artifact has beaten the baseline yet.")
    if int(ledger.get("failed_count") or 0) > 0:
        actions.append("Open failed rows in `reports/hive_solo_learning_ledger.jsonl` and fix the smallest worker/runtime issue.")
    if not actions:
        actions.append("Run `theseus solo loop --execute --cycles 3 --sleep-seconds 60 --max-new-jobs 1 --keep-awake` before a longer offline run.")
    if utilization.get("mode") and training.get_path(utilization, ["offline_contract", "enabled"], False) is not True:
        actions.append("Use `--offline` for travel/flight work so no peer, teacher, or catalog access is required.")
    return actions


def compact_solo_overnight(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "created_utc": report.get("created_utc"),
        "window_hours": report.get("window_hours"),
        "worker_event_count": summary.get("worker_event_count"),
        "improved_count": summary.get("improved_count"),
        "failed_count": summary.get("failed_count"),
        "promoted_arm_count": summary.get("promoted_arm_count"),
        "next_actions": report.get("next_actions") if isinstance(report.get("next_actions"), list) else [],
    }


def compact_promotion_decision(promotion: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": promotion.get("decision"),
        "promoted": bool(promotion.get("promoted")),
        "previous_score": promotion.get("previous_score"),
        "candidate_score": promotion.get("candidate_score"),
        "reason": promotion.get("reason"),
        "active_manifest": training.get_path(promotion, ["best_entry", "active_manifest"], ""),
        "activated_artifact": training.get_path(promotion, ["best_entry", "activated_artifact"], ""),
        "rollback_manifest": training.get_path(promotion, ["best_entry", "rollback_manifest"], ""),
    }


def failure_reason_for(raw: dict[str, Any], row: dict[str, Any]) -> str:
    if row.get("ok", True) is not False:
        return ""
    for key in ["error", "message", "stderr_tail", "stdout_tail"]:
        value = raw.get(key) if isinstance(raw, dict) else None
        if value:
            return str(value)[:500]
    return "worker_report_marked_not_ok"


def score_source(metrics: dict[str, Any]) -> str:
    for key in ["eval_accuracy", "accuracy", "score", "pass_rate", "reward_mean"]:
        if key in metrics:
            return key
    return ""


def ledger_keys() -> set[str]:
    out: set[str] = set()
    if not LEDGER_PATH.exists():
        return out
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event_key"):
            out.add(str(value.get("event_key")))
    return out


def resolve_report_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return out.strip("-_")[:120] or "unknown"


def parse_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
