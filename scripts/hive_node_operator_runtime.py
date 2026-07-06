"""Operator-facing training and utilization summaries for Hive nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hive_node_common import get_path, read_json, read_jsonl_tail
from hive_node_compact_reports import compact_operator_overnight, compact_promoted, compact_training_execution


ROOT = Path(__file__).resolve().parents[1]


def operator_training_summary(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("training_orchestration") if isinstance(policy.get("training_orchestration"), dict) else {}
    state = read_json(ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json"), {})
    last = read_json(ROOT / str(cfg.get("report_path") or "reports/hive_training_orchestrator.json"), {})
    merge = read_json(ROOT / str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")), {})
    overnight = read_json(ROOT / "reports" / "hive_overnight_training_report.json", {})
    arms = state.get("arms") if isinstance(state.get("arms"), list) else []
    best_by_arm = merge.get("best_by_arm") if isinstance(merge.get("best_by_arm"), dict) else {}
    return {
        "policy": "project_theseus_hive_operator_training_v0",
        "enabled": bool(cfg.get("enabled", True)),
        "strategy": cfg.get("strategy") or "deterministic_arm_slot_owner_v0",
        "active_run_id": state.get("active_run_id") or last.get("run_id"),
        "last_round_id": state.get("last_round_id") or last.get("round_id"),
        "profile": state.get("profile") or last.get("profile"),
        "arm_count": len(arms),
        "arms": [
            {
                "arm_id": arm.get("arm_id"),
                "display_name": arm.get("display_name"),
                "task_kind": arm.get("task_kind"),
                "owner_node_name": arm.get("owner_node_name"),
                "slot_type": arm.get("slot_type"),
                "job_id": arm.get("job_id"),
                "lease_expires_utc": arm.get("lease_expires_utc"),
                "best_score": get_path(best_by_arm, [str(arm.get("arm_id") or ""), "score"], None),
            }
            for arm in arms[:12]
            if isinstance(arm, dict)
        ],
        "blocked": state.get("blocked") if isinstance(state.get("blocked"), list) else [],
        "promoted_count": len(merge.get("promoted") or []) if isinstance(merge.get("promoted"), list) else 0,
        "promoted": compact_promoted(merge.get("promoted") if isinstance(merge.get("promoted"), list) else []),
        "recent_jobs": recent_training_jobs_for_operator(policy),
        "mac_local": operator_macos_training_summary(),
        "teacher": operator_full_training_teacher_summary(),
        "overnight": compact_operator_overnight(overnight),
        "last_execution": compact_training_execution(last.get("execution") if isinstance(last.get("execution"), list) else []),
        "last_mode": last.get("mode"),
        "last_created_utc": last.get("created_utc"),
    }


def operator_utilization_summary(status: dict[str, Any]) -> dict[str, Any]:
    full = read_json(ROOT / "reports" / "hive_utilization_manager.json", {})
    summary = full.get("summary") if isinstance(full.get("summary"), dict) else {}
    compact = status.get("utilization") if isinstance(status.get("utilization"), dict) else {}
    return {
        **compact,
        "policy": full.get("policy"),
        "created_utc": full.get("created_utc"),
        "mode": full.get("mode"),
        "trigger_state": full.get("trigger_state") or compact.get("trigger_state"),
        "always_active": full.get("always_active") if isinstance(full.get("always_active"), dict) else {},
        "summary": summary,
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "node_name": node.get("node_name"),
                "api_url": node.get("api_url"),
                "is_local": bool(node.get("is_local")),
                "reachable": node.get("reachable"),
                "reachability_error": node.get("reachability_error"),
                "intended_state": node.get("intended_state"),
                "resource_blockers": node.get("resource_blockers") if isinstance(node.get("resource_blockers"), list) else [],
                "training_blockers": node.get("training_blockers") if isinstance(node.get("training_blockers"), list) else [],
                "idle_slots": len(node.get("idle_slots") or []),
                "busy_slots": len(node.get("busy_slots") or []),
                "queue_pressure": node.get("queue_pressure"),
            }
            for node in full.get("nodes", [])
            if isinstance(node, dict)
        ],
        "actions": [
            {
                "kind": action.get("kind"),
                "title": action.get("title"),
                "priority_lane": action.get("priority_lane"),
                "planned_jobs": action.get("planned_jobs"),
                "target_node_id": get_path(action, ["evidence", "node_id"], ""),
                "task_kind": get_path(action, ["submit", "kind"], ""),
            }
            for action in full.get("actions", [])
            if isinstance(action, dict)
        ],
        "execution": [
            {
                "kind": row.get("kind"),
                "priority_lane": row.get("priority_lane"),
                "status": row.get("status"),
                "runtime_ms": row.get("runtime_ms"),
                "ok": get_path(row, ["result", "ok"], None),
                "error": get_path(row, ["result", "error"], ""),
                "task_kind": get_path(row, ["submit", "kind"], ""),
            }
            for row in full.get("execution", [])
            if isinstance(row, dict)
        ],
        "recent_sweeps": [
            {
                "created_utc": row.get("created_utc"),
                "mode": row.get("mode"),
                "trigger_state": row.get("trigger_state"),
                "planned_actions": row.get("planned_actions"),
                "executed_actions": row.get("executed_actions"),
                "blocked_nodes": row.get("blocked_nodes"),
            }
            for row in read_jsonl_tail(ROOT / "reports" / "hive_utilization_ledger.jsonl", 8)
            if isinstance(row, dict)
        ],
    }


def operator_solo_learning_summary() -> dict[str, Any]:
    status = read_json(ROOT / "reports" / "hive_solo_learning_status.json", {})
    best = read_json(ROOT / "reports" / "hive_solo_best_by_arm.json", {})
    overnight = read_json(ROOT / "reports" / "hive_solo_overnight_report.json", {})
    controls = status.get("controls") if isinstance(status.get("controls"), dict) else {}
    mlx = status.get("mlx") if isinstance(status.get("mlx"), dict) else {}
    ledger = status.get("ledger") if isinstance(status.get("ledger"), dict) else {}
    always_active = status.get("always_active") if isinstance(status.get("always_active"), dict) else {}
    best_by_arm = best.get("best_by_arm") if isinstance(best.get("best_by_arm"), dict) else {}
    promotions = best.get("promotions") if isinstance(best.get("promotions"), list) else []
    overnight_summary = overnight.get("summary") if isinstance(overnight.get("summary"), dict) else {}
    return {
        "policy": "project_theseus_hive_operator_solo_learning_v0",
        "available": bool(status),
        "created_utc": status.get("created_utc"),
        "state": always_active.get("state") or status.get("mode") or "UNKNOWN",
        "standalone": status.get("standalone", True) if status else True,
        "offline_capable": status.get("offline_capable", True) if status else True,
        "local_only": status.get("local_only", True) if status else True,
        "paused": bool(controls.get("paused")),
        "stopped": bool(controls.get("stopped")),
        "mlx_available": bool(mlx.get("available")),
        "mlx_status": mlx.get("status"),
        "ledger_path": ledger.get("path") or "reports/hive_solo_learning_ledger.jsonl",
        "worker_event_count": ledger.get("event_count"),
        "new_event_count": ledger.get("new_event_count"),
        "failed_count": ledger.get("failed_count"),
        "promotion_count": len(promotions),
        "promoted_arm_count": len(best_by_arm),
        "best_by_arm": [
            {
                "arm_id": arm_id,
                "score": row.get("score"),
                "backend": row.get("backend"),
                "task_kind": row.get("task_kind"),
                "active_manifest": row.get("active_manifest"),
                "activated_artifact": row.get("activated_artifact"),
            }
            for arm_id, row in list(best_by_arm.items())[:12]
            if isinstance(row, dict)
        ],
        "recent": ledger.get("recent") if isinstance(ledger.get("recent"), list) else [],
        "overnight": {
            "created_utc": overnight.get("created_utc"),
            "worker_event_count": overnight_summary.get("worker_event_count"),
            "improved_count": overnight_summary.get("improved_count"),
            "failed_count": overnight_summary.get("failed_count"),
            "next_actions": overnight.get("next_actions") if isinstance(overnight.get("next_actions"), list) else [],
        },
        "next_actions": status.get("next_actions") if isinstance(status.get("next_actions"), list) else [],
    }


def operator_full_training_teacher_summary() -> dict[str, Any]:
    preflight = read_json(ROOT / "reports" / "full_training_teacher_preflight.json", {})
    budget = read_json(ROOT / "reports" / "teacher_budget_audit.json", {})
    live = read_json(ROOT / "reports" / "teacher_oracle_live_smoke.json", {})
    queued = read_json(ROOT / "reports" / "teacher_oracle_queue_smoke.json", {})
    readiness = read_json(ROOT / "reports" / "autonomy_launch_readiness_teacher_preflight.json", {})
    summary = preflight.get("summary") if isinstance(preflight.get("summary"), dict) else {}
    return {
        "policy": "project_theseus_hive_operator_full_training_teacher_v0",
        "available": bool(preflight),
        "created_utc": preflight.get("created_utc"),
        "state": preflight.get("trigger_state") or "UNKNOWN",
        "ok": preflight.get("ok"),
        "blocker_count": summary.get("blocker_count"),
        "warning_count": summary.get("warning_count"),
        "teacher_cli_available": summary.get("teacher_cli_available"),
        "teacher_live_status": summary.get("teacher_live_status") or live.get("status"),
        "queue_status": queued.get("status"),
        "apply_mode_blocked": summary.get("apply_mode_blocked"),
        "worker_teacher_invariant": summary.get("worker_teacher_invariant"),
        "budget_state": budget.get("trigger_state"),
        "autonomy_ready": readiness.get("ready_for_autonomous_training"),
        "teacher_enabled_ready": readiness.get("ready_for_teacher_enabled_run"),
        "failed_gates": [
            {
                "name": row.get("name"),
                "severity": row.get("severity"),
                "detail": row.get("detail"),
            }
            for row in (
                (preflight.get("blocker_failures") if isinstance(preflight.get("blocker_failures"), list) else [])
                + (preflight.get("warning_failures") if isinstance(preflight.get("warning_failures"), list) else [])
            )[:8]
            if isinstance(row, dict)
        ],
        "next_commands": preflight.get("next_commands") if isinstance(preflight.get("next_commands"), list) else [],
    }


def operator_macos_training_summary() -> dict[str, Any]:
    preflight = read_json(ROOT / "reports" / "macos_training_preflight.json", {})
    readiness = read_json(ROOT / "reports" / "autonomy_launch_readiness.json", {})
    execution = preflight.get("execution") if isinstance(preflight.get("execution"), dict) else {}
    worker = execution.get("worker_report") if isinstance(execution.get("worker_report"), dict) else {}
    payload = worker.get("payload") if isinstance(worker.get("payload"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    resources = preflight.get("resources") if isinstance(preflight.get("resources"), dict) else {}
    power = resources.get("power") if isinstance(resources.get("power"), dict) else {}
    platform = preflight.get("platform") if isinstance(preflight.get("platform"), dict) else {}
    return {
        "policy": "project_theseus_hive_operator_macos_training_v0",
        "available": bool(preflight),
        "created_utc": preflight.get("created_utc"),
        "state": preflight.get("state") or "UNKNOWN",
        "bounded_smoke_allowed": preflight.get("bounded_smoke_allowed"),
        "long_training_allowed": preflight.get("long_training_allowed"),
        "hard_failures": get_path(preflight, ["summary", "hard_failures"], None),
        "advisory_failures": get_path(preflight, ["summary", "public_or_advisory_failures"], None),
        "worker_canary": execution.get("kind"),
        "worker_report_ok": worker.get("ok"),
        "worker_report_path": worker.get("path"),
        "backend": payload.get("backend"),
        "task_kind": get_path(payload, ["work_receipt", "task_kind"], None) or payload.get("task_kind"),
        "teacher_used": payload.get("teacher_used"),
        "external_inference_calls": payload.get("external_inference_calls"),
        "train_accuracy": metrics.get("train_accuracy"),
        "eval_accuracy": metrics.get("eval_accuracy"),
        "runtime_ms": payload.get("runtime_ms"),
        "is_apple_silicon": platform.get("is_apple_silicon"),
        "is_intel": platform.get("is_intel"),
        "battery_percent": power.get("battery_percent"),
        "on_ac_power": power.get("on_ac_power"),
        "readiness_local_smoke": readiness.get("ready_for_local_macos_smoke_training"),
        "readiness_teacher_enabled": readiness.get("ready_for_teacher_enabled_run"),
        "readiness_autonomous": readiness.get("ready_for_autonomous_training"),
        "smoke_blockers": readiness.get("local_macos_smoke_blockers") if isinstance(readiness.get("local_macos_smoke_blockers"), list) else [],
    }


def recent_training_jobs_for_operator(policy: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    rows = read_jsonl_tail(ROOT / str(get_path(policy, ["node", "job_ledger_path"], "reports/hive_job_ledger.jsonl")), 80)
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        job = row.get("job") if isinstance(row.get("job"), dict) else {}
        orch = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
        kind = str(row.get("kind") or job.get("task_kind") or "")
        if not orch and not (kind.startswith("cuda_") or kind.startswith("mlx_") or kind == "training_smoke"):
            continue
        out.append(
            {
                "created_utc": row.get("created_utc"),
                "finished_utc": row.get("finished_utc"),
                "status": row.get("status"),
                "kind": kind,
                "job_id": job.get("job_id") or payload.get("job_id"),
                "arm_id": orch.get("arm_id") or job.get("arm_id"),
                "round_id": orch.get("round_id"),
                "returncode": row.get("returncode"),
                "runtime_ms": row.get("runtime_ms"),
                "stderr_tail": str(row.get("stderr_tail") or "")[-240:],
            }
        )
    return out[-limit:]
