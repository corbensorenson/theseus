"""Operator notification helpers for the Theseus Hive node."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import hive_version_manager
import update_manager
from hive_node_common import ROOT, get_path, now, read_json, read_jsonl_tail, task_ledger_path, write_json
from hive_node_federation import unique_nonempty


REPORTS = ROOT / "reports"


def _update_policy() -> dict[str, Any]:
    return update_manager.read_json(update_manager.POLICY_PATH, {})


def operator_network_summary() -> dict[str, Any]:
    report = read_json(REPORTS / "hive_network_doctor.json", {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    return {
        "policy": "project_theseus_hive_operator_network_summary_v0",
        "state": report.get("state") or "UNKNOWN",
        "created_utc": report.get("created_utc") or "",
        "coordinator_reachable": summary.get("coordinator_reachable"),
        "remote_peer_reachable_count": summary.get("remote_peer_reachable_count"),
        "remote_peer_inbound_only_count": summary.get("remote_peer_inbound_only_count"),
        "http_status_ok_count": summary.get("http_status_ok_count"),
        "stale_peer_count": summary.get("stale_peer_count"),
        "red_findings": summary.get("red_findings"),
        "yellow_findings": summary.get("yellow_findings"),
        "top_findings": [
            {
                "severity": row.get("severity"),
                "code": row.get("code"),
                "title": row.get("title"),
            }
            for row in findings[:4]
            if isinstance(row, dict)
        ],
        "doctor_endpoint": "/api/hive/network-doctor",
    }

def operator_notification_summary(policy: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    report = operator_notifications(policy, auth_context, limit=6, write_report=False)
    notifications = report.get("notifications") if isinstance(report.get("notifications"), list) else []
    return {
        "policy": "project_theseus_hive_operator_notification_summary_v0",
        "unread_count": report.get("unread_count", 0),
        "notification_count": report.get("notification_count", 0),
        "latest": notifications[:3],
        "endpoint": "/api/hive/operator/notifications",
        "ack_endpoint": "/api/hive/operator/notifications/ack",
    }

def operator_notifications(
    policy: dict[str, Any],
    auth_context: dict[str, Any] | None = None,
    *,
    limit: int = 30,
    since: str = "",
    write_report: bool = True,
) -> dict[str, Any]:
    subject = notification_subject(auth_context)
    acked = notification_ack_ids(subject)
    rows = build_operator_notifications(policy)
    if since:
        rows = [row for row in rows if str(row.get("created_utc") or "") > since]
    rows.sort(key=lambda row: str(row.get("created_utc") or ""), reverse=True)
    max_rows = max(1, min(int(limit or 30), 100))
    for row in rows:
        row["acknowledged"] = str(row.get("id") or "") in acked
    unread = [row for row in rows if not row.get("acknowledged")]
    report = {
        "ok": True,
        "policy": "project_theseus_hive_operator_notifications_v0",
        "created_utc": now(),
        "subject": subject,
        "notification_count": len(rows),
        "unread_count": len(unread),
        "notifications": rows[:max_rows],
    }
    if write_report:
        write_json(REPORTS / "hive_operator_notifications.json", report)
    return report

def acknowledge_operator_notifications(policy: dict[str, Any], payload: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    subject = notification_subject(auth_context)
    feed = operator_notifications(policy, auth_context, limit=100, write_report=False)
    current_ids = [str(row.get("id") or "") for row in feed.get("notifications", []) if isinstance(row, dict) and row.get("id")]
    ids = payload.get("ids") if isinstance(payload.get("ids"), list) else []
    ack_ids = [str(item) for item in ids if str(item) in set(current_ids)]
    if payload.get("all_current"):
        ack_ids = current_ids
    all_before = str(payload.get("all_before_utc") or "")
    if all_before:
        ack_ids.extend(
            str(row.get("id") or "")
            for row in feed.get("notifications", [])
            if isinstance(row, dict) and str(row.get("created_utc") or "") <= all_before and row.get("id")
        )
    ack_ids = unique_nonempty(ack_ids)
    if not ack_ids:
        return {"ok": True, "policy": "project_theseus_hive_operator_notifications_ack_v0", "created_utc": now(), "subject": subject, "acked_count": 0}
    ack = notification_ack_state()
    subjects = ack.setdefault("subjects", {})
    state = subjects.setdefault(subject, {"acked_ids": {}})
    acked = state.setdefault("acked_ids", {})
    for notification_id in ack_ids:
        acked[notification_id] = now()
    if len(acked) > 500:
        kept = sorted(acked.items(), key=lambda item: item[1], reverse=True)[:500]
        state["acked_ids"] = {key: value for key, value in kept}
    state["updated_utc"] = now()
    ack["updated_utc"] = now()
    write_json(notification_ack_path(), ack)
    return {
        "ok": True,
        "policy": "project_theseus_hive_operator_notifications_ack_v0",
        "created_utc": now(),
        "subject": subject,
        "acked_count": len(ack_ids),
        "acked_ids": ack_ids,
    }

def build_operator_notifications(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(network_notifications())
    rows.extend(utilization_notifications())
    rows.extend(training_notifications())
    rows.extend(update_notifications())
    rows.extend(task_failure_notifications(policy))
    return dedupe_notifications(rows)

def network_notifications() -> list[dict[str, Any]]:
    report = read_json(REPORTS / "hive_network_doctor.json", {})
    state = str(report.get("state") or "").upper()
    if state not in {"RED", "YELLOW"}:
        return []
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    top = [row for row in findings if isinstance(row, dict) and str(row.get("severity") or "").upper() in {"RED", "YELLOW"}][:3]
    title = "Hive Network Needs Attention" if state == "YELLOW" else "Hive Network Blocked"
    body = "; ".join(str(row.get("title") or row.get("code") or "") for row in top if row) or f"Network Doctor is {state}."
    return [
        notification_row(
            category="network",
            severity="warning" if state == "YELLOW" else "critical",
            title=title,
            body=body,
            created_utc=str(report.get("created_utc") or now()),
            source="hive_network_doctor",
            dedupe_parts=[state, body],
            interruptive=True,
            data={"state": state, "findings": top},
        )
    ]

def utilization_notifications() -> list[dict[str, Any]]:
    report = read_json(REPORTS / "hive_utilization_manager.json", {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    blocked_nodes = int(summary.get("blocked_nodes") or report.get("blocked_nodes") or 0)
    uncovered = int(summary.get("safe_idle_uncovered_nodes") or report.get("safe_idle_uncovered_nodes") or 0)
    rows: list[dict[str, Any]] = []
    if blocked_nodes > 0:
        rows.append(
            notification_row(
                category="utilization",
                severity="warning",
                title="Hive Work Is Blocked",
                body=f"{blocked_nodes} node(s) are blocked from safe always-active work.",
                created_utc=str(report.get("created_utc") or now()),
                source="hive_utilization_manager",
                dedupe_parts=["blocked", str(blocked_nodes)],
                interruptive=True,
                data={"blocked_nodes": blocked_nodes},
            )
        )
    if uncovered > 0:
        rows.append(
            notification_row(
                category="utilization",
                severity="info",
                title="Idle Hive Capacity Available",
                body=f"{uncovered} safe idle node(s) have no planned work.",
                created_utc=str(report.get("created_utc") or now()),
                source="hive_utilization_manager",
                dedupe_parts=["uncovered", str(uncovered)],
                interruptive=False,
                data={"safe_idle_uncovered_nodes": uncovered},
            )
        )
    return rows

def training_notifications() -> list[dict[str, Any]]:
    report = read_json(REPORTS / "hive_overnight_training_report.json", {})
    if not isinstance(report, dict) or not report:
        return []
    rows: list[dict[str, Any]] = []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    failed = int(summary.get("failed_count") or report.get("failed_count") or 0)
    promoted = int(summary.get("promotion_count") or report.get("promotion_count") or report.get("promoted_count") or 0)
    workers = int(summary.get("worker_report_count") or report.get("worker_report_count") or 0)
    created = str(report.get("created_utc") or now())
    if failed > 0:
        rows.append(
            notification_row(
                category="training",
                severity="warning",
                title="Training Had Failures",
                body=f"Overnight training recorded {failed} failed job(s).",
                created_utc=created,
                source="hive_overnight_training_report",
                dedupe_parts=["failed", created, str(failed)],
                interruptive=True,
                data={"failed_count": failed, "worker_report_count": workers},
            )
        )
    if promoted > 0:
        rows.append(
            notification_row(
                category="training",
                severity="info",
                title="Training Promoted Artifacts",
                body=f"{promoted} training artifact(s) were promoted from {workers} worker report(s).",
                created_utc=created,
                source="hive_overnight_training_report",
                dedupe_parts=["promoted", created, str(promoted)],
                interruptive=True,
                data={"promotion_count": promoted, "worker_report_count": workers},
            )
        )
    return rows

def update_notifications() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        version = hive_version_manager.status_report(write_report=False)
    except Exception:
        version = {}
    convergence = version.get("convergence") if isinstance(version.get("convergence"), dict) else {}
    if convergence and convergence.get("local_matches_target") is False:
        target = str(convergence.get("target_version_id") or "")
        local = str(get_path(version, ["local", "version_id"], ""))
        rows.append(
            notification_row(
                category="updates",
                severity="info",
                title="Hive Update Available",
                body=f"Local version {local or 'unknown'} does not match target {target or 'unknown'}.",
                created_utc=str(version.get("created_utc") or now()),
                source="hive_version_manager",
                dedupe_parts=["version", local, target],
                interruptive=False,
                data={"local_version_id": local, "target_version_id": target},
            )
        )
    try:
        update = update_manager.status_report(policy=_update_policy(), write_report=False)
    except Exception:
        update = {}
    if update.get("restart_required"):
        rows.append(
            notification_row(
                category="updates",
                severity="warning",
                title="Hive Restart Required",
                body=str(update.get("next_action") or "Restart this node to finish the update."),
                created_utc=str(update.get("created_utc") or now()),
                source="update_manager",
                dedupe_parts=["restart_required"],
                interruptive=True,
                data={"restart_required": True},
            )
        )
    return rows

def task_failure_notifications(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in read_jsonl_tail(task_ledger_path(policy), 25):
        if not isinstance(entry, dict):
            continue
        failed = entry.get("ok") is False or str(entry.get("status") or "").lower() in {"failed", "error"}
        if not failed:
            continue
        kind = str(entry.get("kind") or get_path(entry, ["job", "kind"], "task"))
        job_id = str(entry.get("job_id") or get_path(entry, ["job", "job_id"], ""))
        rows.append(
            notification_row(
                category="task",
                severity="warning",
                title="Hive Task Failed",
                body=f"{kind} failed{f' ({job_id})' if job_id else ''}.",
                created_utc=str(entry.get("created_utc") or now()),
                source="hive_task_ledger",
                dedupe_parts=["task_failed", kind, job_id or str(entry.get("created_utc") or "")],
                interruptive=True,
                data={"kind": kind, "job_id": job_id, "returncode": entry.get("returncode")},
            )
        )
    return rows

def notification_row(
    *,
    category: str,
    severity: str,
    title: str,
    body: str,
    created_utc: str,
    source: str,
    dedupe_parts: list[str],
    interruptive: bool,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dedupe_key = "|".join([category, severity] + [str(part) for part in dedupe_parts])
    notification_id = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:24]
    return {
        "id": notification_id,
        "category": category,
        "severity": severity,
        "title": title,
        "body": body[:500],
        "created_utc": created_utc,
        "source": source,
        "dedupe_key": dedupe_key,
        "interruptive": bool(interruptive),
        "data": data or {},
    }

def dedupe_notifications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        notification_id = str(row.get("id") or "")
        if not notification_id or notification_id in seen:
            continue
        seen.add(notification_id)
        out.append(row)
    return out

def notification_subject(auth_context: dict[str, Any] | None) -> str:
    if not auth_context:
        return "anonymous"
    user_id = str(auth_context.get("user_id") or "unknown")
    token_kind = str(auth_context.get("token_kind") or "token")
    return f"{token_kind}:{user_id}"

def notification_ack_path() -> Path:
    return REPORTS / "hive_operator_notification_ack.local.json"

def notification_ack_state() -> dict[str, Any]:
    state = read_json(notification_ack_path(), {})
    if not isinstance(state, dict) or state.get("policy") != "project_theseus_hive_operator_notification_ack_v0":
        state = {
            "policy": "project_theseus_hive_operator_notification_ack_v0",
            "created_utc": now(),
            "updated_utc": now(),
            "subjects": {},
        }
    state.setdefault("subjects", {})
    return state

def notification_ack_ids(subject: str) -> set[str]:
    state = notification_ack_state()
    subjects = state.get("subjects") if isinstance(state.get("subjects"), dict) else {}
    subject_state = subjects.get(subject) if isinstance(subjects.get(subject), dict) else {}
    acked = subject_state.get("acked_ids") if isinstance(subject_state.get("acked_ids"), dict) else {}
    return {str(key) for key in acked.keys()}
