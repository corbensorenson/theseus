"""Decentralized arm-level Hive training orchestration.

Every trusted Hive node can run this planner. It builds deterministic round
leases for independent training arms, assigns each arm to the best currently
visible node/slot, and optionally submits only registered bounded Hive tasks.
Duplicate job IDs are intentional: if multiple nodes submit the same round, the
Hive node enqueue guard collapses duplicates instead of running the same arm
twice.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
import hashlib


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
import hive_security  # noqa: E402
import hive_node_registry  # noqa: E402
import theseus_runtime  # noqa: E402


DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = ROOT / "reports" / "hive_training_orchestrator.json"
DEFAULT_OVERNIGHT_OUT = ROOT / "reports" / "hive_overnight_training_report.json"
DEFAULT_OVERNIGHT_MARKDOWN = ROOT / "reports" / "hive_overnight_training_report.md"
TRAINING_TASKS = {
    "cuda_eval_chunk",
    "cuda_training_chunk",
    "cuda_rollout_chunk",
    "mlx_eval_chunk",
    "mlx_training_chunk",
    "mlx_rollout_chunk",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and run decentralized Project Theseus Hive training rounds.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--payload-json", default="{}")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show the last orchestrator state and promoted arm artifacts.")
    status.add_argument("--out", default="")

    plan = sub.add_parser("plan", help="Build a deterministic arm placement plan without queueing work.")
    add_round_args(plan)

    run = sub.add_parser("run", help="Build a round plan and queue bounded worker chunks.")
    add_round_args(run)
    run.add_argument("--execute", action="store_true", default=True)
    run.add_argument("--no-execute", action="store_true")
    run.add_argument("--sync-artifacts", action="store_true")

    sync = sub.add_parser("sync", help="Fetch and merge Hive training artifacts.")
    sync.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    overnight = sub.add_parser("overnight", help="Write an overnight audit of worker chunks, promotions, failures, and stale leases.")
    overnight.add_argument("--hours", type=float, default=12.0)
    overnight.add_argument("--out", default=str(DEFAULT_OVERNIGHT_OUT.relative_to(ROOT)))
    overnight.add_argument("--markdown-out", default=str(DEFAULT_OVERNIGHT_MARKDOWN.relative_to(ROOT)))

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    payload = parse_json(args.payload_json, {})

    if args.command in {None, "status"}:
        report = status_report(policy=policy)
        out = getattr(args, "out", "") or ""
    elif args.command == "sync":
        report = sync_artifacts(policy)
        out = args.out
    elif args.command == "overnight":
        report = overnight_report(
            policy=policy,
            hours=float(args.hours),
            out=ROOT / args.out,
            markdown_out=ROOT / args.markdown_out,
            write_report=True,
        )
        out = ""
    else:
        execute = bool(getattr(args, "execute", False)) and not bool(getattr(args, "no_execute", False))
        if args.command == "plan":
            execute = False
        report = orchestrate(
            policy,
            profile=str(payload.get("profile") or args.profile),
            run_id=str(payload.get("run_id") or args.run_id or ""),
            round_id=str(payload.get("round_id") or args.round_id or ""),
            execute=execute,
            sync=bool(getattr(args, "sync_artifacts", False) or payload.get("sync_artifacts")),
            max_jobs=int(payload.get("max_jobs") or args.max_jobs or 0),
            allow_wan=bool(getattr(args, "allow_wan", False) or payload.get("allow_wan")),
            local_only=bool(getattr(args, "local_only", False) or payload.get("local_only")),
        )
        out = args.out
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def add_round_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--round-id", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--allow-wan", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))


def orchestrate(
    policy: dict[str, Any],
    *,
    profile: str = "",
    run_id: str = "",
    round_id: str = "",
    execute: bool = False,
    sync: bool = False,
    max_jobs: int = 0,
    allow_wan: bool = False,
    local_only: bool = False,
) -> dict[str, Any]:
    cfg = training_config(policy)
    profile = profile or str(cfg.get("default_profile") or "smoke")
    run_id = safe_id(run_id or active_run_id(cfg))
    round_id = safe_id(round_id or default_round_id(cfg))
    view = operator_view(policy)
    visible = visible_nodes(view)
    nodes, skipped_nodes = reachable_nodes(policy, visible)
    local = nodes[0] if nodes else {}
    lease_recovery = recover_stale_leases(policy, nodes)
    plan = build_plan(
        policy,
        cfg,
        nodes,
        profile=profile,
        run_id=run_id,
        round_id=round_id,
        max_jobs=max_jobs,
        allow_wan=allow_wan,
        local_only=local_only,
    )
    network_doctor = get_path(view, ["node_registry", "network_doctor"], {})
    fleet_readiness = training_fleet_readiness(network_doctor, visible, nodes, skipped_nodes, plan)
    execution: list[dict[str, Any]] = []
    if execute:
        for job in plan.get("jobs", []):
            if job.get("status") != "ready":
                continue
            execution.append(submit_job(policy, job))
    artifact_sync = sync_artifacts(policy) if sync else {}
    execution_ok = not any(row.get("ok") is False for row in execution)
    artifact_sync_ok = not artifact_sync or artifact_sync.get("ok", True) is not False
    report = {
        "ok": bool(execution_ok and artifact_sync_ok),
        "policy": "project_theseus_hive_training_orchestrator_v0",
        "created_utc": now(),
        "mode": "execute" if execute else "plan",
        "decentralized": True,
        "run_id": run_id,
        "round_id": round_id,
        "profile": profile,
        "local_node_id": local.get("node_id"),
        "visible_node_count": len(visible),
        "node_count": len(nodes),
        "network_doctor": network_doctor,
        "fleet_readiness": fleet_readiness,
        "skipped_nodes": skipped_nodes,
        "lease_recovery": lease_recovery,
        "plan": plan,
        "execution": execution,
        "artifact_sync": artifact_sync,
        "safety": {
            "arbitrary_shell": False,
            "task_surface": "registered_hive_worker_chunks_only",
            "duplicate_protection": "deterministic_job_ids_plus_node_enqueue_guard",
            "network_policy": "private_lan_or_tunnel_by_default" if not allow_wan else "wan_allowed_for_bounded_async_chunks",
        },
    }
    write_json(ROOT / str(cfg.get("report_path") or "reports/hive_training_orchestrator.json"), report)
    write_state(policy, report)
    append_jsonl(ROOT / str(cfg.get("ledger_path") or "reports/hive_training_orchestrator_ledger.jsonl"), compact_ledger_row(report))
    return report


def build_plan(
    policy: dict[str, Any],
    cfg: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    profile: str,
    run_id: str,
    round_id: str,
    max_jobs: int,
    allow_wan: bool,
    local_only: bool,
) -> dict[str, Any]:
    local = nodes[0] if nodes else {}
    arms = cfg.get("arms") if isinstance(cfg.get("arms"), list) else default_arms()
    capacity_used: dict[str, int] = {}
    jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    node_rows = [node_summary(node) for node in nodes]
    limit = max_jobs if max_jobs > 0 else int(cfg.get("default_max_jobs_per_round") or len(arms) or 1)

    for arm in sorted((row for row in arms if isinstance(row, dict) and row.get("enabled", True) is not False), key=lambda row: int(row.get("priority") or 100)):
        if len(jobs) >= limit:
            blocked.append({"arm_id": arm.get("arm_id"), "reason": "round_job_limit_reached", "max_jobs": limit})
            continue
        assignment = select_arm_assignment(policy, cfg, arm, nodes, local, capacity_used, allow_wan=allow_wan, local_only=local_only)
        if not assignment.get("ok"):
            blocked.append({"arm_id": arm.get("arm_id"), **assignment})
            continue
        node = assignment["node"]
        task_kind = str(assignment["task_kind"])
        slot_type = str(assignment["slot_type"])
        capacity_key = f"{node.get('node_id')}::{slot_type}"
        capacity_used[capacity_key] = capacity_used.get(capacity_key, 0) + 1
        jobs.append(
            job_from_assignment(
                policy,
                cfg,
                arm,
                node,
                local,
                task_kind=task_kind,
                slot_type=slot_type,
                profile=profile,
                run_id=run_id,
                round_id=round_id,
            )
        )
    return {
        "policy": "project_theseus_hive_training_round_plan_v0",
        "created_utc": now(),
        "run_id": run_id,
        "round_id": round_id,
        "profile": profile,
        "strategy": cfg.get("strategy") or "deterministic_arm_slot_owner_v0",
        "node_count": len(nodes),
        "nodes": node_rows,
        "job_count": len(jobs),
        "jobs": jobs,
        "blocked": blocked,
        "by_arm": {str(job.get("arm_id")): job.get("node_name") for job in jobs},
    }


def training_fleet_readiness(
    network_doctor: dict[str, Any],
    visible_nodes: list[dict[str, Any]],
    reachable_nodes: list[dict[str, Any]],
    skipped_nodes: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    doctor = network_doctor if isinstance(network_doctor, dict) else {}
    doctor_state = str(doctor.get("state") or "MISSING")
    doctor_red = [str(item) for item in doctor.get("red_finding_codes") or []]
    doctor_yellow = [str(item) for item in doctor.get("yellow_finding_codes") or []]
    remote_visible = [node for node in visible_nodes if not node.get("is_local")]
    remote_reachable = [node for node in reachable_nodes if not node.get("is_local")]
    cuda_nodes = [node for node in reachable_nodes if "nvidia_cuda" in set(node.get("capability_ids") or []) and node.get("training_allowed") is not False]
    mlx_nodes = [
        node
        for node in reachable_nodes
        if {"mlx_apple", "apple_mlx", "mlx_cuda"} & set(node.get("capability_ids") or []) and node.get("training_allowed") is not False
    ]
    queued_cuda = [job for job in plan.get("jobs", []) or [] if str(job.get("task_kind") or "").startswith("cuda_")]
    queued_mlx = [job for job in plan.get("jobs", []) or [] if str(job.get("task_kind") or "").startswith("mlx_")]
    skipped_reasons = sorted({str(row.get("reason") or "") for row in skipped_nodes if row.get("reason")})
    blocked_codes = []
    if doctor_state in {"RED", "MISSING", "STALE", "UNKNOWN"}:
        blocked_codes.extend(doctor_red or [f"network_doctor_{doctor_state.lower()}"])
    if remote_visible and not remote_reachable:
        blocked_codes.append("no_remote_peer_live_api")
    if cuda_nodes and not queued_cuda:
        blocked_codes.append("cuda_node_visible_but_no_cuda_job_queued")
    if skipped_reasons:
        blocked_codes.extend([f"skipped_{reason}" for reason in skipped_reasons])
    blocked_codes = list(dict.fromkeys(blocked_codes))
    return {
        "network_doctor_state": doctor_state,
        "network_doctor_fresh": bool(doctor.get("fresh")),
        "network_doctor_red_finding_codes": doctor_red,
        "network_doctor_yellow_finding_codes": doctor_yellow,
        "visible_remote_count": len(remote_visible),
        "reachable_remote_count": len(remote_reachable),
        "reachable_cuda_node_count": len(cuda_nodes),
        "reachable_mlx_node_count": len(mlx_nodes),
        "queued_cuda_job_count": len(queued_cuda),
        "queued_mlx_job_count": len(queued_mlx),
        "distributed_training_ready": bool(remote_reachable and len(reachable_nodes) > 1 and doctor_state not in {"RED", "MISSING", "STALE", "UNKNOWN"}),
        "mixed_cuda_mlx_training_ready": bool(cuda_nodes and mlx_nodes and remote_reachable and doctor_state not in {"RED", "MISSING", "STALE", "UNKNOWN"}),
        "blocked_codes": blocked_codes,
        "interpretation": "CUDA/MLX distributed work is ready only when live reachability and network doctor agree; plans remain local otherwise.",
    }


def select_arm_assignment(
    policy: dict[str, Any],
    cfg: dict[str, Any],
    arm: dict[str, Any],
    nodes: list[dict[str, Any]],
    local: dict[str, Any],
    capacity_used: dict[str, int],
    *,
    allow_wan: bool,
    local_only: bool,
) -> dict[str, Any]:
    task_kinds = [str(item) for item in arm.get("task_kinds", []) if item in TRAINING_TASKS]
    if not task_kinds:
        return {"ok": False, "reason": "arm_has_no_training_task_kinds"}
    candidates: list[dict[str, Any]] = []
    blocked_nodes: list[dict[str, Any]] = []
    for task_kind in task_kinds:
        for node in nodes:
            if local_only and node.get("node_id") != local.get("node_id"):
                continue
            if node.get("training_allowed") is False or node.get("training_blockers"):
                blocked_nodes.append(
                    {
                        "node_id": node.get("node_id"),
                        "node_name": node.get("node_name"),
                        "task_kind": task_kind,
                        "reason": "training_blocked",
                        "training_blockers": node.get("training_blockers") or [],
                    }
                )
                continue
            network = network_profile(node)
            if network.get("class") == "wan" and not allow_wan:
                continue
            slot = matching_slot(policy, node, task_kind)
            if not slot:
                continue
            capacity = int(slot.get("capacity") or 1)
            used = capacity_used.get(f"{node.get('node_id')}::{slot.get('slot_type')}", 0)
            if used >= capacity and not bool(cfg.get("queue_over_capacity", False)):
                continue
            candidates.append(
                {
                    "node": node,
                    "task_kind": task_kind,
                    "slot_type": slot.get("slot_type"),
                    "slot": slot,
                    "score": score_assignment(arm, node, slot, task_kind, network, used),
                    "network": network,
                }
            )
    if not candidates:
        return {"ok": False, "reason": "no_visible_node_with_free_matching_slot", "task_kinds": task_kinds, "blocked_nodes": blocked_nodes[:12]}
    best = sorted(candidates, key=lambda row: (-float(row.get("score") or 0.0), str(row["node"].get("node_id") or "")))[0]
    return {"ok": True, **best}


def job_from_assignment(
    policy: dict[str, Any],
    cfg: dict[str, Any],
    arm: dict[str, Any],
    node: dict[str, Any],
    local: dict[str, Any],
    *,
    task_kind: str,
    slot_type: str,
    profile: str,
    run_id: str,
    round_id: str,
) -> dict[str, Any]:
    arm_id = safe_id(str(arm.get("arm_id") or default_arm_for_task(task_kind)))
    node_id = str(node.get("node_id") or "node")
    chunk_id = safe_id(f"{round_id}_{arm_id}_{task_kind}_{node_id[-8:]}")
    lease_seconds = int(arm.get("lease_seconds") or cfg.get("lease_seconds") or 1800)
    lease_id = safe_id(f"lease_{run_id}_{round_id}_{arm_id}")
    payload = {
        "profile": profile,
        "chunk_id": chunk_id,
        "job_id": safe_id(f"job_{run_id}_{round_id}_{arm_id}_{task_kind}_{node_id[-8:]}"),
        "job_family": str(arm.get("job_family") or task_kind),
        "arm_id": arm_id,
        "backend_requirements": arm.get("backend_requirements") if isinstance(arm.get("backend_requirements"), list) else backend_requirements_for_task(task_kind),
        "merge_policy": str(arm.get("merge_policy") or cfg.get("merge_policy") or "promote_best_for_arm_if_score_improves"),
        "priority": int(arm.get("priority") or 75),
        "lease_seconds": lease_seconds,
        "max_retries": int(arm.get("max_retries") or cfg.get("max_retries") or 1),
        "requester_node_id": str(local.get("node_id") or ""),
        "requester_node_name": str(local.get("node_name") or local.get("hostname") or ""),
        "target_node_id": node_id,
        "target_node_name": str(node.get("node_name") or node.get("hostname") or ""),
        "allowed_task_scope": [task_kind],
        "output_artifacts": [{"type": "worker_report", "path": f"reports/hive_chunks/{chunk_id}.json"}],
        "orchestration": {
            "policy": "project_theseus_hive_training_job_orchestration_v0",
            "run_id": run_id,
            "round_id": round_id,
            "lease_id": lease_id,
            "lease_expires_utc": utc_after_seconds(lease_seconds),
            "arm_id": arm_id,
            "arm_display_name": arm.get("display_name") or arm_id,
            "owner_node_id": node_id,
            "owner_node_name": node.get("node_name"),
            "slot_type": slot_type,
            "strategy": cfg.get("strategy") or "deterministic_arm_slot_owner_v0",
            "artifact_flow": "worker_report_to_best_by_arm_promotion",
            "input_artifacts": active_input_artifacts(arm_id),
        },
    }
    payload.update(default_task_params(task_kind, profile))
    return {
        "status": "ready",
        "arm_id": arm_id,
        "display_name": arm.get("display_name") or arm_id,
        "task_kind": task_kind,
        "slot_type": slot_type,
        "node_id": node_id,
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "target": "local" if node_id == str(local.get("node_id") or "") else "remote",
        "network": network_profile(node),
        "lease": payload["orchestration"],
        "payload": payload,
    }


def submit_job(policy: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    api_url = str(job.get("api_url") or "")
    if not api_url:
        return {"ok": False, "error": "missing_api_url", "job_id": get_path(job, ["payload", "job_id"], "")}
    task_kind = str(job.get("task_kind") or "")
    payload = json.loads(json.dumps(job.get("payload") if isinstance(job.get("payload"), dict) else {}))
    secret = "" if is_loopback_url(api_url) else hive_secret(policy)
    if secret and "manifest" not in payload:
        manifest_payload = {key: value for key, value in payload.items() if key != "manifest"}
        payload["manifest"] = hive_security.build_manifest(
            task_kind,
            manifest_payload,
            hive_id=hive_id(policy),
            join_token=secret,
            scope=[task_kind],
        )
    body = json.dumps({"kind": task_kind, "payload": payload}).encode("utf-8")
    req = urlrequest.Request(
        api_url.rstrip("/") + "/api/hive/tasks",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if secret:
        req.add_header("X-Theseus-Hive-Secret", secret)
    try:
        with urlrequest.urlopen(req, timeout=15) as response:  # noqa: S310 - private Hive endpoint chosen from trusted peer state.
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001 - diagnostics only.
            body = ""
        return {
            "ok": False,
            "error": "submit_failed",
            "message": str(exc),
            "http_status": exc.code,
            "body": body[:2000],
            "api_url": api_url,
            "job_id": payload.get("job_id"),
        }
    except URLError as exc:
        return {"ok": False, "error": "submit_failed", "message": str(exc), "api_url": api_url, "job_id": payload.get("job_id")}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_submit_response", "body": raw[:500], "api_url": api_url, "job_id": payload.get("job_id")}
    if isinstance(result, dict):
        return {"job_id": payload.get("job_id"), "task_kind": task_kind, "target_node_id": job.get("node_id"), **result}
    return {"ok": False, "error": "unexpected_submit_response", "api_url": api_url, "job_id": payload.get("job_id")}


def sync_artifacts(policy: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "scripts/hive_artifact_sync.py", "--out", "reports/hive_artifact_sync.json", "--relay-results"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300, env=theseus_runtime.runtime_env())
    if result.returncode != 0:
        return {"ok": False, "returncode": result.returncode, "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:]}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "stdout_tail": result.stdout[-1000:]}
    return value if isinstance(value, dict) else {"ok": True}


def recover_stale_leases(policy: dict[str, Any], reachable_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = training_config(policy)
    state_path = ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json")
    state = read_json(state_path, {})
    arms = state.get("arms") if isinstance(state.get("arms"), list) else []
    reachable_ids = {str(node.get("node_id") or "") for node in reachable_nodes if node.get("node_id")}
    recovered: list[dict[str, Any]] = []
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        expires = parse_utc(str(arm.get("lease_expires_utc") or ""))
        owner = str(arm.get("owner_node_id") or "")
        if not expires or expires > datetime.now(timezone.utc):
            continue
        reason = "lease_expired"
        if owner and owner not in reachable_ids:
            reason = "lease_expired_owner_offline"
        recovered.append(
            {
                "arm_id": arm.get("arm_id"),
                "job_id": arm.get("job_id"),
                "lease_id": arm.get("lease_id"),
                "owner_node_id": owner,
                "owner_node_name": arm.get("owner_node_name"),
                "lease_expires_utc": arm.get("lease_expires_utc"),
                "reason": reason,
            }
        )
    report = {
        "policy": "project_theseus_hive_training_stale_lease_recovery_v0",
        "created_utc": now(),
        "enabled": True,
        "recovered_count": len(recovered),
        "recovered": recovered,
        "recovery_rule": "expired arm leases do not block the next deterministic round; planning uses currently reachable nodes only",
    }
    if recovered:
        state["stale_lease_recovery"] = report
        state["updated_utc"] = now()
        write_json(state_path, state)
    write_json(REPORTS / "hive_training_stale_lease_recovery.json", report)
    return report


def overnight_report(
    *,
    policy: dict[str, Any] | None = None,
    hours: float = 12.0,
    out: Path = DEFAULT_OVERNIGHT_OUT,
    markdown_out: Path = DEFAULT_OVERNIGHT_MARKDOWN,
    write_report: bool = True,
) -> dict[str, Any]:
    policy = policy or read_json(DEFAULT_POLICY, {})
    since = datetime.now(timezone.utc).timestamp() - max(0.1, hours) * 3600.0
    cfg = training_config(policy)
    state = read_json(ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json"), {})
    last = read_json(ROOT / str(cfg.get("report_path") or "reports/hive_training_orchestrator.json"), {})
    merge = read_json(ROOT / str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")), {})
    worker_reports = collect_worker_reports(since)
    recent_jobs = recent_training_jobs(policy, limit=200)
    recent_jobs = [job for job in recent_jobs if timestamp_in_window(str(job.get("created_utc") or job.get("finished_utc") or ""), since)]
    promotions = [
        compact_promotion(row)
        for row in merge.get("promoted", []) or []
        if isinstance(row, dict) and timestamp_in_window(str(row.get("created_utc") or ""), since)
    ]
    failed = collect_failed_training_events(last, recent_jobs)
    stale = stale_leases_from_state(state, merge)
    by_arm = summarize_worker_by(worker_reports, "arm_id")
    by_backend = summarize_worker_by(worker_reports, "backend")
    report = {
        "ok": True,
        "policy": "project_theseus_hive_overnight_training_report_v0",
        "created_utc": now(),
        "window_hours": hours,
        "window_start_utc": datetime.fromtimestamp(since, tz=timezone.utc).isoformat(),
        "summary": {
            "worker_report_count": len(worker_reports),
            "recent_job_count": len(recent_jobs),
            "promotion_count": len(promotions),
            "failed_count": len(failed),
            "stale_lease_count": len(stale),
            "best_arm_count": len(merge.get("best_by_arm") or {}) if isinstance(merge.get("best_by_arm"), dict) else 0,
        },
        "by_arm": by_arm,
        "by_backend": by_backend,
        "worker_reports": worker_reports[:200],
        "promotions": promotions,
        "best_by_arm": merge.get("best_by_arm") if isinstance(merge.get("best_by_arm"), dict) else {},
        "failed": failed,
        "stale_leases": stale,
        "lease_recovery": {
            "enabled": True,
            "rule": "expired leases are recoverable; the next run replans against currently reachable nodes and overwrites stale ownership",
            "command": "theseus train run --sync-artifacts --execute",
        },
        "next_actions": overnight_next_actions(worker_reports, promotions, failed, stale),
    }
    if write_report:
        write_json(out, report)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_overnight_markdown(report), encoding="utf-8")
    return report


def collect_worker_reports(since_epoch: float) -> list[dict[str, Any]]:
    roots = []
    for root in report_roots():
        roots.extend([root / "hive_chunks", root / "hive_artifact_inbox"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name.endswith(".artifact.json"):
                continue
            if path.stat().st_mtime < since_epoch:
                continue
            payload = read_json(path, {})
            if not isinstance(payload, dict) or not is_worker_report(payload):
                continue
            row = compact_worker_report(payload, path)
            key = str(row.get("job_id") or row.get("path"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("created_utc") or ""))


def is_worker_report(payload: dict[str, Any]) -> bool:
    if payload.get("policy") == "project_theseus_hive_worker_chunk_v0":
        return True
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    return bool(job.get("arm_id") or job.get("task_kind") in TRAINING_TASKS)


def compact_worker_report(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    orch = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
    return {
        "path": display_path(path),
        "created_utc": payload.get("created_utc"),
        "ok": payload.get("ok", True),
        "job_id": job.get("job_id") or payload.get("job_id"),
        "task_kind": job.get("task_kind") or payload.get("task_kind"),
        "worker_kind": job.get("worker_kind") or payload.get("kind"),
        "arm_id": orch.get("arm_id") or job.get("arm_id") or payload.get("arm_id"),
        "run_id": orch.get("run_id"),
        "round_id": orch.get("round_id"),
        "owner_node_id": orch.get("owner_node_id"),
        "owner_node_name": orch.get("owner_node_name"),
        "backend": payload.get("backend") or first_string(job.get("backend_requirements")),
        "backend_requirements": job.get("backend_requirements") if isinstance(job.get("backend_requirements"), list) else [],
        "merge_policy": job.get("merge_policy"),
        "merge_result": merge_result_for_job(str(job.get("job_id") or payload.get("job_id") or "")),
        "score": worker_score(metrics),
        "metrics": compact_metrics(metrics),
        "input_artifacts": orch.get("input_artifacts") if isinstance(orch.get("input_artifacts"), list) else [],
        "train_input": payload.get("train_input"),
        "eval_input": payload.get("eval_input"),
        "output_artifacts": job.get("output_artifacts") if isinstance(job.get("output_artifacts"), list) else [],
        "model_path": telemetry.get("model_path"),
        "runtime_ms": payload.get("runtime_ms"),
    }


def merge_result_for_job(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {"state": "unknown"}
    merge = read_json(ROOT / "reports" / "hive_artifact_merge_summary.json", {})
    for row in merge.get("promoted", []) or []:
        if isinstance(row, dict) and row.get("job_id") == job_id:
            return {"state": "promoted", "score": row.get("score"), "promoted_model_path": row.get("promoted_model_path")}
    best = merge.get("best_by_arm") if isinstance(merge.get("best_by_arm"), dict) else {}
    for row in best.values():
        if isinstance(row, dict) and row.get("job_id") == job_id:
            return {"state": "best_by_arm", "score": row.get("score")}
    return {"state": "merged_or_pending"}


def worker_score(metrics: dict[str, Any]) -> Any:
    for key in ["eval_accuracy", "accuracy", "score", "pass_rate", "reward_mean"]:
        if key in metrics:
            return metrics.get(key)
    return None


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "train_accuracy",
        "eval_accuracy",
        "loss_initial",
        "loss_final",
        "examples_per_second",
        "steps",
        "feature_dim",
        "runtime_ms",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def compact_promotion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": row.get("created_utc"),
        "arm_id": row.get("arm_id"),
        "job_id": row.get("job_id"),
        "run_id": row.get("run_id"),
        "round_id": row.get("round_id"),
        "owner_node_id": row.get("owner_node_id"),
        "backend": row.get("backend"),
        "score": row.get("score"),
        "source_report": row.get("source_report"),
        "promoted_model_path": row.get("promoted_model_path"),
        "merge_policy": row.get("merge_policy"),
    }


def collect_failed_training_events(last: dict[str, Any], recent_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in last.get("execution") or []:
        if isinstance(event, dict) and event.get("ok") is False:
            rows.append(
                {
                    "source": "last_orchestrator_execution",
                    "job_id": event.get("job_id"),
                    "task_kind": event.get("task_kind"),
                    "target_node_id": event.get("target_node_id"),
                    "error": event.get("error") or event.get("message"),
                }
            )
    for job in recent_jobs:
        if str(job.get("status") or "").lower() in {"failed", "timeout", "error"} or int(job.get("returncode") or 0) not in {0}:
            rows.append({**job, "source": "job_ledger"})
    return rows


def stale_leases_from_state(state: dict[str, Any], merge: dict[str, Any]) -> list[dict[str, Any]]:
    promoted_jobs = {str(row.get("job_id") or "") for row in merge.get("promoted", []) or [] if isinstance(row, dict)}
    best_jobs = {
        str(row.get("job_id") or "")
        for row in (merge.get("best_by_arm") or {}).values()
        if isinstance(row, dict)
    } if isinstance(merge.get("best_by_arm"), dict) else set()
    rows = []
    for arm in state.get("arms", []) if isinstance(state.get("arms"), list) else []:
        if not isinstance(arm, dict):
            continue
        expires = parse_utc(str(arm.get("lease_expires_utc") or ""))
        if expires and expires < datetime.now(timezone.utc):
            job_id = str(arm.get("job_id") or "")
            merge_state = "promoted" if job_id in promoted_jobs else ("best_by_arm" if job_id in best_jobs else "not_merged")
            rows.append(
                {
                    "arm_id": arm.get("arm_id"),
                    "job_id": job_id,
                    "owner_node_id": arm.get("owner_node_id"),
                    "owner_node_name": arm.get("owner_node_name"),
                    "lease_expires_utc": arm.get("lease_expires_utc"),
                    "merge_state": merge_state,
                }
            )
    return rows


def summarize_worker_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        group = str(row.get(key) or "unknown")
        entry = out.setdefault(group, {"count": 0, "best_score": None, "promoted": 0})
        entry["count"] += 1
        score = row.get("score")
        if isinstance(score, (int, float)) and (entry["best_score"] is None or float(score) > float(entry["best_score"])):
            entry["best_score"] = score
        if get_path(row, ["merge_result", "state"], "") == "promoted":
            entry["promoted"] += 1
    return out


def overnight_next_actions(
    worker_reports: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    stale: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    blocking_stale = [row for row in stale if row.get("merge_state") not in {"promoted", "best_by_arm"}]
    if failed:
        actions.append("Open the failed rows, fix the smallest reachable-node or worker error, then rerun one short training round.")
    if blocking_stale:
        actions.append("Run `theseus train run --sync-artifacts --execute` to recover expired leases onto currently reachable nodes.")
    if not worker_reports:
        actions.append("Run `theseus utilize sweep --execute` to prove at least one bounded worker chunk completes.")
    if worker_reports and not promotions:
        actions.append("Run artifact sync/merge and inspect best-by-arm thresholds before overnight looping.")
    if not actions:
        actions.append("Run `theseus utilize loop --execute --cycles 3 --sleep-seconds 60 --max-new-jobs 2` before a longer overnight run.")
    return actions


def compact_overnight(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "created_utc": report.get("created_utc"),
        "window_hours": report.get("window_hours"),
        "worker_report_count": summary.get("worker_report_count"),
        "promotion_count": summary.get("promotion_count"),
        "failed_count": summary.get("failed_count"),
        "stale_lease_count": summary.get("stale_lease_count"),
        "next_actions": report.get("next_actions") if isinstance(report.get("next_actions"), list) else [],
    }


def render_overnight_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Hive Overnight Training Report",
        "",
        f"- Window: {report.get('window_hours')} hours since `{report.get('window_start_utc')}`",
        f"- Worker reports: {summary.get('worker_report_count', 0)}",
        f"- Promotions: {summary.get('promotion_count', 0)}",
        f"- Failures: {summary.get('failed_count', 0)}",
        f"- Stale leases: {summary.get('stale_lease_count', 0)}",
        "",
        "## Best By Arm",
        "",
    ]
    by_arm = report.get("by_arm") if isinstance(report.get("by_arm"), dict) else {}
    if by_arm:
        for arm_id, row in by_arm.items():
            lines.append(f"- `{arm_id}`: {row.get('count')} report(s), best score `{row.get('best_score')}`, promoted `{row.get('promoted')}`")
    else:
        lines.append("- No worker reports in this window.")
    lines.extend(["", "## Failures", ""])
    failures = report.get("failed") if isinstance(report.get("failed"), list) else []
    if failures:
        for row in failures[:20]:
            lines.append(f"- `{row.get('job_id') or row.get('task_kind')}`: {row.get('error') or row.get('status') or 'failed'}")
    else:
        lines.append("- No training failures found in this window.")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def status_report(*, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or read_json(DEFAULT_POLICY, {})
    cfg = training_config(policy)
    state = read_json(ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json"), {})
    last = read_json(ROOT / str(cfg.get("report_path") or "reports/hive_training_orchestrator.json"), {})
    merge = read_json(ROOT / str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")), {})
    recent = recent_training_jobs(policy)
    overnight = read_json(DEFAULT_OVERNIGHT_OUT, {})
    return {
        "ok": True,
        "policy": "project_theseus_hive_training_status_v0",
        "created_utc": now(),
        "enabled": bool(cfg.get("enabled", True)),
        "strategy": cfg.get("strategy") or "deterministic_arm_slot_owner_v0",
        "state": state,
        "last_round": compact_last_round(last),
        "fleet_readiness": last.get("fleet_readiness") if isinstance(last.get("fleet_readiness"), dict) else {},
        "network_doctor": last.get("network_doctor") if isinstance(last.get("network_doctor"), dict) else {},
        "active_arms": active_arms_from_state(state, merge),
        "recent_jobs": recent,
        "overnight": compact_overnight(overnight),
        "promoted": merge.get("promoted") if isinstance(merge.get("promoted"), list) else [],
        "best_by_arm": merge.get("best_by_arm") if isinstance(merge.get("best_by_arm"), dict) else {},
    }


def write_state(policy: dict[str, Any], report: dict[str, Any]) -> None:
    cfg = training_config(policy)
    merge_summary = read_json(ROOT / str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")), {})
    plan = report.get("plan") if isinstance(report.get("plan"), dict) else {}
    state = {
        "ok": True,
        "policy": "project_theseus_hive_training_state_v0",
        "updated_utc": now(),
        "active_run_id": report.get("run_id"),
        "last_round_id": report.get("round_id"),
        "profile": report.get("profile"),
        "strategy": get_path(report, ["plan", "strategy"], ""),
        "arms": [
            {
                "arm_id": job.get("arm_id"),
                "display_name": job.get("display_name"),
                "task_kind": job.get("task_kind"),
                "owner_node_id": job.get("node_id"),
                "owner_node_name": job.get("node_name"),
                "slot_type": job.get("slot_type"),
                "lease_id": get_path(job, ["lease", "lease_id"], ""),
                "lease_expires_utc": get_path(job, ["lease", "lease_expires_utc"], ""),
                "job_id": get_path(job, ["payload", "job_id"], ""),
            }
            for job in plan.get("jobs", [])
            if isinstance(job, dict)
        ],
        "blocked": plan.get("blocked") or [],
        "fleet_readiness": report.get("fleet_readiness") if isinstance(report.get("fleet_readiness"), dict) else {},
        "network_doctor": report.get("network_doctor") if isinstance(report.get("network_doctor"), dict) else {},
        "last_execution": report.get("execution") or [],
        "best_by_arm": merge_summary.get("best_by_arm") if isinstance(merge_summary.get("best_by_arm"), dict) else {},
        "promoted": merge_summary.get("promoted") if isinstance(merge_summary.get("promoted"), list) else [],
    }
    write_json(ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json"), state)


def operator_view(policy: dict[str, Any]) -> dict[str, Any]:
    registry = hive_node_registry.build_registry(policy)
    write_json(ROOT / "reports" / "hive_node_registry.json", registry)
    registry_nodes = registry.get("nodes") if isinstance(registry.get("nodes"), list) else []
    if registry_nodes:
        return {
            "ok": True,
            "policy": "project_theseus_hive_operator_status_from_node_registry_v1",
            "node_registry": {
                "report": "reports/hive_node_registry.json",
                "summary": registry.get("summary", {}),
                "network_doctor": registry.get("network_doctor", {}),
            },
            "hive": {
                "hive_id": registry.get("hive_id") or hive_id(policy),
                "local_node": next((node for node in registry_nodes if node.get("is_local")), {}),
                "peers": [node for node in registry_nodes if not node.get("is_local")],
            },
        }
    secret = hive_secret(policy)
    headers = {"X-Theseus-Hive-Secret": secret} if secret else {}
    live = fetch_json(f"http://127.0.0.1:{int(get_path(policy, ['node', 'http_port'], 8791))}/api/hive/operator/status", headers=headers)
    if live.get("ok"):
        return live
    status = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    peers = read_json(ROOT / str(get_path(policy, ["node", "peers_path"], "reports/hive_peers.json")), {})
    return {
        "ok": bool(status),
        "policy": "project_theseus_hive_operator_status_fallback_v0",
        "hive": {
            "hive_id": status.get("hive_id") or hive_id(policy),
            "local_node": peer_from_status(status),
            "peers": peers.get("peers") if isinstance(peers.get("peers"), list) else [],
        },
    }


def visible_nodes(view: dict[str, Any]) -> list[dict[str, Any]]:
    hive = view.get("hive") if isinstance(view.get("hive"), dict) else {}
    local = hive.get("local_node") if isinstance(hive.get("local_node"), dict) else {}
    peers = hive.get("peers") if isinstance(hive.get("peers"), list) else []
    nodes = []
    if local:
        nodes.append({**local, "is_local": True})
    for peer in peers:
        if isinstance(peer, dict) and peer.get("node_id"):
            nodes.append({**peer, "is_local": False})
    seen: set[str] = set()
    out = []
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if node_id and node_id not in seen:
            seen.add(node_id)
            out.append(node)
    return out


def reachable_nodes(policy: dict[str, Any], nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reachable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    secret = hive_secret(policy)
    headers = {"X-Theseus-Hive-Secret": secret} if secret else {}
    for index, node in enumerate(nodes):
        if node.get("is_local"):
            local_url = local_api_url(policy, node)
            check, attempts = fetch_json_with_attempts(local_url + "/api/hive/status", headers={}, timeouts=[1.5, 2.5, 5.0])
            merged = {**node, "api_url": local_url}
            if isinstance(check, dict) and check.get("ok") is not False:
                merged.update(
                    {
                        "node_id": check.get("node_id") or node.get("node_id"),
                        "node_name": check.get("node_name") or node.get("node_name"),
                        "capabilities": check.get("capabilities") or node.get("capabilities") or [],
                        "resources": check.get("resources") or node.get("resources") or {},
                        "slots": check.get("slots") or node.get("slots") or [],
                        "reachability": {"ok": True, "checked": True, "order": index, "local_loopback": True, "attempts": attempts},
                    }
                )
            else:
                merged["reachability"] = {"ok": True, "checked": False, "order": index, "local_loopback": True, "attempts": attempts, "status_error": check.get("error") if isinstance(check, dict) else ""}
            reachable.append(merged)
            continue
        api_url = str(node.get("api_url") or "").rstrip("/")
        if not api_url:
            skipped.append(skipped_node(node, "missing_api_url"))
            continue
        check, attempts = fetch_json_with_attempts(api_url + "/api/hive/status", headers=headers, timeouts=reachability_timeouts(policy, node))
        if check.get("ok") is False and check.get("error"):
            skipped.append(skipped_node(node, "unreachable_api", check.get("error"), attempts=attempts))
            continue
        if not check:
            skipped.append(skipped_node(node, "empty_status_response", attempts=attempts))
            continue
        merged = {**node}
        if isinstance(check, dict):
            merged.update(
                {
                    "node_id": check.get("node_id") or node.get("node_id"),
                    "node_name": check.get("node_name") or node.get("node_name"),
                    "api_url": check.get("api_url") or node.get("api_url"),
                    "capabilities": check.get("capabilities") or node.get("capabilities") or [],
                    "resources": check.get("resources") or node.get("resources") or {},
                    "slots": check.get("slots") or node.get("slots") or [],
                    "reachability": {"ok": True, "checked": True, "order": index, "attempts": attempts},
                }
            )
        reachable.append(merged)
    return reachable, skipped


def reachability_timeouts(policy: dict[str, Any], node: dict[str, Any]) -> list[float]:
    configured = get_path(policy, ["training_orchestration", "reachability_timeouts_seconds"], [])
    if isinstance(configured, list) and configured:
        values: list[float] = []
        for value in configured:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                values.append(parsed)
        if values:
            return values[:5]
    network = network_profile(node)
    if network.get("class") == "wan":
        return [3.0, 8.0, 15.0]
    return [2.5, 5.0, 10.0]


def fetch_json_with_attempts(url: str, *, headers: dict[str, str], timeouts: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for idx, timeout in enumerate(timeouts, start=1):
        started = time.perf_counter()
        last = fetch_json(url, headers=headers, timeout=float(timeout))
        attempts.append(
            {
                "attempt": idx,
                "timeout_seconds": float(timeout),
                "ok": bool(last and last.get("ok") is not False),
                "error": str(last.get("error") or "")[:160] if isinstance(last, dict) else "",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        if last and last.get("ok") is not False:
            return last, attempts
    return last, attempts


def skipped_node(node: dict[str, Any], reason: str, detail: Any = "", *, attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "reason": reason,
        "detail": str(detail)[:300] if detail else "",
        "attempts": attempts or [],
    }


def matching_slot(policy: dict[str, Any], node: dict[str, Any], task_kind: str) -> dict[str, Any] | None:
    wanted = slot_types_for_task(policy, task_kind)
    for slot in node.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        slot_type = str(slot.get("slot_type") or "")
        if slot_type not in wanted:
            continue
        task_kinds = set(str(item) for item in (slot.get("task_kinds") or []))
        if task_kinds and task_kind not in task_kinds and not policy_allows_task_for_slot(policy, slot_type, task_kind):
            continue
        if int(slot.get("capacity") or 0) > 0:
            return slot
    if task_kind.startswith("cuda_") and node_has_capability(node, "nvidia_cuda"):
        return {"slot_type": "cuda", "capacity": 1, "available": True}
    if task_kind.startswith("mlx_") and node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]):
        return {"slot_type": "mlx_apple", "capacity": 1, "available": True}
    return None


def policy_allows_task_for_slot(policy: dict[str, Any], slot_type: str, task_kind: str) -> bool:
    key = slot_policy_key(slot_type)
    kinds = get_path(policy, ["resource_slots", "task_kinds_by_slot", key], [])
    return isinstance(kinds, list) and task_kind in {str(item) for item in kinds}


def slot_policy_key(slot_type: str) -> str:
    if slot_type in {"mlx", "mlx_apple", "mlx_cuda"} or "mlx" in slot_type:
        return "mlx"
    if slot_type.startswith("cuda"):
        return "cuda"
    return "cpu"


def score_assignment(arm: dict[str, Any], node: dict[str, Any], slot: dict[str, Any], task_kind: str, network: dict[str, Any], used: int) -> float:
    score = 0.0
    preferred = set(str(item) for item in arm.get("preferred_backends", []) if item)
    caps = set(node_capability_ids(node)) | set(str(item) for item in get_path(node, ["resources", "mlx", "backend_ids"], []) or [])
    score += 5.0 if preferred & caps else 0.0
    score += 2.0 if slot.get("available") is not False else 0.5
    score -= used * 1.5
    if task_kind.startswith("cuda_") and node_has_capability(node, "nvidia_cuda"):
        score += 4.0
        free = [float(gpu.get("memory_free_mib") or 0) for gpu in get_path(node, ["resources", "nvidia", "gpus"], []) or [] if isinstance(gpu, dict)]
        score += min(1.0, (max(free) if free else 0.0) / 12000.0)
    if task_kind.startswith("mlx_") and node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]):
        score += 4.0
    if node.get("is_local"):
        score += 0.15
    score -= min(2.0, float(network.get("estimated_latency_ms") or 0.0) / 100.0)
    score -= training_power_penalty(node)
    return score


def training_power_penalty(node: dict[str, Any]) -> float:
    power = get_path(node, ["resources", "power"], {})
    if not isinstance(power, dict) or not power:
        return 0.0
    if power.get("on_ac_power") is not False:
        return 0.0
    pct = power.get("battery_percent")
    if isinstance(pct, (int, float)) and float(pct) < 40:
        return 4.0
    return 2.0


def default_task_params(task_kind: str, profile: str) -> dict[str, Any]:
    scale = 2 if profile == "inner_loop" else 1
    if profile == "candidate":
        scale = 3
    if task_kind == "cuda_eval_chunk":
        return {"cases_per_task": 8 * scale, "epochs": 1, "samples_per_launch": 512, "hv_dim": 1024}
    if task_kind == "cuda_training_chunk":
        return {"cases_per_task": 12 * scale, "epochs": 2 * scale, "samples_per_launch": 512, "hv_dim": 1536}
    if task_kind == "cuda_rollout_chunk":
        return {
            "cases_per_task": 6 * scale,
            "epochs": max(1, 1 * scale),
            "state_epochs": 1,
            "samples_per_launch": 512,
            "rollout_batch": 192 * scale,
            "hv_dim": 1536,
            "seq_len": 32,
        }
    if task_kind == "mlx_eval_chunk":
        return {"train_limit": 128 * scale, "eval_limit": 128 * scale, "feature_dim": 512, "steps": 1}
    if task_kind == "mlx_training_chunk":
        return {"train_limit": 512 * scale, "eval_limit": 256 * scale, "feature_dim": 1024, "steps": 24 * scale}
    if task_kind == "mlx_rollout_chunk":
        return {"cases_per_task": 64 * scale, "eval_cases": 64 * scale, "epochs": 6 * scale, "hv_dim": 1024, "obs_dim": 32, "seq_len": 32}
    return {}


def active_input_artifacts(arm_id: str) -> list[dict[str, Any]]:
    active = ROOT / "checkpoints" / "hive_promoted" / safe_id(arm_id) / "active_manifest.json"
    if not active.exists():
        return []
    manifest = read_json(active, {})
    if not isinstance(manifest, dict) or not manifest:
        return []
    return [
        {
            "type": "active_arm_manifest",
            "path": str(active.relative_to(ROOT)).replace("\\", "/"),
            "score": manifest.get("score"),
            "job_id": manifest.get("job_id"),
        }
    ]


def recent_training_jobs(policy: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    ledger_rel = str(get_path(policy, ["node", "job_ledger_path"], "reports/hive_job_ledger.jsonl"))
    rows: list[dict[str, Any]] = []
    seen_ledgers: set[Path] = set()
    for root in report_roots():
        path = (root / Path(ledger_rel).name) if ledger_rel.startswith("reports/") else ROOT / ledger_rel
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen_ledgers:
            continue
        seen_ledgers.add(resolved)
        rows.extend(read_jsonl_tail(path, limit=400))
    out = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        orch = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
        job = row.get("job") if isinstance(row.get("job"), dict) else {}
        if not orch and job.get("task_kind") not in TRAINING_TASKS:
            continue
        out.append(
            {
                "created_utc": row.get("created_utc"),
                "finished_utc": row.get("finished_utc"),
                "status": row.get("status"),
                "task_kind": row.get("kind") or job.get("task_kind"),
                "job_id": job.get("job_id") or payload.get("job_id"),
                "arm_id": orch.get("arm_id") or job.get("arm_id"),
                "run_id": orch.get("run_id"),
                "round_id": orch.get("round_id"),
                "node": get_path(row, ["slot", "slot_type"], ""),
                "returncode": row.get("returncode"),
            }
        )
    return out[-limit:]


def local_api_url(policy: dict[str, Any], node: dict[str, Any] | None = None) -> str:
    raw = str((node or {}).get("api_url") or "")
    parsed = urlparse(raw)
    port = parsed.port or int(get_path(policy, ["node", "http_port"], 8791) or 8791)
    return f"http://127.0.0.1:{port}"


def active_arms_from_state(state: dict[str, Any], merge: dict[str, Any]) -> list[dict[str, Any]]:
    best = merge.get("best_by_arm") if isinstance(merge.get("best_by_arm"), dict) else {}
    rows = []
    for arm in state.get("arms", []) if isinstance(state.get("arms"), list) else []:
        if not isinstance(arm, dict):
            continue
        arm_id = str(arm.get("arm_id") or "")
        rows.append({**arm, "best_candidate": best.get(arm_id, {})})
    return rows


def compact_last_round(last: dict[str, Any]) -> dict[str, Any]:
    plan = last.get("plan") if isinstance(last.get("plan"), dict) else {}
    return {
        "run_id": last.get("run_id"),
        "round_id": last.get("round_id"),
        "mode": last.get("mode"),
        "profile": last.get("profile"),
        "job_count": plan.get("job_count"),
        "fleet_readiness": last.get("fleet_readiness") if isinstance(last.get("fleet_readiness"), dict) else {},
        "network_doctor": last.get("network_doctor") if isinstance(last.get("network_doctor"), dict) else {},
        "blocked": plan.get("blocked") or [],
        "created_utc": last.get("created_utc"),
    }


def compact_ledger_row(report: dict[str, Any]) -> dict[str, Any]:
    plan = report.get("plan") if isinstance(report.get("plan"), dict) else {}
    return {
        "created_utc": report.get("created_utc"),
        "run_id": report.get("run_id"),
        "round_id": report.get("round_id"),
        "mode": report.get("mode"),
        "profile": report.get("profile"),
        "job_count": plan.get("job_count"),
        "execution_count": len(report.get("execution") or []),
    }


def node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "is_local": bool(node.get("is_local")),
        "network": network_profile(node),
        "accelerators": sorted(set(node_capability_ids(node)) & {"nvidia_cuda", "mlx_apple", "apple_mlx", "mlx_cuda"}),
        "slots": [
            {
                "slot_id": slot.get("slot_id"),
                "slot_type": slot.get("slot_type"),
                "capacity": slot.get("capacity"),
                "available": slot.get("available"),
                "task_kinds": slot.get("task_kinds") or [],
            }
            for slot in node.get("slots") or []
            if isinstance(slot, dict)
        ],
    }


def training_config(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("training_orchestration") if isinstance(policy.get("training_orchestration"), dict) else {}
    if not cfg:
        cfg = {}
    cfg.setdefault("enabled", True)
    cfg.setdefault("strategy", "deterministic_arm_slot_owner_v0")
    cfg.setdefault("state_path", "reports/hive_training_state.json")
    cfg.setdefault("report_path", "reports/hive_training_orchestrator.json")
    cfg.setdefault("ledger_path", "reports/hive_training_orchestrator_ledger.jsonl")
    cfg.setdefault("default_profile", "smoke")
    cfg.setdefault("round_minutes", 30)
    cfg.setdefault("lease_seconds", 1800)
    cfg.setdefault("default_max_jobs_per_round", 3)
    cfg.setdefault("arms", default_arms())
    return cfg


def default_arms() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": "apple_mlx_worker_arm",
            "display_name": "Apple MLX language/readout arm",
            "enabled": True,
            "priority": 20,
            "task_kinds": ["mlx_training_chunk", "mlx_eval_chunk"],
            "preferred_backends": ["mlx_apple", "apple_mlx"],
            "backend_requirements": ["mlx_apple_or_mlx_cuda"],
            "job_family": "mlx_readout_chunk",
        },
        {
            "arm_id": "rust_cuda_systems_arm",
            "display_name": "Rust/CUDA systems readout arm",
            "enabled": True,
            "priority": 30,
            "task_kinds": ["cuda_training_chunk", "cuda_eval_chunk"],
            "preferred_backends": ["nvidia_cuda", "rust_cuda"],
            "backend_requirements": ["nvidia_cuda", "rust_cuda"],
            "job_family": "cuda_readout_chunk",
        },
        {
            "arm_id": "apple_mlx_control_arm",
            "display_name": "Apple MLX rollout/control arm",
            "enabled": True,
            "priority": 70,
            "task_kinds": ["mlx_rollout_chunk"],
            "preferred_backends": ["mlx_apple", "apple_mlx"],
            "backend_requirements": ["mlx_apple_or_mlx_cuda"],
            "job_family": "mlx_rollout_chunk",
        },
        {
            "arm_id": "rl_control_arm",
            "display_name": "CUDA rollout/control arm",
            "enabled": True,
            "priority": 80,
            "task_kinds": ["cuda_rollout_chunk"],
            "preferred_backends": ["nvidia_cuda", "rust_cuda"],
            "backend_requirements": ["nvidia_cuda", "rust_cuda"],
            "job_family": "cuda_rollout_chunk",
        },
    ]


def default_round_id(cfg: dict[str, Any]) -> str:
    minutes = max(1, int(cfg.get("round_minutes") or 30))
    bucket = int(time.time() // (minutes * 60)) * minutes * 60
    return datetime.fromtimestamp(bucket, tz=timezone.utc).strftime("round-%Y%m%dT%H%MZ")


def active_run_id(cfg: dict[str, Any]) -> str:
    state = read_json(ROOT / str(cfg.get("state_path") or "reports/hive_training_state.json"), {})
    existing = str(state.get("active_run_id") or "")
    if existing:
        return existing
    return "run-" + datetime.now(timezone.utc).strftime("%Y%m%d")


def slot_types_for_task(policy: dict[str, Any], task_kind: str) -> list[str]:
    by_kind = get_path(policy, ["resource_slots", "slot_types_by_task_kind", task_kind], [])
    if isinstance(by_kind, list) and by_kind:
        return [str(item) for item in by_kind]
    if task_kind.startswith("cuda_"):
        return ["cuda"]
    if task_kind.startswith("mlx_"):
        return ["mlx_apple", "mlx_cuda"]
    return ["cpu"]


def backend_requirements_for_task(task_kind: str) -> list[str]:
    if task_kind.startswith("cuda_"):
        return ["nvidia_cuda", "rust_cuda"]
    if task_kind.startswith("mlx_"):
        return ["mlx_apple_or_mlx_cuda"]
    return ["cpu_worker"]


def default_arm_for_task(task_kind: str) -> str:
    if task_kind.startswith("cuda_rollout"):
        return "rl_control_arm"
    if task_kind.startswith("mlx_rollout"):
        return "apple_mlx_control_arm"
    if task_kind.startswith("cuda_"):
        return "rust_cuda_systems_arm"
    if task_kind.startswith("mlx_"):
        return "apple_mlx_worker_arm"
    return "training_arm"


def network_profile(node: dict[str, Any]) -> dict[str, Any]:
    parsed = urlparse(str(node.get("api_url") or ""))
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return {"class": "local", "estimated_latency_ms": 0, "fit": "interactive_and_training"}
    if host.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.30.", "172.31.")) or host.endswith(".local"):
        return {"class": "lan_or_private_tunnel", "estimated_latency_ms": 8, "fit": "interactive_and_training"}
    if host:
        return {"class": "wan", "estimated_latency_ms": 80, "fit": "bounded_async_chunks"}
    return {"class": "unknown", "estimated_latency_ms": 120, "fit": "bounded_async_chunks"}


def node_capability_ids(node: dict[str, Any]) -> list[str]:
    return [str(cap.get("id") or "") for cap in node.get("capabilities") or [] if isinstance(cap, dict)]


def node_has_capability(node: dict[str, Any], capability: str) -> bool:
    return capability in set(node_capability_ids(node))


def node_has_any_capability(node: dict[str, Any], capabilities: list[str]) -> bool:
    return bool(set(node_capability_ids(node)) & set(capabilities))


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "hostname": status.get("hostname"),
        "api_url": status.get("api_url"),
        "hive_id": status.get("hive_id"),
        "platform": status.get("platform"),
        "capabilities": status.get("capabilities") or [],
        "resources": status.get("resources") or {},
        "slots": status.get("slots") or [],
    }


def fetch_json(url: str, *, headers: dict[str, str], timeout: float = 15.0) -> dict[str, Any]:
    req = urlrequest.Request(url, headers=headers, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/private Hive endpoint.
            raw = response.read().decode("utf-8")
    except (OSError, TimeoutError, URLError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:300], "url": url}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response", "url": url}


def is_loopback_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def hive_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    value = os.environ.get(env_name, "")
    if value:
        return value
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    if isinstance(join, dict) and join.get("join_token"):
        return str(join.get("join_token") or "")
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            token = str(profile.get("join_token") or "")
            if token:
                return token
    return ""


def hive_id(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    return os.environ.get(env_name, "") or str(join.get("hive_id") or get_path(policy, ["federation", "default_hive_id"], ""))


def utc_after_seconds(seconds: int) -> str:
    return datetime.fromtimestamp(time.time() + max(1, seconds), tz=timezone.utc).isoformat()


def safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip().lower())
    if len(cleaned) > 96:
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
        cleaned = cleaned[:80] + "_" + digest
    return cleaned.strip("._-") or f"id_{uuid.uuid4().hex[:8]}"


def parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_in_window(value: str, since_epoch: float) -> bool:
    parsed = parse_utc(value)
    if not parsed:
        return True
    return parsed.timestamp() >= since_epoch


def first_string(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def report_roots() -> list[Path]:
    roots = [REPORTS]
    env_reports = os.environ.get("THESEUS_REPORTS_DIR", "")
    if env_reports:
        roots.append(Path(env_reports).expanduser())
    try:
        runtime = theseus_runtime.runtime_report(create=True)
        runtime_reports = get_path(runtime, ["paths", "reports_dir", "path"], "")
        if runtime_reports:
            roots.append(Path(str(runtime_reports)).expanduser())
    except Exception:
        pass
    roots.extend(
        [
            Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "runtime" / "reports",
            Path.home() / "Library" / "Application Support" / "ProjectTheseus" / "runtime" / "reports",
        ]
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_json(raw: str, default: Any) -> Any:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, type(default)) else default


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
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


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
