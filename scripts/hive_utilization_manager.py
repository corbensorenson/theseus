"""Always-busy utilization manager for the Project Theseus Hive.

This is the queue-filling layer: it watches trusted Hive slots and keeps them
fed with registered, bounded work when policy/resource gates allow it. It does
not run arbitrary shell, train on public benchmark solutions, or launch rented
compute. Cloud expansion remains a reviewed rented-compute plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

import theseus_runtime
import hive_node_registry


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
POLICY_PATH = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = REPORTS / "hive_utilization_manager.json"
LEDGER = REPORTS / "hive_utilization_ledger.jsonl"
TRAINING_TASKS = {"cuda_eval_chunk", "cuda_training_chunk", "cuda_rollout_chunk", "mlx_eval_chunk", "mlx_training_chunk", "mlx_rollout_chunk"}
ACCELERATOR_SLOTS = {"cuda", "mlx", "mlx_apple", "mlx_cuda"}
PAUSE_FLAGS = [REPORTS / "hive_utilization_pause.flag", REPORTS / "sparkstream_pause.flag", REPORTS / "unattended_autonomy_pause.flag"]
STOP_FLAGS = [REPORTS / "hive_utilization_stop.flag", REPORTS / "sparkstream_stop.flag", REPORTS / "unattended_autonomy_stop.flag"]
DEFAULT_MAINTENANCE_TASKS = [
    "training_smoke",
    "storage_index",
    "resource_probe",
    "capability_refresh",
    "network_doctor",
    "voice_following_status",
    "compute_market_status",
    "storage_status",
    "update_status",
    "hive_version_status",
]
DEFAULT_PRIORITY_MODEL = [
    {
        "priority": 10,
        "lane": "user_requested_work",
        "rule": "Existing user, operator, storage, voice, and control tasks keep their slots and queue position.",
    },
    {
        "priority": 20,
        "lane": "accelerator_training",
        "rule": "Idle CUDA/MLX slots receive bounded decentralized training rounds before lower-value background work.",
    },
    {
        "priority": 30,
        "lane": "cpu_self_improvement",
        "rule": "Idle CPU slots rotate bounded smoke training, storage, capability, network, voice, and version refresh tasks.",
    },
    {
        "priority": 40,
        "lane": "grounded_checkpoint",
        "rule": "If no safe worker slot is available, keep a grounded checkpoint summary fresh for the operator.",
    },
    {
        "priority": 50,
        "lane": "blocked_or_paused",
        "rule": "Stop, pause, resource floors, licensing, and public-data guards beat the no-downtime goal.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep Hive compute fed with safe bounded work.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))

    sweep = sub.add_parser("sweep")
    add_common_args(sweep)

    loop = sub.add_parser("loop")
    add_common_args(loop)
    loop.add_argument("--cycles", type=int, default=0, help="0 means run until a stop flag is seen.")
    loop.add_argument("--sleep-seconds", type=int, default=60)

    args = parser.parse_args()
    policy = read_json(resolve(args.policy), {})
    if args.command in {None, "status"}:
        report = build_report(policy, execute=False, args=args)
        write_json(resolve(getattr(args, "out", str(DEFAULT_OUT.relative_to(ROOT)))), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "sweep":
        keep_awake = start_keep_awake_assertion(args)
        try:
            report = build_report(policy, execute=bool(args.execute), args=args)
            write_json(resolve(args.out), report)
            append_jsonl(LEDGER, compact_ledger(report))
            print(json.dumps(report, indent=2))
            return 0 if report.get("ok", True) else 2
        finally:
            stop_keep_awake_assertion(keep_awake)
    if args.command == "loop":
        keep_awake = start_keep_awake_assertion(args)
        try:
            cycles = 0
            last: dict[str, Any] = {}
            while not stop_requested() and (args.cycles <= 0 or cycles < args.cycles):
                cycles += 1
                last = build_report(policy, execute=bool(args.execute), args=args)
                write_json(resolve(args.out), last)
                append_jsonl(LEDGER, compact_ledger(last))
                if args.cycles > 0 and cycles >= args.cycles:
                    break
                sleep_or_stop(max(1, int(args.sleep_seconds)))
            print(json.dumps(last or {"ok": True, "stopped": stop_requested()}, indent=2))
            return 0 if not last or last.get("ok", True) else 2
        finally:
            stop_keep_awake_assertion(keep_awake)
    parser.print_help()
    return 2


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", default="")
    parser.add_argument("--max-new-jobs", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Flight-safe mode: force local-only work and avoid internet/peer maintenance checks.")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--allow-wan", action="store_true")
    parser.add_argument("--allow-battery", action="store_true", help="Allow bounded local work while on battery; still respects --min-battery-percent.")
    parser.add_argument("--min-battery-percent", type=float, default=None)
    parser.add_argument("--keep-awake", action="store_true", help="On macOS, hold a caffeinate assertion while this utilization command runs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))


def build_report(policy: dict[str, Any], *, execute: bool, args: argparse.Namespace) -> dict[str, Any]:
    normalize_runtime_args(args)
    cfg = runtime_utilization_config(utilization_config(policy), args)
    registry = hive_node_registry.build_registry(policy)
    write_json(REPORTS / "hive_node_registry.json", registry)
    registry_nodes = registry.get("nodes") if isinstance(registry.get("nodes"), list) else []
    view = {"ok": True, "hive": {"local_node": next((n for n in registry_nodes if n.get("is_local")), {}), "peers": [n for n in registry_nodes if not n.get("is_local")]}}
    nodes = enrich_nodes(policy, visible_nodes(view), local_only=bool(getattr(args, "local_only", False)))
    node_rows = [summarize_node(policy, cfg, node) for node in nodes]
    idle = idle_capacity(node_rows)
    blockers = global_blockers(cfg)
    planned = plan_actions(policy, cfg, node_rows, args=args, blocked=bool(blockers))
    node_rows = annotate_intended_states(node_rows, planned, blockers)
    coverage = coverage_summary(node_rows)
    execution = execute_actions(planned, policy) if execute and not blockers else []
    report = {
        "ok": not any(row.get("status") == "failed" for row in execution),
        "policy": "project_theseus_hive_utilization_manager_v2",
        "created_utc": now(),
        "mode": "execute" if execute else "plan",
        "trigger_state": trigger_state(blockers, coverage),
        "summary": {
            "node_count": len(nodes),
            "idle_slots": idle,
            "busy_slots": busy_capacity(node_rows),
            "planned_actions": len(planned),
            "executed_actions": len(execution),
            "blocked": bool(blockers),
            "training_profile": cfg.get("training_profile"),
            "target_queue_depth_per_accelerator_slot": cfg.get("target_queue_depth_per_accelerator_slot"),
            "accelerator_lease_seconds": cfg.get("accelerator_lease_seconds"),
            "offline_mode": bool(getattr(args, "offline", False)),
            "local_only": bool(getattr(args, "local_only", False)),
            "allow_battery": bool(cfg.get("allow_on_battery", False)),
            "min_battery_percent": cfg.get("min_battery_percent"),
            "keep_awake": bool(getattr(args, "keep_awake", False)),
            "active_or_planned_nodes": coverage.get("active_or_planned_nodes"),
            "safe_idle_uncovered_nodes": coverage.get("safe_idle_uncovered_nodes"),
            "blocked_nodes": coverage.get("blocked_nodes"),
            "contract": "keep every safe private Hive slot fed with user work first, then bounded training, then maintenance, then checkpointing",
        },
        "always_active": always_active_contract(cfg, args),
        "blockers": blockers,
        "nodes": node_rows,
        "node_registry": {
            "report": "reports/hive_node_registry.json",
            "summary": registry.get("summary", {}),
            "created_utc": registry.get("created_utc"),
        },
        "actions": planned,
        "execution": execution,
        "safety": {
            "arbitrary_shell": False,
            "public_benchmark_training": "forbidden",
            "teacher_use": "not_used_by_utilization_manager",
            "rented_compute": "plan_only_elsewhere; never launched here",
            "task_surface": "registered Hive task kinds only",
            "offline_mode": offline_contract(args, cfg),
            "power": keep_awake_contract(args),
        },
        "external_inference_calls": 0,
    }
    return report


def normalize_runtime_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, "offline", False)):
        setattr(args, "local_only", True)
        setattr(args, "allow_wan", False)


def runtime_utilization_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(cfg)
    if bool(getattr(args, "allow_battery", False)):
        out["allow_on_battery"] = True
    battery_floor = getattr(args, "min_battery_percent", None)
    if battery_floor is not None:
        out["min_battery_percent"] = float(battery_floor)
    if bool(getattr(args, "offline", False)):
        out["adaptive_accelerator_profile"] = False
        out["maintenance_task_order"] = [
            task
            for task in out.get("maintenance_task_order", [])
            if task not in {"network_doctor", "update_status", "hive_version_status", "hive_version_converge"}
        ]
    return out


def offline_contract(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(getattr(args, "offline", False))
    return {
        "enabled": enabled,
        "local_only": bool(getattr(args, "local_only", False)),
        "allow_wan": bool(getattr(args, "allow_wan", False)),
        "teacher_use": "forbidden",
        "external_inference": "forbidden",
        "remote_peer_queueing": "disabled" if enabled else "policy_controlled",
        "internet_dependent_maintenance": "disabled" if enabled else "policy_controlled",
        "allow_battery": bool(cfg.get("allow_on_battery", False)),
        "min_battery_percent": cfg.get("min_battery_percent"),
        "stop_flag": "reports/hive_utilization_stop.flag",
    }


def keep_awake_contract(args: argparse.Namespace) -> dict[str, Any]:
    requested = bool(getattr(args, "keep_awake", False))
    caffeinate = shutil.which("caffeinate") if sys.platform == "darwin" else None
    return {
        "keep_awake_requested": requested,
        "supported": bool(caffeinate),
        "mechanism": "macos_caffeinate_idle_system_disk_sleep_assertion" if caffeinate else "",
        "active_when": "while utilization command process is running" if requested and caffeinate else "",
        "closed_lid_limit": "cannot train while truly asleep; MacBook closed-lid work requires normal macOS clamshell conditions such as AC power and external display/input",
    }


def plan_actions(policy: dict[str, Any], cfg: dict[str, Any], nodes: list[dict[str, Any]], *, args: argparse.Namespace, blocked: bool) -> list[dict[str, Any]]:
    if blocked:
        return []
    max_new = int(getattr(args, "max_new_jobs", 0) or cfg.get("max_new_jobs_per_sweep") or 2)
    actions: list[dict[str, Any]] = []
    accel_slots = [
        (node, slot)
        for node in nodes
        for slot in node.get("idle_slots", [])
        if slot.get("slot_type") in ACCELERATOR_SLOTS and not node.get("training_blockers")
    ]
    active_accel = [slot for node in nodes for slot in node.get("busy_slots", []) if slot.get("slot_type") in ACCELERATOR_SLOTS]
    queued_training = sum(int(get_path(node, ["tasks", "queued_in_memory"], 0) or 0) for node in nodes if node.get("has_training_slot"))
    target_depth = max(1, int(cfg.get("target_queue_depth_per_accelerator_slot") or 1))
    desired = max(0, min(max_new, len(accel_slots) * target_depth - queued_training))
    profile_selection = choose_training_profile(policy, cfg, args)
    if bool(cfg.get("training_enabled", True)) and desired > 0:
        round_id = "util-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]
        profile = str(profile_selection.get("selected_profile") or cfg.get("training_profile") or "smoke")
        command = [
            preferred_python_executable(),
            "scripts/hive_training_orchestrator.py",
            "run",
            "--execute",
            "--profile",
            profile,
            "--run-id",
            "utilization-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
            "--round-id",
            round_id,
            "--max-jobs",
            str(desired),
            "--out",
            "reports/hive_training_orchestrator.json",
        ]
        if not bool(getattr(args, "offline", False)):
            command.append("--sync-artifacts")
        if getattr(args, "allow_wan", False):
            command.append("--allow-wan")
        if getattr(args, "local_only", False):
            command.append("--local-only")
        planned_training_node_ids = sorted(
            {
                str(node.get("node_id") or "")
                for node, _slot in accel_slots[:desired]
                if node.get("node_id")
            }
        )
        actions.append(
            action(
                "training_round",
                "Fill idle accelerator slots with a fresh bounded training round.",
                command=command,
                planned_jobs=desired,
                priority_lane="accelerator_training",
                evidence={
                    "idle_accelerator_slots": len(accel_slots),
                    "busy_accelerator_slots": len(active_accel),
                    "queued_training": queued_training,
                    "planned_node_ids": planned_training_node_ids,
                    "target_queue_depth_per_accelerator_slot": target_depth,
                    "accelerator_lease_seconds": cfg.get("accelerator_lease_seconds"),
                    "profile": profile,
                    "profile_selection": profile_selection,
                },
            )
        )
    remaining = max(0, max_new - sum(int(row.get("planned_jobs") or 1) for row in actions))
    training_round_planned = any(row.get("kind") == "training_round" for row in actions)
    suppress_cpu_maintenance = bool(cfg.get("suppress_cpu_maintenance_when_accelerator_underfed", True)) and (
        training_round_planned or bool(accel_slots and not active_accel)
    )
    if remaining > 0 and bool(cfg.get("maintenance_enabled", True)) and not suppress_cpu_maintenance:
        for node in maintenance_node_order(nodes, avoid_training_accelerators=training_round_planned):
            if remaining <= 0:
                break
            if node.get("resource_blockers"):
                continue
            if node_queue_pressure(node, cfg):
                continue
            cpu_slots = [slot for slot in node.get("idle_slots", []) if slot.get("slot_type") == "cpu"]
            if not cpu_slots:
                continue
            kind = choose_maintenance_task(cfg)
            payload = {
                "job_id": f"util_{kind}_{safe_id(str(node.get('node_id') or 'node'))}_{int(time.time())}",
                "source": "hive_utilization_manager",
                "force_requeue": True,
            }
            actions.append(
                action(
                    "maintenance_task",
                    f"Keep {node.get('node_name') or node.get('node_id')} useful while CPU is idle.",
                    submit={"api_url": node.get("api_url"), "kind": kind, "payload": payload},
                    planned_jobs=1,
                    priority_lane="cpu_self_improvement",
                    evidence={"node_id": node.get("node_id"), "slot_type": "cpu"},
                )
            )
            remaining -= 1
    if not actions and bool(cfg.get("inference_keepalive_enabled", True)) and not accel_slots:
        node = first_available_node(nodes)
        if node and not node_queue_pressure(node, cfg):
            payload = {
                "checkpoint_id": "utilization",
                "prompt": "Summarize current Hive learning state, current blockers, and the next safe local improvement task without teacher escalation.",
                "job_id": f"util_checkpoint_{int(time.time())}",
                "force_requeue": True,
            }
            actions.append(
                action(
                    "inference_keepalive",
                    "Use idle inference capacity for a grounded checkpoint summary.",
                    submit={"api_url": node.get("api_url"), "kind": "checkpoint_chat", "payload": payload},
                    planned_jobs=1,
                    priority_lane="grounded_checkpoint",
                    evidence={"node_id": node.get("node_id")},
                )
            )
    return actions[:max_new]


def maintenance_node_order(nodes: list[dict[str, Any]], *, avoid_training_accelerators: bool = False) -> list[dict[str, Any]]:
    """Prefer light-eligible remote/blocked-training nodes for keepalive work.

    If a Mac or CPU worker cannot safely run heavy training because of a disk or
    resource floor, it should still receive cheap status/index/chat work before
    the coordinator burns the only maintenance slot locally.
    """
    return sorted(
        nodes,
        key=lambda node: (
            1 if avoid_training_accelerators and node_has_unblocked_idle_accelerator(node) else 0,
            1 if node.get("is_local") else 0,
            0 if node.get("training_blockers") else 1,
            str(node.get("node_name") or node.get("node_id") or ""),
        ),
    )


def execute_actions(actions: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in actions:
        started = time.perf_counter()
        if isinstance(row.get("command"), list):
            result = run_command(row["command"], timeout=600)
        elif isinstance(row.get("submit"), dict):
            submit = row["submit"]
            result = submit_task(policy, str(submit.get("api_url") or ""), str(submit.get("kind") or ""), submit.get("payload") if isinstance(submit.get("payload"), dict) else {})
        else:
            result = {"ok": False, "error": "missing_execution_surface"}
        rows.append({**row, "status": "completed" if result.get("ok", False) else "failed", "runtime_ms": int((time.perf_counter() - started) * 1000), "result": result})
    return rows


def summarize_node(policy: dict[str, Any], cfg: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    tasks = node.get("tasks") if isinstance(node.get("tasks"), dict) else {}
    slots = tasks.get("slots") if isinstance(tasks.get("slots"), list) and tasks.get("slots") else node.get("slots") if isinstance(node.get("slots"), list) else []
    normalized_slots = [normalize_slot(slot) for slot in slots if isinstance(slot, dict)]
    resource_blockers = sorted(set((node.get("resource_blockers") or []) + resource_blockers_for_node(cfg, node)))
    training_blockers = sorted(set((node.get("training_blockers") or []) + training_blockers_for_node(cfg, node)))
    schedulable = "api_unreachable" not in resource_blockers
    idle_slots = [slot for slot in normalized_slots if slot.get("available")] if schedulable else []
    busy_slots = [slot for slot in normalized_slots if not slot.get("available")] if schedulable else []
    accelerator_slots = [slot for slot in normalized_slots if slot.get("slot_type") in ACCELERATOR_SLOTS]
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "is_local": bool(node.get("is_local")),
        "reachable": bool(node.get("reachable")),
        "reachability_error": node.get("reachability_error") or "",
        "capabilities": [cap.get("id") for cap in node.get("capabilities") or [] if isinstance(cap, dict)],
        "power": get_path(node, ["resources", "power"], {}),
        "has_training_slot": bool(accelerator_slots and not training_blockers),
        "slots": normalized_slots,
        "idle_slots": idle_slots,
        "busy_slots": busy_slots,
        "resource_blockers": resource_blockers,
        "training_blockers": training_blockers,
        "tasks": {"queued_in_memory": tasks.get("queued_in_memory", 0), "recent_results": len(tasks.get("recent_results") or [])},
    }


def resource_blockers_for_node(cfg: dict[str, Any], node: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if node.get("reachable") is False:
        blockers.append("api_unreachable")
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    disk_free = get_path(resources, ["disk", "free_gib"], None)
    if disk_free is not None and number(disk_free) < float(cfg.get("min_disk_free_gib") or 5):
        blockers.append("disk_below_utilization_floor")
    mem_load = get_path(resources, ["memory", "load_percent"], None)
    if mem_load is not None and number(mem_load) > float(cfg.get("max_memory_load_percent") or 92):
        blockers.append("memory_load_above_floor")
    power = resources.get("power") if isinstance(resources.get("power"), dict) else {}
    if power:
        if power.get("on_ac_power") is False and not bool(cfg.get("allow_on_battery", False)):
            blockers.append("battery_power")
        pct = power.get("battery_percent")
        if isinstance(pct, (int, float)) and float(pct) < float(cfg.get("min_battery_percent") or 40):
            blockers.append("battery_below_utilization_floor")
    return blockers


def training_blockers_for_node(cfg: dict[str, Any], node: dict[str, Any]) -> list[str]:
    blockers = resource_blockers_for_node({**cfg, "min_disk_free_gib": cfg.get("min_training_disk_free_gib", 8)}, node)
    for gpu in get_path(node, ["resources", "nvidia", "gpus"], []) or []:
        if not isinstance(gpu, dict):
            continue
        total = number(gpu.get("memory_total_mib"))
        used = number(gpu.get("memory_used_mib"))
        mem_pct = used / total * 100.0 if total else 0.0
        if number(gpu.get("utilization_gpu_percent")) > float(cfg.get("max_gpu_utilization_percent_to_enqueue") or 85):
            blockers.append("gpu_busy")
        if mem_pct > float(cfg.get("max_gpu_memory_used_percent_to_enqueue") or 88):
            blockers.append("gpu_vram_busy")
    return sorted(set(blockers))


def normalize_slot(slot: dict[str, Any]) -> dict[str, Any]:
    capacity = max(1, int(slot.get("capacity") or 1))
    running = int(slot.get("running") or 0)
    available = bool(slot.get("available", running < capacity))
    return {
        "slot_id": slot.get("slot_id"),
        "slot_type": slot.get("slot_type") or ("mlx" if str(slot.get("slot_type") or "").startswith("mlx") else slot.get("slot_type")),
        "capacity": capacity,
        "running": running,
        "available": available,
        "task_kinds": slot.get("task_kinds") or [],
    }


def idle_capacity(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        for slot in node.get("idle_slots", []):
            key = str(slot.get("slot_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts


def busy_capacity(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        for slot in node.get("busy_slots", []):
            key = str(slot.get("slot_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts


def choose_maintenance_task(cfg: dict[str, Any]) -> str:
    tasks = cfg.get("maintenance_task_order") if isinstance(cfg.get("maintenance_task_order"), list) else []
    allowed = [str(task) for task in tasks if task]
    if not allowed:
        allowed = DEFAULT_MAINTENANCE_TASKS
    return allowed[int(time.time() // 60) % len(allowed)]


def first_available_node(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for node in nodes:
        if node.get("idle_slots") and not node.get("resource_blockers"):
            return node
    return None


def node_has_unblocked_idle_accelerator(node: dict[str, Any]) -> bool:
    if node.get("training_blockers"):
        return False
    return any(slot.get("slot_type") in ACCELERATOR_SLOTS for slot in node.get("idle_slots", []))


def node_queue_pressure(node: dict[str, Any], cfg: dict[str, Any]) -> bool:
    max_depth = int(cfg.get("max_background_queue_depth_per_node") or 2)
    return int(get_path(node, ["tasks", "queued_in_memory"], 0) or 0) >= max_depth


def annotate_intended_states(nodes: list[dict[str, Any]], actions: list[dict[str, Any]], blockers: list[str]) -> list[dict[str, Any]]:
    global_state = "paused_or_stopped" if blockers else ""
    training_planned = any(row.get("kind") == "training_round" for row in actions)
    planned_by_node = {
        str(get_path(row, ["evidence", "node_id"], ""))
        for row in actions
        if get_path(row, ["evidence", "node_id"], "")
    }
    out = []
    for node in nodes:
        row = dict(node)
        node_id = str(row.get("node_id") or "")
        idle_slots = row.get("idle_slots") if isinstance(row.get("idle_slots"), list) else []
        idle_accel = [slot for slot in idle_slots if slot.get("slot_type") in ACCELERATOR_SLOTS]
        if global_state:
            intended = global_state
        elif "api_unreachable" in set(row.get("resource_blockers") or []):
            intended = "blocked_api_unreachable"
        elif row.get("resource_blockers"):
            intended = "blocked_resource_floor"
        elif row.get("busy_slots"):
            intended = "active_running_work"
        elif node_id in planned_by_node:
            intended = "background_task_planned"
        elif idle_accel and training_planned and not row.get("training_blockers"):
            intended = "training_round_planned"
        elif idle_slots:
            intended = "safe_idle_waiting_for_next_sweep"
        else:
            intended = "no_advertised_capacity"
        row["intended_state"] = intended
        row["queue_pressure"] = int(get_path(row, ["tasks", "queued_in_memory"], 0) or 0)
        out.append(row)
    return out


def coverage_summary(nodes: list[dict[str, Any]]) -> dict[str, int]:
    active_states = {"active_running_work", "background_task_planned", "training_round_planned"}
    safe_idle = {"safe_idle_waiting_for_next_sweep"}
    return {
        "active_or_planned_nodes": sum(1 for row in nodes if row.get("intended_state") in active_states),
        "safe_idle_uncovered_nodes": sum(1 for row in nodes if row.get("intended_state") in safe_idle),
        "blocked_nodes": sum(1 for row in nodes if str(row.get("intended_state") or "").startswith(("blocked", "paused"))),
    }


def trigger_state(blockers: list[str], coverage: dict[str, int]) -> str:
    if "stop_flag_present" in blockers:
        return "RED"
    if blockers:
        return "YELLOW"
    if int(coverage.get("blocked_nodes") or 0) > 0:
        return "YELLOW"
    if int(coverage.get("safe_idle_uncovered_nodes") or 0) > 0:
        return "YELLOW"
    return "GREEN"


def always_active_contract(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sleep_seconds = int(getattr(args, "sleep_seconds", 0) or cfg.get("sweep_interval_seconds") or 60)
    max_new_jobs = int(getattr(args, "max_new_jobs", 0) or cfg.get("max_new_jobs_per_sweep") or 2)
    loop_args = f"--execute --sleep-seconds {sleep_seconds} --max-new-jobs {max_new_jobs}"
    if bool(getattr(args, "offline", False)):
        loop_args += " --offline"
    elif bool(getattr(args, "local_only", False)):
        loop_args += " --local-only"
    if bool(getattr(args, "allow_battery", False)):
        loop_args += " --allow-battery"
    if getattr(args, "min_battery_percent", None) is not None:
        loop_args += f" --min-battery-percent {float(getattr(args, 'min_battery_percent')):g}"
    if bool(getattr(args, "keep_awake", False)):
        loop_args += " --keep-awake"
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "mode": cfg.get("always_active_mode") or "user_work_first_bounded_self_improvement",
        "sweep_interval_seconds": sleep_seconds,
        "max_new_jobs_per_sweep": max_new_jobs,
        "priority_model": cfg.get("priority_model")
        if isinstance(cfg.get("priority_model"), list)
        else DEFAULT_PRIORITY_MODEL,
        "overnight_command": f"{Path(sys.executable).name} scripts/hive_utilization_manager.py loop {loop_args}",
        "operator_command": f"theseus utilize loop {loop_args}",
        "stop_flags": [str(path.relative_to(ROOT)) for path in STOP_FLAGS],
        "pause_flags": [str(path.relative_to(ROOT)) for path in PAUSE_FLAGS],
    }


def global_blockers(cfg: dict[str, Any]) -> list[str]:
    blockers = []
    if stop_requested():
        blockers.append("stop_flag_present")
    if paused():
        blockers.append("pause_flag_present")
    if not bool(cfg.get("enabled", True)):
        blockers.append("utilization_manager_disabled")
    return blockers


def action(
    kind: str,
    title: str,
    *,
    command: list[str] | None = None,
    submit: dict[str, Any] | None = None,
    planned_jobs: int,
    priority_lane: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "title": title,
        "command": command,
        "submit": submit,
        "planned_jobs": planned_jobs,
        "priority_lane": priority_lane,
        "evidence": evidence,
        "public_data_rule": "public_benchmarks_calibration_only_not_training",
        "side_effect_tier": "registered_hive_task_queue",
    }
    payload["action_id"] = "hive_util_" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return payload


def operator_view(policy: dict[str, Any]) -> dict[str, Any]:
    headers = secret_headers(policy)
    live = fetch_json(f"http://127.0.0.1:{int(get_path(policy, ['node', 'http_port'], 8791))}/api/hive/operator/status", headers=headers, timeout=4)
    if live.get("ok"):
        return live
    status = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    peers = read_json(ROOT / str(get_path(policy, ["node", "peers_path"], "reports/hive_peers.json")), {})
    return {"ok": bool(status), "hive": {"local_node": peer_from_status(status), "peers": peers.get("peers") if isinstance(peers.get("peers"), list) else []}}


def visible_nodes(view: dict[str, Any]) -> list[dict[str, Any]]:
    hive = view.get("hive") if isinstance(view.get("hive"), dict) else {}
    rows = []
    local = hive.get("local_node") if isinstance(hive.get("local_node"), dict) else {}
    if local:
        rows.append({**local, "is_local": True})
    for peer in hive.get("peers") or []:
        if isinstance(peer, dict) and peer.get("node_id"):
            rows.append({**peer, "is_local": False})
    seen: set[str] = set()
    out = []
    for row in rows:
        node_id = str(row.get("node_id") or "")
        if node_id and node_id not in seen:
            seen.add(node_id)
            out.append(row)
    return out


def enrich_nodes(policy: dict[str, Any], nodes: list[dict[str, Any]], *, local_only: bool) -> list[dict[str, Any]]:
    out = []
    for node in nodes:
        if local_only and not node.get("is_local"):
            continue
        api_url = str(node.get("api_url") or "")
        if node.get("is_local"):
            api_url = f"http://127.0.0.1:{int(get_path(policy, ['node', 'http_port'], 8791))}"
        tasks: dict[str, Any] = {}
        reachable = False
        reachability_error = "missing_api_url"
        if api_url:
            tasks, reachable, reachability_error = fetch_json_checked(
                api_url.rstrip("/") + "/api/hive/tasks",
                headers=secret_headers(policy),
                timeout=3,
            )
        out.append(
            {
                **node,
                "api_url": api_url or node.get("api_url"),
                "reachable": reachable,
                "reachability_error": "" if reachable else reachability_error,
                "tasks": tasks if isinstance(tasks, dict) else {},
            }
        )
    return out


def submit_task(policy: dict[str, Any], api_url: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not api_url:
        return {"ok": False, "error": "missing_api_url"}
    body = json.dumps({"kind": kind, "payload": payload}).encode("utf-8")
    req = urlrequest.Request(api_url.rstrip("/") + "/api/hive/tasks", data=body, headers={"Content-Type": "application/json", **secret_headers(policy)}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=10) as response:  # noqa: S310 - private Hive endpoint.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        return {"ok": False, "error": str(exc), "kind": kind, "api_url": api_url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:500]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response"}


def run_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=theseus_runtime.runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "command": command}
    payload = parse_json(result.stdout.strip(), {})
    return {"ok": result.returncode == 0, "returncode": result.returncode, "payload": payload, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}


def utilization_config(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("hive_utilization") if isinstance(policy.get("hive_utilization"), dict) else {}
    return {
        "enabled": cfg.get("enabled", True),
        "always_active_mode": cfg.get("always_active_mode", "user_work_first_bounded_self_improvement"),
        "training_enabled": cfg.get("training_enabled", True),
        "maintenance_enabled": cfg.get("maintenance_enabled", True),
        "inference_keepalive_enabled": cfg.get("inference_keepalive_enabled", True),
        "training_profile": cfg.get("training_profile", "smoke"),
        "adaptive_accelerator_profile": bool(cfg.get("adaptive_accelerator_profile", True)),
        "min_accelerator_profile": cfg.get("min_accelerator_profile", "inner_loop"),
        "max_accelerator_profile": cfg.get("max_accelerator_profile", "candidate"),
        "suppress_cpu_maintenance_when_accelerator_underfed": bool(
            cfg.get("suppress_cpu_maintenance_when_accelerator_underfed", True)
        ),
        "max_new_jobs_per_sweep": int(cfg.get("max_new_jobs_per_sweep") or 2),
        "sweep_interval_seconds": int(cfg.get("sweep_interval_seconds") or 60),
        "target_queue_depth_per_accelerator_slot": int(cfg.get("target_queue_depth_per_accelerator_slot") or 1),
        "accelerator_lease_seconds": int(cfg.get("accelerator_lease_seconds") or 900),
        "max_background_queue_depth_per_node": int(cfg.get("max_background_queue_depth_per_node") or 2),
        "min_disk_free_gib": float(cfg.get("min_disk_free_gib") or 5),
        "min_training_disk_free_gib": float(cfg.get("min_training_disk_free_gib") or 8),
        "max_memory_load_percent": float(cfg.get("max_memory_load_percent") or 92),
        "allow_on_battery": bool(cfg.get("allow_on_battery", False)),
        "min_battery_percent": float(cfg.get("min_battery_percent") or 40),
        "max_gpu_utilization_percent_to_enqueue": float(cfg.get("max_gpu_utilization_percent_to_enqueue") or 85),
        "max_gpu_memory_used_percent_to_enqueue": float(cfg.get("max_gpu_memory_used_percent_to_enqueue") or 88),
        "maintenance_task_order": cfg.get("maintenance_task_order") if isinstance(cfg.get("maintenance_task_order"), list) else [],
        "priority_model": cfg.get("priority_model") if isinstance(cfg.get("priority_model"), list) else DEFAULT_PRIORITY_MODEL,
    }


def choose_training_profile(policy: dict[str, Any], cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    requested = str(getattr(args, "profile", "") or "").strip()
    configured = str(cfg.get("training_profile") or "smoke")
    floor = str(cfg.get("min_accelerator_profile") or configured or "smoke")
    ceiling = str(cfg.get("max_accelerator_profile") or "candidate")
    governor = run_resource_governor(requested or configured)
    recommended = str(get_path(governor, ["decision", "recommended_profile"], "") or "")
    order = {"smoke": 0, "inner_loop": 1, "candidate": 2}
    allowed = {name for name in order if order[name] <= order.get(ceiling, order["candidate"])}
    selected = requested or configured
    candidates = [selected]
    if bool(cfg.get("adaptive_accelerator_profile", True)):
        candidates.extend([floor, recommended])
    ranked = [
        name
        for name in candidates
        if name in allowed and name in order
    ]
    if ranked:
        selected = sorted(ranked, key=lambda name: order[name], reverse=True)[0]
    return {
        "requested_profile": requested,
        "configured_profile": configured,
        "resource_governor_recommended_profile": recommended,
        "min_accelerator_profile": floor,
        "max_accelerator_profile": ceiling,
        "selected_profile": selected,
        "adaptive": bool(cfg.get("adaptive_accelerator_profile", True)),
        "resource_governor_can_run_requested": get_path(governor, ["decision", "can_run_requested_profile"], None),
        "resource_governor_throttle_reasons": get_path(governor, ["decision", "throttle_reasons"], []),
    }


def run_resource_governor(profile: str) -> dict[str, Any]:
    command = [
        preferred_python_executable(),
        "scripts/resource_governor.py",
        "--profile",
        profile or "inner_loop",
        "--out",
        "reports/resource_governor.json",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60, env=theseus_runtime.runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if result.returncode != 0:
        return {"ok": False, "returncode": result.returncode, "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:]}
    payload = parse_json(result.stdout.strip(), {})
    return payload if isinstance(payload, dict) and payload else read_json(REPORTS / "resource_governor.json", {})


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "api_url": status.get("api_url"),
        "capabilities": status.get("capabilities") or [],
        "resources": status.get("resources") or {},
        "slots": status.get("slots") or [],
    }


def secret_headers(policy: dict[str, Any]) -> dict[str, str]:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    secret = os.environ.get(env_name, "")
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    token = secret or str(join.get("join_token") or "")
    return {"X-Theseus-Hive-Secret": token} if token else {}


def fetch_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 3.0) -> dict[str, Any]:
    value, ok, _error = fetch_json_checked(url, headers=headers, timeout=timeout)
    return value if ok else {}


def fetch_json_checked(url: str, *, headers: dict[str, str] | None = None, timeout: float = 3.0) -> tuple[dict[str, Any], bool, str]:
    try:
        req = urlrequest.Request(url, headers=headers or {})
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/private Hive endpoints only.
            value = json.loads(response.read().decode("utf-8"))
            if isinstance(value, dict):
                return value, True, ""
            return {}, False, "non_object_json_response"
    except json.JSONDecodeError as exc:
        return {}, False, f"json_decode_error:{exc}"
    except (OSError, URLError) as exc:
        return {}, False, str(exc)


def compact_ledger(report: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": report.get("created_utc"), "mode": report.get("mode"), "trigger_state": report.get("trigger_state"), **(report.get("summary") if isinstance(report.get("summary"), dict) else {})}


def stop_requested() -> bool:
    return any(path.exists() for path in STOP_FLAGS)


def paused() -> bool:
    return any(path.exists() for path in PAUSE_FLAGS)


def sleep_or_stop(seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline and not stop_requested():
        time.sleep(min(1.0, deadline - time.time()))


def start_keep_awake_assertion(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if not bool(getattr(args, "keep_awake", False)):
        return None
    if sys.platform != "darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        return None
    command = [caffeinate, "-i", "-m", "-s", "-w", str(os.getpid())]
    try:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def stop_keep_awake_assertion(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def safe_id(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug or "id"


def preferred_python_executable() -> str:
    rows = theseus_runtime.python_runtime_checks()
    preferred = theseus_runtime.preferred_python_runtime(rows)
    python = str(preferred.get("python") or "")
    return python if python else sys.executable


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
