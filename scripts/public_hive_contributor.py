"""Opt-in public compute contribution manager for Theseus Hive.

This keeps public contribution separate from private Hive membership. A private
Hive can lend idle compute outward, but public work never receives private join
tokens, private peer lists, local data, ROM assets, teacher/self-edit access, or
arbitrary shell.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

import license_manager
import compute_market


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "hive_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    configure = sub.add_parser("configure")
    configure.add_argument("--mode", choices=["off", "idle", "always"], default="")
    configure.add_argument("--gateway-url", default="")
    configure.add_argument("--worker-name", default="")
    configure.add_argument("--allow", action="store_true")
    configure.add_argument("--out", default="")

    status = sub.add_parser("status")
    status.add_argument("--out", default="")

    poll = sub.add_parser("poll-once")
    poll.add_argument("--out", default="")

    work = sub.add_parser("work-smoke")
    work.add_argument("--out", default="reports/public_hive_work_smoke.json")

    blocked = sub.add_parser("blocked-task")
    blocked.add_argument("--kind", default="unknown")
    blocked.add_argument("--out", default="reports/public_hive_blocked_task.json")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "configure":
        report = configure_contribution(policy, args)
    elif args.command == "status" or args.command is None:
        report = status_report(policy)
    elif args.command == "poll-once":
        report = poll_once(policy)
    elif args.command == "work-smoke":
        report = work_smoke(policy)
    elif args.command == "blocked-task":
        report = blocked_task(policy, args.kind)
    else:
        parser.print_help()
        return 2
    out = getattr(args, "out", "") or status_path(policy)
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def configure_contribution(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(policy)
    if not cfg:
        cfg = {
            "policy": "theseus_hive_public_contribution_config_v0",
            "created_utc": now(),
            "worker_id": f"public-worker-{uuid.uuid4().hex}",
        }
    if args.mode:
        cfg["mode"] = args.mode
    else:
        cfg.setdefault("mode", get_path(policy, ["public_contribution", "default_mode"], "off"))
    if args.gateway_url:
        cfg["gateway_url"] = args.gateway_url
    else:
        cfg.setdefault("gateway_url", get_path(policy, ["public_contribution", "gateway_url"], ""))
    if args.worker_name:
        cfg["worker_name"] = args.worker_name
    else:
        cfg.setdefault("worker_name", default_worker_name())
    if args.allow:
        cfg["explicit_opt_in"] = True
    else:
        cfg.setdefault("explicit_opt_in", False)
    cfg["updated_utc"] = now()
    write_json(config_path(policy), cfg)
    report = status_report(policy)
    append_jsonl(ledger_path(policy), event("public_contribution_configure", {"mode": cfg.get("mode"), "gateway_configured": bool(cfg.get("gateway_url")), "explicit_opt_in": bool(cfg.get("explicit_opt_in"))}))
    return report


def status_report(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(policy)
    public = policy.get("public_contribution") if isinstance(policy.get("public_contribution"), dict) else {}
    mode = str(cfg.get("mode") or public.get("default_mode") or "off")
    resources = read_json(ROOT / "reports" / "hive_status.json", {}).get("resources") or {}
    resource_governor = read_json(ROOT / "reports" / "resource_governor.json", {})
    gates = evaluate_gates(policy, cfg, resources, resource_governor)
    license_check = license_manager.check_feature("public_contribution_worker", write_report=True)
    if mode != "off":
        gates.append(gate("license_allows_public_contribution_worker", bool(license_check.get("allowed")), license_check.get("next_action") or "license check"))
    can_connect = all(row["ok"] for row in gates)
    report = {
        "ok": True,
        "policy": "theseus_hive_public_contribution_status_v0",
        "created_utc": now(),
        "mode": mode,
        "enabled": mode != "off" and bool(cfg.get("explicit_opt_in")),
        "can_connect_now": can_connect,
        "gateway_url": str(cfg.get("gateway_url") or ""),
        "gateway_url_configured": bool(cfg.get("gateway_url")),
        "explicit_opt_in": bool(cfg.get("explicit_opt_in")),
        "public_hive_id": public.get("public_hive_id", "theseus-public"),
        "worker": public_worker(cfg),
        "privacy_boundary": {
            "private_hive_data_shared": False,
            "teacher_access_shared": False,
            "arbitrary_shell_allowed": False,
            "local_training_data_shared": False,
            "rom_assets_shared": False,
        },
        "allowed_task_kinds": public.get("allowed_public_task_kinds", []),
        "license": {
            "allowed": bool(license_check.get("allowed")),
            "tier": get_path(license_check, ["entitlement", "tier"], ""),
            "source": get_path(license_check, ["entitlement", "source"], ""),
            "next_action": license_check.get("next_action"),
        },
        "compute_market": compact_market(compute_market.status_report(write_report=True)),
        "gates": gates,
        "next_action": next_action(mode, cfg, gates),
    }
    write_json(ROOT / status_path(policy), report)
    return report


def poll_once(policy: dict[str, Any]) -> dict[str, Any]:
    report = status_report(policy)
    cfg = load_config(policy)
    if not report.get("can_connect_now"):
        append_jsonl(ledger_path(policy), event("public_contribution_skip", {"reason": report.get("next_action")}))
        return report
    gateway = str(cfg.get("gateway_url") or "").rstrip("/")
    probe: dict[str, Any] = {"attempted": False}
    if gateway:
        probe = gateway_probe(gateway)
    result = {
        **report,
        "connectivity_probe": probe,
        "task_polling": {
            "status": "not_enabled_until_public_gateway_protocol_and_signed_task_manifests_are_live",
            "signed_manifest_required": bool(get_path(policy, ["public_contribution", "require_signed_task_manifest"], True)),
            "sandbox_required": "sandboxed_execution_runtime" in get_path(policy, ["public_contribution", "promotion_requirements"], []),
        },
    }
    append_jsonl(ledger_path(policy), event("public_contribution_poll_once", {"gateway": bool(gateway), "probe_ok": bool(probe.get("ok"))}))
    write_json(ROOT / status_path(policy), result)
    return result


def work_smoke(policy: dict[str, Any]) -> dict[str, Any]:
    worker = public_worker(load_config(policy))
    receipt = {
        "version": "theseus_verified_work_receipt_v0",
        "accounting_only": True,
        "accepted": True,
        "task_kind": "public_training_smoke",
        "worker_kind": "public_training_smoke",
        "backend": "cpu",
        "profile": "public",
        "difficulty_class": "public",
        "claimed_work_units": 1_000_000,
        "verifier": "public_contribution_smoke",
        "anti_cheat_status": "local_report_only",
        "provider_account": compute_market.load_wallet(compute_market.read_json(compute_market.POLICY_PATH, {})).get("account_id"),
        "worker": worker,
        "created_utc": now(),
    }
    settlement = compute_market.settle_receipt(
        receipt,
        context={"source": "public_hive_contributor.work_smoke"},
        policy=compute_market.read_json(compute_market.POLICY_PATH, {}),
        write_report=True,
    )
    report = {
        "ok": True,
        "policy": "theseus_hive_public_work_smoke_v0",
        "created_utc": now(),
        "task_kind": "public_training_smoke",
        "result": "completed_report_only_smoke",
        "work_receipt": receipt,
        "compute_market": settlement,
        "private_data_access": False,
        "teacher_access": False,
        "arbitrary_shell": False,
        "notes": "This smoke proves the public contribution task lane can execute bounded report-only work. Real public training shards remain gated by signed manifests and sandboxing.",
    }
    append_jsonl(ledger_path(policy), event("public_work_smoke", {"ok": True}))
    return report


def blocked_task(policy: dict[str, Any], kind: str) -> dict[str, Any]:
    report = {
        "ok": True,
        "policy": "theseus_hive_public_task_block_v0",
        "created_utc": now(),
        "task_kind": kind,
        "status": "blocked",
        "reason": "public_task_requires_signed_manifest_sandbox_and_gateway_protocol",
        "private_data_access": False,
        "teacher_access": False,
        "arbitrary_shell": False,
    }
    append_jsonl(ledger_path(policy), event("public_task_blocked", {"task_kind": kind}))
    return report


def compact_market(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report.get("mode"),
        "currency_symbol": get_path(report, ["currency", "symbol"], "TWC"),
        "available_micro_twc": get_path(report, ["balances", "available_micro_twc"], 0),
        "earned_micro_twc": get_path(report, ["balances", "earned_micro_twc"], 0),
        "exchange_enabled": bool(report.get("exchange_enabled")),
        "next_action": report.get("next_action"),
    }


def evaluate_gates(policy: dict[str, Any], cfg: dict[str, Any], resources: dict[str, Any], resource_governor: dict[str, Any]) -> list[dict[str, Any]]:
    public = policy.get("public_contribution") if isinstance(policy.get("public_contribution"), dict) else {}
    idle = public.get("idle_policy") if isinstance(public.get("idle_policy"), dict) else {}
    mode = str(cfg.get("mode") or public.get("default_mode") or "off")
    gates: list[dict[str, Any]] = [
        gate("mode_not_off", mode != "off", mode),
        gate("explicit_opt_in", bool(cfg.get("explicit_opt_in")), "user opted in" if cfg.get("explicit_opt_in") else "missing explicit opt-in"),
        gate("gateway_url_configured", bool(cfg.get("gateway_url")), str(cfg.get("gateway_url") or "missing")),
        gate("worker_only", bool(public.get("worker_only", True)), "public mode cannot control private hive"),
        gate("signed_task_manifest_required", bool(public.get("require_signed_task_manifest", True)), "required"),
    ]
    idle_seconds = idle_seconds_platform()
    if mode == "idle":
        required = int(idle.get("idle_minutes_before_connect", 20)) * 60
        gates.append(gate("machine_idle", idle_seconds >= required, f"idle_seconds={idle_seconds} required={required}"))
    if idle.get("allow_on_ac_power_only", True):
        ac = ac_power_status()
        gates.append(gate("ac_power", ac is True, f"ac_power={ac}"))
    disk = resources.get("disk") if isinstance(resources.get("disk"), dict) else {}
    free_gib = float(disk.get("free_gib") or 0.0)
    gates.append(gate("disk_free", free_gib >= float(idle.get("min_disk_free_gib", 25)), f"{free_gib:.2f} GiB"))
    memory = resources.get("memory") if isinstance(resources.get("memory"), dict) else {}
    mem_load = float(memory.get("load_percent") or 0.0)
    gates.append(gate("memory_load", mem_load <= float(idle.get("max_memory_load_percent", 82)), f"{mem_load:.1f}%"))
    gpu_rows = get_path(resources, ["nvidia", "gpus"], [])
    gpu_ok = True
    gpu_detail = "no nvidia gpu"
    if gpu_rows:
        details = []
        for gpu in gpu_rows:
            if not isinstance(gpu, dict):
                continue
            util = float(gpu.get("utilization_gpu_percent") or 0.0)
            used = float(gpu.get("memory_used_mib") or 0.0)
            total = float(gpu.get("memory_total_mib") or 1.0)
            mem_pct = used / max(1.0, total) * 100.0
            details.append(f"{gpu.get('name','gpu')}: util={util:.1f}% mem={mem_pct:.1f}%")
            if util > float(idle.get("max_gpu_utilization_percent", 35)) or mem_pct > float(idle.get("max_gpu_memory_used_percent", 70)):
                gpu_ok = False
        gpu_detail = "; ".join(details) or gpu_detail
    gates.append(gate("gpu_available_or_not_busy", gpu_ok, gpu_detail))
    if resource_governor:
        can_run = get_path(resource_governor, ["decision", "can_run_requested_profile"], True)
        gates.append(gate("resource_governor_green", bool(can_run), str(can_run)))
    return gates


def gate(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def next_action(mode: str, cfg: dict[str, Any], gates: list[dict[str, Any]]) -> str:
    if mode == "off":
        return "Enable public contribution from the setup wizard or configure command."
    failed = [row["name"] for row in gates if not row["ok"]]
    if failed:
        return "Waiting on gates: " + ", ".join(failed)
    if not cfg.get("gateway_url"):
        return "Add a public gateway URL when the public Hive coordinator is available."
    return "Ready to contribute bounded public compute when signed public tasks are available."


def public_worker(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": cfg.get("worker_id", ""),
        "worker_name": cfg.get("worker_name") or default_worker_name(),
        "platform": platform.system(),
        "hostname": socket.gethostname(),
    }


def gateway_probe(gateway: str) -> dict[str, Any]:
    candidates = [
        gateway.rstrip("/") + "/api/public-hive/status",
        gateway.rstrip("/") + "/api/hive/relay/status",
    ]
    for url in candidates:
        try:
            with urlrequest.urlopen(url, timeout=5) as response:  # noqa: S310 - user configured public Hive gateway.
                raw = response.read().decode("utf-8")
            return {"attempted": True, "ok": True, "url": url, "status": json.loads(raw) if raw.startswith("{") else raw[:500]}
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = {"attempted": True, "ok": False, "url": url, "error": str(exc)}
    return last


def idle_seconds_platform() -> int:
    if platform.system() == "Windows":
        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
            tick = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
            return max(0, int((tick - info.dwTime) / 1000))
    return 0


def ac_power_status() -> bool | None:
    if platform.system() != "Windows":
        return None

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_byte),
            ("BatteryFlag", ctypes.c_byte),
            ("BatteryLifePercent", ctypes.c_byte),
            ("SystemStatusFlag", ctypes.c_byte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return status.ACLineStatus == 1
    return None


def config_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["public_contribution", "config_path"], "configs/public_hive.local.json"))


def status_path(policy: dict[str, Any]) -> str:
    return str(get_path(policy, ["public_contribution", "status_path"], "reports/public_hive_contribution_status.json"))


def ledger_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["public_contribution", "ledger_path"], "reports/public_hive_contribution_ledger.jsonl"))


def load_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = config_path(policy)
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def default_worker_name() -> str:
    return f"{socket.gethostname()}-{platform.system().lower()}"


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), "kind": kind, **payload}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
