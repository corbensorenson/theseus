"""Cross-machine Hive readiness probe.

Use this on the coordinator and on each Windows/macOS worker before expecting
the Hive to run unattended. It reports the practical blockers: runtime paths,
Python, optional Node, launch scripts, shared secret/relay posture, ports, and
worker-task policy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
import hive_node_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/hive_policy.json")
    parser.add_argument("--out", default="reports/hive_fleet_readiness.json")
    parser.add_argument("--markdown-out", default="reports/hive_fleet_readiness.md")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    registry = hive_node_registry.build_registry(policy)
    write_json(REPORTS / "hive_node_registry.json", registry)
    system = platform.system().lower()
    node_version = command_version(["node", "--version"])
    python_version = command_version([sys.executable, "--version"])
    secret_present = bool(hive_node_registry.hive_secret(policy))
    relay_url = os.environ.get(str(get_path(policy, ["federation", "relay_url_env"], "THESEUS_HIVE_RELAY_URL")), "")
    hive_id = hive_node_registry.hive_id(policy)
    port = int(get_path(policy, ["node", "http_port"], 8791) or 8791)
    launch = launch_command(system, port=port)
    blockers = blockers_for(system, policy, secret_present=secret_present, relay_url=relay_url, registry=registry)

    gates = [
        gate("policy_loaded", policy.get("policy") == "project_theseus_hive_policy_v0", policy.get("policy")),
        gate("python_available", bool(python_version), python_version),
        gate("launch_script_available", launch["script_exists"], launch),
        gate("runtime_root_available", bool(runtime_root()), runtime_root()),
        gate("remote_secret_or_loopback_only", secret_present or not remote_tasks_enabled(policy), {"secret_present": secret_present, "remote_tasks": remote_tasks_enabled(policy)}),
        gate("node_optional_or_present", bool(node_version) or not node_required_for_current_hive(policy), node_version or "node optional for Hive core; needed for some JS benchmark adapters"),
    ]
    report = {
        "policy": "project_theseus_hive_fleet_readiness_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) and not blockers else "YELLOW",
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": system,
            "machine": platform.machine(),
            "python": python_version,
            "node": node_version,
            "runtime_root": runtime_root(),
            "cwd": str(ROOT),
        },
        "hive": {
            "port": port,
            "port_open_now": port_listening(port),
            "hive_id_present": bool(hive_id),
            "relay_url_present": bool(relay_url),
            "shared_secret_present": secret_present,
            "remote_tasks_enabled_by_policy": remote_tasks_enabled(policy),
            "worker_task_kinds": sorted((policy.get("task_kinds") or {}).keys())[:64],
            "launch": launch,
            "node_registry": {
                "report": "reports/hive_node_registry.json",
                "summary": registry.get("summary", {}),
                "trusted_nodes": registry.get("summary", {}).get("trusted_node_count"),
            },
        },
        "blockers": blockers,
        "next_actions": next_actions(blockers, launch),
        "gates": gates,
        "score_semantics": "deployment readiness only; not capability evidence",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def blockers_for(system: str, policy: dict[str, Any], *, secret_present: bool, relay_url: str, registry: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    trusted_nodes = int(summary.get("trusted_node_count") or 0)
    remote_trusted = any(isinstance(node, dict) and not node.get("is_local") and get_path(node, ["trust", "trusted"], False) for node in registry.get("nodes", []))
    if policy.get("policy") != "project_theseus_hive_policy_v0":
        blockers.append("hive_policy_missing_or_invalid")
    if not runtime_root():
        blockers.append("runtime_root_unavailable")
    if remote_tasks_enabled(policy) and not secret_present:
        blockers.append("remote_task_secret_missing")
    if remote_tasks_enabled(policy) and not relay_url and not remote_trusted:
        blockers.append("same_lan_or_private_tunnel_required_without_relay")
    if remote_tasks_enabled(policy) and trusted_nodes <= 1:
        blockers.append("no_trusted_remote_worker_visible")
    if not launch_command(system, port=8791)["script_exists"]:
        blockers.append(f"launcher_missing_for_{system}")
    if node_required_for_current_hive(policy) and not shutil.which("node"):
        blockers.append("node_missing_for_js_benchmark_adapters")
    return blockers


def next_actions(blockers: list[str], launch: dict[str, Any]) -> list[str]:
    actions = []
    if "remote_task_secret_missing" in blockers:
        actions.append("Apply a machine Hive invite or set THESEUS_HIVE_SECRET before remote task submission.")
    if "same_lan_or_private_tunnel_required_without_relay" in blockers:
        actions.append("Use the same LAN, hotspot, self-hosted WireGuard/private tunnel, or configure THESEUS_HIVE_RELAY_URL.")
    if any(item.startswith("launcher_missing") for item in blockers):
        actions.append("Use scripts/start_theseus_hive.ps1 on Windows or scripts/start_theseus_hive.sh on macOS/Linux from this repo.")
    if "node_missing_for_js_benchmark_adapters" in blockers:
        actions.append("Install or upgrade Node.js only for JS-heavy benchmark adapters; Hive core is Python.")
    if not actions:
        actions.append(f"Start this node with: {launch.get('command')}")
    return actions


def launch_command(system: str, *, port: int) -> dict[str, Any]:
    if system == "windows":
        script = ROOT / "scripts" / "start_theseus_hive.ps1"
        command = f"powershell -ExecutionPolicy Bypass -File scripts\\start_theseus_hive.ps1 -Restart -HivePort {port}"
    else:
        script = ROOT / "scripts" / "start_theseus_hive.sh"
        command = f"bash scripts/start_theseus_hive.sh --restart --hive-port {port}"
    return {"script": rel(script), "script_exists": script.exists(), "command": command}


def runtime_root() -> str:
    value = os.environ.get("THESEUS_RUNTIME_ROOT")
    if value:
        return value
    if platform.system().lower() == "windows" and Path("D:/").exists():
        return "D:/ProjectTheseus/runtime"
    return str(Path.home() / ".project-theseus" / "runtime")


def remote_tasks_enabled(policy: dict[str, Any]) -> bool:
    return bool(get_path(policy, ["federation", "tiers", "private", "remote_tasks"], False))


def node_required_for_current_hive(policy: dict[str, Any]) -> bool:
    kinds = policy.get("task_kinds") if isinstance(policy.get("task_kinds"), dict) else {}
    return any("js" in key or "node" in key or "opencode" in key for key in kinds)


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def command_version(command: list[str]) -> str:
    if not command or not shutil.which(command[0]) and Path(command[0]).name == command[0]:
        return ""
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else ""


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hive Fleet Readiness",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- host: `{report.get('host', {}).get('hostname')}`",
        f"- system: `{report.get('host', {}).get('system')}`",
        f"- shared_secret_present: `{report.get('hive', {}).get('shared_secret_present')}`",
        f"- relay_url_present: `{report.get('hive', {}).get('relay_url_present')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("- none")
    else:
        for item in blockers:
            lines.append(f"- `{item}`")
    lines.extend(["", "## Next Actions", ""])
    for item in report.get("next_actions", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
