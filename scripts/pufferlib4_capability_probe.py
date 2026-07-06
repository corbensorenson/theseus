"""Probe PufferLib 4 readiness for Theseus RL training.

This keeps the fast-RL frontier governed: Ocean/native environments are
preferred first, Atari/ALE stays disabled unless the user has explicitly
provided legal assets, and runtime blockers become useful residuals instead of
silent training churn.
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

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
VENDOR_PUFFER = ROOT / "vendor" / "pufferlib"
RUNTIME_PATHS = theseus_runtime.runtime_report(create=False)["paths"]
DEFAULT_RUNTIME_ROOT = Path(RUNTIME_PATHS["runtime_root"]["path"])
DEFAULT_DATA_DIR = Path(RUNTIME_PATHS["data_dir"]["path"])
DEFAULT_CACHE_DIR = Path(RUNTIME_PATHS["cache_dir"]["path"])
PUFFER_VENV_PY = ROOT / ".venv-puffer" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
D_RUNTIME = DEFAULT_RUNTIME_ROOT / "pufferlib4"
D_DATA = DEFAULT_DATA_DIR / "rl" / "pufferlib4"
D_TMP = DEFAULT_RUNTIME_ROOT / "tmp" / "pufferlib4"
DEFAULT_OUT = REPORTS / "pufferlib4_capability_probe.json"
DEFAULT_MARKDOWN = REPORTS / "pufferlib4_capability_probe.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--skip-help-probe", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    for path in [REPORTS, D_RUNTIME, D_DATA, D_TMP]:
        path.mkdir(parents=True, exist_ok=True)

    python_path = PUFFER_VENV_PY if PUFFER_VENV_PY.exists() else Path(sys.executable)
    module_probe = probe_python_modules(python_path)
    cli_probe = None if args.skip_help_probe else probe_puffer_cli_help(python_path)
    podman_probe = probe_podman()
    ocean_inventory = discover_ocean_envs()
    configs = discover_puffer_configs()
    rom_policy = atari_policy(module_probe)
    build_plan = puffer_build_plan(module_probe, podman_probe, ocean_inventory)
    trigger_state, blockers = classify_state(module_probe, ocean_inventory, podman_probe)

    report = {
        "policy": "project_theseus_pufferlib4_capability_probe_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "puffer_python": rel(python_path),
            "pufferlib_import_ok": bool(get_path(module_probe, ["modules", "pufferlib", "ok"])),
            "pufferlib_version": get_path(module_probe, ["modules", "pufferlib", "version"]),
            "native_backend_ok": bool(get_path(module_probe, ["modules", "pufferlib._C", "ok"])),
            "torch_version": get_path(module_probe, ["modules", "torch", "version"]),
            "torch_cuda_available": bool(get_path(module_probe, ["modules", "torch", "cuda_available"])),
            "ocean_env_count": len(ocean_inventory),
            "puffer_config_count": len(configs),
            "atari_enabled": rom_policy["atari_enabled"],
            "podman_connected": bool(podman_probe.get("connected")),
            "blocker_count": len(blockers),
        },
        "blockers": blockers,
        "python_modules": module_probe,
        "puffer_cli_help": cli_probe,
        "ocean_inventory": ocean_inventory,
        "puffer_configs": configs[:80],
        "storage": {
            "runtime_dir": str(D_RUNTIME),
            "training_data_dir": str(D_DATA),
            "tmp_dir": str(D_TMP),
            "all_on_d_drive": all(str(path).upper().startswith("D:") for path in [D_RUNTIME, D_DATA, D_TMP]),
        },
        "license_policy": {
            "pufferlib_license": puffer_license_signal(),
            "commercial_rom_fetching": "forbidden_without_explicit_user_rights",
            "atari_roms": "disabled_until_ale_plus_legal_user_supplied_or_permissive_assets_exist",
            "autonomous_downloads": "no_commercial_rom_or_uncertain_license_downloads",
        },
        "atari_policy": rom_policy,
        "podman": podman_probe,
        "build_plan": build_plan,
        "recommended_next_actions": recommended_actions(trigger_state, blockers, module_probe, podman_probe),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def probe_python_modules(python_path: Path) -> dict[str, Any]:
    code = r"""
import importlib, json, sys

def probe(name):
    row = {"ok": False}
    try:
        mod = importlib.import_module(name)
        row["ok"] = True
        row["path"] = getattr(mod, "__file__", "")
        row["version"] = getattr(mod, "__version__", "")
        if name == "torch":
            row["cuda_available"] = bool(mod.cuda.is_available())
            row["cuda_device_count"] = int(mod.cuda.device_count()) if mod.cuda.is_available() else 0
            row["cuda_devices"] = [
                mod.cuda.get_device_name(i) for i in range(mod.cuda.device_count())
            ] if mod.cuda.is_available() else []
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row

mods = {
    "pufferlib": probe("pufferlib"),
    "pufferlib._C": probe("pufferlib._C"),
    "torch": probe("torch"),
    "gymnasium": probe("gymnasium"),
    "ale_py": probe("ale_py"),
    "pettingzoo": probe("pettingzoo"),
    "numpy": probe("numpy"),
}
print(json.dumps({"python": sys.executable, "modules": mods}, sort_keys=True))
"""
    result = run([str(python_path), "-c", code], timeout=45, pythonpath=True)
    payload = parse_json_output(result.get("stdout", ""))
    if not payload:
        payload = {"python": str(python_path), "modules": {}, "probe_error": result}
    payload["command"] = result["command"]
    payload["returncode"] = result["returncode"]
    return payload


def probe_puffer_cli_help(python_path: Path) -> dict[str, Any]:
    result = run([str(python_path), "-m", "pufferlib.pufferl", "--help"], timeout=30, pythonpath=True)
    return {
        "ok": result["returncode"] == 0,
        "returncode": result["returncode"],
        "stdout_tail": tail(result.get("stdout", ""), 3000),
        "stderr_tail": tail(result.get("stderr", ""), 3000),
        "command": result["command"],
    }


def probe_podman() -> dict[str, Any]:
    version = run(["podman", "--version"], timeout=10, pythonpath=False) if shutil.which("podman") else None
    info = run(["podman", "info", "--format", "json"], timeout=20, pythonpath=False) if shutil.which("podman") else None
    machine = run(["podman", "machine", "list", "--format", "json"], timeout=20, pythonpath=False) if shutil.which("podman") else None
    wsl = run(["wsl", "--status"], timeout=20, pythonpath=False) if shutil.which("wsl") else None
    return {
        "installed": bool(version and version["returncode"] == 0),
        "version": tail((version or {}).get("stdout", ""), 500).strip(),
        "connected": bool(info and info["returncode"] == 0),
        "info_error_tail": tail((info or {}).get("stderr", ""), 1200),
        "machine_list": parse_json_output((machine or {}).get("stdout", "")) if machine else None,
        "wsl_status_ok": bool(wsl and wsl["returncode"] == 0),
        "wsl_status_tail": tail(((wsl or {}).get("stdout", "") + "\n" + (wsl or {}).get("stderr", "")), 1200).strip(),
    }


def discover_ocean_envs() -> list[dict[str, Any]]:
    roots = [
        ("vendored_pufferlib", VENDOR_PUFFER / "ocean"),
        ("public_benchmark_cache", ROOT / "data" / "public_benchmarks" / "pufferlib" / "ocean"),
    ]
    rows: list[dict[str, Any]] = []
    for source, root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            c_files = sorted(child.glob("*.c"))
            h_files = sorted(child.glob("*.h"))
            binding = child / "binding.c"
            rows.append(
                {
                    "name": child.name,
                    "source": source,
                    "path": rel(child),
                    "has_binding": binding.exists(),
                    "c_file_count": len(c_files),
                    "h_file_count": len(h_files),
                    "status": "available_source" if c_files and h_files else "incomplete_source",
                }
            )
    return rows


def discover_puffer_configs() -> list[dict[str, Any]]:
    config_root = VENDOR_PUFFER / "config"
    rows: list[dict[str, Any]] = []
    if not config_root.exists():
        return rows
    for path in sorted(config_root.rglob("*.ini")):
        env_name = ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("env_name"):
                env_name = line.split("=", 1)[-1].strip()
                break
        rows.append({"path": rel(path), "env_name": env_name})
    return rows


def atari_policy(module_probe: dict[str, Any]) -> dict[str, Any]:
    local_rom_registry = read_json(REPORTS / "local_rom_registry.json")
    rom_count = int(get_path(local_rom_registry, ["summary", "rom_count"], 0) or 0)
    ale_ok = bool(get_path(module_probe, ["modules", "ale_py", "ok"]))
    recommendations = local_rom_registry.get("recommendations") if isinstance(local_rom_registry.get("recommendations"), list) else []
    matched = [
        item for item in recommendations
        if isinstance(item, dict) and int(item.get("matched_rom_count") or 0) > 0
    ]
    explicit_enable = (ROOT / "configs" / "allow_user_supplied_atari.flag").exists()
    runtime_ready = bool(ale_ok and matched)
    return {
        "atari_enabled": bool(runtime_ready and explicit_enable),
        "atari_runtime_ready": runtime_ready,
        "explicit_user_enable_present": explicit_enable,
        "ale_py_ok": ale_ok,
        "local_user_rom_count": rom_count,
        "matched_licensed_profile_count": len(matched),
        "autonomous_rom_download": "forbidden",
        "disabled_reasons": [
            reason
            for reason, active in [
                ("ale_py_missing", not ale_ok),
                ("no_explicit_licensed_user_supplied_or_permissive_rom_profile", not matched),
                ("missing_configs_allow_user_supplied_atari_flag", not explicit_enable),
                ("commercial_rom_fetch_forbidden_without_explicit_rights", True),
            ]
            if active
        ],
    }


def puffer_license_signal() -> dict[str, Any]:
    license_path = VENDOR_PUFFER / "LICENSE"
    if not license_path.exists():
        return {"ok": False, "path": rel(license_path), "reason": "missing_license_file"}
    data = license_path.read_bytes()
    return {
        "ok": True,
        "path": rel(license_path),
        "sha256_16": hashlib.sha256(data).hexdigest()[:16],
        "kind": "MIT",
        "first_line": license_path.read_text(encoding="utf-8", errors="replace").splitlines()[0] if data else "",
    }


def puffer_build_plan(module_probe: dict[str, Any], podman_probe: dict[str, Any], ocean_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    native_ok = bool(get_path(module_probe, ["modules", "pufferlib._C", "ok"]))
    preferred = {"tmaze", "cartpole", "minimal", "memory", "chain_mdp"}
    first_env = next((row["name"] for row in ocean_inventory if row.get("has_binding") and row.get("name") in preferred), None)
    first_env = first_env or next((row["name"] for row in ocean_inventory if row.get("has_binding")), "tmaze")
    return {
        "native_backend_ready": native_ok,
        "preferred_env_for_first_build": first_env,
        "windows_direct_build": "admitted_for_cpu_cartpole_backend via scripts/build_pufferlib_windows_cpu_backend.py",
        "preferred_build_host": "native local build when supported; Windows MSVC CPU backend and Podman/Linux remain fallbacks",
        "podman_ready": bool(podman_probe.get("connected")),
        "safe_commands": [
            "python scripts/build_pufferlib_windows_cpu_backend.py --env cartpole --out reports/pufferlib4_windows_cpu_build.json",
            f"cd vendor/pufferlib && bash build.sh {first_env} --cpu",
            f"cd vendor/pufferlib && bash build.sh {first_env} --float",
        ],
        "post_build_probe": "python scripts/pufferlib4_capability_probe.py --out reports/pufferlib4_capability_probe.json --markdown-out reports/pufferlib4_capability_probe.md",
    }


def classify_state(module_probe: dict[str, Any], ocean_inventory: list[dict[str, Any]], podman_probe: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    puffer_ok = bool(get_path(module_probe, ["modules", "pufferlib", "ok"]))
    native_ok = bool(get_path(module_probe, ["modules", "pufferlib._C", "ok"]))
    ocean_ok = any(row.get("has_binding") for row in ocean_inventory)
    if not puffer_ok:
        blockers.append({"id": "pufferlib_import_failed", "severity": "red", "detail": get_path(module_probe, ["modules", "pufferlib", "error"])})
    if not native_ok:
        blockers.append({"id": "pufferlib_native_backend_missing", "severity": "yellow", "detail": get_path(module_probe, ["modules", "pufferlib._C", "error"])})
    if not ocean_ok:
        blockers.append({"id": "puffer_ocean_source_missing", "severity": "red", "detail": "No local Ocean env source with binding.c was found."})
    if not native_ok and not podman_probe.get("connected"):
        blockers.append({"id": "podman_linux_builder_not_connected", "severity": "yellow", "detail": podman_probe.get("info_error_tail") or "Podman machine/socket is not active."})
    if puffer_ok and native_ok and ocean_ok:
        return "GREEN", blockers
    if puffer_ok and ocean_ok:
        return "YELLOW", blockers
    return "RED", blockers


def recommended_actions(trigger_state: str, blockers: list[dict[str, Any]], module_probe: dict[str, Any], podman_probe: dict[str, Any]) -> list[str]:
    actions = []
    blocker_ids = {row.get("id") for row in blockers}
    if "pufferlib_native_backend_missing" in blocker_ids:
        actions.append("Use the local synthetic RL fallback now; build a Puffer/Ocean native backend later, then rerun this probe.")
    if "podman_linux_builder_not_connected" in blocker_ids:
        actions.append("Use Podman/Linux only for later upstream CUDA/Ocean variants; the Windows CPU backend is the first local unblock.")
    if not bool(get_path(module_probe, ["modules", "ale_py", "ok"])):
        actions.append("Keep Atari disabled for now; install ALE only after deciding the legal asset path. Do not fetch commercial ROMs automatically.")
    if trigger_state == "GREEN":
        actions.append("Admit pufferlib4_rl_lane for bounded Ocean rollouts and STS capsule generation.")
    else:
        actions.append("Use board_game_rl, long_horizon_tool_use, and local synthetic RL as active transfer pressure until Puffer native backend is built.")
    return actions


def run(cmd: list[str], *, timeout: int, pythonpath: bool) -> dict[str, Any]:
    env = os.environ.copy()
    env["TEMP"] = str(D_TMP)
    env["TMP"] = str(D_TMP)
    env["PIP_CACHE_DIR"] = str(DEFAULT_CACHE_DIR / "pip-cache")
    if pythonpath:
        current = env.get("PYTHONPATH", "")
        paths = [str(VENDOR_PUFFER)]
        if current:
            paths.append(current)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "timeout",
        }
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": 125,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def parse_json_output(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    blockers = report.get("blockers", [])
    actions = report.get("recommended_next_actions", [])
    lines = [
        "# PufferLib 4 Capability Probe",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- pufferlib: `{summary.get('pufferlib_version')}` import_ok=`{summary.get('pufferlib_import_ok')}`",
        f"- native_backend_ok: `{summary.get('native_backend_ok')}`",
        f"- torch: `{summary.get('torch_version')}` cuda=`{summary.get('torch_cuda_available')}`",
        f"- ocean_env_count: `{summary.get('ocean_env_count')}`",
        f"- atari_enabled: `{summary.get('atari_enabled')}`",
        f"- podman_connected: `{summary.get('podman_connected')}`",
        "",
        "## Blockers",
    ]
    if blockers:
        lines.extend(f"- `{row.get('id')}` ({row.get('severity')}): {row.get('detail')}" for row in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Actions"])
    lines.extend(f"- {action}" for action in actions)
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
        if cur is None:
            return default
    return cur


def tail(text: str, limit: int) -> str:
    text = (text or "").replace("\x00", "")
    return text[-limit:] if len(text) > limit else text


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
