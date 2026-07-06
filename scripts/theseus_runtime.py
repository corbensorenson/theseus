"""Runtime path and environment management for Project Theseus.

The repository root stays source-owned. Large generated runtime surfaces can be
redirected through local config or environment variables so a nearly-full source
drive does not block CUDA/MLX training work.
"""

from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "runtime_paths.json"
LOCAL_CONFIG_PATH = ROOT / "configs" / "runtime_paths.local.json"
REPORT_PATH = ROOT / "reports" / "runtime_paths.json"
DOCTOR_REPORT_PATH = ROOT / "reports" / "macos_runtime_doctor.json"


def runtime_report(*, create: bool = False, write_report: bool = False) -> dict[str, Any]:
    config = effective_config()
    paths = resolve_paths(config)
    if create:
        for row in paths.values():
            Path(row["path"]).mkdir(parents=True, exist_ok=True)
    report = {
        "ok": True,
        "policy": "project_theseus_runtime_paths_v0",
        "created_utc": now(),
        "root": str(ROOT),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "local_config_path": str(LOCAL_CONFIG_PATH.relative_to(ROOT)),
        "env_names": config.get("env_names", {}),
        "paths": with_disk_status(paths),
        "environment": runtime_env_overrides(config, paths),
        "migration": migration_status(config, paths),
    }
    if write_report:
        write_json(REPORT_PATH, report)
    return report


def runtime_doctor_report(*, write_report: bool = False) -> dict[str, Any]:
    """Report runtime consistency across source, installed app, and LaunchAgents."""
    runtime = runtime_report(create=False, write_report=True)
    python_checks = python_runtime_checks()
    preferred = preferred_python_runtime(python_checks)
    active = next((row for row in python_checks if row.get("name") == "active_python"), {})
    source = next((row for row in python_checks if row.get("name") == "source_venv"), {})
    installed = next((row for row in python_checks if row.get("name") == "installed_app_venv"), {})
    launchagents = macos_launchagent_report() if platform.system() == "Darwin" else {}
    installed_app = installed_app_report()
    runtime_roots = runtime_root_comparison(runtime, installed_app)
    runtime_state = runtime_state_comparison(runtime, installed_app)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    source_or_installed_mlx = any(
        bool(get_path(row, ["mlx", "available"], False))
        for row in [source, installed]
    )
    active_false_negative = bool(
        is_apple_silicon
        and not get_path(active, ["mlx", "available"], False)
        and source_or_installed_mlx
    )
    blockers: list[str] = []
    warnings: list[str] = []
    false_negatives: list[str] = []
    if is_apple_silicon and not source_or_installed_mlx:
        blockers.append("mlx_missing_from_source_and_installed_runtime")
    if active_false_negative:
        false_negatives.append("active_shell_python_lacks_mlx_but_hive_runtime_has_it")
    if platform.system() == "Darwin" and not get_path(launchagents, ["local.project-theseus.hive", "loaded"], False):
        warnings.append("hive_launchagent_not_loaded")
    if get_path(runtime_roots, ["source_vs_installed", "same"], None) is False:
        warnings.append("source_and_installed_runtime_roots_differ")
    warnings.extend(runtime_state.get("warnings", []))
    blockers.extend(runtime_state.get("blockers", []))
    state = "RED" if blockers else ("YELLOW" if warnings else "GREEN")
    report = {
        "ok": state != "RED",
        "policy": "project_theseus_macos_runtime_doctor_v0",
        "created_utc": now(),
        "state": state,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "is_apple_silicon": is_apple_silicon,
        },
        "preferred_runtime": preferred,
        "python_runtimes": python_checks,
        "runtime_paths": runtime,
        "runtime_roots": runtime_roots,
        "runtime_state": runtime_state,
        "installed_app": installed_app,
        "launchagents": launchagents,
        "blockers": blockers,
        "warnings": warnings,
        "false_negatives": false_negatives,
        "next_actions": runtime_doctor_next_actions(blockers, warnings, preferred),
    }
    if write_report:
        write_json(DOCTOR_REPORT_PATH, report)
    return report


def effective_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH, {})
    local = read_json(LOCAL_CONFIG_PATH, {})
    merged = deep_merge(config, local)
    env_names = merged.setdefault(
        "env_names",
        {
            "runtime_root": "THESEUS_RUNTIME_ROOT",
            "data_dir": "THESEUS_DATA_DIR",
            "cache_dir": "THESEUS_CACHE_DIR",
            "reports_dir": "THESEUS_REPORTS_DIR",
            "checkpoints_dir": "THESEUS_CHECKPOINTS_DIR",
            "cargo_target_dir": "CARGO_TARGET_DIR",
        },
    )
    runtime_root = os.environ.get(str(env_names.get("runtime_root") or "THESEUS_RUNTIME_ROOT"))
    if runtime_root:
        merged["runtime_root"] = runtime_root
    for key in ["data_dir", "cache_dir", "reports_dir", "checkpoints_dir", "cargo_target_dir"]:
        env_value = os.environ.get(str(env_names.get(key) or ""))
        if env_value:
            merged[key] = env_value
    if not merged.get("runtime_root"):
        merged["runtime_root"] = str(default_runtime_root())
    return merged


def resolve_paths(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    root = expand_path(str(config.get("runtime_root") or default_runtime_root()))
    defaults = {
        "data_dir": root / "data",
        "cache_dir": root / "cache",
        "reports_dir": root / "reports",
        "checkpoints_dir": root / "checkpoints",
        "cargo_target_dir": root / "cargo-target",
    }
    paths: dict[str, dict[str, str]] = {}
    for key, default in defaults.items():
        path = expand_path(str(config.get(key) or default))
        paths[key] = {"path": str(path), "source": "configured" if config.get(key) else "runtime_root_default"}
    paths["runtime_root"] = {"path": str(root), "source": "configured" if config.get("runtime_root") else "default"}
    return paths


def runtime_env(*, create: bool = True) -> dict[str, str]:
    config = effective_config()
    paths = resolve_paths(config)
    if create:
        for row in paths.values():
            Path(row["path"]).mkdir(parents=True, exist_ok=True)
    return {**os.environ.copy(), **runtime_env_overrides(config, paths)}


def runtime_env_overrides(config: dict[str, Any], paths: dict[str, dict[str, str]]) -> dict[str, str]:
    env_names = config.get("env_names") if isinstance(config.get("env_names"), dict) else {}
    values = {
        "runtime_root": paths["runtime_root"]["path"],
        "data_dir": paths["data_dir"]["path"],
        "cache_dir": paths["cache_dir"]["path"],
        "reports_dir": paths["reports_dir"]["path"],
        "checkpoints_dir": paths["checkpoints_dir"]["path"],
        "cargo_target_dir": paths["cargo_target_dir"]["path"],
    }
    env: dict[str, str] = {}
    for key, value in values.items():
        name = str(env_names.get(key) or "")
        if name:
            env[name] = value
    return env


def write_local_config(runtime_root: str = "", *, create: bool = True) -> dict[str, Any]:
    root = expand_path(runtime_root) if runtime_root else default_runtime_root()
    env_names = read_json(CONFIG_PATH, {}).get("env_names", {})
    if not isinstance(env_names, dict):
        env_names = {}

    def configured_path(key: str, default_env: str, default: Path) -> str:
        env_name = str(env_names.get(key) or default_env)
        env_value = os.environ.get(env_name)
        return str(expand_path(env_value)) if env_value else str(default)

    config = {
        "policy": "project_theseus_runtime_paths_local_v0",
        "created_utc": now(),
        "runtime_root": str(root),
        "data_dir": configured_path("data_dir", "THESEUS_DATA_DIR", root / "data"),
        "cache_dir": configured_path("cache_dir", "THESEUS_CACHE_DIR", root / "cache"),
        "reports_dir": configured_path("reports_dir", "THESEUS_REPORTS_DIR", root / "reports"),
        "checkpoints_dir": configured_path("checkpoints_dir", "THESEUS_CHECKPOINTS_DIR", root / "checkpoints"),
        "cargo_target_dir": configured_path("cargo_target_dir", "CARGO_TARGET_DIR", root / "cargo-target"),
    }
    write_json(LOCAL_CONFIG_PATH, config)
    return runtime_report(create=create, write_report=True)


def migration_status(config: dict[str, Any], paths: dict[str, dict[str, str]]) -> dict[str, Any]:
    managed = {
        "reports": ROOT / "reports",
        "checkpoints": ROOT / "checkpoints",
        "target": ROOT / "target",
    }
    targets = {
        "reports": Path(paths["reports_dir"]["path"]),
        "checkpoints": Path(paths["checkpoints_dir"]["path"]),
        "target": Path(paths["cargo_target_dir"]["path"]),
    }
    rows = []
    for name, source in managed.items():
        target = targets[name]
        rows.append(
            {
                "name": name,
                "source": str(source),
                "target": str(target),
                "source_exists": source.exists(),
                "target_exists": target.exists(),
                "source_is_reparse_point": is_reparse_point(source),
                "source_size_gib": directory_size_gib(source) if source.exists() and not is_reparse_point(source) else None,
                "status": migration_row_status(source, target),
            }
        )
    return {
        "managed_directories": rows,
        "mode": str(config.get("migration_mode") or "manual_or_launcher"),
        "note": "Use scripts/runtime_paths.py migrate-junctions to move ignored generated directories onto the runtime drive.",
    }


def default_runtime_root() -> Path:
    env_root = os.environ.get("THESEUS_RUNTIME_ROOT")
    if env_root:
        return expand_path(env_root)
    if platform.system() == "Windows" and Path("D:/").exists():
        return Path("D:/ProjectTheseus/runtime")
    if platform.system() == "Darwin":
        return macos_hive_support_root() / "runtime"
    return Path.home() / ".local" / "share" / "project-theseus" / "runtime"


def python_runtime_checks() -> list[dict[str, Any]]:
    candidates = [
        {"name": "active_python", "python": Path(sys.executable), "role": "current_process"},
        {"name": "source_venv", "python": source_venv_python(), "role": "source_cli_preferred"},
        {"name": "installed_app_venv", "python": installed_app_python(), "role": "launchagent_preferred"},
    ]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        python = Path(candidate["python"])
        key = str(python)
        if key in seen:
            continue
        seen.add(key)
        rows.append(python_runtime_check(str(candidate["name"]), python, str(candidate["role"])))
    return rows


def source_venv_python() -> Path:
    return ROOT / ".venv-puffer" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def installed_app_root() -> Path:
    return macos_hive_support_root() / "app" / "current"


def installed_app_python() -> Path:
    return installed_app_root() / ".venv-puffer" / "bin" / "python"


def macos_hive_support_root() -> Path:
    env_root = os.environ.get("THESEUS_APP_SUPPORT_ROOT") or os.environ.get("THESEUS_MACOS_SUPPORT_ROOT")
    if env_root:
        return expand_path(env_root)
    return Path.home() / "Library" / "Application Support" / "Project Theseus Hive"


def python_runtime_check(name: str, python: Path, role: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "role": role,
        "python": str(python),
        "exists": python.exists(),
        "executable": os.access(python, os.X_OK) if python.exists() else False,
    }
    if not python.exists():
        row["available"] = False
        row["error"] = "python_missing"
        return row
    code = (
        "import importlib.util, json, platform, sys\n"
        "out={'available': True, 'version': sys.version.split()[0], 'reported_executable': sys.executable, "
        "'platform': platform.platform(), 'machine': platform.machine()}\n"
        "parent=importlib.util.find_spec('mlx')\n"
        "spec=importlib.util.find_spec('mlx.core') if parent else None\n"
        "out['mlx']={'module': 'mlx.core', 'module_available': bool(spec), 'available': False, 'probe': []}\n"
        "if spec:\n"
        "    try:\n"
        "        import mlx.core as mx\n"
        "        x=mx.array([1.0, 2.0]); mx.eval(x)\n"
        "        out['mlx'].update({'available': True, 'probe': [float(v) for v in x.tolist()]})\n"
        "    except Exception as exc:\n"
        "        out['mlx'].update({'available': False, 'error': type(exc).__name__, 'message': str(exc)})\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            env=runtime_env(create=False),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**row, "available": False, "error": type(exc).__name__, "message": str(exc)}
    row.update({"returncode": result.returncode})
    if result.returncode != 0:
        row.update({"available": False, "stdout_tail": result.stdout[-500:], "stderr_tail": result.stderr[-1000:]})
        return row
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"available": True, "stdout_tail": result.stdout[-500:]}
    return {**row, **payload}


def preferred_python_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for name in ["source_venv", "installed_app_venv", "active_python"]:
        row = next((item for item in rows if item.get("name") == name and item.get("available")), None)
        if row:
            return row
    return rows[0] if rows else {}


def installed_app_report() -> dict[str, Any]:
    root = installed_app_root()
    runtime_root = macos_hive_support_root() / "runtime"
    reports = runtime_root / "reports"
    return {
        "install_root": file_or_dir_row(root),
        "python": file_or_dir_row(installed_app_python()),
        "runtime_root": file_or_dir_row(runtime_root),
        "reports_dir": file_or_dir_row(reports),
        "status_report": file_or_dir_row(reports / "hive_status.json"),
        "version_status_report": file_or_dir_row(reports / "hive_version_status.json"),
        "license_registration": file_or_dir_row(root / "configs" / "theseus_registration.local.json"),
        "join_config": file_or_dir_row(root / "configs" / "hive_join.local.json"),
    }


def runtime_root_comparison(runtime: dict[str, Any], installed: dict[str, Any]) -> dict[str, Any]:
    source_root = str(get_path(runtime, ["paths", "runtime_root", "path"], ""))
    installed_root = str(get_path(installed, ["runtime_root", "path"], ""))
    return {
        "source_runtime_root": source_root,
        "installed_runtime_root": installed_root,
        "source_vs_installed": {
            "same": bool(source_root and installed_root and Path(source_root).expanduser() == Path(installed_root).expanduser()),
            "interpretation": "different is acceptable for source-dev versus installed-app mode, but release gates must say which mode they checked",
        },
    }


def runtime_state_comparison(runtime: dict[str, Any], installed: dict[str, Any]) -> dict[str, Any]:
    source_reports = ROOT / "reports"
    installed_reports = Path(str(get_path(installed, ["reports_dir", "path"], "") or macos_hive_support_root() / "runtime" / "reports"))
    contexts = {
        "source_checkout": runtime_state_context("source_checkout", ROOT, source_reports),
        "installed_app": runtime_state_context("installed_app", installed_app_root(), installed_reports),
    }
    warnings: list[str] = []
    blockers: list[str] = []
    source = contexts["source_checkout"]
    app = contexts["installed_app"]

    compare_versions(
        warnings,
        "source_verified_catalog_mismatch",
        get_path(source, ["verified_version", "version_id"], ""),
        get_path(source, ["update_catalog", "latest_version_id"], ""),
    )
    compare_versions(
        warnings,
        "installed_verified_catalog_mismatch",
        get_path(app, ["verified_version", "version_id"], ""),
        get_path(app, ["update_catalog", "latest_version_id"], ""),
    )
    compare_versions(
        warnings,
        "source_installed_verified_version_mismatch",
        get_path(source, ["verified_version", "version_id"], ""),
        get_path(app, ["verified_version", "version_id"], ""),
    )
    compare_versions(
        warnings,
        "source_installed_catalog_version_mismatch",
        get_path(source, ["update_catalog", "latest_version_id"], ""),
        get_path(app, ["update_catalog", "latest_version_id"], ""),
    )
    compare_versions(
        warnings,
        "installed_version_status_local_version_stale",
        get_path(app, ["verified_version", "version_id"], ""),
        get_path(app, ["version_status", "local_version_id"], ""),
    )
    compare_versions(
        warnings,
        "installed_version_status_verified_version_stale",
        get_path(app, ["verified_version", "version_id"], ""),
        get_path(app, ["version_status", "verified_version_id"], ""),
    )
    compare_versions(
        warnings,
        "installed_update_checkin_checkpoint_stale",
        get_path(app, ["verified_version", "version_id"], ""),
        get_path(app, ["update_checkin", "applied_checkpoint_id"], "") or get_path(app, ["update_checkin", "selected_checkpoint_id"], ""),
    )
    if get_path(installed, ["install_root", "exists"], False):
        for key, warning in [
            ("verified_version", "installed_mirrored_verified_version_missing"),
            ("update_catalog", "installed_mirrored_update_catalog_missing"),
        ]:
            if not get_path(app, [key, "exists"], False):
                warnings.append(warning)

    return {
        "ok": not blockers,
        "source_runtime_root": get_path(runtime, ["paths", "runtime_root", "path"], ""),
        "installed_runtime_root": get_path(installed, ["runtime_root", "path"], ""),
        "contexts": contexts,
        "warnings": warnings,
        "blockers": blockers,
        "interpretation": "Source and installed app may write separate reports, but verified version/catalog/checkin summaries must name the context and should converge before fleet rollout.",
    }


def runtime_state_context(name: str, root: Path, reports: Path) -> dict[str, Any]:
    return {
        "name": name,
        "root": file_or_dir_row(root),
        "reports_dir": file_or_dir_row(reports),
        "verified_version": verified_version_summary(reports / "hive_verified_version.json"),
        "update_catalog": update_catalog_summary(reports / "hive_update_catalog.json"),
        "version_status": version_status_summary(reports / "hive_version_status.json"),
        "update_checkin": update_checkin_summary(reports / "update_checkin.json"),
        "license_registration": file_or_dir_row(root / "configs" / "theseus_registration.local.json"),
        "join_config": file_or_dir_row(root / "configs" / "hive_join.local.json"),
    }


def verified_version_summary(path: Path) -> dict[str, Any]:
    row = json_file_base_summary(path)
    payload = read_json(path, {})
    if isinstance(payload, dict):
        row.update(
            {
                "version_id": payload.get("version_id", ""),
                "app_version": payload.get("app_version", ""),
                "git_commit": get_path(payload, ["git", "commit"], "") or get_path(payload, ["git", "short_commit"], ""),
                "promotion_state": payload.get("promotion_state", ""),
            }
        )
    return row


def update_catalog_summary(path: Path) -> dict[str, Any]:
    row = json_file_base_summary(path)
    payload = read_json(path, {})
    latest = payload.get("latest_hive_version") if isinstance(payload.get("latest_hive_version"), dict) else {}
    offers = payload.get("offers") if isinstance(payload.get("offers"), list) else []
    offer = offers[0] if offers and isinstance(offers[0], dict) else {}
    offer_version = offer.get("hive_version") if isinstance(offer.get("hive_version"), dict) else {}
    latest_offer = payload.get("latest_offer") if isinstance(payload.get("latest_offer"), dict) else {}
    if isinstance(payload, dict):
        row.update(
            {
                "catalog_id": payload.get("catalog_id", ""),
                "channel": payload.get("channel", ""),
                "track": payload.get("track", ""),
                "offer_count": payload.get("offer_count", len(offers)),
                "latest_version_id": latest.get("version_id") or offer_version.get("version_id") or latest_offer.get("checkpoint_id") or offer.get("checkpoint_id", ""),
                "latest_git_commit": latest.get("git_commit") or offer_version.get("git_commit", ""),
                "latest_update_id": latest_offer.get("update_id") or offer.get("update_id", ""),
            }
        )
    return row


def version_status_summary(path: Path) -> dict[str, Any]:
    row = json_file_base_summary(path)
    payload = read_json(path, {})
    if isinstance(payload, dict):
        row.update(
            {
                "local_version_id": get_path(payload, ["local", "version_id"], ""),
                "local_git_commit": get_path(payload, ["local", "git", "commit"], "") or get_path(payload, ["local", "git", "short_commit"], ""),
                "local_dirty": get_path(payload, ["local", "git", "dirty"], None),
                "verified_version_id": get_path(payload, ["verified_version", "version_id"], ""),
                "update_available": get_path(payload, ["updates", "update_available"], None),
                "soft_update_available": get_path(payload, ["updates", "soft_update_available"], None),
                "hard_update_available": get_path(payload, ["updates", "hard_update_available"], None),
                "installed_update_id": get_path(payload, ["updates", "installed_update_id"], ""),
                "offer_update_id": get_path(payload, ["updates", "offer_update_id"], ""),
            }
        )
    return row


def update_checkin_summary(path: Path) -> dict[str, Any]:
    row = json_file_base_summary(path)
    payload = read_json(path, {})
    if isinstance(payload, dict):
        row.update(
            {
                "catalog_ok": payload.get("catalog_ok", None),
                "catalog_source": payload.get("catalog_source", ""),
                "catalog_url": payload.get("catalog_url", ""),
                "client_mode": get_path(payload, ["client", "mode"], ""),
                "selected_update_id": get_path(payload, ["selected", "update_id"], ""),
                "selected_checkpoint_id": get_path(payload, ["selected", "checkpoint_id"], ""),
                "installed_update_id": get_path(payload, ["installed", "active_update_id"], ""),
                "installed_checkpoint_id": get_path(payload, ["installed", "active_checkpoint_id"], ""),
                "applied_update_id": get_path(payload, ["auto_apply", "update_id"], ""),
                "applied_checkpoint_id": get_path(payload, ["auto_apply", "checkpoint_id"], ""),
                "next_action": payload.get("next_action", ""),
            }
        )
    return row


def json_file_base_summary(path: Path) -> dict[str, Any]:
    row = file_or_dir_row(path)
    payload = read_json(path, {})
    if isinstance(payload, dict) and payload:
        row.update(
            {
                "json_ok": True,
                "ok": payload.get("ok", None),
                "policy": payload.get("policy", ""),
                "created_utc": payload.get("created_utc", ""),
            }
        )
    elif row.get("exists"):
        row.update({"json_ok": False})
    return row


def compare_versions(warnings: list[str], warning: str, expected: Any, actual: Any) -> None:
    expected_text = str(expected or "")
    actual_text = str(actual or "")
    if expected_text and actual_text and expected_text != actual_text:
        warnings.append(warning)


def macos_launchagent_report() -> dict[str, Any]:
    labels = ["local.project-theseus.hive", "local.project-theseus.hive-menubar", "local.project-theseus.update"]
    uid = os.getuid()
    rows: dict[str, Any] = {}
    for label in labels:
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        capture = run_capture(["launchctl", "print", f"gui/{uid}/{label}"], timeout=8)
        plist = read_plist(plist_path)
        rows[label] = {
            "loaded": bool(capture.get("ok")),
            "print": compact_capture(capture),
            "plist": file_or_dir_row(plist_path),
            "program": plist.get("Program") if isinstance(plist, dict) else "",
            "program_arguments": plist.get("ProgramArguments") if isinstance(plist, dict) else [],
            "working_directory": plist.get("WorkingDirectory") if isinstance(plist, dict) else "",
            "environment": redact_env(plist.get("EnvironmentVariables") if isinstance(plist, dict) else {}),
        }
    return rows


def read_plist(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = plistlib.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, plistlib.InvalidFileException):
        return {}


def run_capture(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
    }


def compact_capture(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(capture.get("ok")),
        "returncode": capture.get("returncode"),
        "error": capture.get("error", ""),
        "message": capture.get("message", ""),
        "stdout_tail": str(capture.get("stdout_tail") or "")[-600:],
        "stderr_tail": str(capture.get("stderr_tail") or "")[-600:],
    }


def file_or_dir_row(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "size_bytes": stat.st_size if path.is_file() else None,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def redact_env(env: Any) -> dict[str, Any]:
    if not isinstance(env, dict):
        return {}
    redacted: dict[str, Any] = {}
    for key, value in env.items():
        name = str(key)
        if any(token in name.upper() for token in ["SECRET", "TOKEN", "PASSWORD", "KEY"]):
            redacted[name] = "***"
        else:
            redacted[name] = value
    return redacted


def runtime_doctor_next_actions(blockers: list[str], warnings: list[str], preferred: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if "mlx_missing_from_source_and_installed_runtime" in blockers:
        actions.append("Run `python3 scripts/macos_dependency_bootstrap.py --venv .venv-puffer --install-missing --require-mlx` on Apple Silicon.")
    if "active_shell_python_lacks_mlx_but_hive_runtime_has_it" in warnings:
        actions.append("Use `./.venv-puffer/bin/python` or the installed app venv for MLX gates; active shell python is not the Hive MLX runtime.")
    if "hive_launchagent_not_loaded" in warnings:
        actions.append("Run `./scripts/install_theseus_hive_macos.sh --install-service --enable-service --start` or reload the LaunchAgent.")
    if "source_and_installed_runtime_roots_differ" in warnings:
        actions.append("Keep source-dev and installed-app reports labeled separately; do not compare their version/license state without naming the checked runtime.")
    if any(item in warnings for item in ["source_verified_catalog_mismatch", "installed_verified_catalog_mismatch", "source_installed_verified_version_mismatch", "source_installed_catalog_version_mismatch"]):
        actions.append("Run `python3 scripts/hive_version_manager.py verify`, `publish`, `installer-artifacts`, and `status`, then reinstall or restart the Mac canary so source and installed runtime metadata converge.")
    if any(item in warnings for item in ["installed_version_status_local_version_stale", "installed_version_status_verified_version_stale", "installed_update_checkin_checkpoint_stale"]):
        actions.append("Refresh installed-app runtime reports with the installed venv or LaunchAgent update check; stale status reports should not be used as fleet evidence.")
    if any(item in warnings for item in ["installed_mirrored_verified_version_missing", "installed_mirrored_update_catalog_missing"]):
        actions.append("Publish the Hive update catalog again so private catalog reports are mirrored into the installed runtime reports directory.")
    if not actions:
        actions.append(f"Use `{preferred.get('python') or sys.executable}` for Mac MLX/control-plane checks.")
    return actions


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def expand_path(value: str) -> Path:
    if not value:
        return default_runtime_root()
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else ROOT / path


def with_disk_status(paths: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, row in paths.items():
        path = Path(row["path"])
        status = dict(row)
        try:
            anchor = path.anchor or str(ROOT.anchor)
            usage = shutil.disk_usage(anchor)
            status.update(
                {
                    "disk_root": anchor,
                    "disk_total_gib": round(usage.total / 1024**3, 2),
                    "disk_free_gib": round(usage.free / 1024**3, 2),
                }
            )
        except OSError as exc:
            status.update({"disk_error": str(exc)})
        out[key] = status
    return out


def directory_size_gib(path: Path) -> float:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    except OSError:
        return 0.0
    return round(total / 1024**3, 3)


def is_reparse_point(path: Path) -> bool:
    try:
        return bool(path.exists() and path.lstat().st_file_attributes & 0x400)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return path.is_symlink()


def migration_row_status(source: Path, target: Path) -> str:
    if not source.exists():
        return "source_missing"
    if is_reparse_point(source):
        return "redirected"
    if target.exists() and source.resolve() == target.resolve():
        return "already_target"
    return "local_generated_directory"


def deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override not in ({}, None) else base
    result = dict(base)
    for key, value in override.items():
        result[key] = deep_merge(result.get(key), value)
    return result


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if value is not None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
