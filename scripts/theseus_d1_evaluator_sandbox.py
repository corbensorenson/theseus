#!/usr/bin/env python3
"""Qualify and invoke the fail-closed D1 untrusted evaluator sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import host_resource_safety as host_safety  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_evaluator_sandbox.json"
POLICY = "project_theseus_d1_untrusted_evaluator_sandbox_v1"
RESULT_MARKER = "__THESEUS_D1_SANDBOX_RESULT__"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = qualify(config, config_path=config_path) if args.qualify else preflight(config, config_path=config_path)
    out = resolve(args.out or str(config["qualification_report"]))
    write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    faults = validate_config(config)
    backend = Path(str(config.get("backend") or ""))
    python = Path(str(config.get("python") or ""))
    if not backend.is_file():
        faults.append("sandbox_backend_missing")
    if not python.is_file() or sha256_file(python) != config.get("python_sha256"):
        faults.append("python_identity_invalid")
    return {
        "policy": POLICY,
        "created_utc": utc_now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "activation_state": "CONTRACT_INVALID" if faults else "READY_FOR_LOCAL_SANDBOX_QUALIFICATION",
        "faults": sorted(set(faults)),
        "config": source_identity(config_path),
        "backend": source_identity(backend),
        "python": source_identity(python),
        "qualification_executed": False,
        "untrusted_execution_authorized": False,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def qualify(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    if before["trigger_state"] == "RED":
        return before
    required_root = Path(str(config["required_work_root"])).resolve()
    with tempfile.TemporaryDirectory(prefix="theseus-d1-evaluator-", dir=required_root) as temporary:
        workdir = Path(temporary).resolve()
        result = run_sandboxed(
            [str(config["python"]), "-c", canary_source(config, workdir)],
            workdir=workdir,
            config=config,
        )
    canary, canary_faults = parse_canary(result)
    faults = list(canary_faults)
    if result.get("boundary_hit") is True:
        faults.append("qualification_physical_boundary_hit")
    passed = not faults and result.get("returncode") == 0
    return {
        **before,
        "created_utc": utc_now(),
        "trigger_state": "GREEN" if passed else "RED",
        "activation_state": (
            "D1_UNTRUSTED_EVALUATOR_SANDBOX_QUALIFIED"
            if passed
            else "D1_UNTRUSTED_EVALUATOR_SANDBOX_NOT_QUALIFIED"
        ),
        "faults": sorted(set(faults)),
        "qualification_executed": True,
        "untrusted_execution_authorized": passed,
        "sandbox_profile_sha256": stable_hash(sandbox_profile(config, Path("/private/tmp/placeholder"))),
        "run_receipt": result,
        "canary": canary,
        "parent_target_or_evaluator_executions": 0,
    }


def run_sandboxed(
    command: list[str], *, workdir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    workdir = workdir.resolve()
    required_root = Path(str(config["required_work_root"])).resolve()
    if workdir == required_root or required_root not in workdir.parents:
        raise ValueError("sandbox_workdir_outside_required_root")
    if not command or Path(command[0]).resolve() != Path(str(config["python"])).resolve():
        raise ValueError("sandbox_command_not_pinned_python")
    profile = sandbox_profile(config, workdir)
    limits = mapping(config.get("qualification_limits"))
    stdout_path = workdir / "stdout.bin"
    stderr_path = workdir / "stderr.bin"
    env = {
        "HOME": str(workdir),
        "LC_CTYPE": "UTF-8",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(workdir),
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def apply_limits() -> None:
        mib = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds"]), int(limits["cpu_seconds"])))
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["output_file_mib"]) * mib,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(limits["open_files"]),) * 2)
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["user_processes"]),) * 2)

    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [str(config["backend"]), "-p", profile, *command],
            cwd=workdir,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=apply_limits,
        )
        boundary_reason = ""
        maximum_rss_mib = 0.0
        telemetry_fault = ""
        deadline = started + float(limits["wall_seconds"])
        while process.poll() is None:
            if time.monotonic() >= deadline:
                boundary_reason = "wall_boundary_hit"
                break
            try:
                rss_mib = host_safety.process_rss_mib(process.pid)
                maximum_rss_mib = max(maximum_rss_mib, rss_mib)
            except (OSError, subprocess.SubprocessError, host_safety.HostResourceSafetyFault) as exc:
                telemetry_fault = f"{type(exc).__name__}:{exc}"[:1000]
                boundary_reason = "rss_telemetry_unavailable"
                break
            if maximum_rss_mib > float(limits["max_process_group_rss_mib"]):
                boundary_reason = "process_group_rss_boundary_hit"
                break
            time.sleep(0.05)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=float(limits["terminate_grace_seconds"]))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()
        else:
            returncode = int(process.returncode or 0)
    maximum = int(limits["output_file_mib"]) * 1024 * 1024
    stdout_bytes = stdout_path.stat().st_size
    stderr_bytes = stderr_path.stat().st_size
    output_boundary_hit = stdout_bytes >= maximum or stderr_bytes >= maximum
    if output_boundary_hit and not boundary_reason:
        boundary_reason = "output_file_boundary_hit"
    return {
        "returncode": returncode,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "boundary_hit": bool(boundary_reason),
        "boundary_reason": boundary_reason,
        "rss_telemetry_fault": telemetry_fault,
        "maximum_process_group_rss_mib": round(maximum_rss_mib, 3),
        "max_process_group_rss_mib": float(limits["max_process_group_rss_mib"]),
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        # Historical field names are retained for receipt compatibility. When
        # the physical output-file boundary is untouched, both values contain
        # the complete verifier streams, not a project-selected tail.
        "stdout_tail": read_tail(stdout_path, maximum),
        "stderr_tail": read_tail(stderr_path, maximum),
        "stdout_complete": not output_boundary_hit,
        "stderr_complete": not output_boundary_hit,
        "project_selected_character_cap": None,
        "profile_sha256": hashlib.sha256(profile.encode()).hexdigest(),
        "environment_keys": sorted(env),
        "limits": limits,
    }


def sandbox_profile(config: dict[str, Any], workdir: Path) -> str:
    python_root = Path(str(config["python"])).resolve().parents[1]
    denied_roots = " ".join(
        f'(subpath "{Path(value).resolve()}")'
        for value in strings(config.get("denied_read_roots"))
    )
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny mach-lookup)",
            f"(deny file-read* {denied_roots})",
            f'(deny file-write* (require-not (subpath "{workdir.resolve()}")))',
            '(allow file-write* (literal "/dev/null"))',
            f'(deny process-exec (require-not (subpath "{python_root}")))',
        ]
    )


def canary_source(config: dict[str, Any], workdir: Path) -> str:
    denied_read = ROOT / "AGENTS.md"
    denied_write = ROOT / "runtime" / "control" / "d1_sandbox_escape_canary"
    python = str(config["python"])
    limits = mapping(config.get("qualification_limits"))
    payload = {
        "denied_read": str(denied_read),
        "denied_write": str(denied_write),
        "workdir": str(workdir),
        "python": python,
        "expected_environment_keys": sorted(strings(config.get("environment_allowlist"))),
        "limits": limits,
        "marker": RESULT_MARKER,
        "child_source": (
            "import pathlib\n"
            "try:\n"
            f"    pathlib.Path({str(denied_read)!r}).read_bytes()\n"
            "    print('OPEN')\n"
            "except (PermissionError, OSError):\n"
            "    print('DENIED')\n"
        ),
    }
    encoded = json.dumps(payload)
    return f'''import json, os, pathlib, resource, socket, subprocess
p = json.loads({encoded!r})
r = {{}}
def denied(name, fn):
    try:
        fn(); r[name] = False
    except (PermissionError, OSError):
        r[name] = True
denied("host_read_denied", lambda: pathlib.Path(p["denied_read"]).read_bytes())
link = pathlib.Path(p["workdir"]) / "host-link"
try: link.symlink_to(p["denied_read"])
except FileExistsError: pass
denied("symlink_host_read_denied", lambda: link.read_bytes())
denied("outside_write_denied", lambda: pathlib.Path(p["denied_write"]).write_text("escape"))
def network():
    s = socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 9))
denied("network_denied", network)
denied("shell_exec_denied", lambda: subprocess.run(["/bin/sh", "-c", "echo escape"], check=False))
try:
    c = subprocess.run([p["python"], "-c", p["child_source"]], capture_output=True, text=True, check=False)
    r["child_python_cannot_escape_read_denial"] = c.stdout.strip() == "DENIED" or (c.returncode != 0 and "OPEN" not in c.stdout)
except OSError:
    r["child_python_cannot_escape_read_denial"] = True
inside = pathlib.Path(p["workdir"]) / "inside.txt"; inside.write_text("ok")
r["inside_write_allowed"] = inside.read_text() == "ok"
r["environment_minimized"] = sorted(os.environ) == p["expected_environment_keys"]
r["cpu_limit_present"] = resource.getrlimit(resource.RLIMIT_CPU)[0] == int(p["limits"]["cpu_seconds"])
r["file_size_limit_present"] = resource.getrlimit(resource.RLIMIT_FSIZE)[0] == int(p["limits"]["output_file_mib"]) * 1024 * 1024
r["open_file_limit_present"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0] == int(p["limits"]["open_files"])
r["process_limit_present"] = (not hasattr(resource, "RLIMIT_NPROC")) or resource.getrlimit(resource.RLIMIT_NPROC)[0] == int(p["limits"]["user_processes"])
print(p["marker"] + json.dumps(r, sort_keys=True))
'''


def parse_canary(result: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    lines = [
        line
        for line in str(result.get("stdout_tail") or "").splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(lines) != 1:
        return {}, ["sandbox_canary_result_missing_or_duplicated"]
    try:
        value = json.loads(lines[0][len(RESULT_MARKER) :])
    except json.JSONDecodeError:
        return {}, ["sandbox_canary_result_invalid_json"]
    required = {
        "host_read_denied",
        "symlink_host_read_denied",
        "outside_write_denied",
        "network_denied",
        "shell_exec_denied",
        "child_python_cannot_escape_read_denial",
        "inside_write_allowed",
        "environment_minimized",
        "cpu_limit_present",
        "file_size_limit_present",
        "open_file_limit_present",
        "process_limit_present",
    }
    faults = [f"sandbox_canary_failed:{key}" for key in sorted(required) if value.get(key) is not True]
    if set(value) != required:
        faults.append("sandbox_canary_keyset_invalid")
    return {key: bool(value.get(key)) for key in sorted(required)}, faults


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVE_QUALIFICATION_BEFORE_ANY_D1_PARENT_TARGET_OR_EVALUATOR_EXECUTION":
        faults.append("state_invalid")
    if config.get("required_work_root") != "/private/tmp":
        faults.append("required_work_root_invalid")
    runner = resolve(str(config.get("sandbox_runner") or ""))
    if (
        not runner.is_file()
        or runner != Path(__file__).resolve()
        or sha256_file(runner) != str(config.get("sandbox_runner_sha256") or "")
    ):
        faults.append("sandbox_runner_binding_invalid")
    if set(strings(config.get("denied_read_roots"))) != {"/Users", "/private/var/folders", "/Volumes", "/Network"}:
        faults.append("denied_read_roots_invalid")
    expected_env = {"HOME", "LC_CTYPE", "PATH", "TMPDIR", "NO_COLOR", "PYTHONDONTWRITEBYTECODE"}
    if set(strings(config.get("environment_allowlist"))) != expected_env:
        faults.append("environment_allowlist_invalid")
    limits = mapping(config.get("qualification_limits"))
    for key in ("cpu_seconds", "max_process_group_rss_mib", "output_file_mib", "open_files", "user_processes", "wall_seconds", "terminate_grace_seconds"):
        if int(limits.get(key) or 0) <= 0:
            faults.append(f"qualification_limit_invalid:{key}")
    authority = mapping(config.get("authority"))
    if authority.get("user_or_operator_approval_required") is not False:
        faults.append("user_gate_present")
    for key, value in authority.items():
        if key != "user_or_operator_approval_required" and value is not False:
            faults.append(f"forbidden_authority_present:{key}")
    return sorted(set(faults))


def source_identity(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_tail(path: Path, maximum: int) -> str:
    with path.open("rb") as handle:
        if path.stat().st_size > maximum:
            handle.seek(-maximum, os.SEEK_END)
        return handle.read().decode("utf-8", errors="replace")


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def strings(value: Any) -> list[str]:
    return [str(row) for row in value or [] if isinstance(row, str)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "qualification_executed": report.get("qualification_executed"),
        "untrusted_execution_authorized": report.get("untrusted_execution_authorized"),
        "faults": report.get("faults", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
