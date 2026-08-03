#!/usr/bin/env python3
"""Qualify bounded Python, Node, and Rust-binary VCM evaluator containment."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import host_resource_safety as host_safety

ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_vcm_multilang_sandbox_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_multilang_sandbox.json"
MARKER = "__THESEUS_VCM_SANDBOX_RESULT__"
REQUIRED = {
    "environment_minimized",
    "host_read_denied",
    "inside_write_allowed",
    "network_denied",
    "outside_write_denied",
    "shell_exec_denied",
    "symlink_host_read_denied",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = qualify(config, config_path) if args.qualify else preflight(config, config_path)
    out = resolve(args.out or config["report"])
    write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVE_CONTAINMENT_QUALIFICATION_BEFORE_VCM_REPOSITORY_EXECUTION":
        faults.append("state_invalid")
    if config.get("required_work_root") != "/private/tmp":
        faults.append("work_root_invalid")
    owner = resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or sha256_file(owner) != config.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    for name in ("backend", "python", "node", "rustc"):
        row = mapping(config.get("executables")).get(name)
        row = mapping(row)
        path = Path(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            faults.append(f"executable_identity_invalid:{name}")
    authority = mapping(config.get("authority"))
    allowed_true = {"trusted_canary_compilation_authorized", "trusted_canary_execution_authorized"}
    for key, value in authority.items():
        expected = key in allowed_true
        if value is not expected:
            faults.append(f"authority_invalid:{key}")
    limits = mapping(config.get("limits"))
    for key in ("cpu_seconds", "wall_seconds", "output_file_mib", "open_files", "user_processes", "max_process_group_rss_mib", "terminate_grace_seconds"):
        if float(limits.get(key) or 0) <= 0:
            faults.append(f"limit_invalid:{key}")
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_TRUSTED_MULTILANG_CANARIES",
        "faults": sorted(set(faults)),
        "config": identity(config_path),
        "qualification_executed": False,
        "repository_execution_authorized": False,
        "dependency_installation_authorized": False,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def qualify(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    results: dict[str, Any] = {}
    faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-sandbox-", dir="/private/tmp") as tmp:
        root = Path(tmp).resolve()
        python = str(mapping(mapping(config["executables"])["python"])["path"])
        node = str(mapping(mapping(config["executables"])["node"])["path"])
        rustc = str(mapping(mapping(config["executables"])["rustc"])["path"])
        commands = {
            "python": [python, "-c", python_source(root, config)],
            "node": [node, "-e", node_source(root, config)],
        }
        rust_source_path = root / "canary.rs"
        rust_binary = root / "rust-canary"
        rust_source_path.write_text(rust_source(root, config), encoding="utf-8")
        compile_receipt = subprocess.run(
            [rustc, str(rust_source_path), "-o", str(rust_binary)],
            cwd=root,
            env={"HOME": str(root), "PATH": "/usr/bin:/bin", "TMPDIR": str(root)},
            capture_output=True,
            check=False,
        )
        results["rust_trusted_compile"] = {
            "returncode": compile_receipt.returncode,
            "stdout_sha256": hashlib.sha256(compile_receipt.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(compile_receipt.stderr).hexdigest(),
            "source_sha256": sha256_file(rust_source_path),
            "binary_sha256": sha256_file(rust_binary),
            "untrusted_source_compiled": False,
        }
        if compile_receipt.returncode != 0 or not rust_binary.is_file():
            faults.append("trusted_rust_canary_compile_failed")
        else:
            commands["rust_binary"] = [str(rust_binary)]
        for runtime, command in commands.items():
            receipt = run_sandboxed(command, root / runtime, config, runtime)
            canary, parse_faults = parse(receipt)
            receipt["canary"] = canary
            results[runtime] = receipt
            faults.extend(f"{runtime}:{fault}" for fault in parse_faults)
            if receipt.get("boundary_hit"):
                faults.append(f"{runtime}:physical_boundary_hit")
            if receipt.get("returncode") != 0:
                faults.append(f"{runtime}:nonzero_returncode")
    passed = not faults and set(results) == {"python", "node", "rust_binary", "rust_trusted_compile"}
    return {
        **before,
        "created_utc": now(),
        "trigger_state": "GREEN" if passed else "RED",
        "state": "MULTILANG_CONTAINMENT_CANARIES_QUALIFIED" if passed else "MULTILANG_CONTAINMENT_CANARIES_FAILED",
        "faults": sorted(set(faults)),
        "qualification_executed": True,
        "runtime_results": results,
        "qualified_runtime_scopes": ["python_interpreter", "node_runtime", "rust_precompiled_binary"] if passed else [],
        "explicitly_unqualified_scopes": ["npm_dependency_installation", "typescript_transpilation", "cargo_dependency_resolution", "rust_untrusted_compilation", "repository_runner_adequacy"],
        "repository_execution_authorized": False,
    }


def run_sandboxed(command: list[str], workdir: Path, config: dict[str, Any], runtime: str) -> dict[str, Any]:
    workdir.mkdir()
    profile = sandbox_profile(command[0], workdir, config, runtime)
    backend = str(mapping(mapping(config["executables"])["backend"])["path"])
    limits = mapping(config["limits"])
    stdout_path, stderr_path = workdir / "stdout.bin", workdir / "stderr.bin"
    env = {"HOME": str(workdir), "LC_CTYPE": "UTF-8", "PATH": "/usr/bin:/bin", "TMPDIR": str(workdir), "NO_COLOR": "1"}

    def set_limits() -> None:
        mib = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds"]),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["output_file_mib"]) * mib,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(limits["open_files"]),) * 2)
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["user_processes"]),) * 2)

    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen([backend, "-p", profile, *command], cwd=workdir, env=env, stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=set_limits)
        boundary, telemetry, maximum_rss = "", "", 0.0
        deadline = started + float(limits["wall_seconds"])
        while process.poll() is None:
            if time.monotonic() >= deadline:
                boundary = "wall_boundary_hit"; break
            try:
                maximum_rss = max(maximum_rss, host_safety.process_rss_mib(process.pid))
            except Exception as exc:
                telemetry = f"{type(exc).__name__}:{exc}"[:1000]; boundary = "rss_telemetry_unavailable"; break
            if maximum_rss > float(limits["max_process_group_rss_mib"]):
                boundary = "rss_boundary_hit"; break
            time.sleep(0.05)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: returncode = process.wait(timeout=float(limits["terminate_grace_seconds"]))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); returncode = process.wait()
        else:
            returncode = int(process.returncode or 0)
    maximum = int(limits["output_file_mib"]) * 1024 * 1024
    stdout_bytes, stderr_bytes = stdout_path.read_bytes(), stderr_path.read_bytes()
    if (len(stdout_bytes) >= maximum or len(stderr_bytes) >= maximum) and not boundary:
        boundary = "output_file_boundary_hit"
    return {
        "returncode": returncode, "duration_ms": round((time.monotonic()-started)*1000, 3),
        "boundary_hit": bool(boundary), "boundary_reason": boundary, "rss_telemetry_fault": telemetry,
        "maximum_process_group_rss_mib": round(maximum_rss, 3), "stdout_complete": len(stdout_bytes) < maximum,
        "stderr_complete": len(stderr_bytes) < maximum, "stdout": stdout_bytes.decode(errors="replace"),
        "stderr": stderr_bytes.decode(errors="replace"), "project_selected_character_cap": None,
        "profile_sha256": hashlib.sha256(profile.encode()).hexdigest(), "runtime": runtime,
    }


def sandbox_profile(executable: str, workdir: Path, config: dict[str, Any], runtime: str) -> str:
    allowed = Path(executable).resolve().parents[1] if runtime != "rust_binary" else workdir.parent
    denied = " ".join(f'(subpath "{Path(p).resolve()}")' for p in config["denied_read_roots"])
    return "\n".join(["(version 1)", "(allow default)", "(deny network*)", "(deny mach-lookup)", f"(deny file-read* {denied})", f'(deny file-write* (require-not (subpath "{workdir}")))', '(allow file-write* (literal "/dev/null"))', f'(deny process-exec (require-not (subpath "{allowed}")))'])


def payload(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {"host": str(ROOT / "AGENTS.md"), "outside": str(ROOT / "runtime/control/vcm_sandbox_escape"), "work": str(root), "env": sorted(config["environment_allowlist"]), "marker": MARKER}


def python_source(root: Path, config: dict[str, Any]) -> str:
    p = json.dumps(payload(root / "python", config))
    return f'''import json,os,pathlib,socket,subprocess
p=json.loads({p!r}); r={{}}
def d(k,f):
 try:f();r[k]=False
 except (OSError,PermissionError):r[k]=True
d("host_read_denied",lambda:pathlib.Path(p["host"]).read_bytes()); l=pathlib.Path(p["work"])/"link"; l.symlink_to(p["host"]); d("symlink_host_read_denied",lambda:l.read_bytes()); d("outside_write_denied",lambda:pathlib.Path(p["outside"]).write_text("x")); d("network_denied",lambda:socket.create_connection(("127.0.0.1",9),.2)); d("shell_exec_denied",lambda:subprocess.run(["/bin/sh","-c","true"])); q=pathlib.Path(p["work"])/"ok";q.write_text("ok");r["inside_write_allowed"]=q.read_text()=="ok";r["environment_minimized"]=sorted(os.environ)==p["env"];print(p["marker"]+json.dumps(r,sort_keys=True))'''


def node_source(root: Path, config: dict[str, Any]) -> str:
    p = json.dumps(payload(root / "node", config))
    return f'''const fs=require("fs"),net=require("net"),cp=require("child_process");const p={p};const r={{}};function d(k,f){{try{{f();r[k]=false}}catch(e){{r[k]=true}}}}d("host_read_denied",()=>fs.readFileSync(p.host));fs.symlinkSync(p.host,p.work+"/link");d("symlink_host_read_denied",()=>fs.readFileSync(p.work+"/link"));d("outside_write_denied",()=>fs.writeFileSync(p.outside,"x"));d("shell_exec_denied",()=>cp.execFileSync("/bin/sh",["-c","true"]));fs.writeFileSync(p.work+"/ok","ok");r.inside_write_allowed=fs.readFileSync(p.work+"/ok","utf8")==="ok";r.environment_minimized=JSON.stringify(Object.keys(process.env).sort())===JSON.stringify(p.env);const s=net.connect({{host:"127.0.0.1",port:9}});let done=false;function finish(v){{if(done)return;done=true;r.network_denied=v;console.log(p.marker+JSON.stringify(r));}}s.on("connect",()=>finish(false));s.on("error",()=>finish(true));setTimeout(()=>finish(true),250);'''


def rust_source(root: Path, config: dict[str, Any]) -> str:
    p = payload(root / "rust_binary", config)
    return f'''use std::fs;use std::net::TcpStream;use std::process::Command;fn main(){{let host={p['host']!r};let outside={p['outside']!r};let work={p['work']!r};let mut rows=Vec::new();rows.push(("host_read_denied",fs::read(host).is_err()));let link=format!("{{}}/link",work);let _=std::os::unix::fs::symlink(host,&link);rows.push(("symlink_host_read_denied",fs::read(&link).is_err()));rows.push(("outside_write_denied",fs::write(outside,b"x").is_err()));rows.push(("network_denied",TcpStream::connect("127.0.0.1:9").is_err()));rows.push(("shell_exec_denied",Command::new("/bin/sh").arg("-c").arg("true").status().is_err()));let inside=format!("{{}}/ok",work);let ok=fs::write(&inside,b"ok").is_ok()&&fs::read(&inside).ok()==Some(b"ok".to_vec());rows.push(("inside_write_allowed",ok));let mut keys:Vec<String>=std::env::vars().map(|x|x.0).collect();keys.sort();rows.push(("environment_minimized",keys==vec!["HOME","LC_CTYPE","NO_COLOR","PATH","TMPDIR"]));let body=rows.iter().map(|(k,v)|format!("\\\"{{}}\\\":{{}}",k,v)).collect::<Vec<_>>().join(",");println!("{MARKER}{{{{{{}}}}}}",body);}}'''.replace("'", "\"")


def parse(receipt: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    lines = [x for x in str(receipt.get("stdout") or "").splitlines() if x.startswith(MARKER)]
    if len(lines) != 1:
        return {}, ["result_missing_or_duplicated"]
    try: value = json.loads(lines[0][len(MARKER):])
    except json.JSONDecodeError: return {}, ["result_invalid_json"]
    faults = [f"canary_failed:{key}" for key in REQUIRED if value.get(key) is not True]
    if set(value) != REQUIRED: faults.append("canary_keyset_invalid")
    return {key: bool(value.get(key)) for key in sorted(REQUIRED)}, sorted(faults)


def mapping(value: Any) -> dict[str, Any]: return value if isinstance(value, dict) else {}
def resolve(value: str | Path) -> Path: p=Path(value); return p if p.is_absolute() else ROOT/p
def relative(path: Path) -> str:
    try: return path.resolve().relative_to(ROOT).as_posix()
    except ValueError: return str(path)
def sha256_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
def identity(path: Path) -> dict[str, str]: return {"path": relative(path), "sha256": sha256_file(path)}
def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def write_json(path: Path, value: dict[str, Any]) -> None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def summary(r: dict[str, Any]) -> dict[str, Any]: return {k:r.get(k) for k in ("trigger_state","state","qualification_executed","qualified_runtime_scopes","faults")}

if __name__ == "__main__": raise SystemExit(main())
