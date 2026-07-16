#!/usr/bin/env python3
"""Independent functional verifiers for the frozen neural-seed utility suite."""

from __future__ import annotations

import ast
import base64
import hashlib
import html.parser
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
ALLOWED_ENV = {"HOME", "PATH", "TMPDIR", "NO_COLOR"}


def _result(case: dict[str, Any], *, passed: bool, stage: str, started: float, **extra: Any) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "arm_id": case["arm_id"],
        "verifier_kind": case["verifier"]["kind"],
        "passed": bool(passed),
        "stage": stage,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        **extra,
    }


def _sandbox_profile(workdir: Path, *, browser: bool = False) -> str:
    canonical = Path(os.path.realpath(workdir))
    # Toolchains need ordinary Mach/IPC/runtime access. Constrain the two
    # dangerous effects directly: all network operations and writes outside
    # the case directory are denied by the OS sandbox.
    write_deny = f'(deny file-write* (require-not (subpath "{canonical}")))'
    if browser:
        darwin_temp = Path(os.path.realpath(tempfile.gettempdir()))
        crashpad = Path.home() / "Library/Application Support/Google/Chrome/Crashpad"
        write_deny = (
            "(deny file-write* (require-all "
            f'(require-not (subpath "{canonical}")) '
            f'(require-not (subpath "{darwin_temp}")) '
            f'(require-not (subpath "{crashpad}"))))'
        )
    network_deny = "(deny network*)"
    if browser:
        # Chrome needs local Unix sockets for its process singleton. Deny all
        # IP transport; external resources are also rejected by the DOM audit.
        network_deny = "(deny network* (remote ip))"
    common = [
        "(version 1)",
        "(allow default)",
        network_deny,
    ]
    common.append(write_deny)
    common.append('(allow file-write* (literal "/dev/null"))')
    return "\n".join(common)


def _run_sandboxed(
    command: list[str], workdir: Path, timeout: int, *, browser: bool = False
) -> dict[str, Any]:
    allowed_bins = {
        "/usr/bin/python3",
        shutil.which("python3") or "",
        shutil.which("deno") or "",
        shutil.which("cargo") or "",
        shutil.which("rustc") or "",
        shutil.which("tidy") or "/usr/bin/tidy",
        str(CHROME),
    }
    executable = command[0]
    resolved = shutil.which(executable) if not executable.startswith("/") else executable
    if not resolved or resolved not in allowed_bins:
        return {"ok": False, "fault": "command_not_allowlisted", "command": executable}
    env = {key: value for key, value in os.environ.items() if key in ALLOWED_ENV}
    env["TMPDIR"] = str(workdir)
    if browser:
        env["HOME"] = str(workdir)
    wrapped = ["/usr/bin/sandbox-exec", "-p", _sandbox_profile(workdir, browser=browser), *command]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            wrapped,
            cwd=workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "fault": "timeout",
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }


def _render_chrome(command: list[str], workdir: Path, screenshot: Path, timeout: int) -> dict[str, Any]:
    env = {key: value for key, value in os.environ.items() if key in ALLOWED_ENV}
    env["TMPDIR"] = str(workdir)
    wrapped = ["/usr/bin/sandbox-exec", "-p", _sandbox_profile(workdir, browser=True), *command]
    started = time.monotonic()
    process = subprocess.Popen(
        wrapped,
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = started + timeout
    while time.monotonic() < deadline and process.poll() is None:
        if screenshot.exists() and screenshot.stat().st_size >= 512:
            break
        time.sleep(0.05)
    terminated_after_render = False
    try:
        # A complete screenshot is the browser verifier's durable output. Chrome
        # often keeps background processes alive for seconds after flushing it;
        # a short grace preserves stderr while avoiding that fixed per-render tax.
        stdout, stderr = process.communicate(timeout=min(0.5, max(0.1, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        terminated_after_render = screenshot.exists() and screenshot.stat().st_size >= 512
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)
            stdout, stderr = process.communicate(timeout=2)
    rendered = screenshot.exists() and screenshot.stat().st_size >= 512
    completed_cleanly = process.returncode == 0 or terminated_after_render
    return {
        "ok": rendered and completed_cleanly,
        "returncode": process.returncode,
        "terminated_after_render": terminated_after_render,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "fault": None if rendered and completed_cleanly else "browser_render_failure",
    }


def verify_candidate(case: dict[str, Any], source: str, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    maximum = int(config["sandbox"]["maximum_candidate_bytes"])
    encoded = source.encode("utf-8", errors="replace")
    if not source.strip():
        return _result(case, passed=False, stage="candidate_integrity", started=started, fault="no_output")
    if len(encoded) > maximum:
        return _result(case, passed=False, stage="candidate_integrity", started=started, fault="candidate_too_large")
    if re.search(r"^\s*```", source) or re.search(r"```\s*$", source):
        return _result(case, passed=False, stage="candidate_integrity", started=started, fault="markdown_fence")
    side_effect = _prohibited_side_effect(case["arm_id"], source)
    if side_effect:
        return _result(case, passed=False, stage="candidate_integrity", started=started, fault="prohibited_side_effect", detail=side_effect)
    kind = case["verifier"]["kind"]
    if kind == "python_isolated_tests_v1":
        return _verify_python(case, source, config, started)
    if kind == "deno_typescript_tests_v1":
        return _verify_deno(case, source, config, started)
    if kind == "rust_cargo_tests_v1":
        return _verify_rust(case, source, config, started)
    if kind == "html_dom_a11y_render_v1":
        return _verify_html(case, source, config, started)
    if kind == "blind_rubric_v1":
        return _result(case, passed=False, stage="blind_rubric_pending", started=started, pending_human_judgment=True)
    return _result(case, passed=False, stage="dispatch", started=started, fault="unknown_verifier")


def _prohibited_side_effect(arm_id: str, source: str) -> str:
    patterns = {
        "python": [r"\b(?:open|exec|eval|compile)\s*\(", r"\bimport\s+(?:os|socket|subprocess|pathlib|http|urllib|requests)\b", r"\bfrom\s+(?:os|socket|subprocess|pathlib|http|urllib|requests)\b"],
        "javascript_typescript": [r"\b(?:fetch|XMLHttpRequest|WebSocket|eval)\s*\(?", r"\bDeno\s*\.", r"\b(?:require|import)\s*\(?[\"'](?:node:|fs|net|http|https|child_process)"],
        "rust": [
            r"std::(?:net|fs|process)::",
            r"\b(?:File|TcpStream|UdpSocket|Command)::",
            r"\b(?:include|include_str|include_bytes|env|option_env)!\s*\(",
            r"#\s*\[\s*path\s*=",
        ],
    }
    for pattern in patterns.get(arm_id, []):
        if re.search(pattern, source):
            return pattern
    return ""


def _verify_python(case: dict[str, Any], source: str, config: dict[str, Any], started: float) -> dict[str, Any]:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return _result(case, passed=False, stage="syntax", started=started, fault="syntax_error", detail=str(exc))
    spec = case["verifier"]
    with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
        workdir = Path(os.path.realpath(raw))
        (workdir / "candidate.py").write_text(source, encoding="utf-8")
        runner = """import importlib.util,json,sys
s=importlib.util.spec_from_file_location('candidate','candidate.py')
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
p=json.loads(sys.argv[1])
try:
    if p['mode']=='call':
        args=p['args']
        before=json.loads(json.dumps(args,sort_keys=True))
        value=getattr(m,p['function'])(*args)
        row={'state':'returned','value':value,'args_after':args,'input_unchanged':args==before}
    else:
        row={'state':'returned','retry_limit':m.RETRY_LIMIT,'timeout_ms':m.DEFAULT_TIMEOUT_MS,'lower':m.should_retry(p['lower']),'boundary':m.should_retry(p['boundary'])}
except BaseException as exc:
    row={'state':'raised','exception_type':type(exc).__name__}
print('__THESEUS_RESULT__'+json.dumps(row,sort_keys=True,separators=(',',':')))
"""
        (workdir / "runner.py").write_text(runner, encoding="utf-8")
        timeout = int(config["sandbox"]["timeout_seconds"])
        python = shutil.which("python3") or "/usr/bin/python3"
        executions = []
        if spec["family"] == "repository_edit":
            payload = spec["cases"]
            probe = {
                "mode": "repository",
                "lower": int(payload["new_limit"]) - 1,
                "boundary": int(payload["new_limit"]),
            }
            run, observed = _run_python_probe(python, workdir, probe, timeout)
            executions.append(run)
            passed = bool(run["ok"]) and observed == {
                "state": "returned",
                "retry_limit": int(payload["new_limit"]),
                "timeout_ms": int(payload["timeout_ms"]),
                "lower": True,
                "boundary": False,
            }
        else:
            passed = True
            for hidden in spec["cases"]:
                probe = {"mode": "call", "function": spec["function_name"], "args": hidden["args"]}
                run, observed = _run_python_probe(python, workdir, probe, timeout)
                executions.append(run)
                expected = {
                    "state": "returned",
                    "value": hidden["expected"],
                    "args_after": hidden["args"],
                    "input_unchanged": True,
                }
                if not run["ok"] or observed != expected:
                    passed = False
            invalid_args = {
                "clamp_values": [[1], 2, 1],
                "chunk_values": [[1], 0],
                "parse_duration": ["not a duration"],
            }.get(spec["family"])
            if invalid_args is not None:
                probe = {"mode": "call", "function": spec["function_name"], "args": invalid_args}
                run, observed = _run_python_probe(python, workdir, probe, timeout)
                executions.append(run)
                if not run["ok"] or observed.get("state") != "raised" or observed.get("exception_type") not in {"ValueError", "TypeError"}:
                    passed = False
        fault = None if passed else next((run.get("fault") for run in executions if run.get("fault")), "test_failure")
        return _result(case, passed=passed, stage="input_only_subprocess_tests", started=started, execution=executions, hidden_expected_visible_to_candidate=False, fault=fault)


def _run_python_probe(
    python: str, workdir: Path, payload: dict[str, Any], timeout: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _run_sandboxed(
        [python, "-I", "runner.py", json.dumps(payload, separators=(",", ":"))],
        workdir,
        timeout,
    )
    observed: dict[str, Any] = {}
    marker = "__THESEUS_RESULT__"
    if run["ok"]:
        lines = [line for line in run.get("stdout", "").splitlines() if line.startswith(marker)]
        if len(lines) != 1:
            run = {**run, "ok": False, "fault": "candidate_protocol_violation"}
        else:
            try:
                observed = json.loads(lines[0][len(marker) :])
            except json.JSONDecodeError:
                run = {**run, "ok": False, "fault": "candidate_protocol_invalid_json"}
    return run, observed


def _verify_deno(case: dict[str, Any], source: str, config: dict[str, Any], started: float) -> dict[str, Any]:
    spec = case["verifier"]
    filename = spec["candidate_filename"]
    with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
        workdir = Path(os.path.realpath(raw))
        (workdir / filename).write_text(source, encoding="utf-8")
        runner = """const safeParse=JSON.parse.bind(JSON);
const safeStringify=JSON.stringify.bind(JSON);
const safeLog=console.log.bind(console);
const p=safeParse(Deno.args[0]);
const m=await import(p.module);
let row;
try {
  if (p.mode === 'call') {
    const args=p.args;
    const before=safeParse(safeStringify(args));
    const value=await m[p.function](...args);
    row={state:'returned',value,args_after:args,input_unchanged:safeStringify(args)===safeStringify(before)};
  } else {
    row={state:'returned',retry_limit:m.RETRY_LIMIT,timeout_ms:m.DEFAULT_TIMEOUT_MS,lower:await m.shouldRetry(p.lower),boundary:await m.shouldRetry(p.boundary)};
  }
} catch (error) {
  row={state:'raised',exception_type:error?.constructor?.name || 'Error'};
}
try {
  safeLog('__THESEUS_RESULT__'+safeStringify(row));
} catch (_) {
  safeLog('__THESEUS_RESULT__'+safeStringify({state:'serialization_fault'}));
}
"""
        (workdir / "runner.ts").write_text(runner, encoding="utf-8")
        deno = shutil.which("deno") or "deno"
        timeout = int(config["sandbox"]["timeout_seconds"])
        checks = [
            _run_sandboxed(
                [deno, "check", "--no-config", "--deny-import", filename],
                workdir,
                timeout,
            )
        ]
        if not checks[0]["ok"]:
            return _result(
                case,
                passed=False,
                stage="typecheck_and_input_only_subprocess_tests",
                started=started,
                execution=checks,
                hidden_expected_visible_to_candidate=False,
                fault=checks[0].get("fault", "typecheck_failure"),
            )
        executions: list[dict[str, Any]] = []
        passed = True
        if spec["family"] == "repository_edit":
            payload = spec["cases"]
            probe = {
                "mode": "repository",
                "module": "./" + filename,
                "lower": int(payload["new_limit"]) - 1,
                "boundary": int(payload["new_limit"]),
            }
            run, observed = _run_deno_probe(deno, workdir, probe, timeout)
            executions.append(run)
            passed = bool(run["ok"]) and observed == {
                "state": "returned",
                "retry_limit": int(payload["new_limit"]),
                "timeout_ms": int(payload["timeout_ms"]),
                "lower": True,
                "boundary": False,
            }
        else:
            name = spec["function_name"]
            for hidden in spec["cases"]:
                probe = {
                    "mode": "call",
                    "module": "./" + filename,
                    "function": name,
                    "args": hidden["args"],
                }
                run, observed = _run_deno_probe(deno, workdir, probe, timeout)
                executions.append(run)
                expected = {
                    "state": "returned",
                    "value": hidden["expected"],
                    "args_after": hidden["args"],
                    "input_unchanged": True,
                }
                if not run["ok"] or observed != expected:
                    passed = False
            invalid_args = {
                "clamp_values": [[1], 2, 1],
                "chunk_values": [[1], 0],
                "parse_duration": ["not a duration"],
            }.get(spec["family"])
            if invalid_args is not None:
                probe = {
                    "mode": "call",
                    "module": "./" + filename,
                    "function": name,
                    "args": invalid_args,
                }
                run, observed = _run_deno_probe(deno, workdir, probe, timeout)
                executions.append(run)
                if not run["ok"] or observed.get("state") != "raised" or observed.get("exception_type") not in {"Error", "RangeError", "TypeError"}:
                    passed = False
        checks.extend(executions)
        fault = None if passed else next((run.get("fault") for run in checks if run.get("fault")), "test_failure")
        return _result(
            case,
            passed=passed,
            stage="typecheck_and_input_only_subprocess_tests",
            started=started,
            execution=checks,
            hidden_expected_visible_to_candidate=False,
            fault=fault,
        )


def _run_deno_probe(
    deno: str, workdir: Path, payload: dict[str, Any], timeout: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _run_sandboxed(
        [
            deno,
            "run",
            "--cached-only",
            "--no-config",
            "--no-lock",
            f"--allow-read={workdir}",
            "runner.ts",
            json.dumps(payload, separators=(",", ":")),
        ],
        workdir,
        timeout,
    )
    observed: dict[str, Any] = {}
    marker = "__THESEUS_RESULT__"
    if run["ok"]:
        lines = [line for line in run.get("stdout", "").splitlines() if line.startswith(marker)]
        if len(lines) != 1:
            run = {**run, "ok": False, "fault": "candidate_protocol_violation"}
        else:
            try:
                observed = json.loads(lines[0][len(marker) :])
            except json.JSONDecodeError:
                run = {**run, "ok": False, "fault": "candidate_protocol_invalid_json"}
    return run, observed


def _rust_string(value: str) -> str:
    return f"String::from({json.dumps(value)})"


def _rust_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _rust_string(value)
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "vec![" + ",".join(f"({_rust_string(v['name'])},{_rust_value(v['enabled'])},{v['priority']})" for v in value) + "]"
        return "vec![" + ",".join(_rust_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "vec![" + ",".join(f"({_rust_string(k)},{v})" for k, v in sorted(value.items())) + "]"
    raise TypeError(f"unsupported Rust literal: {type(value)}")


def _rust_assertion(spec: dict[str, Any], row: dict[str, Any]) -> str:
    name, family, args, expected = spec["function_name"], spec["family"], row["args"], row["expected"]
    if family in {"stable_unique", "normalize_slug", "select_active"}:
        call = f"{name}(&{_rust_value(args[0])})" if family != "normalize_slug" else f"{name}({json.dumps(args[0])})"
    elif family == "clamp_values":
        call = f"{name}(&{_rust_value(args[0])}, {args[1]}, {args[2]}).unwrap()"
    elif family == "merge_counts":
        call = f"{name}(&{_rust_value(args[0])}, &{_rust_value(args[1])})"
    elif family == "chunk_values":
        call = f"{name}(&{_rust_value(args[0])}, {args[1]}).unwrap()"
    elif family == "parse_duration":
        call = f"{name}({json.dumps(args[0])}).unwrap()"
    else:
        raise ValueError(f"unsupported Rust family: {family}")
    expected_expr = _rust_value(expected)
    if family == "parse_duration":
        expected_expr += "u64"
    return f"assert_eq!({call}, {expected_expr});"


def _verify_rust(case: dict[str, Any], source: str, config: dict[str, Any], started: float) -> dict[str, Any]:
    spec = case["verifier"]
    with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
        workdir = Path(os.path.realpath(raw))
        (workdir / "src").mkdir()
        (workdir / "src/lib.rs").write_text(source, encoding="utf-8")
        (workdir / "Cargo.toml").write_text('[package]\nname="functional_case"\nversion="0.1.0"\nedition="2021"\n', encoding="utf-8")
        if spec["family"] == "repository_edit":
            p = spec["cases"]
            assertions = [
                f"assert_eq!(RETRY_LIMIT, {int(p['new_limit'])});",
                f"assert_eq!(DEFAULT_TIMEOUT_MS, {int(p['timeout_ms'])});",
                f"assert!(should_retry({int(p['new_limit']) - 1}));",
                f"assert!(!should_retry({int(p['new_limit'])}));",
            ]
        else:
            assertions = [_rust_assertion(spec, row) for row in spec["cases"]]
        tests = "use functional_case::*;\n#[test]\nfn functional() {\n" + "\n".join(assertions) + "\n}\n"
        (workdir / "tests").mkdir()
        (workdir / "tests/functional.rs").write_text(tests, encoding="utf-8")
        cargo = shutil.which("cargo") or "cargo"
        timeout = int(config["sandbox"]["timeout_seconds"])
        checks = []
        for args in ([cargo, "check", "--offline", "--quiet"], [cargo, "clippy", "--offline", "--quiet", "--", "-D", "warnings"], [cargo, "test", "--offline", "--quiet"]):
            run = _run_sandboxed(list(args), workdir, timeout)
            checks.append(run)
            if not run["ok"]:
                break
        passed = len(checks) == 3 and all(run["ok"] for run in checks)
        return _result(case, passed=passed, stage="cargo_check_clippy_test", started=started, execution=checks, fault=None if passed else checks[-1].get("fault", "test_failure"))


class _DOMAudit(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: list[tuple[str, dict[str, str]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attrs.append((tag, {key: "" if value is None else value for key, value in attrs}))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _verify_html(case: dict[str, Any], source: str, config: dict[str, Any], started: float) -> dict[str, Any]:
    spec = case["verifier"]
    audit = _DOMAudit()
    try:
        audit.feed(source)
    except Exception as exc:
        return _result(case, passed=False, stage="parse", started=started, fault="html_parse_error", detail=str(exc))
    lower = source.lower()
    failures: list[str] = []
    if "<script" in lower or re.search(r"\bon[a-z]+\s*=", lower):
        failures.append("javascript_forbidden")
    if re.search(r"(?:src|href)\s*=\s*['\"](?:https?:)?//", lower):
        failures.append("external_resource_forbidden")
    if re.search(r"(?:src|href)\s*=\s*['\"]file:", lower) or re.search(r"@import\b", lower):
        failures.append("local_resource_forbidden")
    for tag in spec["required_tags"]:
        if tag not in audit.tags:
            failures.append(f"missing_tag:{tag}")
    for tag, key, value in spec["required_attrs"]:
        if not any(found_tag == tag and attrs.get(key) == value for found_tag, attrs in audit.attrs):
            failures.append(f"missing_attr:{tag}:{key}:{value}")
    joined_text = " ".join(audit.text)
    for required in spec["required_text"]:
        if required not in joined_text:
            failures.append(f"missing_text:{required}")
    normalized_css = re.sub(r"\s+", " ", lower)
    for required in spec["required_css"]:
        if required.lower() not in normalized_css:
            failures.append(f"missing_css:{required}")
    tidy: dict[str, Any] = {"ok": False, "fault": "not_run"}
    renders: list[dict[str, Any]] = []
    if not failures and CHROME.exists():
        with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
            workdir = Path(os.path.realpath(raw))
            (workdir / "index.html").write_text(source, encoding="utf-8")
            (workdir / "tidy.conf").write_text(
                "new-blocklevel-tags: article, aside, dialog, figcaption, figure, footer, header, main, nav, section\n",
                encoding="utf-8",
            )
            tidy_run = _run_sandboxed(
                [shutil.which("tidy") or "/usr/bin/tidy", "-config", "tidy.conf", "-errors", "-quiet", "index.html"],
                workdir,
                int(config["sandbox"]["timeout_seconds"]),
            )
            tidy = {**tidy_run, "ok": tidy_run.get("returncode") in {0, 1}}
            if not tidy["ok"]:
                failures.append("tidy_parse_error")
            audited = _inject_browser_audit(source)
            (workdir / "audit.html").write_text(audited, encoding="utf-8")
            for width, height, label in ((800, 600, "wide"), (375, 667, "narrow")):
                screenshot = workdir / f"render-{label}.png"
                render = _render_chrome(
                    [str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter", "--disable-crashpad", "--disable-breakpad", "--disable-background-networking", "--no-first-run", "--no-default-browser-check", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=1000", "--dump-dom", f"--user-data-dir={workdir / f'chrome-{label}'}", f"--window-size={width},{height}", f"--screenshot={screenshot}", (workdir / "audit.html").as_uri()],
                    workdir,
                    screenshot,
                    int(config["sandbox"]["timeout_seconds"]),
                )
                audit = _extract_browser_audit(render.get("stdout", ""))
                render["viewport"] = {"width": width, "height": height, "label": label}
                render["browser_audit"] = audit
                render["browser_assertions"] = _browser_assertions(case["task_family"], audit, label)
                if render["ok"]:
                    render["screenshot_sha256"] = hashlib.sha256(screenshot.read_bytes()).hexdigest()
                    render["screenshot_bytes"] = screenshot.stat().st_size
                if not render["ok"] or not all(render["browser_assertions"].values()):
                    failures.append(f"browser_behavior_failure:{label}")
                renders.append(render)
    elif not CHROME.exists():
        failures.append("chrome_unavailable")
    passed = not failures and len(renders) == 2 and all(render["ok"] for render in renders)
    return _result(case, passed=passed, stage="parse_dom_a11y_responsive_render", started=started, failures=failures, tidy=tidy, renders=renders, fault=None if passed else (failures[0] if failures else "render_failure"))


def _inject_browser_audit(source: str) -> str:
    script = r"""<script>
(() => {
  const visible = (node) => { const r=node.getBoundingClientRect(); const s=getComputedStyle(node); return r.width>0 && r.height>0 && s.display!=='none' && s.visibility!=='hidden'; };
  const controls=[...document.querySelectorAll('button,input,select,textarea,a[href]')];
  const inputs=[...document.querySelectorAll('input,select,textarea')];
  const labeled=inputs.filter((node) => node.getAttribute('aria-label') || node.getAttribute('aria-labelledby') || (node.id && document.querySelector(`label[for="${CSS.escape(node.id)}"]`)) || node.closest('label'));
  const buttons=[...document.querySelectorAll('button')];
  const articles=[...document.querySelectorAll('article')].filter(visible);
  const lefts=[...new Set(articles.map((node) => Math.round(node.getBoundingClientRect().left)))];
  const images=[...document.querySelectorAll('img')];
  const root=getComputedStyle(document.documentElement);
  const result={
    visibleTextChars:(document.body.innerText||'').trim().length,
    horizontalOverflow:document.documentElement.scrollWidth>window.innerWidth+2,
    controlCount:controls.filter(visible).length,
    inputCount:inputs.length,
    labeledInputCount:labeled.length,
    namedButtonCount:buttons.filter((node) => (node.innerText||node.getAttribute('aria-label')||'').trim()).length,
    articleCount:articles.length,
    articleColumnCount:lefts.length,
    tableRowCount:document.querySelectorAll('table tr').length,
    tableHeaderCount:document.querySelectorAll('table th').length,
    alertVisible:[...document.querySelectorAll('[role="alert"]')].some(visible),
    dialogVisible:[...document.querySelectorAll('dialog[open],[role="dialog"]')].some(visible),
    imageCount:images.length,
    loadedImageCount:images.filter((node) => node.complete && node.naturalWidth>0).length,
    landmarkCount:document.querySelectorAll('main,nav,header,footer,aside').length,
    themeVariableCount:['--surface','--text','--accent'].filter((name) => root.getPropertyValue(name).trim()).length
  };
  document.documentElement.setAttribute('data-theseus-browser-audit', btoa(JSON.stringify(result)));
})();
</script>"""
    match = list(re.finditer(r"</body\s*>", source, flags=re.IGNORECASE))
    if match:
        index = match[-1].start()
        return source[:index] + script + source[index:]
    return source + script


def _extract_browser_audit(dump: str) -> dict[str, Any]:
    match = re.search(r'data-theseus-browser-audit="([A-Za-z0-9+/=]+)"', dump)
    if not match:
        return {}
    try:
        return json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}


def _browser_assertions(family: str, audit: dict[str, Any], viewport: str) -> dict[str, bool]:
    assertions = {
        "audit_present": bool(audit),
        "visible_content": int(audit.get("visibleTextChars") or 0) > 0,
        "no_horizontal_overflow": audit.get("horizontalOverflow") is False,
    }
    if family == "accessible_form":
        assertions.update(
            inputs_present=int(audit.get("inputCount") or 0) >= 2,
            inputs_labeled=int(audit.get("labeledInputCount") or 0) == int(audit.get("inputCount") or 0),
            named_submit=int(audit.get("namedButtonCount") or 0) >= 1,
        )
    elif family == "landmark_navigation":
        assertions["landmarks_present"] = int(audit.get("landmarkCount") or 0) >= 4
    elif family == "data_table":
        assertions["table_rows_present"] = int(audit.get("tableRowCount") or 0) >= 3
        assertions["table_headers_present"] = int(audit.get("tableHeaderCount") or 0) >= 3
    elif family == "status_alert":
        assertions["visible_alert"] = audit.get("alertVisible") is True
        assertions["named_retry"] = int(audit.get("namedButtonCount") or 0) >= 1
    elif family == "responsive_cards":
        assertions["three_cards_present"] = int(audit.get("articleCount") or 0) >= 3
        assertions["responsive_columns"] = (
            int(audit.get("articleColumnCount") or 0) >= 3
            if viewport == "wide"
            else int(audit.get("articleColumnCount") or 0) == 1
        )
    elif family == "theme_variables":
        assertions["theme_variables_resolve"] = int(audit.get("themeVariableCount") or 0) == 3
    elif family == "modal_dialog":
        assertions["dialog_visible"] = audit.get("dialogVisible") is True
        assertions["dialog_actions_named"] = int(audit.get("namedButtonCount") or 0) >= 2
    elif family == "media_figure":
        assertions["image_present"] = int(audit.get("imageCount") or 0) >= 1
        assertions["image_loaded"] = int(audit.get("loadedImageCount") or 0) >= 1
    return assertions


def score_english_judgments(
    cases: list[dict[str, Any]], candidates: dict[str, str], judgments: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    english = {case["case_id"]: case for case in cases if case["arm_id"] == "english"}
    dimensions = list(config["english_scoring"]["dimensions"])
    minimum = int(config["english_scoring"]["minimum_raters"])
    delta = int(config["english_scoring"]["adjudication_required_score_delta"])
    by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in english}
    faults: list[str] = []
    seen_judgments: set[tuple[str, str, bool]] = set()
    for row in judgments:
        case_id = str(row.get("case_id", ""))
        if case_id not in english:
            faults.append(f"unknown_case:{case_id}")
            continue
        if any(key in row for key in ("model_id", "checkpoint_id", "architecture", "reference_answer")):
            faults.append(f"identity_or_reference_exposed:{case_id}")
        binding = english_candidate_binding(case_id, candidates.get(case_id, ""))
        if row.get("candidate_sha256") != binding["candidate_sha256"]:
            faults.append(f"candidate_binding_mismatch:{case_id}")
            continue
        if row.get("blind_item_id") != binding["blind_item_id"]:
            faults.append(f"blind_item_binding_mismatch:{case_id}")
            continue
        rater_id = str(row.get("rater_id") or "").strip()
        if not rater_id:
            faults.append(f"missing_rater_id:{case_id}")
            continue
        judgment_key = (case_id, rater_id, row.get("adjudicator") is True)
        if judgment_key in seen_judgments:
            faults.append(f"duplicate_judgment:{case_id}:{rater_id}")
            continue
        seen_judgments.add(judgment_key)
        scores = row.get("scores", {})
        if set(scores) != set(dimensions) or any(not isinstance(scores[d], int) or not 0 <= scores[d] <= 4 for d in dimensions):
            faults.append(f"invalid_scores:{case_id}")
            continue
        by_case[case_id].append({**row, "rater_id": rater_id})
    results = []
    pair_values: list[tuple[int, int]] = []
    for case_id, case in english.items():
        rows = by_case[case_id]
        primary_by_rater = {
            str(row["rater_id"]): row for row in rows if row.get("adjudicator") is not True
        }
        if len(primary_by_rater) != minimum:
            faults.append(f"insufficient_raters:{case_id}")
            continue
        first, second = [primary_by_rater[key] for key in sorted(primary_by_rater)]
        requires_adjudication = any(abs(first["scores"][d] - second["scores"][d]) >= delta for d in dimensions)
        adjudicators = [row for row in rows if row.get("adjudicator") is True]
        if requires_adjudication and len(adjudicators) != 1:
            faults.append(f"missing_adjudication:{case_id}")
            continue
        if not requires_adjudication and adjudicators:
            faults.append(f"unexpected_adjudication:{case_id}")
            continue
        if adjudicators and str(adjudicators[0]["rater_id"]) in primary_by_rater:
            faults.append(f"adjudicator_not_independent:{case_id}")
            continue
        adjudicated = adjudicators[0] if adjudicators else None
        final_scores = adjudicated["scores"] if adjudicated else {d: (first["scores"][d] + second["scores"][d]) / 2 for d in dimensions}
        pair_values.extend((first["scores"][d], second["scores"][d]) for d in dimensions)
        required = case["verifier"]["required_concepts"]
        forbidden = case["verifier"]["forbidden_claims"]
        text = candidates.get(case_id, "").lower()
        concept_diagnostic = sum(term.lower() in text for term in required) / max(1, len(required))
        forbidden_hits = [term for term in forbidden if term.lower() in text]
        mean = sum(final_scores.values()) / len(dimensions)
        results.append({"case_id": case_id, "passed": mean >= 3.0 and not forbidden_hits, "mean_score": mean, "scores": final_scores, "required_concept_recall_diagnostic": concept_diagnostic, "forbidden_hits": forbidden_hits})
    kappa = _quadratic_weighted_kappa(pair_values)
    return {"valid": not faults and len(results) == len(english), "faults": faults, "results": results, "quadratic_weighted_kappa": kappa, "passed": sum(bool(row["passed"]) for row in results), "total": len(english)}


def english_candidate_binding(case_id: str, candidate: str) -> dict[str, str]:
    candidate_sha256 = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    blind_item_id = hashlib.sha256(
        json.dumps(
            {
                "policy": "project_theseus_blind_english_item_v1",
                "case_id": case_id,
                "candidate_sha256": candidate_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"candidate_sha256": candidate_sha256, "blind_item_id": blind_item_id}


def _quadratic_weighted_kappa(pairs: list[tuple[int, int]], categories: int = 5) -> float | None:
    if not pairs:
        return None
    observed = [[0 for _ in range(categories)] for _ in range(categories)]
    left = [0] * categories
    right = [0] * categories
    for a, b in pairs:
        observed[a][b] += 1
        left[a] += 1
        right[b] += 1
    total = len(pairs)
    denominator = numerator = 0.0
    for i in range(categories):
        for j in range(categories):
            weight = ((i - j) / (categories - 1)) ** 2
            numerator += weight * observed[i][j] / total
            denominator += weight * (left[i] * right[j]) / (total * total)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - numerator / denominator
