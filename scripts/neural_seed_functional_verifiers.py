#!/usr/bin/env python3
"""Independent functional verifiers for the frozen neural-seed utility suite."""

from __future__ import annotations

import ast
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
        # Chromium creates a trusted process-singleton socket in Darwin's user
        # temp root even when its profile and screenshot are case-local.
        darwin_temp = Path(os.path.realpath(tempfile.gettempdir()))
        write_deny = (
            f'(deny file-write* (require-not (subpath "{canonical}")) '
            f'(require-not (subpath "{darwin_temp}")))'
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
    if not browser:
        common.extend([write_deny, '(allow file-write* (literal "/dev/null"))'])
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
        str(CHROME),
    }
    executable = command[0]
    resolved = shutil.which(executable) if not executable.startswith("/") else executable
    if not resolved or resolved not in allowed_bins:
        return {"ok": False, "fault": "command_not_allowlisted", "command": executable}
    env = {key: value for key, value in os.environ.items() if key in ALLOWED_ENV}
    env["TMPDIR"] = str(workdir)
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
    try:
        while time.monotonic() < deadline:
            if screenshot.exists() and screenshot.stat().st_size >= 512:
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=2)
    rendered = screenshot.exists() and screenshot.stat().st_size >= 512
    return {
        "ok": rendered,
        "returncode": process.returncode,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "fault": None if rendered else "blank_or_missing_screenshot",
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
        "rust": [r"std::(?:net|fs|process)::", r"\b(?:File|TcpStream|UdpSocket|Command)::"],
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
        if spec["family"] == "repository_edit":
            payload = spec["cases"]
            assertions = [
                f"assert m.RETRY_LIMIT == {int(payload['new_limit'])}",
                f"assert m.DEFAULT_TIMEOUT_MS == {int(payload['timeout_ms'])}",
                f"assert m.should_retry({int(payload['new_limit']) - 1}) is True",
                f"assert m.should_retry({int(payload['new_limit'])}) is False",
            ]
        else:
            assertions = [
                f"assert m.{spec['function_name']}(*{repr(row['args'])}) == {repr(row['expected'])}"
                for row in spec["cases"]
            ]
            assertions.extend(_python_negative_assertions(spec))
        harness = (
            "import importlib.util\n"
            "s=importlib.util.spec_from_file_location('candidate','candidate.py')\n"
            "m=importlib.util.module_from_spec(s)\n"
            "s.loader.exec_module(m)\n"
            + "\n".join(assertions)
            + "\nprint('FUNCTIONAL_PASS')\n"
        )
        (workdir / "verify.py").write_text(harness, encoding="utf-8")
        run = _run_sandboxed([shutil.which("python3") or "/usr/bin/python3", "-I", "verify.py"], workdir, int(config["sandbox"]["timeout_seconds"]))
        return _result(case, passed=bool(run["ok"]), stage="execute", started=started, execution=run, fault=None if run["ok"] else run.get("fault", "test_failure"))


def _python_negative_assertions(spec: dict[str, Any]) -> list[str]:
    name = spec["function_name"]
    if spec["family"] == "clamp_values":
        return [f"\ntry:\n m.{name}([1], 2, 1)\n raise AssertionError('must reject')\nexcept (ValueError, TypeError): pass"]
    if spec["family"] == "chunk_values":
        return [f"\ntry:\n m.{name}([1], 0)\n raise AssertionError('must reject')\nexcept (ValueError, TypeError): pass"]
    if spec["family"] == "parse_duration":
        return [f"\ntry:\n m.{name}('not a duration')\n raise AssertionError('must reject')\nexcept (ValueError, TypeError): pass"]
    return []


def _verify_deno(case: dict[str, Any], source: str, config: dict[str, Any], started: float) -> dict[str, Any]:
    spec = case["verifier"]
    filename = spec["candidate_filename"]
    with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
        workdir = Path(os.path.realpath(raw))
        (workdir / filename).write_text(source, encoding="utf-8")
        import_name = "./" + filename
        if spec["family"] == "repository_edit":
            payload = spec["cases"]
            imports = f"import {{ RETRY_LIMIT, DEFAULT_TIMEOUT_MS, shouldRetry }} from {json.dumps(import_name)};"
            body = f"""
if (RETRY_LIMIT !== {int(payload['new_limit'])}) throw new Error('limit');
if (DEFAULT_TIMEOUT_MS !== {int(payload['timeout_ms'])}) throw new Error('timeout');
if (!shouldRetry({int(payload['new_limit']) - 1}) || shouldRetry({int(payload['new_limit'])})) throw new Error('boundary');
"""
        else:
            name = spec["function_name"]
            imports = f"import {{ {name} }} from {json.dumps(import_name)};"
            lines = []
            for row in spec["cases"]:
                lines.append(
                    f"if (JSON.stringify({name}(...{json.dumps(row['args'])})) !== JSON.stringify({json.dumps(row['expected'])})) throw new Error('case');"
                )
            if spec["family"] in {"clamp_values", "chunk_values", "parse_duration"}:
                invalid_args = {
                    "clamp_values": [[1], 2, 1],
                    "chunk_values": [[1], 0],
                    "parse_duration": ["not a duration"],
                }[spec["family"]]
                lines.append(
                    f"let rejected=false; try {{ {name}(...{json.dumps(invalid_args)}); }} catch (_) {{ rejected=true; }} if (!rejected) throw new Error('must reject invalid input');"
                )
            body = "\n".join(lines)
        test_source = f"{imports}\nDeno.test('functional', () => {{\n{body}\n}});\n"
        (workdir / "candidate_test.ts").write_text(test_source, encoding="utf-8")
        deno = shutil.which("deno") or "deno"
        run = _run_sandboxed([deno, "test", "--cached-only", "--no-config", "--no-lock", "candidate_test.ts"], workdir, int(config["sandbox"]["timeout_seconds"]))
        return _result(case, passed=bool(run["ok"]), stage="compile_and_test", started=started, execution=run, fault=None if run["ok"] else run.get("fault", "test_failure"))


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
    render: dict[str, Any] = {"ok": False, "fault": "not_run"}
    if not failures and CHROME.exists():
        with tempfile.TemporaryDirectory(prefix="theseus-fu-") as raw:
            workdir = Path(os.path.realpath(raw))
            (workdir / "index.html").write_text(source, encoding="utf-8")
            screenshot = workdir / "render.png"
            viewport = spec["viewport"]
            render = _render_chrome(
                [str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter", "--disable-breakpad", "--disable-background-networking", "--no-first-run", "--no-default-browser-check", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=1000", f"--user-data-dir={workdir / 'chrome'}", f"--window-size={viewport['width']},{viewport['height']}", f"--screenshot={screenshot}", (workdir / "index.html").as_uri()],
                workdir,
                screenshot,
                int(config["sandbox"]["timeout_seconds"]),
            )
            if render["ok"]:
                if not screenshot.exists() or screenshot.stat().st_size < 512:
                    render = {**render, "ok": False, "fault": "blank_or_missing_screenshot"}
                else:
                    render["screenshot_sha256"] = hashlib.sha256(screenshot.read_bytes()).hexdigest()
                    render["screenshot_bytes"] = screenshot.stat().st_size
    elif not CHROME.exists():
        render = {"ok": False, "fault": "chrome_unavailable"}
    passed = not failures and bool(render["ok"])
    return _result(case, passed=passed, stage="dom_a11y_render", started=started, failures=failures, render=render, fault=None if passed else (failures[0] if failures else render.get("fault", "render_failure")))


def score_english_judgments(
    cases: list[dict[str, Any]], candidates: dict[str, str], judgments: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    english = {case["case_id"]: case for case in cases if case["arm_id"] == "english"}
    dimensions = list(config["english_scoring"]["dimensions"])
    minimum = int(config["english_scoring"]["minimum_raters"])
    delta = int(config["english_scoring"]["adjudication_required_score_delta"])
    by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in english}
    faults: list[str] = []
    for row in judgments:
        case_id = str(row.get("case_id", ""))
        if case_id not in english:
            faults.append(f"unknown_case:{case_id}")
            continue
        if any(key in row for key in ("model_id", "checkpoint_id", "architecture", "reference_answer")):
            faults.append(f"identity_or_reference_exposed:{case_id}")
        scores = row.get("scores", {})
        if set(scores) != set(dimensions) or any(not isinstance(scores[d], int) or not 0 <= scores[d] <= 4 for d in dimensions):
            faults.append(f"invalid_scores:{case_id}")
            continue
        by_case[case_id].append(row)
    results = []
    pair_values: list[tuple[int, int]] = []
    for case_id, case in english.items():
        rows = by_case[case_id]
        raters = {str(row.get("rater_id", "")) for row in rows}
        if len(raters) < minimum:
            faults.append(f"insufficient_raters:{case_id}")
            continue
        first, second = rows[0], rows[1]
        requires_adjudication = any(abs(first["scores"][d] - second["scores"][d]) >= delta for d in dimensions)
        adjudicated = next((row for row in rows if row.get("adjudicator") is True), None)
        if requires_adjudication and adjudicated is None:
            faults.append(f"missing_adjudication:{case_id}")
            continue
        final_scores = adjudicated["scores"] if adjudicated else {d: round((first["scores"][d] + second["scores"][d]) / 2) for d in dimensions}
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
