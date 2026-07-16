#!/usr/bin/env python3
"""Source-frozen Rust function-hole discovery and executable verification.

`cargo-mutants` supplies function spans from Rust syntax. Candidate selection is
completed before any tests run. A unit receives credit only when the pinned
package baseline passes, the exact body hole still compiles, and the same test
command rejects that hole.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


VERIFIER_ABI = "project_theseus_rust_test_killed_function_body_v3"
SELECTION_ABI = "content_hash_file_round_robin_v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def public_run_receipt(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": run.get("command", []),
        "returncode": run.get("returncode"),
        "ok": bool(run.get("ok")),
        "timed_out": bool(run.get("timed_out")),
        "duration_ms": int(run.get("duration_ms", 0)),
        "stdout_tail": str(run.get("stdout", ""))[-2000:],
        "stderr_tail": str(run.get("stderr", ""))[-4000:],
        "timeout_termination": run.get("timeout_termination"),
    }


def run_command(
    command: list[str],
    workdir: Path,
    timeout_seconds: int,
    *,
    target_dir: Path | None = None,
    cargo_home: Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TERM_COLOR": "never",
        "RUST_BACKTRACE": "0",
    })
    if target_dir is not None:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    if cargo_home is not None:
        env["CARGO_HOME"] = str(cargo_home)
    if environment:
        env.update(environment)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "command": command,
            "returncode": process.returncode,
            "ok": process.returncode == 0,
            "timed_out": False,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "stdout": stdout,
            "stderr": stderr,
            "timeout_termination": None,
        }
    except subprocess.TimeoutExpired as exc:
        termination = "process_group_sigterm"
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            termination = "process_group_sigkill_after_grace"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        return {
            "command": command,
            "returncode": process.returncode,
            "ok": False,
            "timed_out": True,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "stdout": stdout or _decode_timeout_stream(exc.stdout),
            "stderr": stderr or _decode_timeout_stream(exc.stderr),
            "timeout_termination": termination,
        }


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@contextmanager
def resilient_temporary_directory(
    *, prefix: str, directory: Path | None = None, cleanup_attempts: int = 8
):
    """Remove cargo work trees despite short-lived late filesystem writes."""

    raw = tempfile.mkdtemp(prefix=prefix, dir=directory)
    path = Path(raw)
    try:
        yield raw
    finally:
        last_error: OSError | None = None
        for attempt in range(max(1, cleanup_attempts)):
            try:
                shutil.rmtree(path)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.05 * (2**attempt), 1.0))
        if last_error is not None:
            raise last_error


def ensure_locked_cargo_home(
    source_root: Path,
    source: dict[str, Any],
    toolchain: dict[str, Any],
    *,
    prepare: bool,
) -> dict[str, Any]:
    snapshot = Path(source["cargo_lock_snapshot"])
    if not snapshot.is_absolute():
        snapshot = (Path.cwd() / snapshot).resolve()
    expected = str(source["cargo_lock_sha256"])
    if not snapshot.is_file() or file_sha256(snapshot) != expected:
        raise ValueError(f"Rust lock snapshot identity mismatch: {source['id']}")
    source_lock = source_root / "Cargo.lock"
    if source_lock.is_file() and file_sha256(source_lock) != expected:
        raise ValueError(f"source Cargo.lock differs from frozen snapshot: {source['id']}")
    if not source_lock.is_file():
        shutil.copy2(snapshot, source_lock)

    cargo = Path(shutil.which("cargo") or "cargo").absolute()
    cargo_home_root = Path(toolchain["cargo_home_root"])
    if not cargo_home_root.is_absolute():
        cargo_home_root = (Path.cwd() / cargo_home_root).resolve()
    cargo_home = cargo_home_root / str(source["id"])
    cargo_home.mkdir(parents=True, exist_ok=True)
    fetch_command = [str(cargo), "fetch", "--locked"]
    prepared_with_network = False
    if prepare:
        env = os.environ.copy()
        env.update({"CARGO_HOME": str(cargo_home), "CARGO_TERM_COLOR": "never"})
        completed = subprocess.run(
            fetch_command,
            cwd=source_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(toolchain.get("prepare_timeout_seconds", 1200)),
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Rust dependency preparation failed: {completed.stderr[-3000:]}")
        prepared_with_network = True
    offline = run_command(
        [str(cargo), "metadata", "--offline", "--locked", "--format-version", "1"],
        source_root,
        int(toolchain.get("offline_replay_timeout_seconds", 180)),
        cargo_home=cargo_home,
    )
    if not offline["ok"]:
        raise RuntimeError(
            "Rust dependencies are not offline-replayable; rerun with "
            f"--prepare-rust-toolchain: {offline.get('stderr', '')[-2000:]}"
        )
    inventory_rows = []
    for pattern in ("registry/cache/**/*.crate", "git/checkouts/**/.git/HEAD"):
        for path in sorted(cargo_home.glob(pattern)):
            if path.is_file():
                inventory_rows.append({
                    "path": path.relative_to(cargo_home).as_posix(),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                })
    identity = {
        "cargo_lock_snapshot": str(snapshot),
        "cargo_lock_sha256": expected,
        "cargo_home_inventory_sha256": stable_hash(inventory_rows),
        "cargo_home_artifact_count": len(inventory_rows),
        "cargo_home_artifact_bytes": sum(int(row["bytes"]) for row in inventory_rows),
        "offline_replay_valid": True,
        "network_during_verification": "denied",
    }
    return {
        **identity,
        "cargo_home_path": str(cargo_home),
        "prepared_with_network_this_run": prepared_with_network,
        "offline_replay": public_run_receipt(offline),
    }


def rust_toolchain_identity(
    source_root: Path,
    source: dict[str, Any],
    toolchain: dict[str, Any],
    locked_environment: dict[str, Any],
) -> dict[str, Any]:
    cargo = Path(shutil.which("cargo") or "cargo")
    mutants = Path(toolchain["cargo_mutants_binary"])
    if not mutants.is_absolute():
        mutants = (Path.cwd() / mutants).resolve()
    lockfile = source_root / "Cargo.lock"
    if not cargo.exists() or not mutants.is_file() or not lockfile.is_file():
        missing = [
            name for name, ready in (
                ("cargo", cargo.exists()),
                ("cargo_mutants", mutants.is_file()),
                ("cargo_lock", lockfile.is_file()),
            ) if not ready
        ]
        raise FileNotFoundError(f"missing Rust verifier dependency: {','.join(missing)}")
    cargo_version = subprocess.run(
        [str(cargo), "--version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    mutants_version = subprocess.run(
        [str(mutants), "mutants", "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    identity = {
        "verifier_abi": VERIFIER_ABI,
        # rustup multiplexes cargo/rustc by argv[0]; retain the cargo symlink for
        # invocation while binding the resolved executable bytes separately.
        "cargo_path": str(cargo.absolute()),
        "cargo_resolved_path": str(cargo.resolve()),
        "cargo_sha256": file_sha256(cargo.resolve()),
        "cargo_version": cargo_version,
        "cargo_mutants_path": str(mutants),
        "cargo_mutants_sha256": file_sha256(mutants),
        "cargo_mutants_version": mutants_version,
        "cargo_lock_sha256": file_sha256(lockfile),
        "locked_environment": {
            key: value for key, value in locked_environment.items()
            if key not in {"prepared_with_network_this_run", "offline_replay"}
        },
        "archive_sha256": source["archive_sha256"],
        "network": "denied",
        "verification_runtime_contract": {
            "baseline_timeout_seconds": int(
                toolchain.get("baseline_timeout_seconds", 300)
            ),
            "mutation_test_timeout_seconds": int(
                toolchain.get("test_timeout_seconds", 60)
            ),
            "compile_timeout_seconds": int(
                toolchain.get("compile_timeout_seconds", 120)
            ),
            "verification_parallelism": int(
                toolchain.get("verification_parallelism", 1)
            ),
            "verification_checkpoint_interval": int(
                toolchain.get("verification_checkpoint_interval", 25)
            ),
            "worker_temp_isolation": "unique_os_tmp_outside_project_v1",
            "timeout_termination": "process_group_sigterm_then_sigkill_v1",
        },
        "baseline_command": ["cargo", "test", "--offline", "--locked", "-p", "PACKAGE", "--quiet"],
        "compile_command": ["cargo", "check", "--offline", "--locked", "-p", "PACKAGE", "--quiet"],
    }
    expected = {
        "cargo_mutants_sha256": toolchain.get("cargo_mutants_sha256"),
        "cargo_mutants_version": toolchain.get("cargo_mutants_version"),
        "cargo_lock_sha256": source.get("cargo_lock_sha256"),
    }
    for key, value in expected.items():
        if value and identity[key] != value:
            raise ValueError(f"Rust toolchain identity mismatch for {key}: {identity[key]} != {value}")
    return identity


def discover_function_holes(
    source_root: Path,
    source: dict[str, Any],
    toolchain: dict[str, Any],
    locked_environment: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    identity = rust_toolchain_identity(source_root, source, toolchain, locked_environment)
    mutants = Path(identity["cargo_mutants_path"])
    command = [
        str(mutants), "mutants", "--list", "--json", "--workspace", "--no-config",
    ]
    cargo_features = [str(value) for value in source.get("cargo_features") or []]
    if source.get("cargo_all_features"):
        command.append("--all-features")
    elif cargo_features:
        command.extend(["--features", ",".join(cargo_features)])
    command.extend(["--dir", str(source_root)])
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
        timeout=int(toolchain.get("inventory_timeout_seconds", 180)),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cargo-mutants inventory failed: {completed.stderr[-2000:]}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError("cargo-mutants inventory is not a JSON list")

    include_globs = source.get("source_globs") or ["**/src/**/*.rs", "src/**/*.rs"]
    exclude_globs = source.get("exclude_source_globs") or []
    minimum_bytes = int(source.get("minimum_target_bytes", 256))
    maximum_bytes = int(source.get("maximum_target_bytes", 32768))
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    rejection_counts: defaultdict[str, int] = defaultdict(int)
    for mutant in payload:
        function = mutant.get("function") or {}
        span = function.get("span") or {}
        path_text = str(mutant.get("file") or "")
        if not isinstance(span.get("start"), dict) or not isinstance(span.get("end"), dict):
            rejection_counts["missing_function_span"] += 1
            continue
        key = (
            path_text,
            str(function.get("function_name") or ""),
            _span_key(span),
            str(mutant.get("package") or ""),
        )
        if key in unique:
            continue
        if not path_text or not _matches_any(path_text, include_globs):
            rejection_counts["outside_source_globs"] += 1
            continue
        if _matches_any(path_text, exclude_globs):
            rejection_counts["excluded_source_path"] += 1
            continue
        path = source_root / path_text
        if not path.is_file():
            rejection_counts["source_missing"] += 1
            continue
        try:
            source_text = path.read_text(encoding="utf-8")
            function_start, function_end = span_to_char_range(source_text, span)
            body_start, body_end = find_function_body(source_text, function_start, function_end)
        except (UnicodeDecodeError, ValueError, KeyError):
            rejection_counts["span_or_parse_fault"] += 1
            continue
        target_body = source_text[body_start:body_end]
        target_bytes = len(target_body.encode("utf-8"))
        if not minimum_bytes <= target_bytes <= maximum_bytes:
            rejection_counts["target_size_outside_bounds"] += 1
            continue
        function_name = str(function.get("function_name") or "")
        package = str(mutant.get("package") or "")
        if not function_name or not package:
            rejection_counts["missing_function_or_package"] += 1
            continue
        context = int(source.get("context_characters_each_side", 12000))
        excerpt_start = max(0, function_start - context)
        excerpt_end = min(len(source_text), function_end + context)
        visible_source = (
            source_text[excerpt_start:body_start]
            + '{ unimplemented!("THESEUS_TASK_COMPLETE_FUNCTION_BODY_HOLE") }'
            + source_text[body_end:excerpt_end]
        )
        hole = {
            "path": path_text,
            "package": package,
            "verification_root_package": str(source.get("verification_root_package") or package),
            "cargo_all_features": bool(source.get("cargo_all_features")),
            "cargo_features": cargo_features,
            "test_target_args": [str(value) for value in source.get("test_target_args") or []],
            "function_name": function_name,
            "function_start_char": function_start,
            "function_end_char": function_end,
            "body_start_char": body_start,
            "body_end_char": body_end,
            "body_start_byte": len(source_text[:body_start].encode("utf-8")),
            "body_end_byte": len(source_text[:body_end].encode("utf-8")),
            "target_body": target_body,
            "target_sha256": sha256_text(target_body),
            "target_bytes": target_bytes,
            "source_sha256": sha256_text(source_text),
            "visible_source": visible_source,
            "visible_excerpt_start_char": excerpt_start,
            "visible_excerpt_end_char": excerpt_end,
        }
        hole["candidate_id"] = stable_hash({
            "path": path_text,
            "package": package,
            "function_name": function_name,
            "function_span": [function_start, function_end],
            "body_span": [body_start, body_end],
            "target_sha256": hole["target_sha256"],
        })[:24]
        unique[key] = hole

    holes = sorted(unique.values(), key=_hole_sort_key)
    discovery = {
        "cargo_mutant_count": len(payload),
        "distinct_eligible_function_count": len(holes),
        "eligible_target_bytes": sum(int(row["target_bytes"]) for row in holes),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "inventory_command": command,
        "inventory_stderr_tail": completed.stderr[-2000:],
    }
    return holes, discovery, identity


def _span_key(span: dict[str, Any]) -> tuple[int, int, int, int]:
    start, end = span["start"], span["end"]
    return int(start["line"]), int(start["column"]), int(end["line"]), int(end["column"])


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _hole_sort_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return str(row["path"]), str(row["package"]), int(row["body_start_char"]), str(row["candidate_id"])


def span_to_char_range(text: str, span: dict[str, Any]) -> tuple[int, int]:
    start = _line_byte_column_to_char(text, int(span["start"]["line"]), int(span["start"]["column"]))
    end = _line_byte_column_to_char(text, int(span["end"]["line"]), int(span["end"]["column"]))
    if start < 0 or end <= start or end > len(text):
        raise ValueError("invalid function span")
    return start, end


def _line_byte_column_to_char(text: str, line_number: int, byte_column: int) -> int:
    lines = text.splitlines(keepends=True)
    if line_number < 1 or line_number > len(lines):
        raise ValueError("line outside source")
    line = lines[line_number - 1]
    prefix_bytes = line.encode("utf-8")[: max(0, byte_column - 1)]
    try:
        prefix = prefix_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("column splits UTF-8 scalar") from exc
    return sum(len(value) for value in lines[: line_number - 1]) + len(prefix)


def find_function_body(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the exact outer body range, including braces, for one function span."""

    index = start
    depth = 0
    body_start: int | None = None
    block_comment_depth = 0
    state = "normal"
    raw_hashes = 0
    while index < end:
        char = text[index]
        next_char = text[index + 1] if index + 1 < end else ""
        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if char == "/" and next_char == "*":
                block_comment_depth += 1
                index += 2
            elif char == "*" and next_char == "/":
                block_comment_depth -= 1
                index += 2
                if block_comment_depth == 0:
                    state = "normal"
            else:
                index += 1
            continue
        if state in {"string", "byte_string"}:
            if char == "\\":
                index += 2
            elif char == '"':
                state = "normal"
                index += 1
            else:
                index += 1
            continue
        if state == "char":
            if char == "\\":
                index += 2
            elif char == "'":
                state = "normal"
                index += 1
            else:
                index += 1
            continue
        if state == "raw_string":
            if char == '"' and text.startswith("#" * raw_hashes, index + 1):
                index += raw_hashes + 1
                state = "normal"
            index += 1
            continue

        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            block_comment_depth = 1
            index += 2
            continue
        raw = _raw_string_prefix(text, index, end)
        if raw is not None:
            raw_hashes, consumed = raw
            state = "raw_string"
            index += consumed
            continue
        if char == '"' or (char == "b" and next_char == '"'):
            state = "byte_string" if char == "b" else "string"
            index += 2 if char == "b" else 1
            continue
        if char == "'" and _looks_like_char_literal(text, index, end):
            state = "char"
            index += 1
            continue
        if char == "{":
            if body_start is None:
                body_start = index
                depth = 1
            else:
                depth += 1
        elif char == "}" and body_start is not None:
            depth -= 1
            if depth == 0:
                return body_start, index + 1
        index += 1
    raise ValueError("function body braces not found")


def _raw_string_prefix(text: str, index: int, end: int) -> tuple[int, int] | None:
    cursor = index
    if text.startswith("br", cursor):
        cursor += 2
    elif text.startswith("r", cursor):
        cursor += 1
    else:
        return None
    hashes = 0
    while cursor < end and text[cursor] == "#":
        hashes += 1
        cursor += 1
    if cursor < end and text[cursor] == '"':
        return hashes, cursor - index + 1
    return None


def _looks_like_char_literal(text: str, index: int, end: int) -> bool:
    cursor = index + 1
    if cursor >= end:
        return False
    if text[cursor] == "\\":
        cursor += 2
        if cursor < end and text[cursor] in {"u", "x"}:
            while cursor < end and text[cursor] != "'" and cursor - index < 16:
                cursor += 1
    else:
        cursor += 1
    return cursor < end and text[cursor] == "'"


def select_verification_candidates(source: dict[str, Any], holes: list[dict[str, Any]]) -> dict[str, Any]:
    policy = source["verification_candidate_selection"]
    if policy.get("kind") != SELECTION_ABI or policy.get("selection_uses_verifier_outcomes") is not False:
        raise ValueError("Rust candidate selection must be source-only content-hash round-robin")
    maximum = int(policy["maximum_candidates"])
    per_file = int(policy.get("maximum_candidates_per_file", maximum))
    per_package = int(policy.get("maximum_candidates_per_package", maximum))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hole in holes:
        grouped[str(hole["path"])].append(hole)
    for rows in grouped.values():
        rows.sort(key=lambda row: stable_hash({
            "source_id": source["id"], "candidate_id": row["candidate_id"]
        }))
    paths = sorted(grouped, key=lambda path: stable_hash({"source_id": source["id"], "path": path}))
    selected: list[dict[str, Any]] = []
    selected_per_file: defaultdict[str, int] = defaultdict(int)
    selected_per_package: defaultdict[str, int] = defaultdict(int)
    progress = True
    while len(selected) < maximum and progress:
        progress = False
        for path in paths:
            if len(selected) >= maximum:
                break
            rows = grouped[path]
            if not rows or selected_per_file[path] >= per_file:
                continue
            candidate = rows.pop(0)
            package = str(candidate["package"])
            if selected_per_package[package] >= per_package:
                continue
            selected.append(candidate)
            selected_per_file[path] += 1
            selected_per_package[package] += 1
            progress = True

    inventory_rows = [
        _selection_identity(row) for row in sorted(holes, key=_hole_sort_key)
    ]
    selected_rows = [_selection_identity(row) for row in selected]
    return {
        "selected": selected,
        "public_receipt": {
            "kind": SELECTION_ABI,
            "candidate_count": len(holes),
            "selected_count": len(selected),
            "selected_target_bytes": sum(int(row["target_bytes"]) for row in selected),
            "ordered_inventory_sha256": stable_hash(inventory_rows),
            "selected_inventory_sha256": stable_hash(selected_rows),
            "maximum_candidates": maximum,
            "maximum_candidates_per_file": per_file,
            "maximum_candidates_per_package": per_package,
            "selection_uses_verifier_outcomes": False,
            "selected_package_counts": dict(sorted(selected_per_package.items())),
        },
    }


def _selection_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "path": row["path"],
        "package": row["package"],
        "function_name": row["function_name"],
        "body_span": [row["body_start_char"], row["body_end_char"]],
        "target_sha256": row["target_sha256"],
        "target_bytes": row["target_bytes"],
    }


def verify_selected_holes(
    source_root: Path,
    holes: list[dict[str, Any]],
    toolchain_identity: dict[str, Any],
    toolchain: dict[str, Any],
    *,
    package_completed: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hole in holes:
        by_package[str(hole["package"])].append(hole)
    results: dict[str, dict[str, Any]] = {}
    package_receipts: dict[str, dict[str, Any]] = {}
    # Package groups use independent trees, so source mutation and incremental
    # cargo state cannot race across workers.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parallelism = max(1, int(toolchain.get("verification_parallelism", 2)))
    callback_lock = threading.Lock()

    def persist_completed(batch: dict[str, dict[str, Any]]) -> None:
        if package_completed is None:
            return
        with callback_lock:
            package_completed(batch)

    with ThreadPoolExecutor(max_workers=min(parallelism, max(1, len(by_package)))) as executor:
        futures = {
            executor.submit(
                _verify_package_group,
                source_root,
                package,
                rows,
                toolchain_identity,
                toolchain,
                persist_completed,
            ): package
            for package, rows in by_package.items()
        }
        for future in as_completed(futures):
            package = futures[future]
            package_results, package_receipt = future.result()
            results.update(package_results)
            package_receipts[package] = package_receipt
    return results, {
        "package_count": len(by_package),
        "verification_parallelism": parallelism,
        "package_receipts": package_receipts,
    }


def _verify_package_group(
    source_root: Path,
    package: str,
    holes: list[dict[str, Any]],
    toolchain_identity: dict[str, Any],
    toolchain: dict[str, Any],
    progress_completed: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    timeout = int(toolchain.get("test_timeout_seconds", 240))
    baseline_timeout = int(toolchain.get("baseline_timeout_seconds", max(300, timeout)))
    compile_timeout = int(toolchain.get("compile_timeout_seconds", timeout))
    checkpoint_interval = max(
        1, int(toolchain.get("verification_checkpoint_interval", 25))
    )
    cargo = toolchain_identity["cargo_path"]
    verification_packages = sorted({
        package,
        str(holes[0].get("verification_root_package") or package),
    })
    package_args = [value for name in verification_packages for value in ("-p", name)]
    cargo_features = [str(value) for value in holes[0].get("cargo_features") or []]
    feature_args = ["--all-features"] if holes[0].get("cargo_all_features") else []
    if not feature_args and cargo_features:
        feature_args = ["--features", ",".join(cargo_features)]
    test_target_args = [str(value) for value in holes[0].get("test_target_args") or []]
    test_command = [
        cargo, "test", "--offline", "--locked", *package_args,
        *feature_args, *test_target_args, "--quiet",
    ]
    check_command = [
        cargo, "check", "--offline", "--locked", *package_args, *feature_args, "--quiet",
    ]
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    work_root = Path(toolchain.get("work_root", "runtime/task_complete_workdirs/rust"))
    if not work_root.is_absolute():
        work_root = (Path.cwd() / work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with (
        resilient_temporary_directory(
            prefix=f"{package.replace('/', '_')}-", directory=work_root
        ) as raw,
        # Some upstream tests inspect Git ancestry. Keep their temporary files
        # outside the Theseus worktree while still giving every worker a unique
        # root, otherwise ripgrep workers collide on `ripgrep-tests/<name>/<id>`.
        tempfile.TemporaryDirectory(prefix="theseus-rust-tests-") as worker_tmp_raw,
    ):
        workspace = Path(raw) / "source"
        shutil.copytree(
            source_root,
            workspace,
            ignore=shutil.ignore_patterns("target", ".git", ".jj", ".hg"),
        )
        target_dir = Path(raw) / "target"
        worker_tmp = Path(worker_tmp_raw).resolve()
        worker_environment = {"TMPDIR": str(worker_tmp)}
        cargo_home = Path(toolchain_identity["locked_environment"]["cargo_home_path"])
        baseline = run_command(
            test_command,
            workspace,
            baseline_timeout,
            target_dir=target_dir,
            cargo_home=cargo_home,
            environment=worker_environment,
        )
        baseline_receipt = public_run_receipt(baseline)
        if not baseline["ok"]:
            for hole in holes:
                results[hole["candidate_id"]] = failed_receipt(
                    "package_baseline_failed", toolchain_identity,
                    baseline=baseline_receipt,
                )
            if progress_completed is not None:
                progress_completed(dict(results))
            return results, {
                "baseline": baseline_receipt,
                "final_baseline": None,
                "verified_count": 0,
                "passed_count": 0,
                "checkpoint_write_count": 1 if progress_completed is not None else 0,
                "verification_packages": verification_packages,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }

        pending_checkpoint: dict[str, dict[str, Any]] = {}
        checkpoint_write_count = 0
        final_receipt = baseline_receipt

        def record(candidate_id: str, receipt: dict[str, Any]) -> None:
            results[candidate_id] = receipt
            pending_checkpoint[candidate_id] = receipt

        def flush_checkpoint() -> None:
            nonlocal checkpoint_write_count, final_receipt
            if not pending_checkpoint:
                return
            final_baseline = run_command(
                test_command,
                workspace,
                baseline_timeout,
                target_dir=target_dir,
                cargo_home=cargo_home,
                environment=worker_environment,
            )
            final_receipt = public_run_receipt(final_baseline)
            for candidate_id, receipt in list(pending_checkpoint.items()):
                if not final_baseline["ok"] and receipt.get("state") == "passed":
                    receipt = failed_receipt(
                        "checkpoint_package_baseline_failed",
                        toolchain_identity,
                        baseline=baseline_receipt,
                        final_baseline=final_receipt,
                    )
                    results[candidate_id] = receipt
                    pending_checkpoint[candidate_id] = receipt
                else:
                    receipt["checkpoint_baseline_run"] = final_receipt
            if progress_completed is not None:
                progress_completed(dict(pending_checkpoint))
                checkpoint_write_count += 1
            pending_checkpoint.clear()

        for hole in holes:
            path = workspace / hole["path"]
            original = path.read_text(encoding="utf-8")
            source_hash_before = sha256_text(original)
            if source_hash_before != hole["source_sha256"]:
                record(
                    hole["candidate_id"],
                    failed_receipt(
                        "source_hash_mismatch_before_mutation",
                        toolchain_identity,
                        baseline=baseline_receipt,
                    ),
                )
                if len(pending_checkpoint) >= checkpoint_interval:
                    flush_checkpoint()
                continue
            start, end = int(hole["body_start_char"]), int(hole["body_end_char"])
            if sha256_text(original[start:end]) != hole["target_sha256"]:
                record(
                    hole["candidate_id"],
                    failed_receipt(
                        "target_span_hash_mismatch",
                        toolchain_identity,
                        baseline=baseline_receipt,
                    ),
                )
                if len(pending_checkpoint) >= checkpoint_interval:
                    flush_checkpoint()
                continue
            mutated = (
                original[:start]
                + '{ unimplemented!("THESEUS_TASK_COMPLETE_FUNCTION_BODY_HOLE") }'
                + original[end:]
            )
            path.write_text(mutated, encoding="utf-8")
            try:
                # `cargo test` already compiles the changed package. Run it first
                # so surviving holes do not pay for a redundant `cargo check`.
                # A nonzero test result is ambiguous, so only then run the check
                # command to distinguish a behavioral kill from invalid Rust.
                test_run = run_command(
                    test_command,
                    workspace,
                    timeout,
                    target_dir=target_dir,
                    cargo_home=cargo_home,
                    environment=worker_environment,
                )
                if test_run["ok"]:
                    compile_run = {
                        **test_run,
                        "compile_evidence": "cargo_test_completed",
                    }
                elif not test_run["timed_out"]:
                    compile_run = run_command(
                        check_command,
                        workspace,
                        compile_timeout,
                        target_dir=target_dir,
                        cargo_home=cargo_home,
                        environment=worker_environment,
                    )
                    compile_run["compile_evidence"] = "cargo_check_after_test_failure"
                else:
                    compile_run = {
                        "command": check_command,
                        "returncode": None,
                        "ok": False,
                        "timed_out": True,
                        "duration_ms": 0,
                        "stdout": "",
                        "stderr": "not run because starter tests timed out",
                        "compile_evidence": "not_run_after_test_timeout",
                    }
            finally:
                path.write_text(original, encoding="utf-8")
            restored = sha256_text(path.read_text(encoding="utf-8")) == source_hash_before
            passed = bool(
                compile_run["ok"]
                and not test_run["ok"]
                and not test_run["timed_out"]
                and restored
            )
            reason = "test_suite_killed_body_hole" if passed else _failure_reason(
                compile_run, test_run, restored
            )
            record(hole["candidate_id"], {
                "kind": VERIFIER_ABI,
                "strength": "executable_target_pass_starter_fail",
                "state": "passed" if passed else "failed",
                "reason": reason,
                "target_passed": True,
                "starter_compiled": bool(compile_run["ok"]),
                "starter_test_failed": bool(not test_run["ok"] and not test_run["timed_out"]),
                "starter_compile_evidence": compile_run.get("compile_evidence"),
                "source_restored": restored,
                "baseline_run": baseline_receipt,
                "starter_compile_run": public_run_receipt(compile_run),
                "starter_test_run": public_run_receipt(test_run),
                "toolchain": toolchain_identity,
            })
            if len(pending_checkpoint) >= checkpoint_interval:
                flush_checkpoint()

        flush_checkpoint()
        return results, {
            "baseline": baseline_receipt,
            "final_baseline": final_receipt,
            "verified_count": len(holes),
            "passed_count": sum(row.get("state") == "passed" for row in results.values()),
            "checkpoint_write_count": checkpoint_write_count,
            "checkpoint_interval": checkpoint_interval,
            "verification_packages": verification_packages,
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }


def _failure_reason(compile_run: dict[str, Any], test_run: dict[str, Any], restored: bool) -> str:
    if not restored:
        return "source_restore_failed"
    if test_run.get("timed_out"):
        return "starter_test_timeout"
    if compile_run.get("timed_out"):
        return "starter_compile_timeout"
    if not compile_run.get("ok"):
        return "starter_compile_failed"
    if test_run.get("ok"):
        return "starter_tests_passed"
    return "starter_test_fault"


def failed_receipt(
    reason: str,
    toolchain_identity: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    final_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": VERIFIER_ABI,
        "strength": "executable_target_pass_starter_fail",
        "state": "failed",
        "reason": reason,
        "target_passed": bool(baseline and baseline.get("ok")),
        "starter_compiled": False,
        "starter_test_failed": False,
        "source_restored": reason != "source_restore_failed",
        "baseline_run": baseline,
        "final_baseline_run": final_baseline,
        "toolchain": toolchain_identity,
    }
