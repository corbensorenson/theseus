#!/usr/bin/env python3
"""Run the prospectively sealed eight-row dependency-free evaluator segment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_static_evaluator_segment_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_static_evaluator_segment.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = execute(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("report") or "")), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "task_count", "qualified_task_count", "inconclusive_task_count", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def preflight(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != "PROSPECTIVE_K2_05_EIGHT_STATIC_EVALUATORS_BEFORE_EXECUTION":
        faults.append("policy_or_state_invalid")
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    sources: dict[str, dict[str, Any]] = {}
    for binding in p2a.dicts(cfg.get("sources")):
        source_id = str(binding.get("id") or "")
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source_id or not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{source_id}")
            sources[source_id] = {}
        else:
            sources[source_id] = p2a.read_json(source)
    executables: dict[str, str] = {}
    for name, raw in p2a.mapping(cfg.get("executables")).items():
        binding = p2a.mapping(raw)
        executable = p2a.resolve(str(binding.get("path") or ""))
        if not executable.is_file() or p2a.sha256_file(executable) != binding.get("sha256"):
            faults.append(f"executable_binding_invalid:{name}")
        executables[name] = str(executable)
    rows = p2a.dicts(cfg.get("rows"))
    if len(rows) != 8 or len({int(row.get("index") or 0) for row in rows}) != 8:
        faults.append("static_row_denominator_invalid")
    closure_rows = {int(row.get("campaign_index") or 0): row for row in p2a.dicts(sources.get("repository_closures", {}).get("tasks"))}
    runner_rows = {int(row.get("index") or 0): row for row in p2a.dicts(sources.get("runner_inventory", {}).get("rows"))}
    class_rows = {int(row.get("index") or 0): row for row in p2a.dicts(sources.get("dependency_classes", {}).get("rows"))}
    for row in rows:
        index = int(row.get("index") or 0)
        closure, runner, classification = closure_rows.get(index, {}), runner_rows.get(index, {}), class_rows.get(index, {})
        repositories = {str(value.get("repository") or "") for value in (closure, runner, classification)}
        if len(repositories) != 1 or "" in repositories or classification.get("dependency_class") != "LOCK_NOT_REQUIRED_FOR_STATIC_EVALUATOR_CLOSURE":
            faults.append(f"static_source_alignment_invalid:{index}")
        verifier = str(row.get("selected_verifier_path") or "")
        if verifier not in p2a.strings(runner.get("selected_verifier_paths")):
            faults.append(f"selected_verifier_binding_invalid:{index}")
        artifacts = {str(item.get("label") or ""): item for item in p2a.dicts(closure.get("artifacts"))}
        for side in ("parent", "target"):
            artifact = p2a.mapping(artifacts.get(side))
            archive = p2a.resolve(str(artifact.get("normalized") or ""))
            if not archive.is_file() or p2a.sha256_file(archive) != artifact.get("normalized_sha256"):
                faults.append(f"archive_binding_invalid:{index}:{side}")
    return cfg, {"sources": sources, "executables": executables, "closures": closure_rows}, faults


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path)
    if faults:
        return finish(cfg, path, [], faults)
    results: list[dict[str, Any]] = []
    limits = p2a.mapping(cfg.get("limits"))
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-static-", dir="/private/tmp") as temp:
        temp_root = Path(temp).resolve()
        for row in p2a.dicts(cfg.get("rows")):
            index = int(row.get("index") or 0)
            artifacts = {str(item.get("label") or ""): item for item in p2a.dicts(bound["closures"][index].get("artifacts"))}
            sides: dict[str, dict[str, Any]] = {}
            for side in ("parent", "target"):
                work = temp_root / f"task-{index:02d}-{side}"
                archive = p2a.resolve(str(p2a.mapping(artifacts.get(side)).get("normalized") or ""))
                source_root, extract_faults = extract_regular_archive(archive, work)
                faults.extend(f"task_{index}:{side}:{fault}" for fault in extract_faults)
                runtime = str(row.get("runtime") or "")
                executable = str(bound["executables"].get(runtime) or "")
                command = [executable, *p2a.strings(row.get("arguments"))]
                receipt = run(command, source_root, work, limits)
                receipt.update({"archive": p2a.rel(archive), "archive_sha256": p2a.sha256_file(archive), "command": command, "selected_verifier_path": row.get("selected_verifier_path")})
                sides[side] = receipt
            parent_failed = sides["parent"].get("returncode") not in (None, 0) and not sides["parent"].get("boundary_hit")
            target_passed = sides["target"].get("returncode") == 0 and not sides["target"].get("boundary_hit")
            disposition = "QUALIFIED_PARENT_FAIL_TARGET_PASS" if parent_failed and target_passed else "INCONCLUSIVE_EXPERIMENT_STATIC_EVALUATOR_CONSTRUCT"
            results.append({"index": index, "repository": bound["closures"][index].get("repository"), "parent": sides["parent"], "target": sides["target"], "parent_failed": parent_failed, "target_passed": target_passed, "disposition": disposition})
    return finish(cfg, path, results, faults)


def extract_regular_archive(archive: Path, destination: Path) -> tuple[Path, list[str]]:
    faults: list[str] = []
    destination.mkdir(parents=True)
    roots: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                faults.append("unsafe_member_path"); continue
            roots.add(parts[0])
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True); continue
            if not member.isfile():
                faults.append("non_regular_member"); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = handle.extractfile(member)
            if extracted is None:
                faults.append("unreadable_regular_member"); continue
            target.write_bytes(extracted.read())
    if len(roots) != 1:
        faults.append("archive_root_count_invalid")
    return destination / next(iter(roots), ""), sorted(set(faults))


def run(command: list[str], cwd: Path, work_root: Path, limits: dict[str, Any]) -> dict[str, Any]:
    profile = "\n".join(["(version 1)", "(allow default)", "(deny network*)", "(deny mach-lookup)", f'(deny file-write* (require-not (subpath "{work_root}")))', '(allow file-write* (literal "/dev/null"))'])
    stdout_path, stderr_path = work_root / "stdout.bin", work_root / "stderr.bin"
    env = {"HOME": str(work_root), "TMPDIR": str(work_root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(cwd), "NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds"]),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["output_mib"]) * 1024**2,) * 2)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(["/usr/bin/sandbox-exec", "-p", profile, *command], cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=set_limits)
        try:
            returncode = process.wait(timeout=float(limits["wall_seconds"]))
            boundary = ""
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); returncode = process.wait()
            boundary = "wall_boundary_hit"
    stdout_bytes, stderr_bytes = stdout_path.read_bytes(), stderr_path.read_bytes()
    return {"returncode": returncode, "duration_ms": round((time.monotonic() - started) * 1000, 3), "boundary_hit": bool(boundary), "boundary_reason": boundary, "stdout_bytes": len(stdout_bytes), "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(), "stderr_bytes": len(stderr_bytes), "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(), "stdout_complete": True, "stderr_complete": True, "profile_sha256": hashlib.sha256(profile.encode()).hexdigest(), "project_selected_output_cap": None}


def finish(cfg: dict[str, Any], path: Path, rows: list[dict[str, Any]], faults: list[str]) -> dict[str, Any]:
    qualified = sum(row.get("disposition") == "QUALIFIED_PARENT_FAIL_TARGET_PASS" for row in rows)
    mechanically_valid = not faults and len(rows) == int(cfg.get("expected_task_count") or 0)
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if mechanically_valid else "RED", "state": "K2_05_STATIC_SEGMENT_EXECUTED_WITH_SCOPED_DISPOSITIONS" if mechanically_valid else "K2_05_STATIC_SEGMENT_EXECUTION_INVALID", "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "task_count": len(rows), "qualified_task_count": qualified, "inconclusive_task_count": len(rows) - qualified, "rows": rows, "panel_admitted": False, "network_or_dependency_execution_performed": False, "parent_target_or_evaluator_executions": len(rows) * 2, "candidate_or_control_calls": 0, "external_reference_calls": 0, "maximum_inference": cfg.get("maximum_inference")}


if __name__ == "__main__":
    raise SystemExit(main())
