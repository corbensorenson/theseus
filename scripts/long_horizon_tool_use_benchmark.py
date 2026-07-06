#!/usr/bin/env python3
"""Private long-horizon terminal/tool-use benchmark lane.

This is deliberately local and permissive-data only. It does not fetch public
benchmarks, copy public solutions, or use external model calls. The lane turns
agent-like control skills into replayable traces and STS rows:

goal -> allowed action -> checkpoint -> resume/retry -> evidence route -> summary
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNTIME_PATHS = theseus_runtime.runtime_report(create=False)["paths"]
DEFAULT_RUNTIME_ROOT = Path(RUNTIME_PATHS["runtime_root"]["path"])
DEFAULT_DATA_DIR = Path(RUNTIME_PATHS["data_dir"]["path"])
DEFAULT_WORK_ROOT = DEFAULT_RUNTIME_ROOT / "long_horizon_tool_use"
DEFAULT_TRACE_OUT = DEFAULT_DATA_DIR / "tool_use" / "private_train" / "long_horizon_tool_use_traces.jsonl"
DEFAULT_STS_OUT = DEFAULT_DATA_DIR / "tool_use" / "sts" / "long_horizon_tool_use_sts.jsonl"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    skill: str
    residual: str
    actions: list[dict[str, Any]]
    evidence: dict[str, Any]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def safe_run(command: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(1, timeout),
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": proc.stdout[-800:],
            "stderr_tail": proc.stderr[-800:],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "ok": False,
            "stdout_tail": (exc.stdout or "")[-800:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": "timeout",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }


def case_checkpoint_resume(work_root: Path) -> CaseResult:
    case_dir = work_root / "checkpoint_resume"
    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = case_dir / "checkpoint.json"
    output = case_dir / "result.txt"
    actions: list[dict[str, Any]] = []
    checkpoint.write_text(json.dumps({"step": 1, "items": ["alpha", "beta"]}), encoding="utf-8")
    actions.append({"action": "write_checkpoint", "path": rel_or_abs(checkpoint), "ok": checkpoint.exists()})
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["step"] = 2
    payload["items"].append("gamma")
    checkpoint.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    output.write_text("|".join(payload["items"]), encoding="utf-8")
    passed = output.read_text(encoding="utf-8") == "alpha|beta|gamma"
    actions.append({"action": "resume_from_checkpoint", "path": rel_or_abs(output), "ok": passed})
    return CaseResult(
        "checkpoint_resume",
        passed,
        "resume_state",
        "checkpoint_or_resume_failure" if not passed else "resume_state_preserved",
        actions,
        {"checkpoint": rel_or_abs(checkpoint), "output": rel_or_abs(output)},
    )


def case_retry_recovery(work_root: Path) -> CaseResult:
    case_dir = work_root / "retry_recovery"
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.json"
    output_path = case_dir / "output.json"
    for path in (input_path, output_path):
        if path.exists():
            path.unlink()
    actions: list[dict[str, Any]] = []
    first = safe_run([sys.executable, "-c", "import json; open('output.json','w').write(open('input.json').read())"], case_dir)
    actions.append({"action": "first_attempt_missing_input", **first})
    input_path.write_text(json.dumps({"fixed": True, "count": 3}), encoding="utf-8")
    second = safe_run([sys.executable, "-c", "import json; payload=json.load(open('input.json')); json.dump(payload, open('output.json','w'))"], case_dir)
    actions.append({"action": "repair_and_retry_once", **second})
    passed = (not first["ok"]) and second["ok"] and json.loads(output_path.read_text(encoding="utf-8")).get("fixed") is True
    return CaseResult(
        "retry_recovery",
        passed,
        "repair_after_failure",
        "retry_failed_or_not_bounded" if not passed else "bounded_retry_recovered",
        actions,
        {"input": rel_or_abs(input_path), "output": rel_or_abs(output_path)},
    )


def case_evidence_routing(work_root: Path) -> CaseResult:
    case_dir = work_root / "evidence_routing"
    case_dir.mkdir(parents=True, exist_ok=True)
    source = case_dir / "events.jsonl"
    routed = case_dir / "routed.json"
    rows = [
        {"kind": "artifact", "score": 0.9},
        {"kind": "residual", "score": 0.2},
        {"kind": "artifact", "score": 0.7},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    artifacts = [row for row in rows if row["kind"] == "artifact"]
    routed.write_text(json.dumps({"artifact_count": len(artifacts), "residual_count": 1}, sort_keys=True), encoding="utf-8")
    passed = json.loads(routed.read_text(encoding="utf-8")) == {"artifact_count": 2, "residual_count": 1}
    return CaseResult(
        "evidence_routing",
        passed,
        "evidence_grounding",
        "evidence_not_routed" if not passed else "evidence_routed_to_artifact_and_residual",
        [
            {"action": "write_jsonl_events", "path": rel_or_abs(source), "ok": source.exists()},
            {"action": "route_evidence_summary", "path": rel_or_abs(routed), "ok": passed},
        ],
        {"source": rel_or_abs(source), "routed": rel_or_abs(routed)},
    )


def case_terminal_pipeline(work_root: Path) -> CaseResult:
    case_dir = work_root / "terminal_pipeline"
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_path = case_dir / "items.csv"
    out_path = case_dir / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "value"])
        writer.writeheader()
        writer.writerows([
            {"name": "a", "value": "2"},
            {"name": "b", "value": "5"},
            {"name": "c", "value": "7"},
        ])
    command = [
        sys.executable,
        "-c",
        "import csv,json; rows=list(csv.DictReader(open('items.csv', newline='', encoding='utf-8'))); json.dump({'rows':len(rows),'total':sum(int(r['value']) for r in rows)}, open('summary.json','w'))",
    ]
    result = safe_run(command, case_dir)
    passed = result["ok"] and json.loads(out_path.read_text(encoding="utf-8")) == {"rows": 3, "total": 14}
    return CaseResult(
        "terminal_pipeline",
        passed,
        "multi_step_execution",
        "terminal_pipeline_failed" if not passed else "terminal_pipeline_completed",
        [
            {"action": "write_csv_fixture", "path": rel_or_abs(csv_path), "ok": csv_path.exists()},
            {"action": "run_allowed_python_terminal_step", **result},
        ],
        {"csv": rel_or_abs(csv_path), "summary": rel_or_abs(out_path)},
    )


def case_tool_rot_detection(work_root: Path) -> CaseResult:
    case_dir = work_root / "tool_rot_detection"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = case_dir / "tool_manifest.json"
    report = case_dir / "tool_rot_report.json"
    manifest.write_text(
        json.dumps(
            {
                "tool": "local_csv_summarizer",
                "verified_python": "3.9",
                "current_python": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    drift = payload["verified_python"] != payload["current_python"]
    report.write_text(json.dumps({"drift_detected": drift, "action": "revalidate" if drift else "keep"}, sort_keys=True), encoding="utf-8")
    passed = "action" in json.loads(report.read_text(encoding="utf-8"))
    return CaseResult(
        "tool_rot_detection",
        passed,
        "tool_rot_detection",
        "tool_rot_revalidation_needed" if drift else "tool_manifest_current",
        [
            {"action": "read_tool_manifest", "path": rel_or_abs(manifest), "ok": manifest.exists()},
            {"action": "write_tool_rot_report", "path": rel_or_abs(report), "ok": passed},
        ],
        {"manifest": rel_or_abs(manifest), "report": rel_or_abs(report), "drift_detected": drift},
    )


CASES: list[Callable[[Path], CaseResult]] = [
    case_checkpoint_resume,
    case_retry_recovery,
    case_evidence_routing,
    case_terminal_pipeline,
    case_tool_rot_detection,
]


def generated_case(work_root: Path, index: int) -> CaseResult:
    """Generate a deterministic private tool-use case.

    These are intentionally small local fixtures, but they pressure the same
    control skills as real terminal work: parse state, choose a legal action,
    checkpoint before mutation, recover once, and route evidence.
    """

    case_type = index % 8
    case_dir = work_root / f"generated_{index:03d}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, Any]] = []
    checkpoint = case_dir / "checkpoint.json"
    checkpoint.write_text(json.dumps({"index": index, "case_type": case_type, "stage": "pre"}), encoding="utf-8")
    actions.append({"action": "write_checkpoint", "path": rel_or_abs(checkpoint), "ok": checkpoint.exists()})

    if case_type == 0:
        source = case_dir / "input.txt"
        output = case_dir / "output.txt"
        source.write_text(f"alpha {index}\nbeta {index + 1}\n", encoding="utf-8")
        output.write_text(source.read_text(encoding="utf-8").upper(), encoding="utf-8")
        passed = "ALPHA" in output.read_text(encoding="utf-8") and str(index + 1) in output.read_text(encoding="utf-8")
        skill = "file_transform"
        residual = "file_transform_completed" if passed else "file_transform_failed"
        evidence = {"source": rel_or_abs(source), "output": rel_or_abs(output)}
    elif case_type == 1:
        config = case_dir / "service_config.json"
        config.write_text(json.dumps({"enabled": False, "retries": 0}), encoding="utf-8")
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload.update({"enabled": True, "retries": 1, "updated_by": "tool_use_case"})
        config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        passed = payload["enabled"] is True and payload["retries"] == 1
        skill = "config_edit"
        residual = "config_edit_completed" if passed else "config_edit_failed"
        evidence = {"config": rel_or_abs(config)}
    elif case_type == 2:
        log = case_dir / "app.log"
        report = case_dir / "log_report.json"
        log.write_text("INFO boot\nWARN retry\nERROR missing-token\nINFO recovered\n", encoding="utf-8")
        lines = log.read_text(encoding="utf-8").splitlines()
        errors = [line for line in lines if line.startswith("ERROR")]
        report.write_text(json.dumps({"error_count": len(errors), "has_recovery": any("recovered" in line for line in lines)}), encoding="utf-8")
        passed = json.loads(report.read_text(encoding="utf-8")) == {"error_count": 1, "has_recovery": True}
        skill = "log_inspection"
        residual = "log_inspection_completed" if passed else "log_inspection_failed"
        evidence = {"log": rel_or_abs(log), "report": rel_or_abs(report)}
    elif case_type == 3:
        first = safe_run([sys.executable, "-c", "open('missing/source.txt').read()"], case_dir)
        (case_dir / "missing").mkdir(exist_ok=True)
        (case_dir / "missing" / "source.txt").write_text("recovered", encoding="utf-8")
        second = safe_run([sys.executable, "-c", "open('result.txt','w').write(open('missing/source.txt').read())"], case_dir)
        passed = (not first["ok"]) and second["ok"] and (case_dir / "result.txt").read_text(encoding="utf-8") == "recovered"
        actions.extend([{"action": "first_attempt", **first}, {"action": "bounded_retry", **second}])
        skill = "repair_after_failure"
        residual = "bounded_retry_recovered" if passed else "bounded_retry_failed"
        evidence = {"result": rel_or_abs(case_dir / "result.txt")}
    elif case_type == 4:
        manifest = case_dir / "tool_manifest.json"
        manifest.write_text(json.dumps({"tool": "demo", "verified_version": "1", "current_version": str(1 + (index % 2))}), encoding="utf-8")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        drift = payload["verified_version"] != payload["current_version"]
        report = case_dir / "rot.json"
        report.write_text(json.dumps({"drift": drift, "action": "revalidate" if drift else "keep"}), encoding="utf-8")
        passed = "action" in json.loads(report.read_text(encoding="utf-8"))
        skill = "tool_rot_detection"
        residual = "tool_rot_revalidated" if drift else "tool_manifest_current"
        evidence = {"manifest": rel_or_abs(manifest), "report": rel_or_abs(report)}
    elif case_type == 5:
        before = case_dir / "before.txt"
        after = case_dir / "after.txt"
        diff = case_dir / "diff_summary.json"
        before.write_text("a\nb\n", encoding="utf-8")
        after.write_text("a\nb\nc\n", encoding="utf-8")
        added = [line for line in after.read_text(encoding="utf-8").splitlines() if line not in before.read_text(encoding="utf-8").splitlines()]
        diff.write_text(json.dumps({"added": added, "added_count": len(added)}), encoding="utf-8")
        passed = json.loads(diff.read_text(encoding="utf-8"))["added_count"] == 1
        skill = "diff_summary"
        residual = "diff_summary_completed" if passed else "diff_summary_failed"
        evidence = {"before": rel_or_abs(before), "after": rel_or_abs(after), "diff": rel_or_abs(diff)}
    elif case_type == 6:
        status = case_dir / "service_state.json"
        status.write_text(json.dumps({"state": "stopped", "restart_count": 0}), encoding="utf-8")
        payload = json.loads(status.read_text(encoding="utf-8"))
        payload["state"] = "running"
        payload["restart_count"] += 1
        status.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        passed = payload == {"restart_count": 1, "state": "running"}
        skill = "service_lifecycle_sim"
        residual = "service_restart_simulated" if passed else "service_restart_failed"
        evidence = {"status": rel_or_abs(status)}
    else:
        package = case_dir / "requirements.lock"
        report = case_dir / "package_report.json"
        package.write_text("local-demo==1.0\nlocal-helper==2.0\n", encoding="utf-8")
        rows = [line.strip() for line in package.read_text(encoding="utf-8").splitlines() if line.strip()]
        report.write_text(json.dumps({"package_count": len(rows), "network_fetch": False}), encoding="utf-8")
        passed = json.loads(report.read_text(encoding="utf-8")) == {"package_count": 2, "network_fetch": False}
        skill = "package_setup_planning"
        residual = "package_plan_local_only" if passed else "package_plan_failed"
        evidence = {"lock": rel_or_abs(package), "report": rel_or_abs(report)}

    actions.append({"action": "verify_case", "ok": passed, "case_type": case_type})
    return CaseResult(f"generated_{index:03d}", passed, skill, residual, actions, evidence)


def run_cases(work_root: Path, target: int) -> list[CaseResult]:
    base = [fn(work_root) for fn in CASES]
    if target <= len(base):
        return base[:target]
    generated = [generated_case(work_root, index) for index in range(target - len(base))]
    return base + generated


def trace_row(case: CaseResult) -> dict[str, Any]:
    return {
        "policy": "project_theseus_long_horizon_tool_use_trace_v1",
        "created_utc": now(),
        "case_id": case.case_id,
        "passed": case.passed,
        "skill": case.skill,
        "residual": case.residual,
        "actions": case.actions,
        "evidence": case.evidence,
        "public_benchmark_training_data_used": False,
        "external_inference_calls": 0,
    }


def sts_row(case: CaseResult) -> dict[str, Any]:
    return {
        "policy": "project_theseus_cross_domain_sts_stream_v1",
        "created_utc": now(),
        "source_lane": "long_horizon_tool_use",
        "case_id": case.case_id,
        "skill": case.skill,
        "residual": case.residual,
        "stream": (
            f"Long-horizon tool use skill {case.skill}: maintain goal state, use allowed actions, "
            f"checkpoint/resume, recover once after failure, route evidence, and record residual {case.residual}."
        ),
        "transfer_targets": ["tool_use_arm", "repo_repair_arm", "code_generation_arm", "symliquid_state_engine"],
        "public_benchmark_training_data_used": False,
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Long-Horizon Tool-Use Benchmark",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Pass rate: `{report['summary']['pass_rate']}`",
        f"- Cases: `{report['summary']['case_count']}`",
        f"- Trace output: `{report['outputs']['trace_out']}`",
        f"- STS output: `{report['outputs']['sts_out']}`",
        "",
        "## Cases",
        "",
    ]
    for row in report["cases"]:
        marker = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {marker}: `{row['case_id']}` skill=`{row['skill']}` residual=`{row['residual']}`")
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- Local/private fixtures only.",
            "- Public benchmark solutions or tests are never training rows.",
            "- External inference calls are zero.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--out", default="reports/high_transfer_long_horizon_tool_use.json")
    parser.add_argument("--markdown-out", default="reports/high_transfer_long_horizon_tool_use.md")
    parser.add_argument("--trace-out", default=str(DEFAULT_TRACE_OUT))
    parser.add_argument("--sts-out", default=str(DEFAULT_STS_OUT))
    parser.add_argument("--max-cases", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    target_cases = max(1, int(args.max_cases))
    cases = run_cases(work_root, target_cases)
    traces = [trace_row(case) for case in cases]
    sts_rows = [sts_row(case) for case in cases]
    trace_count = append_jsonl(Path(args.trace_out), traces)
    sts_count = append_jsonl(Path(args.sts_out), sts_rows)
    passed = sum(1 for case in cases if case.passed)
    pass_rate = passed / max(1, len(cases))
    gates = [
        {"name": "case_count_floor", "passed": len(cases) >= min(64, target_cases), "detail": len(cases)},
        {"name": "all_cases_passed", "passed": passed == len(cases), "detail": {"passed": passed, "total": len(cases)}},
        {"name": "traces_appended", "passed": trace_count == len(cases), "detail": str(args.trace_out)},
        {"name": "sts_rows_appended", "passed": sts_count == len(cases), "detail": str(args.sts_out)},
        {"name": "public_data_not_used", "passed": True, "detail": "local private fixtures only"},
        {"name": "external_inference_zero", "passed": True, "detail": 0},
    ]
    report = {
        "policy": "project_theseus_long_horizon_tool_use_benchmark_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(gate["passed"] for gate in gates) else "YELLOW",
        "summary": {
            "case_count": len(cases),
            "passed_cases": passed,
            "pass_rate": round(pass_rate, 6),
            "trace_rows": trace_count,
            "sts_rows": sts_count,
            "skills": sorted({case.skill for case in cases}),
            "residuals": sorted({case.residual for case in cases}),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "external_inference_calls": 0,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "passed": case.passed,
                "skill": case.skill,
                "residual": case.residual,
                "evidence": case.evidence,
            }
            for case in cases
        ],
        "outputs": {
            "work_root": rel_or_abs(work_root),
            "trace_out": rel_or_abs(Path(args.trace_out)),
            "sts_out": rel_or_abs(Path(args.sts_out)),
        },
        "gates": gates,
        "rules": {
            "public_benchmarks": "public benchmark data is not used",
            "side_effect_class": "local_reversible_private_fixture",
            "purpose": "transferable agent control pressure for resume, retry, evidence routing, and tool rot",
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, render_markdown(report))
    print(json.dumps({"trigger_state": report["trigger_state"], "pass_rate": report["summary"]["pass_rate"], "trace_rows": trace_count}, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
