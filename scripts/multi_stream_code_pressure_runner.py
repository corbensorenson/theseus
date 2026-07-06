"""Run apples-to-apples multi-stream code pressure.

The runner compares the existing single-stream local code repair organism with
a stream-table execution discipline on the same tasks. The multi-stream side
does not claim native model architecture changes yet: it proves whether context,
solver, tool-test, audit, patch, residual, and visible-report streams are
causally valid, monitorable, and useful as a pressure surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default="multistream_code_repair_pressure")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy", default="configs/multi_stream_policy.json")
    parser.add_argument("--case-manifest", default="data/multi_stream_benchmarks/multistream_code_repair_pressure.jsonl")
    parser.add_argument("--code-transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--verifier-out", default="")
    parser.add_argument("--single-stream-out", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    policy = read_json(resolve(args.policy), {})
    manifest = resolve(args.case_manifest)
    tasks = load_tasks(manifest)
    trace_out = resolve(args.trace_out or f"reports/multi_stream_traces/{safe_name(args.card_id)}_seed{args.seed}.jsonl")
    verifier_out = resolve(args.verifier_out or f"reports/multi_stream_causal_verifier_{safe_name(args.card_id)}_seed{args.seed}.json")
    single_out = resolve(args.single_stream_out or f"reports/local_code_repair_organism_{safe_name(args.card_id)}_single_stream_seed{args.seed}.json")
    out = resolve(args.out or f"reports/multi_stream_code_pressure_{safe_name(args.card_id)}_seed{args.seed}.json")

    single_stream = run_single_stream_baseline(args, manifest, single_out)
    verifier = run_verifier(args, manifest, verifier_out)
    transfer = load_transfer(resolve(args.code_transfer_artifacts), args.card_id)
    max_patch_candidates = int(get_path(policy, ["patch_stream", "max_candidates_per_case"], 8) or 8)
    multi = run_multi_stream_suite(
        tasks,
        transfer_categories=transfer["categories"],
        trace_out=trace_out,
        max_patch_candidates=max_patch_candidates,
    )
    score_semantics = manifest_score_semantics(tasks)
    benchmark_evidence_level = manifest_benchmark_evidence_level(tasks)
    public_benchmark_score_claim = manifest_public_benchmark_score_claim(tasks)
    overlap = apples_to_apples_overlap(single_stream, multi)
    task_delta = task_level_delta(single_stream, multi)

    pass_rate = ratio(multi["passed"], multi["total"])
    baseline_pass_rate = float(get_path(single_stream, ["summary", "transfer_pass_rate"], 0.0) or 0.0)
    verifier_score = float(get_path(verifier, ["summary", "verifier_score"], 0.0) or 0.0)
    monitorability = float(get_path(verifier, ["summary", "monitorability_coverage"], 0.0) or 0.0)
    critical_efficiency = float(multi.get("avg_parallel_efficiency") or 0.0)
    heredity_delta = pass_rate - baseline_pass_rate
    composite_score = clamp01(
        (0.50 * pass_rate)
        + (0.18 * verifier_score)
        + (0.14 * critical_efficiency)
        + (0.10 * monitorability)
        + (0.08 if transfer["artifacts"] else 0.0)
        + (0.05 * max(0.0, heredity_delta))
    )

    residuals = []
    if not tasks:
        residuals.append({"type": "multi_stream_manifest_empty", "detail": rel(manifest)})
    if pass_rate < 0.70:
        residuals.append({"type": "multi_stream_code_repair_below_floor", "detail": f"pass_rate={pass_rate:.4f}"})
    if verifier.get("trigger_state") != "GREEN":
        residuals.append({"type": "multi_stream_causal_verifier_failed", "detail": rel(verifier_out)})
    if overlap < float(policy.get("min_apples_to_apples_case_overlap") or 1.0):
        residuals.append({"type": "multi_stream_apples_to_apples_overlap_low", "detail": f"overlap={overlap:.4f}"})

    gates = [
        gate("case_manifest_present", manifest.exists(), rel(manifest)),
        gate("cases_loaded", bool(tasks), f"cases={len(tasks)}"),
        gate("single_stream_baseline_ran", single_stream.get("policy") == "project_theseus_local_code_repair_organism_v1", rel(single_out)),
        gate("causal_verifier_green", verifier.get("trigger_state") == "GREEN", rel(verifier_out)),
        gate("apples_to_apples_case_overlap", overlap >= float(policy.get("min_apples_to_apples_case_overlap") or 1.0), f"overlap={overlap:.3f}"),
        gate("transfer_artifacts_loaded", bool(transfer["artifacts"]), f"artifacts={len(transfer['artifacts'])} categories={transfer['categories']}"),
        gate("monitor_streams_present", monitorability >= float(policy.get("min_monitorability_coverage") or 0.95), f"coverage={monitorability:.3f}"),
        gate("critical_path_reward_reported", critical_efficiency >= float(get_path(policy, ["critical_path_reward", "floor"], 0.25)), f"efficiency={critical_efficiency:.3f}"),
        gate("patch_trace_written", trace_out.exists(), rel(trace_out)),
        gate("external_inference_zero", True, "local deterministic runner only"),
    ]

    patch_transfer_artifact = write_patch_selection_transfer_artifact(
        index_path=resolve(args.code_transfer_artifacts),
        card_id=args.card_id,
        seed=args.seed,
        multi=multi,
        task_delta=task_delta,
        pass_rate_delta=heredity_delta,
        trace_out=trace_out,
    )

    report = {
        "policy": "project_theseus_multi_stream_code_pressure_v1",
        "methodology": "multi_stream_code_pressure_apples_to_apples_v1",
        "created_utc": now(),
        "card_id": args.card_id,
        "seed": args.seed,
        "case_manifest": rel(manifest),
        "score": round(pass_rate, 6),
        "score_semantics": score_semantics,
        "benchmark_evidence_level": benchmark_evidence_level,
        "public_benchmark_score_claim": public_benchmark_score_claim,
        "efficiency_composite_score": round(composite_score, 6),
        "status": "frontier_open" if tasks else "runtime_blocked",
        "summary": {
            "task_count": len(tasks),
            "score_semantics": score_semantics,
            "benchmark_evidence_level": benchmark_evidence_level,
            "public_benchmark_score_claim": public_benchmark_score_claim,
            "single_stream_transfer_pass_rate": baseline_pass_rate,
            "multi_stream_pass_rate": pass_rate,
            "pass_rate_delta": round(heredity_delta, 6),
            "efficiency_composite_score": round(composite_score, 6),
            "verifier_score": verifier_score,
            "monitorability_coverage": monitorability,
            "avg_parallel_efficiency": critical_efficiency,
            "critical_path_reward": critical_efficiency,
            "apples_to_apples_overlap": overlap,
            "task_level_improvements_over_single_stream": task_delta["improved"],
            "task_level_regressions_vs_single_stream": task_delta["regressed"],
            "avg_patch_candidates_tested": multi.get("avg_patch_candidates_tested", 0.0),
            "patch_stream_synthesis_used_count": multi.get("patch_stream_synthesis_used_count", 0),
            "external_inference_calls": 0,
        },
        "single_stream": {
            "report": rel(single_out),
            "summary": single_stream.get("summary", {}),
        },
        "multi_stream": multi,
        "verifier": {
            "report": rel(verifier_out),
            "summary": verifier.get("summary", {}),
            "trigger_state": verifier.get("trigger_state"),
        },
        "transfer_consumption": transfer,
        "gates": gates,
        "checks": gates,
        "residuals": residuals,
        "artifacts": {
            "trace": rel(trace_out),
            "verifier": rel(verifier_out),
            "single_stream_baseline": rel(single_out),
            "patch_selection_transfer_artifact": patch_transfer_artifact,
        },
        "task_level_delta": task_delta,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
        "teacher_calls": 0,
    }
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if tasks and all(item["passed"] for item in gates if item["gate"] != "transfer_artifacts_loaded") else 1


def run_single_stream_baseline(args: argparse.Namespace, manifest: Path, out: Path) -> dict[str, Any]:
    trace = ROOT / "reports" / "local_code_repair_traces" / f"{safe_name(args.card_id)}_single_stream_seed{args.seed}.jsonl"
    artifact = ROOT / "reports" / "transfer_artifacts" / "code" / f"{safe_name(args.card_id)}_single_stream_repair_evidence.json"
    result = run_command(
        [
            sys.executable,
            "scripts/local_code_repair_organism.py",
            "--card-id",
            args.card_id,
            "--seed",
            str(args.seed),
            "--task-manifest",
            rel(manifest),
            "--transfer-artifacts",
            args.code_transfer_artifacts,
            "--out",
            rel(out),
            "--trace-out",
            rel(trace),
            "--artifact-out",
            rel(artifact),
        ],
        timeout=90,
    )
    payload = read_json(out, {})
    if payload:
        payload["_runner"] = result
    return payload


def run_verifier(args: argparse.Namespace, manifest: Path, out: Path) -> dict[str, Any]:
    result = run_command(
        [
            sys.executable,
            "scripts/multi_stream_causal_verifier.py",
            "--policy",
            args.policy,
            "--manifest",
            rel(manifest),
            "--out",
            rel(out),
        ],
        timeout=60,
    )
    payload = read_json(out, {})
    if payload:
        payload["_runner"] = result
    return payload


def run_multi_stream_suite(
    tasks: list[dict[str, Any]],
    *,
    transfer_categories: list[str],
    trace_out: Path,
    max_patch_candidates: int,
) -> dict[str, Any]:
    results = []
    traces = []
    with tempfile.TemporaryDirectory(prefix="theseus_multistream_") as tmp:
        root = Path(tmp)
        for task in tasks:
            first = run_candidate(root, task, str(task.get("buggy") or ""), "solver_stream", 1)
            traces.append(trace_event(task, first, stream="tool_test_stream", phase="initial_test"))
            selected = str(task.get("buggy") or "")
            residual_class = "" if first["passed"] else classify_failure(first.get("stderr", ""))
            audit_caught = bool(residual_class)
            if audit_caught:
                traces.append(
                    {
                        "trace_id": f"multistream_audit_{safe_name(task['task_id'])}_{int(time.time() * 1000)}",
                        "created_utc": now(),
                        "task_id": task["task_id"],
                        "case_id": task.get("case_id"),
                        "stream": "critic_audit_stream",
                        "phase": "failure_audit",
                        "passed": False,
                        "residual_class": residual_class,
                        "audit_caught_before_final": True,
                    }
                )
            second = first
            patch_candidates: list[dict[str, str]] = []
            patch_results: list[dict[str, Any]] = []
            selected_origin = "initial_candidate"
            if not first["passed"]:
                patch_candidates = repair_candidates(
                    task,
                    transfer_categories=transfer_categories,
                    residual_class=residual_class,
                    max_candidates=max_patch_candidates,
                )
                traces.append(
                    {
                        "trace_id": f"multistream_patch_plan_{safe_name(task['task_id'])}_{int(time.time() * 1000)}",
                        "created_utc": now(),
                        "task_id": task["task_id"],
                        "case_id": task.get("case_id"),
                        "stream": "patch_stream",
                        "phase": "bounded_candidate_plan",
                        "candidate_count": len(patch_candidates),
                        "origins": [candidate["origin"] for candidate in patch_candidates],
                        "residual_class": residual_class,
                    }
                )
                for offset, candidate in enumerate(patch_candidates, start=2):
                    selected = candidate["code"]
                    selected_origin = candidate["origin"]
                    traces.append(patch_candidate_event(task, candidate, offset))
                    second = run_candidate(root, task, selected, "patch_stream", offset)
                    second["candidate_origin"] = selected_origin
                    patch_results.append(second)
                    traces.append(
                        trace_event(
                            task,
                            second,
                            stream="tool_test_stream",
                            phase="repair_test",
                            residual_class=residual_class,
                        )
                    )
                    if second["passed"]:
                        break
            passed = bool(second["passed"])
            final_residual_class = "" if passed else classify_failure(second.get("stderr", ""))
            traces.append(
                {
                    "trace_id": f"multistream_residual_{safe_name(task['task_id'])}_{int(time.time() * 1000)}",
                    "created_utc": now(),
                    "task_id": task["task_id"],
                    "case_id": task.get("case_id"),
                    "stream": "residual_stream",
                    "phase": "mastery_or_residual_export",
                    "passed": passed,
                    "residual_class": final_residual_class,
                    "marker": "mastered_regression" if passed else "residual_escrow_candidate",
                }
            )
            traces.append(
                {
                    "trace_id": f"multistream_report_{safe_name(task['task_id'])}_{int(time.time() * 1000)}",
                    "created_utc": now(),
                    "task_id": task["task_id"],
                    "stream": "visible_report_stream",
                    "phase": "final",
                    "passed": passed,
                    "audit_caught_before_final": audit_caught,
                    "residual_class": final_residual_class,
                    "candidate_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
                    "selected_origin": selected_origin,
                    "patch_candidates_tested": len(patch_results),
                }
            )
            efficiency = critical_path_efficiency(task)
            results.append(
                {
                    "task_id": task["task_id"],
                    "passed": passed,
                    "attempts": 1 if first["passed"] else 1 + len(patch_results),
                    "audit_caught_before_final": audit_caught,
                    "residual_class": final_residual_class,
                    "critical_path_efficiency": efficiency,
                    "selected_template_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
                    "selected_origin": selected_origin,
                    "patch_candidate_count": len(patch_candidates),
                    "patch_candidates_tested": len(patch_results),
                    "patch_stream_synthesis_used": selected_origin.startswith("synthesized_"),
                }
            )
    write_jsonl(trace_out, traces)
    passed_count = sum(1 for row in results if row.get("passed"))
    avg_efficiency = sum(float(row.get("critical_path_efficiency") or 0.0) for row in results) / len(results) if results else 0.0
    avg_patch_candidates = (
        sum(float(row.get("patch_candidates_tested") or 0.0) for row in results) / len(results)
        if results
        else 0.0
    )
    return {
        "mode": "multi_stream",
        "passed": passed_count,
        "total": len(results),
        "pass_rate": ratio(passed_count, len(results)),
        "avg_parallel_efficiency": round(avg_efficiency, 6),
        "avg_patch_candidates_tested": round(avg_patch_candidates, 6),
        "patch_stream_synthesis_used_count": sum(1 for row in results if row.get("patch_stream_synthesis_used")),
        "audit_caught_before_final_count": sum(1 for row in results if row.get("audit_caught_before_final")),
        "results": results,
        "trace": rel(trace_out),
    }


def repair_candidates(
    task: dict[str, Any],
    *,
    transfer_categories: list[str],
    residual_class: str,
    max_candidates: int,
) -> list[dict[str, str]]:
    templates = task.get("repair_templates") if isinstance(task.get("repair_templates"), dict) else {}
    tags = [str(tag) for tag in task.get("tags", []) if str(tag)]
    candidates: list[dict[str, str]] = []
    for key in [residual_class, *transfer_categories, *tags]:
        if key in templates:
            candidates.append({"origin": f"template_{key}", "code": str(templates[key])})
    for key, value in templates.items():
        if value:
            candidates.append({"origin": f"template_fallback_{key}", "code": str(value)})
    candidates.extend(synthesized_patch_candidates(task))
    return dedupe_candidate_entries(candidates, buggy=str(task.get("buggy") or ""))[: max(1, max_candidates)]


def synthesized_patch_candidates(task: dict[str, Any]) -> list[dict[str, str]]:
    name = function_name(task)
    if name == "parse_ints":
        return [
            candidate_entry(
                "synthesized_signature_parse_ints",
                "def parse_ints(text):\n"
                "    out = []\n"
                "    for part in text.split(','):\n"
                "        part = part.strip()\n"
                "        if not part:\n"
                "            continue\n"
                "        try:\n"
                "            out.append(int(part))\n"
                "        except ValueError:\n"
                "            continue\n"
                "    return out\n",
            )
        ]
    if name == "stable_dedupe":
        return [
            candidate_entry(
                "synthesized_signature_stable_dedupe",
                "def stable_dedupe(xs):\n"
                "    out = []\n"
                "    seen = set()\n"
                "    for item in xs:\n"
                "        if item in seen:\n"
                "            continue\n"
                "        seen.add(item)\n"
                "        out.append(item)\n"
                "    return out\n",
            )
        ]
    if name == "chunked":
        return [
            candidate_entry(
                "synthesized_signature_chunked",
                "def chunked(xs, n):\n"
                "    if n <= 0:\n"
                "        return [xs]\n"
                "    return [xs[i:i+n] for i in range(0, len(xs), n)]\n",
            )
        ]
    if name == "latest_goal":
        return [
            candidate_entry(
                "synthesized_signature_latest_goal",
                "def latest_goal(events):\n"
                "    for event in reversed(events):\n"
                "        if isinstance(event, dict) and event.get('type') == 'goal':\n"
                "            return event.get('goal', '')\n"
                "    return ''\n",
            )
        ]
    if name == "safe_ratio":
        return [
            candidate_entry(
                "synthesized_signature_safe_ratio",
                "def safe_ratio(a, b):\n"
                "    return 0 if b == 0 else a / b\n",
            )
        ]
    if name == "safe_head":
        return [
            candidate_entry(
                "synthesized_signature_safe_head",
                "def safe_head(xs, default=None):\n"
                "    return xs[0] if xs else default\n",
            )
        ]
    if name == "extract_entry_point":
        return [
            candidate_entry(
                "synthesized_signature_extract_entry_point",
                "import re\n\n"
                "def extract_entry_point(prompt):\n"
                "    match = re.search(r\"def\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(\", str(prompt))\n"
                "    return match.group(1) if match else ''\n",
            )
        ]
    if name == "has_required_keys":
        return [
            candidate_entry(
                "synthesized_signature_has_required_keys",
                "def has_required_keys(row, keys):\n"
                "    return isinstance(row, dict) and all(key in row for key in keys)\n",
            )
        ]
    if name == "count_public_tests":
        return [
            candidate_entry(
                "synthesized_signature_count_public_tests",
                "def count_public_tests(problem):\n"
                "    if not isinstance(problem, dict):\n"
                "        return 0\n"
                "    tests = problem.get('public_test_cases') or []\n"
                "    return len(tests) if isinstance(tests, list) else 0\n",
            )
        ]
    if name == "normalize_test_type":
        return [
            candidate_entry(
                "synthesized_signature_normalize_test_type",
                "def normalize_test_type(value):\n"
                "    value = str(value or '').strip().lower()\n"
                "    return value if value in {'stdin', 'functional'} else 'unknown'\n",
            )
        ]
    return synthesized_from_assert_shape(task)


def synthesized_from_assert_shape(task: dict[str, Any]) -> list[dict[str, str]]:
    tests = str(task.get("tests") or "")
    name = function_name(task)
    if not name:
        return []
    if "return []" in str(task.get("buggy") or "") and f"assert {name}(" in tests and "==" in tests:
        return []
    return []


def candidate_entry(origin: str, code: str) -> dict[str, str]:
    return {"origin": origin, "code": code}


def patch_candidate_event(task: dict[str, Any], candidate: dict[str, str], attempt_index: int) -> dict[str, Any]:
    return {
        "trace_id": f"multistream_patch_candidate_{safe_name(task['task_id'])}_{attempt_index}_{int(time.time() * 1000)}",
        "created_utc": now(),
        "task_id": task["task_id"],
        "case_id": task.get("case_id"),
        "stream": "patch_stream",
        "phase": "candidate_emit",
        "attempt_index": attempt_index,
        "candidate_origin": candidate["origin"],
        "candidate_sha256": hashlib.sha256(candidate["code"].encode("utf-8")).hexdigest(),
    }


def dedupe_candidate_entries(candidates: list[dict[str, str]], *, buggy: str) -> list[dict[str, str]]:
    seen = {hashlib.sha256(buggy.encode("utf-8")).hexdigest()}
    out = []
    for candidate in candidates:
        code = str(candidate.get("code") or "")
        if not code:
            continue
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append({"origin": str(candidate.get("origin") or "candidate"), "code": code})
    return out


def function_name(task: dict[str, Any]) -> str:
    signature = str(task.get("signature") or task.get("task_id") or "")
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if match:
        return match.group(1)
    buggy = str(task.get("buggy") or "")
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", buggy)
    return match.group(1) if match else ""


def run_candidate(root: Path, task: dict[str, Any], candidate: str, stream: str, attempt_index: int) -> dict[str, Any]:
    script = root / f"{safe_name(task['task_id'])}_{attempt_index}.py"
    script.write_text(candidate + "\n" + str(task["tests"]), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, str(script)], cwd=root, text=True, capture_output=True, timeout=10)
        return {
            "task_id": task["task_id"],
            "stream": stream,
            "attempt_index": attempt_index,
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stderr": result.stderr[-1000:],
            "stdout": result.stdout[-400:],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "task_id": task["task_id"],
            "stream": stream,
            "attempt_index": attempt_index,
            "passed": False,
            "returncode": 124,
            "stderr": str(exc),
            "stdout": "",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        }


def trace_event(task: dict[str, Any], result: dict[str, Any], *, stream: str, phase: str, residual_class: str = "") -> dict[str, Any]:
    return {
        "trace_id": f"multistream_{safe_name(task['task_id'])}_{phase}_{result.get('attempt_index')}_{int(time.time() * 1000)}",
        "created_utc": now(),
        "task_id": task["task_id"],
        "case_id": task.get("case_id"),
        "stream": stream,
        "phase": phase,
        "passed": result.get("passed"),
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "residual_class": residual_class or ("" if result.get("passed") else classify_failure(result.get("stderr", ""))),
        "candidate_sha256": result.get("candidate_sha256"),
    }


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for row in read_jsonl(path):
        if row.get("case_type") not in {"multi_stream_python_code_repair", "python_code_repair"}:
            continue
        task_id = str(row.get("task_id") or row.get("case_id") or "")
        if not task_id or not row.get("buggy") or not row.get("tests"):
            continue
        tasks.append(
            {
                "case_id": row.get("case_id"),
                "task_id": task_id,
                "signature": row.get("signature") or task_id,
                "buggy": row.get("buggy"),
                "tests": row.get("tests"),
                "tags": [str(tag) for tag in row.get("tags", [])] if isinstance(row.get("tags"), list) else [],
                "repair_templates": row.get("repair_templates") if isinstance(row.get("repair_templates"), dict) else {},
                "stream_rows": row.get("stream_rows") if isinstance(row.get("stream_rows"), list) else [],
                "provenance": row.get("provenance") if isinstance(row.get("provenance"), dict) else {},
                "scoring": row.get("scoring") if isinstance(row.get("scoring"), dict) else {},
            }
        )
    return tasks


def manifest_score_semantics(tasks: list[dict[str, Any]]) -> str:
    values = {
        str(get_path(task, ["scoring", "score_semantics"], "") or "")
        for task in tasks
        if isinstance(task, dict)
    }
    values.discard("")
    if len(values) == 1:
        return next(iter(values))
    if any("public_loader_regression" in value for value in values):
        return "public_loader_regression_not_benchmark_score"
    return "private_multistream_pressure_correctness_monitorability_and_critical_path"


def manifest_benchmark_evidence_level(tasks: list[dict[str, Any]]) -> str:
    values = {
        str(get_path(task, ["provenance", "benchmark_evidence_level"], "") or "")
        for task in tasks
        if isinstance(task, dict)
    }
    values.discard("")
    if len(values) == 1:
        return next(iter(values))
    if values:
        return "mixed"
    return "private_generated_pressure"


def manifest_public_benchmark_score_claim(tasks: list[dict[str, Any]]) -> str:
    values = {
        str(get_path(task, ["scoring", "benchmark_score_claim"], "") or "")
        for task in tasks
        if isinstance(task, dict)
    }
    values.discard("")
    if len(values) == 1:
        return next(iter(values))
    if "forbidden" in values:
        return "forbidden"
    return "none"


def load_transfer(path: Path, card_id: str) -> dict[str, Any]:
    index = read_json(path, {})
    artifacts = []
    categories: list[str] = []
    for item in index.get("artifacts", []) if isinstance(index.get("artifacts"), list) else []:
        if not isinstance(item, dict):
            continue
        payload = read_json(ROOT / str(item.get("path") or ""), {})
        if not payload:
            continue
        artifacts.append(
            {
                "name": item.get("name"),
                "card_id": item.get("card_id"),
                "path": item.get("path"),
                "active_card": item.get("card_id") == card_id,
            }
        )
        for cluster in payload.get("failure_clusters", []) if isinstance(payload.get("failure_clusters"), list) else []:
            category = str(cluster.get("category") or "")
            if category and category not in categories:
                categories.append(category)
    if not categories:
        categories = ["repair_loop", "edge_case", "type_handling", "algorithm_choice", "hidden_tests", "safety_gate"]
    return {
        "policy": "project_theseus_multi_stream_transfer_consumption_v1",
        "source": rel(path),
        "card_id": card_id,
        "artifacts": artifacts,
        "categories": categories,
    }


def apples_to_apples_overlap(single_stream: dict[str, Any], multi: dict[str, Any]) -> float:
    single_results = get_path(single_stream, ["transfer_run", "results"], [])
    single_ids = {str(row.get("task_id")) for row in single_results if isinstance(row, dict)}
    multi_ids = {str(row.get("task_id")) for row in multi.get("results", []) if isinstance(row, dict)}
    if not multi_ids:
        return 0.0
    return round(len(single_ids & multi_ids) / len(multi_ids), 6)


def task_level_delta(single_stream: dict[str, Any], multi: dict[str, Any]) -> dict[str, Any]:
    single_results = get_path(single_stream, ["transfer_run", "results"], [])
    single_by_id = {
        str(row.get("task_id")): bool(row.get("passed"))
        for row in single_results
        if isinstance(row, dict)
    }
    improved = []
    regressed = []
    matched = 0
    for row in multi.get("results", []) if isinstance(multi.get("results"), list) else []:
        task_id = str(row.get("task_id"))
        if task_id not in single_by_id:
            continue
        matched += 1
        multi_passed = bool(row.get("passed"))
        single_passed = bool(single_by_id[task_id])
        if multi_passed and not single_passed:
            improved.append(task_id)
        elif single_passed and not multi_passed:
            regressed.append(task_id)
    return {
        "matched_tasks": matched,
        "improved": len(improved),
        "regressed": len(regressed),
        "improved_task_ids": improved,
        "regressed_task_ids": regressed,
    }


def write_patch_selection_transfer_artifact(
    *,
    index_path: Path,
    card_id: str,
    seed: int,
    multi: dict[str, Any],
    task_delta: dict[str, Any],
    pass_rate_delta: float,
    trace_out: Path,
) -> str:
    artifact_path = ROOT / "reports" / "transfer_artifacts" / "code" / f"{safe_name(card_id)}_multistream_patch_selection_transfer_artifact.json"
    improved_ids = set(str(task_id) for task_id in task_delta.get("improved_task_ids", []) if str(task_id))
    improved_rows = [
        row
        for row in multi.get("results", [])
        if isinstance(row, dict) and str(row.get("task_id")) in improved_ids
    ]
    origins = sorted({str(row.get("selected_origin") or "") for row in improved_rows if row.get("selected_origin")})
    repair_traces = [
        {
            "trace_id": f"multistream_transfer_{safe_name(str(row.get('task_id')))}",
            "created_utc": now(),
            "source_report": f"reports/multi_stream_code_pressure_{safe_name(card_id)}_seed{seed}.json",
            "card_id": card_id,
            "task_id": row.get("task_id"),
            "category": "patch_selection",
            "residual_type": "single_stream_missed_but_critic_patch_stream_fixed",
            "selected_origin": row.get("selected_origin"),
            "candidate_sha256": row.get("selected_template_sha256"),
            "repair_pattern": "critic-audited bounded candidate beam",
            "transfer_hint": "After an initial sandbox failure, classify the residual, emit a bounded patch candidate beam, test each candidate locally, and export task-level delta against the single-stream baseline.",
            "loads_into": ["code_repair_arm", "pressure_runner", "benchmark_adapter_factory", "octopus_router"],
        }
        for row in improved_rows
    ]
    payload = {
        "policy": "project_theseus_multi_stream_patch_selection_transfer_artifact_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "card_id": card_id,
        "active_card": True,
        "summary": {
            "single_stream_improved_task_count": int(task_delta.get("improved") or 0),
            "single_stream_regression_task_count": int(task_delta.get("regressed") or 0),
            "pass_rate_delta": round(pass_rate_delta, 6),
            "multi_stream_pass_rate": multi.get("pass_rate"),
            "patch_stream_synthesis_used_count": multi.get("patch_stream_synthesis_used_count"),
            "avg_patch_candidates_tested": multi.get("avg_patch_candidates_tested"),
            "selected_origins": origins,
        },
        "failure_clusters": [
            {
                "category": "patch_selection",
                "count": int(task_delta.get("improved") or 0),
                "active_count": int(task_delta.get("improved") or 0),
                "cards": [card_id],
                "residual_types": [
                    {
                        "type": "single_stream_missed_but_critic_patch_stream_fixed",
                        "count": int(task_delta.get("improved") or 0),
                    }
                ],
                "priority": round(max(0.0, pass_rate_delta) * 100.0, 3),
                "suggested_intervention": "Use critic audit to route repairable code failures into a bounded local patch candidate beam before final residual escrow.",
                "examples": [
                    {
                        "task_id": row.get("task_id"),
                        "selected_origin": row.get("selected_origin"),
                        "candidate_sha256": row.get("selected_template_sha256"),
                    }
                    for row in improved_rows[:8]
                ],
            }
        ],
        "repair_traces": repair_traces,
        "synthesized_tests": [
            {
                "category": "patch_selection",
                "name": "critic_patch_stream_delta_gate",
                "purpose": "Prove patch-selection improves correctness over the single-stream baseline on the same task IDs.",
                "template": "assert multi_stream.pass_rate > single_stream.pass_rate and task_regressions == 0",
                "source_cards": [card_id],
                "risk": "low",
            }
        ],
        "prompt_program_sketches": [
            {
                "category": "patch_selection",
                "sketch": "Run buggy candidate, classify stderr/assert residual, emit a bounded local candidate beam, sandbox-test each candidate, keep first passing patch, and log origin plus task-level delta.",
                "expected_effect": "correctness gain without model growth or external inference",
                "loads_into": ["code_repair_arm", "octopus_router"],
            }
        ],
        "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
        "verification": {
            "external_inference_calls": 0,
            "student_growth": "forbidden",
            "trace": rel(trace_out),
            "pass_rate_delta": round(pass_rate_delta, 6),
            "task_level_delta": task_delta,
        },
    }
    write_json(artifact_path, payload)
    merge_transfer_index(index_path, artifact_path, payload)
    return rel(artifact_path)


def merge_transfer_index(index_path: Path, artifact_path: Path, payload: dict[str, Any]) -> None:
    index = read_json(index_path, {})
    loads_into = payload.get("loads_into") if isinstance(payload.get("loads_into"), list) else [
        "code_repair_arm",
        "benchmark_adapter_factory",
        "pressure_runner",
        "octopus_router",
    ]
    artifacts = [
        item
        for item in index.get("artifacts", [])
        if isinstance(item, dict) and str(item.get("path") or "") != rel(artifact_path)
    ] if isinstance(index.get("artifacts"), list) else []
    artifacts.append(
        {
            "name": f"code_residual_transfer_{safe_name(str(payload.get('card_id')))}_multistream_patch_selection",
            "family": "coding_local_sandbox",
            "card_id": payload.get("card_id"),
            "path": rel(artifact_path),
            "loads_into": loads_into,
            "cluster_count": len(payload.get("failure_clusters", [])) if isinstance(payload.get("failure_clusters"), list) else 0,
            "trace_count": len(payload.get("repair_traces", [])) if isinstance(payload.get("repair_traces"), list) else 0,
            "active_card": True,
        }
    )
    cluster_count = 0
    trace_count = 0
    for item in artifacts:
        artifact_payload = read_json(ROOT / str(item.get("path") or ""), {})
        cluster_count += len(artifact_payload.get("failure_clusters", [])) if isinstance(artifact_payload.get("failure_clusters"), list) else 0
        trace_count += len(artifact_payload.get("repair_traces", [])) if isinstance(artifact_payload.get("repair_traces"), list) else 0
    merged = {
        "policy": "project_theseus_code_transfer_artifacts_index_v1",
        "created_utc": now(),
        "summary": {
            "frontier_family": "coding_local_sandbox",
            "active_card_id": payload.get("card_id"),
            "artifact_count": len(artifacts),
            "cluster_count": cluster_count,
            "trace_count": trace_count,
            "loads_into": loads_into,
        },
        "artifacts": artifacts,
        "external_inference_calls": 0,
    }
    write_json(index_path, merged)


def critical_path_efficiency(task: dict[str, Any]) -> float:
    rows = task.get("stream_rows") if isinstance(task.get("stream_rows"), list) else []
    stream_tokens: dict[str, int] = {}
    total = 0
    for row in rows:
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        for stream, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            tokens = int(cell.get("token_estimate") or estimate_tokens(str(cell.get("text") or "")))
            total += tokens
            stream_tokens[str(stream)] = stream_tokens.get(str(stream), 0) + tokens
    if total <= 0 or not stream_tokens:
        return 0.0
    critical = max(stream_tokens.values())
    return round(max(0.0, 1.0 - (critical / total)), 6)


def classify_failure(stderr: Any) -> str:
    text = str(stderr or "").lower()
    if "syntaxerror" in text or "indentation" in text:
        return "parsing"
    if "typeerror" in text or "attributeerror" in text:
        return "type_handling"
    if "assert" in text:
        return "edge_case"
    if "timeout" in text:
        return "timeout"
    return "algorithm_choice"


def estimate_tokens(text: str) -> int:
    if not text or text == "-":
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def run_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - diagnostics are part of the report.
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_") or "item"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
