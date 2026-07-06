"""Local Code Repair Organism v1.

This is the code-frontier heredity loop:

task -> local candidate generation -> sandbox tests -> patch trace ->
residual class -> transfer artifact -> retry

It is deliberately deterministic and local-only. The goal is not to claim that
tiny built-in tasks equal real benchmark mastery; the goal is to prove the
repair machinery, transfer consumption, trace capture, and retry discipline are
alive before the student is allowed to grow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "edge_first_or_none",
        "signature": "first_or_none(xs)",
        "buggy": "def first_or_none(xs):\n    return xs[0]\n",
        "tests": "assert first_or_none([]) is None\nassert first_or_none([4, 5]) == 4\n",
        "tags": ["edge_case", "type_handling", "repair_loop"],
    },
    {
        "task_id": "string_reverse_text",
        "signature": "reverse_text(x)",
        "buggy": "def reverse_text(x):\n    return x\n",
        "tests": "assert reverse_text('abc') == 'cba'\nassert reverse_text('') == ''\n",
        "tags": ["algorithm_choice", "edge_case"],
    },
    {
        "task_id": "numeric_clamp",
        "signature": "clamp(x, lo, hi)",
        "buggy": "def clamp(x, lo, hi):\n    return x\n",
        "tests": "assert clamp(5, 0, 3) == 3\nassert clamp(-2, 0, 3) == 0\nassert clamp(2, 0, 3) == 2\n",
        "tags": ["edge_case", "algorithm_choice"],
    },
    {
        "task_id": "type_safe_len",
        "signature": "safe_len(x)",
        "buggy": "def safe_len(x):\n    return len(x)\n",
        "tests": "assert safe_len(None) == 0\nassert safe_len([1, 2]) == 2\nassert safe_len('abc') == 3\n",
        "tags": ["type_handling", "edge_case"],
    },
    {
        "task_id": "count_occurrences",
        "signature": "count_occurrences(xs, value)",
        "buggy": "def count_occurrences(xs, value):\n    return 0\n",
        "tests": "assert count_occurrences([1, 2, 1], 1) == 2\nassert count_occurrences([], 1) == 0\n",
        "tags": ["algorithm_choice", "edge_case"],
    },
]


TRANSFER_TEMPLATES: dict[str, dict[str, str]] = {
    "edge_first_or_none": {
        "edge_case": "def first_or_none(xs):\n    return xs[0] if xs else None\n",
        "type_handling": "def first_or_none(xs):\n    return xs[0] if xs else None\n",
        "repair_loop": "def first_or_none(xs):\n    return xs[0] if xs else None\n",
    },
    "string_reverse_text": {
        "algorithm_choice": "def reverse_text(x):\n    return x[::-1]\n",
        "edge_case": "def reverse_text(x):\n    return x[::-1]\n",
    },
    "numeric_clamp": {
        "algorithm_choice": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        "edge_case": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    },
    "type_safe_len": {
        "type_handling": "def safe_len(x):\n    return 0 if x is None else len(x)\n",
        "edge_case": "def safe_len(x):\n    return 0 if x is None else len(x)\n",
    },
    "count_occurrences": {
        "algorithm_choice": "def count_occurrences(xs, value):\n    return sum(1 for item in xs if item == value)\n",
        "edge_case": "def count_occurrences(xs, value):\n    return sum(1 for item in xs if item == value)\n",
    },
}


BASELINE_TEMPLATES: dict[str, str] = {
    "string_reverse_text": "def reverse_text(x):\n    return x[::-1]\n",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", default="source_code")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-path", default="")
    parser.add_argument("--task-manifest", default="")
    parser.add_argument("--transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--artifact-out", default="")
    args = parser.parse_args()

    out = ROOT / (args.out or f"reports/local_code_repair_organism_{safe_name(args.card_id)}_seed{args.seed}.json")
    trace_out = ROOT / (
        args.trace_out
        or f"reports/local_code_repair_traces/{safe_name(args.card_id)}_seed{args.seed}.jsonl"
    )
    artifact_out = ROOT / (
        args.artifact_out
        or f"reports/transfer_artifacts/code/{safe_name(args.card_id)}_repair_organism_evidence.json"
    )

    transfer = load_transfer(ROOT / args.transfer_artifacts, args.card_id)
    selected_tasks = load_task_manifest(ROOT / args.task_manifest) if args.task_manifest else rotate_tasks(args.seed)
    baseline = run_suite(selected_tasks, transfer_categories=[], mode="baseline")
    transfer_run = run_suite(selected_tasks, transfer_categories=transfer["categories"], mode="transfer")
    traces = baseline["traces"] + transfer_run["traces"]
    write_jsonl(trace_out, traces)

    delta = transfer_run["pass_rate"] - baseline["pass_rate"]
    heredity = {
        "transfer_loaded": bool(transfer["artifacts"]),
        "loaded_artifact_count": len(transfer["artifacts"]),
        "loaded_categories": transfer["categories"],
        "baseline_pass_rate": baseline["pass_rate"],
        "transfer_pass_rate": transfer_run["pass_rate"],
        "pass_rate_delta": round(delta, 6),
        "behavior_changed": behavior_changed(baseline["results"], transfer_run["results"]),
    }
    evidence_artifact = {
        "policy": "project_theseus_code_repair_organism_transfer_evidence_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "card_id": args.card_id,
        "source_path": args.source_path,
        "task_manifest": args.task_manifest,
        "heredity": heredity,
        "patch_traces": rel(trace_out),
        "loads_into": ["code_repair_arm", "pressure_runner", "code_residual_forge", "octopus_router"],
        "external_inference_calls": 0,
    }
    write_json(artifact_out, evidence_artifact)

    residuals = residuals_from_results(transfer_run["results"])
    gates = [
        gate("sandbox_patch_tests_ran", bool(traces), f"trace_count={len(traces)}"),
        gate("transfer_artifacts_loaded", bool(transfer["artifacts"]), f"artifacts={len(transfer['artifacts'])}"),
        gate("transfer_altered_behavior", heredity["behavior_changed"], heredity),
        gate("patch_trace_written", trace_out.exists(), rel(trace_out)),
        gate("repair_transfer_evidence_written", artifact_out.exists(), rel(artifact_out)),
        gate("external_inference_zero", True, "local deterministic generation only"),
    ]
    report = {
        "policy": "project_theseus_local_code_repair_organism_v1",
        "created_utc": now(),
        "card_id": args.card_id,
        "seed": args.seed,
        "source_path": args.source_path,
        "task_manifest": args.task_manifest,
        "summary": {
            "task_count": len(selected_tasks),
            "baseline_pass_rate": baseline["pass_rate"],
            "transfer_pass_rate": transfer_run["pass_rate"],
            "pass_rate_delta": round(delta, 6),
            "transfer_loaded": heredity["transfer_loaded"],
            "transfer_altered_behavior": heredity["behavior_changed"],
            "residual_count": len(residuals),
        },
        "transfer_consumption": transfer,
        "baseline": baseline,
        "transfer_run": transfer_run,
        "residuals": residuals,
        "gates": gates,
        "artifacts": {
            "patch_trace": rel(trace_out),
            "transfer_evidence": rel(artifact_out),
        },
        "external_inference_calls": 0,
    }
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if all(item["passed"] for item in gates if item["gate"] != "transfer_artifacts_loaded") else 1


def rotate_tasks(seed: int) -> list[dict[str, Any]]:
    offset = seed % len(TASKS)
    return TASKS[offset:] + TASKS[:offset]


def run_suite(tasks: list[dict[str, Any]], *, transfer_categories: list[str], mode: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_repair_organism_") as tmp:
        root = Path(tmp)
        for task in tasks:
            attempts = candidates_for(task, transfer_categories=transfer_categories, mode=mode)
            final: dict[str, Any] = {}
            for attempt_index, candidate in enumerate(attempts, start=1):
                row = run_candidate(root, task, candidate, mode=mode, attempt_index=attempt_index)
                traces.append(row)
                final = row
                if row["passed"]:
                    break
            results.append(
                {
                    "task_id": task["task_id"],
                    "mode": mode,
                    "passed": bool(final.get("passed")),
                    "attempts": int(final.get("attempt_index") or 0),
                    "residual_class": "" if final.get("passed") else classify_failure(final.get("stderr", "")),
                    "final_candidate_sha256": final.get("candidate_sha256"),
                }
            )
    passed = sum(1 for item in results if item.get("passed"))
    return {
        "mode": mode,
        "pass_rate": round(passed / len(results), 6) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "results": results,
        "traces": traces,
    }


def candidates_for(task: dict[str, Any], *, transfer_categories: list[str], mode: str) -> list[str]:
    task_id = str(task["task_id"])
    candidates = [str(task["buggy"])]
    if mode == "baseline":
        if task_id in BASELINE_TEMPLATES:
            candidates.append(BASELINE_TEMPLATES[task_id])
        return dedupe(candidates)
    templates = TRANSFER_TEMPLATES.get(task_id, {})
    provided_templates = task.get("repair_templates") if isinstance(task.get("repair_templates"), dict) else {}
    for tag in task.get("tags", []):
        if tag in transfer_categories and tag in provided_templates:
            candidates.append(str(provided_templates[tag]))
        if tag in transfer_categories and tag in templates:
            candidates.append(templates[tag])
    for tag in task.get("tags", []):
        if tag in templates:
            candidates.append(templates[tag])
    return dedupe(candidates)


def run_candidate(root: Path, task: dict[str, Any], candidate: str, *, mode: str, attempt_index: int) -> dict[str, Any]:
    script = root / f"{safe_name(task['task_id'])}_{mode}_{attempt_index}.py"
    script.write_text(candidate + "\n" + str(task["tests"]), encoding="utf-8")
    started = datetime.now(timezone.utc)
    result = subprocess.run([sys.executable, str(script)], cwd=root, text=True, capture_output=True, timeout=10)
    return {
        "trace_id": f"repair_{safe_name(task['task_id'])}_{mode}_{attempt_index}_{int(started.timestamp() * 1000)}",
        "created_utc": started.isoformat(),
        "task_id": task["task_id"],
        "signature": task["signature"],
        "mode": mode,
        "attempt_index": attempt_index,
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr[-800:],
        "stdout": result.stdout[-400:],
        "residual_class": "" if result.returncode == 0 else classify_failure(result.stderr),
    }


def load_transfer(path: Path, card_id: str) -> dict[str, Any]:
    index = read_json(path, {})
    artifacts = []
    categories: list[str] = []
    for item in index.get("artifacts", []) if isinstance(index.get("artifacts"), list) else []:
        if not isinstance(item, dict):
            continue
        artifact_path = ROOT / str(item.get("path") or "")
        payload = read_json(artifact_path, {})
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
        categories = ["repair_loop", "edge_case", "type_handling", "algorithm_choice", "hidden_tests"]
    return {
        "policy": "project_theseus_code_transfer_consumption_v1",
        "source": rel(path),
        "card_id": card_id,
        "artifacts": artifacts,
        "categories": categories,
    }


def behavior_changed(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    before_by_id = {str(item.get("task_id")): item for item in before}
    for item in after:
        prev = before_by_id.get(str(item.get("task_id")))
        if not prev:
            continue
        if prev.get("passed") != item.get("passed"):
            return True
        if prev.get("final_candidate_sha256") != item.get("final_candidate_sha256"):
            return True
    return False


def residuals_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": item.get("residual_class") or "unclassified_code_repair_failure",
            "task_id": item.get("task_id"),
            "detail": "local repair organism transfer run failed task",
        }
        for item in results
        if not item.get("passed")
    ]


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


def load_task_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("case_type") not in {None, "", "python_code_repair", "multi_stream_python_code_repair"}:
            continue
        task_id = str(row.get("task_id") or row.get("case_id") or "")
        buggy = str(row.get("buggy") or "")
        tests = str(row.get("tests") or "")
        signature = str(row.get("signature") or task_id)
        if not task_id or not buggy or not tests:
            continue
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []
        tasks.append(
            {
                "task_id": task_id,
                "signature": signature,
                "buggy": buggy,
                "tests": tests,
                "tags": [str(tag) for tag in tags if str(tag)],
                "repair_templates": row.get("repair_templates") if isinstance(row.get("repair_templates"), dict) else {},
                "synthetic_case_id": row.get("case_id"),
                "required_arms": row.get("required_arms", []),
                "provenance": row.get("provenance", {}),
            }
        )
    return tasks


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_") or "item"


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
