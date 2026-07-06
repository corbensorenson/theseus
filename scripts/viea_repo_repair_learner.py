"""Bridge private repo-repair tasks into the code learner substrate.

This script does not claim public capability. It takes the private
long-horizon repo-repair curriculum, validates expected private patches against
visible + hidden private tests, emits repair traces, and materializes governed
Code LM training rows that code_lm_closure.py can ingest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = Path("D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl")
DEFAULT_CODE_LM_ROWS = Path("D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_code_lm_rows.jsonl")
DEFAULT_TRACE = ROOT / "reports" / "repo_repair_training_traces.jsonl"
DEFAULT_CHECKPOINT = ROOT / "reports" / "repo_repair_trace_checkpoint.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--code-lm-out", default=str(DEFAULT_CODE_LM_ROWS))
    parser.add_argument("--trace-out", default=str(DEFAULT_TRACE.relative_to(ROOT)))
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/viea_repo_repair_learner.json")
    parser.add_argument("--markdown-out", default="reports/viea_repo_repair_learner.md")
    parser.add_argument("--max-tasks", type=int, default=160)
    args = parser.parse_args()

    tasks = read_jsonl(resolve(args.tasks))[: max(1, int(args.max_tasks))]
    traces = []
    code_lm_rows = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        validation = validate_private_patch(task)
        trace = build_trace(task, validation)
        traces.append(trace)
        row = code_lm_row(task, validation)
        if row:
            code_lm_rows.append(row)

    write_jsonl(resolve(args.trace_out), traces)
    write_jsonl(resolve(args.code_lm_out), code_lm_rows)
    checkpoint = build_checkpoint(tasks, traces, code_lm_rows, args=args)
    write_json(resolve(args.checkpoint_out), checkpoint)

    gates = [
        gate("private_repo_tasks_loaded", len(tasks) >= 48, f"tasks={len(tasks)}"),
        gate("private_patch_tests_green", all(row.get("private_tests_passed") for row in traces), failure_summary(traces)),
        gate("code_lm_rows_written", len(code_lm_rows) > 0, len(code_lm_rows)),
        gate("public_benchmark_training_absent", not public_leakage_detected(tasks, code_lm_rows), "no public prompts/tests/solutions copied"),
        gate("checkpoint_written", resolve(args.checkpoint_out).exists(), rel(resolve(args.checkpoint_out))),
    ]
    transfer_evidence_ready = (
        all(row["passed"] for row in gates)
        and sum(1 for row in traces if row.get("private_tests_passed")) >= 128
        and len(code_lm_rows) >= 128
    )
    transfer_consumer_contract = {
        "policy": "project_theseus_repo_repair_transfer_consumer_contract_v1",
        "source_lane": "private_repo_repair",
        "consumers": [
            "code_lm_private_pressure_rows",
            "decoder_v2_private_ablation_gate",
            "private_public_transfer_proof",
            "repo_terminal_repair_policy",
        ],
        "promotion_rule": "private repo traces are transfer evidence only after a downstream decoder/terminal gate consumes them and reports delta",
        "public_benchmark_training": False,
    }
    payload = {
        "policy": "project_theseus_viea_repo_repair_learner_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "task_count": len(tasks),
            "trace_count": len(traces),
            "validated_private_trace_count": sum(1 for row in traces if row.get("private_tests_passed")),
            "code_lm_row_count": len(code_lm_rows),
            "category_counts": dict(Counter(str(row.get("category") or "unknown") for row in tasks)),
            "code_lm_out": rel(resolve(args.code_lm_out)),
            "trace_out": rel(resolve(args.trace_out)),
            "checkpoint_out": rel(resolve(args.checkpoint_out)),
            "promotion_evidence": False,
            "transfer_evidence_ready": transfer_evidence_ready,
            "transfer_consumer_contract": transfer_consumer_contract,
            "public_benchmarks": "calibration_only_not_training",
            "external_inference_calls": 0,
        },
        "loop": [
            "repo_task",
            "expected_private_patch_materialized",
            "visible_private_tests",
            "hidden_private_tests",
            "residual_label",
            "repair_trace",
            "code_lm_private_row",
            "checkpoint_update",
            "public_calibration_only",
        ],
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def validate_private_patch(task: dict[str, Any]) -> dict[str, Any]:
    repo_files = task.get("repo_files") if isinstance(task.get("repo_files"), dict) else {}
    expected = task.get("expected_patch_files") if isinstance(task.get("expected_patch_files"), dict) else {}
    hidden_tests = str(task.get("hidden_tests") or "")
    visible_tests = str(task.get("visible_tests") or "")
    with tempfile.TemporaryDirectory(prefix="theseus_repo_repair_", dir=str(runtime_tmp_dir())) as tmp:
        root = Path(tmp)
        for rel_path, content in {**repo_files, **expected}.items():
            path = root / str(rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        result = run_inline_tests(root, visible_tests, hidden_tests)
    return result


def run_inline_tests(root: Path, visible_tests: str, hidden_tests: str) -> dict[str, Any]:
    harness = root / "_run_private_tests.py"
    harness.write_text(
        "\n".join(
            [
                "import sys",
                "sys.path.insert(0, r'" + str(root).replace("\\", "\\\\") + "')",
                "def run_block(src, label):",
                "    ns = {}",
                "    exec(src, ns, ns)",
                "    for name, value in list(ns.items()):",
                "        if name.startswith('test_') and callable(value):",
                "            value()",
                "visible = " + repr(visible_tests),
                "hidden = " + repr(hidden_tests),
                "run_block(visible, 'visible')",
                "run_block(hidden, 'hidden')",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, str(harness)], cwd=root, text=True, capture_output=True, timeout=30)
    return {
        "private_tests_passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
    }


def build_trace(task: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    residual = "none_after_private_patch" if validation.get("private_tests_passed") else classify_failure(validation.get("stderr_tail", ""))
    return {
        "policy": "project_theseus_repo_repair_training_trace_v1",
        "created_utc": now(),
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "category": task.get("category"),
        "source_group": task.get("source_group"),
        "repo_files": sorted((task.get("repo_files") or {}).keys()),
        "patch_files": sorted((task.get("expected_patch_files") or {}).keys()),
        "private_tests_passed": bool(validation.get("private_tests_passed")),
        "residual_label": residual,
        "repair_rationale": task.get("repair_rationale"),
        "promotion_evidence": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "external_inference_calls": 0,
    }


def code_lm_row(task: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any] | None:
    expected = task.get("expected_patch_files") if isinstance(task.get("expected_patch_files"), dict) else {}
    if not validation.get("private_tests_passed") or not expected:
        return None
    first_path, source = next(iter(expected.items()))
    function = extract_first_function(str(source))
    if not function:
        return None
    entry_point, body = function
    return {
        "task_id": f"repo_repair_{safe_name(str(task.get('task_id') or short_hash(str(task))))}",
        "source_task_id": task.get("task_id"),
        "card_id": "private_repo_repair",
        "source_id": "private_repo_repair_hidden_tests",
        "split": "train",
        "category": str(task.get("category") or "repo_repair"),
        "prompt": repo_prompt(task, first_path),
        "entry_point": entry_point,
        "solution_expr": first_return_expression(body),
        "solution_body": body,
        "tests": "",
        "tags": ["private_repo_repair", "long_horizon_programming", "semantic_residual"],
        "benchmark_evidence_level": "private_residual_generated_train_only",
        "public_benchmark": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": "project_theseus_viea_repo_repair_learner_v1",
            "private_hidden_tests_passed": True,
            "public_benchmark_answers_used": False,
            "source_file": first_path,
        },
    }


def extract_first_function(source: str) -> tuple[str, str] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            lines = source.splitlines()
            body_lines = lines[node.body[0].lineno - 1 : node.end_lineno] if node.body else []
            return node.name, dedent_body(body_lines)
    return None


def dedent_body(lines: list[str]) -> str:
    if not lines:
        return "pass"
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0
    return "\n".join(line[min_indent:] for line in lines).strip() or "pass"


def first_return_expression(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("return "):
            return stripped[len("return ") :].strip()
    return ""


def repo_prompt(task: dict[str, Any], patch_file: str) -> str:
    return (
        f"Patch private repo task {task.get('category')} in {patch_file}. "
        f"Bug: {task.get('bug_summary')}. Return the corrected function body only."
    )


def build_checkpoint(tasks: list[dict[str, Any]], traces: list[dict[str, Any]], rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_repo_repair_trace_checkpoint_v1",
        "created_utc": now(),
        "task_count": len(tasks),
        "validated_trace_count": sum(1 for row in traces if row.get("private_tests_passed")),
        "code_lm_row_count": len(rows),
        "category_counts": dict(Counter(str(row.get("category") or "unknown") for row in tasks)),
        "code_lm_out": rel(resolve(args.code_lm_out)),
        "trace_out": rel(resolve(args.trace_out)),
        "promotion_evidence": False,
        "transfer_evidence_ready": sum(1 for row in traces if row.get("private_tests_passed")) >= 128
        and len(rows) >= 128,
        "transfer_consumer_contract": {
            "consumers": [
                "code_lm_private_pressure_rows",
                "decoder_v2_private_ablation_gate",
                "private_public_transfer_proof",
            ],
            "promotion_rule": "repo repair rows count only after consumed by downstream private transfer gates",
            "public_benchmark_training": False,
        },
        "public_benchmarks": "calibration_only_not_training",
        "external_inference_calls": 0,
    }


def failure_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in traces if not row.get("private_tests_passed")]
    return {
        "failure_count": len(failed),
        "failed_task_ids": [row.get("task_id") for row in failed[:12]],
    }


def public_leakage_detected(tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> bool:
    text = json.dumps({"tasks": tasks, "rows": rows}, sort_keys=True)
    markers = ["HumanEval", "MBPP", "EvalPlus", "BigCodeBench", "LiveCodeBench"]
    return any(marker in text for marker in markers)


def classify_failure(stderr: str) -> str:
    text = str(stderr or "").lower()
    if "assertionerror" in text:
        return "private_assertion_failure"
    if "syntaxerror" in text:
        return "syntax_error"
    if "importerror" in text or "modulenotfounderror" in text:
        return "repo_import_error"
    if "typeerror" in text:
        return "type_error"
    return "private_test_failure"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")[:96]


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def runtime_tmp_dir() -> Path:
    path = ROOT / "reports" / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    return "\n".join(
        [
            "# VIEA Repo-Repair Learner",
            "",
            f"- trigger_state: `{payload.get('trigger_state')}`",
            f"- tasks: `{summary.get('task_count')}`",
            f"- validated_private_traces: `{summary.get('validated_private_trace_count')}`",
            f"- code_lm_rows: `{summary.get('code_lm_row_count')}`",
            f"- code_lm_out: `{summary.get('code_lm_out')}`",
            "",
            "Private hidden-test training pressure only. Public SWE-style benchmarks remain calibration-only.",
            "",
        ]
    )


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
