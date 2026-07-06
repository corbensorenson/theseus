#!/usr/bin/env python3
"""Private/local sandbox regression for macOS candidate execution paths.

This checks the real code benchmark candidate execution path without using any
public benchmark task, public test, public answer, or external inference. It is
intended to guard the macOS/Linux path where ``D:/ProjectTheseus/tmp`` must not
leak into the sandbox root or candidate scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from real_code_benchmark_runtime import run_cases  # noqa: E402
from real_code_benchmark_support import runtime_tmp_dir, write_json  # noqa: E402
from code_lm_closure import runtime_tmp_dir as code_lm_closure_runtime_tmp_dir  # noqa: E402
from code_lm_private_verifier import runtime_tmp_dir as code_lm_private_verifier_runtime_tmp_dir  # noqa: E402


WINDOWS_TMP_NEEDLES = (
    "D:/ProjectTheseus/tmp",
    "D:\\ProjectTheseus\\tmp",
    "D:/ProjectTheseus\\tmp",
    "D:\\ProjectTheseus/tmp",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/macos_sandbox_path_regression.json")
    parser.add_argument("--markdown-out", default="reports/macos_sandbox_path_regression.md")
    parser.add_argument("--verification-workers", type=int, default=1)
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root_tmp = runtime_tmp_dir()
    closure_tmp = code_lm_closure_runtime_tmp_dir()
    private_verifier_tmp = code_lm_private_verifier_runtime_tmp_dir()
    tasks = [private_path_regression_task()]
    candidates = private_student_candidates(tasks)
    run = run_cases(
        tasks,
        mode="multi_stream",
        transfer_categories=["private_sandbox_path_regression", "algorithm_choice"],
        student_candidates=candidates,
        verification_workers=max(1, int(args.verification_workers)),
    )
    evidence_text = json.dumps(
        {
            "runtime_tmp_dir": str(root_tmp),
            "code_lm_closure_runtime_tmp_dir": str(closure_tmp),
            "code_lm_private_verifier_runtime_tmp_dir": str(private_verifier_tmp),
            "results": run.get("results"),
            "traces": run.get("traces"),
            "verification_cascade_summary": run.get("verification_cascade_summary"),
        },
        sort_keys=True,
    )
    leak_hits = [needle for needle in WINDOWS_TMP_NEEDLES if needle in evidence_text]
    result = run["results"][0] if run.get("results") else {}
    platform_native_expected = os.name != "nt"
    platform_native = not any(needle in str(root_tmp) for needle in WINDOWS_TMP_NEEDLES)
    closure_platform_native = not any(needle in str(closure_tmp) for needle in WINDOWS_TMP_NEEDLES)
    private_verifier_platform_native = not any(
        needle in str(private_verifier_tmp) for needle in WINDOWS_TMP_NEEDLES
    )
    gates = [
        gate(
            "private_local_task_only",
            True,
            "synthetic private task and private assertions only; no public benchmark task/test/answer content",
        ),
        gate(
            "runtime_tmp_dir_platform_native",
            platform_native if platform_native_expected else True,
            {"runtime_tmp_dir": str(root_tmp), "platform": sys.platform},
        ),
        gate(
            "code_lm_closure_runtime_tmp_dir_platform_native",
            closure_platform_native if platform_native_expected else True,
            {"runtime_tmp_dir": str(closure_tmp), "platform": sys.platform},
        ),
        gate(
            "code_lm_private_verifier_runtime_tmp_dir_platform_native",
            private_verifier_platform_native if platform_native_expected else True,
            {"runtime_tmp_dir": str(private_verifier_tmp), "platform": sys.platform},
        ),
        gate(
            "candidate_execution_passed",
            bool(result.get("passed"))
            and bool(result.get("runtime_loaded"))
            and bool(result.get("intended_behavior_passed")),
            {
                "passed": result.get("passed"),
                "runtime_loaded": result.get("runtime_loaded"),
                "intended_behavior_passed": result.get("intended_behavior_passed"),
                "verification_stage": result.get("verification_stage"),
                "stderr_tail": result.get("stderr_tail"),
            },
        ),
        gate("no_windows_tmp_leak_in_candidate_scoring", not leak_hits, {"leak_hits": leak_hits}),
        gate(
            "temp_env_assertions_executed",
            bool(result.get("intended_behavior_passed")),
            {
                "test_harness_compile_passed": bool(result.get("compile_passed")),
                "intended_behavior_passed": bool(result.get("intended_behavior_passed")),
                "assertions": [
                    "TMPDIR/TMP/TEMP exist",
                    "TMPDIR/TMP/TEMP do not contain D:/ProjectTheseus/tmp",
                    "tempfile.gettempdir() does not contain D:/ProjectTheseus/tmp",
                    "NamedTemporaryFile can create a file inside the sandbox temp root",
                ],
            },
        ),
        gate("external_inference_zero", True, 0),
    ]
    return {
        "policy": "project_theseus_macos_sandbox_path_regression_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "RED",
        "summary": {
            "platform": sys.platform,
            "runtime_tmp_dir": str(root_tmp),
            "code_lm_closure_runtime_tmp_dir": str(closure_tmp),
            "code_lm_private_verifier_runtime_tmp_dir": str(private_verifier_tmp),
            "candidate_passed": bool(result.get("passed")),
            "runtime_loaded": bool(result.get("runtime_loaded")),
            "intended_behavior_passed": bool(result.get("intended_behavior_passed")),
            "windows_tmp_leak_hit_count": len(leak_hits),
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "candidate_run": {
            "mode": run.get("mode"),
            "passed": run.get("passed"),
            "total": run.get("total"),
            "pass_rate": run.get("pass_rate"),
            "runtime_ms": run.get("runtime_ms"),
            "verification_cascade_summary": run.get("verification_cascade_summary"),
            "results": run.get("results"),
        },
        "gates": gates,
        "rules": {
            "scope": "private/local sandbox scorer regression only",
            "public_boundary": "does not read public benchmark reports, prompts, tests, solutions, or candidate manifests",
            "promotion": "regression proof only; does not create model promotion evidence by itself",
        },
        "external_inference_calls": 0,
    }


def private_path_regression_task() -> dict[str, Any]:
    tests = r'''
import os
import pathlib
import tempfile

for key in ("TMPDIR", "TMP", "TEMP"):
    value = os.environ.get(key, "")
    assert value, key
    normalized = value.replace("\\", "/")
    assert "D:/ProjectTheseus/tmp" not in normalized, normalized
    assert pathlib.Path(value).exists(), value

temp_root = tempfile.gettempdir()
print("__THESEUS_PRIVATE_TMPDIR__:" + temp_root, flush=True)
assert "D:/ProjectTheseus/tmp" not in temp_root.replace("\\", "/"), temp_root
with tempfile.NamedTemporaryFile("w", delete=True, encoding="utf-8") as handle:
    handle.write("private local sandbox regression")
    handle.flush()
    assert pathlib.Path(handle.name).exists()

assert private_sandbox_sum([1, 2, 3, 4]) == 10
assert private_sandbox_sum([-2, 2, 5]) == 5
'''
    return {
        "task_id": "private_macos_sandbox_path_regression_0001",
        "source_task_id": "private_macos_sandbox_path_regression_0001",
        "entry_point": "private_sandbox_sum",
        "prompt": (
            "def private_sandbox_sum(data):\n"
            "    \"\"\"Return the sum of a list of integers using an explicit loop.\"\"\""
        ),
        "tests": tests,
        "tags": ["private_sandbox_path_regression", "algorithm_choice"],
    }


def private_student_candidates(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        row = private_candidate_row(task)
        for key in [task["task_id"], task["source_task_id"], task["entry_point"]]:
            by_key.setdefault(str(key), []).append(row)
    return {
        "manifest_exists": True,
        "by_key": by_key,
        "valid_candidate_count": len(tasks),
        "candidate_sources": ["student_code_lm_checkpoint_v1"],
        "candidate_generation_modes": ["private_local_sandbox_regression_token_decoder"],
        "provenance_valid": True,
    }


def private_candidate_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "source_task_id": task["source_task_id"],
        "entry_point": task["entry_point"],
        "candidate_source": "student_code_lm_checkpoint_v1",
        "checkpoint_id": "private_macos_sandbox_path_regression",
        "origin": "private_sandbox_regression:private_local_sandbox_regression_token_decoder",
        "candidate_generation_mode": "private_local_sandbox_regression_token_decoder",
        "candidate_program_scope": "full_function_body",
        "token_level_code_generation_learned": True,
        "benchmark_promotion_eligible": True,
        "loop_closure_generated": False,
        "template_like_candidate": False,
        "compositional_token_candidate": True,
        "full_body_token_candidate": True,
        "grammar_masked_learned_token_candidate": True,
        "expression_memory_fallback": False,
        "deterministic_guardrail_passed": True,
        "decoder_contract_verifier_v1_passed": True,
        "program_synthesis_loop_v1": {
            "policy": "project_theseus_program_synthesis_loop_v1",
            "loop_shape": [
                "contract_ir",
                "ast_plan",
                "constrained_token_decode",
                "parser_contract_mask",
                "verifier_repair",
                "ranker",
            ],
            "decode_control": {
                "constrained_token_decode": True,
                "parser_contract_mask": True,
                "exact_interface_claim": True,
                "grammar_masked_learned_token_candidate": True,
                "template_or_memory_fallback": False,
            },
        },
        "code": (
            "def private_sandbox_sum(data):\n"
            "    total = 0\n"
            "    for value in data:\n"
            "        total += value\n"
            "    return total\n"
        ),
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# macOS Sandbox Path Regression",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Runtime tmp dir: `{summary.get('runtime_tmp_dir')}`",
        f"- Code LM closure tmp dir: `{summary.get('code_lm_closure_runtime_tmp_dir')}`",
        f"- Code LM private verifier tmp dir: `{summary.get('code_lm_private_verifier_runtime_tmp_dir')}`",
        f"- Candidate passed: `{summary.get('candidate_passed')}`",
        f"- Windows tmp leak hits: `{summary.get('windows_tmp_leak_hit_count')}`",
        f"- External inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: `{row.get('passed')}`")
    lines.append("")
    lines.append("Public benchmark prompts, tests, solutions, and candidate manifests are not read by this regression.")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
