#!/usr/bin/env python3
"""Private-only verifier for edge_contract_v2 residual curriculum.

This is a gate, not a public benchmark claim. It checks that v2 rows are
private, carry generation-plan contracts, pass their private solution tests,
and, once a private Code LM closure exists, that the held-out private signal is
strong enough to permit one controlled public receiver calibration.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402


SOURCE_PRIVATE_IN = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    )
)
REPAIRED_PRIVATE_IN = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_repaired_code_lm_tasks.jsonl",
    )
)
DEFAULT_PRIVATE_IN = REPAIRED_PRIVATE_IN if REPAIRED_PRIVATE_IN.exists() else SOURCE_PRIVATE_IN
DEFAULT_CLOSURE = ROOT / "reports" / "code_lm_closure_edge_contract_v2_private.json"
DEFAULT_PRIVATE_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_edge_contract_v2_private.jsonl"
DEFAULT_OUT = ROOT / "reports" / "edge_contract_v2_private_verifier.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "edge_contract_v2_private_verifier.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-in", default=str(DEFAULT_PRIVATE_IN))
    parser.add_argument("--closure-report", default=str(DEFAULT_CLOSURE.relative_to(ROOT)))
    parser.add_argument("--private-candidates", default=str(DEFAULT_PRIVATE_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--min-private-delta", type=float, default=0.05)
    args = parser.parse_args()

    private_path = resolve(args.private_in)
    closure_path = resolve(args.closure_report)
    candidates_path = resolve(args.private_candidates)
    rows = read_jsonl(private_path)
    v2_rows = [row for row in rows if is_v2_row(row)]
    solution_check = verify_solution_rows(v2_rows)
    contract_summary = contract_coverage(v2_rows)
    closure = read_json(closure_path, {})
    closure_gate = closure_readiness(closure, min_delta=float(args.min_private_delta))
    candidate_gate = candidate_verifier_summary(candidates_path)
    unsafe_rows = [
        row.get("task_id")
        for row in v2_rows
        if row.get("public_benchmark")
        or row.get("public_tests_included")
        or row.get("public_benchmark_solutions_included")
        or str(row.get("benchmark_evidence_level") or "").startswith("public")
    ]
    benchmark_named = [
        row.get("task_id")
        for row in v2_rows
        if any(str(tag).lower().startswith(("mbpp_", "evalplus_", "bigcodebench_", "livecodebench_")) for tag in row.get("tags", []))
    ]
    gates = [
        gate("private_rows_present", len(v2_rows) > 0, {"rows": len(v2_rows), "path": rel_or_abs(private_path)}),
        gate("private_solution_tests_pass", solution_check["failure_count"] == 0, solution_check),
        gate("generation_plan_contracts_complete", contract_summary["generation_plan_rows"] == len(v2_rows) and len(v2_rows) > 0, contract_summary),
        gate("no_public_benchmark_training_rows", not unsafe_rows, unsafe_rows[:20]),
        gate("benchmark_named_private_rows_zero", not benchmark_named, benchmark_named[:20]),
        gate("closure_completed_green", closure_gate["closure_completed_green"], closure_gate),
        gate("private_delta_above_floor", closure_gate["private_delta"] >= float(args.min_private_delta), closure_gate),
        gate("sts_nonregressive", closure_gate["sts_regressions"] == 0, closure_gate),
        gate("candidate_contract_verifier_active", candidate_gate["verifier_rows"] > 0 or not candidates_path.exists(), candidate_gate, severity="soft" if not candidates_path.exists() else "hard"),
    ]
    hard_gates = [row for row in gates if row.get("severity") != "soft"]
    ready = all(row["passed"] for row in hard_gates)
    row_ready = all(row["passed"] for row in gates[:5])
    report = {
        "policy": "project_theseus_edge_contract_v2_private_verifier_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else ("YELLOW" if row_ready else "RED"),
        "ready_for_public_calibration": ready,
        "summary": {
            "private_rows": len(v2_rows),
            "solution_test_failures": solution_check["failure_count"],
            "generation_plan_rows": contract_summary["generation_plan_rows"],
            "benchmark_named_private_rows": len(benchmark_named),
            "closure_report": rel_or_abs(closure_path),
            "closure_trigger_state": closure.get("trigger_state"),
            "closure_run_status": closure.get("run_status"),
            "private_baseline_pass_rate": closure_gate["private_baseline_pass_rate"],
            "private_trained_pass_rate": closure_gate["private_trained_pass_rate"],
            "private_delta": closure_gate["private_delta"],
            "sts_regressions": closure_gate["sts_regressions"],
            "candidate_verifier_rows": candidate_gate["verifier_rows"],
            "candidate_verifier_pass_rows": candidate_gate["verifier_pass_rows"],
            "public_benchmarks": "calibration_only_not_training",
            "external_inference_calls": 0,
        },
        "inputs": {
            "private_in": rel_or_abs(private_path),
            "closure_report": rel_or_abs(closure_path),
            "private_candidates": rel_or_abs(candidates_path),
        },
        "contract_summary": contract_summary,
        "solution_check": solution_check,
        "closure_gate": closure_gate,
        "candidate_gate": candidate_gate,
        "gates": gates,
        "next_actions": [
            "run edge_contract_v2_private_closure if rows are green but closure is missing or stale",
            "only run one public 4-card receiver calibration after ready_for_public_calibration is true",
            "if public receiver stays flat, enqueue teacher architecture experiment spec only",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def is_v2_row(row: dict[str, Any]) -> bool:
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    return (
        any("edge_contract_v2" in str(tag) for tag in tags)
        or "edge_contract_v2" in str(row.get("residual_concept") or "")
    )


def contract_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shape_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    generation_plan_rows = 0
    missing: list[str] = []
    for row in rows:
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        shape = str(contract.get("return_shape") or "unknown")
        family = str(contract.get("type_family") or "unknown")
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
        if plan.get("skeleton_bias") and plan.get("repair_strategy") and plan.get("verifier_feedback"):
            generation_plan_rows += 1
        else:
            missing.append(str(row.get("task_id") or "unknown"))
    return {
        "row_count": len(rows),
        "generation_plan_rows": generation_plan_rows,
        "missing_generation_plan": missing[:20],
        "return_shape_counts": shape_counts,
        "type_family_counts": family_counts,
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def closure_readiness(report: dict[str, Any], *, min_delta: float) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    baseline = number(summary.get("private_baseline_pass_rate"))
    trained = number(summary.get("private_trained_pass_rate"))
    delta = number(summary.get("private_pass_rate_delta"), trained - baseline)
    sts_regressions = int(number(summary.get("private_sts_repair_task_level_regressions")))
    return {
        "closure_completed_green": bool(
            report.get("trigger_state") == "GREEN"
            and report.get("run_status") == "completed"
        ),
        "private_baseline_pass_rate": baseline,
        "private_trained_pass_rate": trained,
        "private_delta": delta,
        "min_private_delta": min_delta,
        "sts_regressions": sts_regressions,
        "score_semantics": "held-out private gate only; not public promotion evidence",
    }


def candidate_verifier_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    verifier_rows = [row for row in rows if "decoder_contract_verifier_v1_passed" in row]
    pass_rows = [row for row in verifier_rows if row.get("decoder_contract_verifier_v1_passed") is True]
    reasons: dict[str, int] = {}
    for row in verifier_rows:
        row_reasons = row.get("decoder_contract_verifier_v1_reasons", [])
        if not isinstance(row_reasons, list):
            row_reasons = []
        for reason in row_reasons:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "path": rel_or_abs(path),
        "exists": path.exists(),
        "candidate_rows": len(rows),
        "verifier_rows": len(verifier_rows),
        "verifier_pass_rows": len(pass_rows),
        "verifier_reason_counts": reasons,
    }


def verify_solution_rows(rows: list[dict[str, Any]], *, max_failures: int = 8) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_edge_contract_v2_") as tmp:
        root = Path(tmp)
        for idx, row in enumerate(rows):
            entry = safe_identifier(str(row.get("entry_point") or f"private_task_{idx}"))
            body = str(row.get("solution_body") or "").strip()
            tests = str(row.get("tests") or "").strip()
            if not body or not tests:
                failures.append({"task_id": str(row.get("task_id") or ""), "error": "missing_body_or_tests"})
                continue
            path = root / f"{entry}.py"
            code = "import collections, itertools, functools, math, re\n\n" + f"def {entry}(*args):\n"
            code += "    data = args[0] if len(args) > 0 else None\n"
            code += "    other = args[1] if len(args) > 1 else None\n"
            code += "    extra = args[2:] if len(args) > 2 else ()\n"
            for line in body.splitlines():
                code += f"    {line}\n"
            code += "\n" + tests + "\n"
            path.write_text(code, encoding="utf-8")
            result = subprocess.run([sys.executable, str(path)], cwd=root, text=True, capture_output=True, timeout=20)
            if result.returncode != 0:
                failures.append(
                    {
                        "task_id": str(row.get("task_id") or ""),
                        "category": str(row.get("category") or ""),
                        "stderr": result.stderr[-600:],
                    }
                )
            if len(failures) >= max_failures:
                break
    return {"checked_rows": len(rows), "failure_count": len(failures), "sample_failures": failures}


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Edge Contract V2 Private Verifier",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- ready_for_public_calibration: `{report.get('ready_for_public_calibration')}`",
        f"- private_rows: `{summary.get('private_rows')}`",
        f"- generation_plan_rows: `{summary.get('generation_plan_rows')}`",
        f"- private_delta: `{summary.get('private_delta')}`",
        f"- sts_regressions: `{summary.get('sts_regressions')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    return "\n".join(lines)


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_identifier(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
