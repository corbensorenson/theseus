"""STS repair ablation report.

This script promotes a simple truth: STS only matters if the same public
calibration surface improves versus the single-stream baseline without
regressions. It reads the real-code graduation report and traces; it does not
generate candidates or access public tests.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-code-report", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--trace-in", default="reports/real_code_benchmark_traces.jsonl")
    parser.add_argument("--code-lm-report", default="reports/code_lm_closure.json")
    parser.add_argument("--out", default="reports/sts_repair_ablation.json")
    parser.add_argument("--markdown-out", default="reports/sts_repair_ablation.md")
    args = parser.parse_args()

    real_code = read_json(resolve(args.real_code_report), {})
    code_lm = read_json(resolve(args.code_lm_report), {})
    traces = read_jsonl(resolve(args.trace_in))
    summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    private_eval = code_lm.get("private_eval") if isinstance(code_lm.get("private_eval"), dict) else {}
    task_rows = task_level_trace_summary(traces)
    single = float(summary.get("single_stream_pass_rate") or 0.0)
    multi = float(summary.get("multi_stream_pass_rate") or 0.0)
    delta = round(multi - single, 6)
    regressions = int(summary.get("task_level_regressions_vs_single_stream") or 0)
    improvements = int(summary.get("task_level_improvements_over_single_stream") or 0)
    gates = [
        gate("same_task_overlap", int(summary.get("total_case_count") or 0) > 0, f"cases={summary.get('total_case_count')}"),
        gate("sts_delta_positive", delta > 0.0, f"delta={delta}"),
        gate("zero_task_regressions", regressions == 0, f"regressions={regressions}"),
        gate("token_level_student_generation_valid", bool(summary.get("token_level_code_generation_learned")), summary.get("candidate_generation_modes")),
        gate("private_concept_sts_nonregressing", float(private_eval.get("sts_repair_pass_rate_delta") or 0.0) >= 0.0 and int(private_eval.get("sts_repair_task_level_regressions") or 0) == 0, private_eval),
        gate("public_score_quarantined", real_code.get("public_benchmark_score_claim") == "student_code_lm_checkpoint_public_task_calibration_only", real_code.get("public_benchmark_score_claim")),
        gate("external_inference_zero", int(real_code.get("external_inference_calls") or 0) == 0, real_code.get("external_inference_calls")),
    ]
    report = {
        "policy": "project_theseus_sts_repair_ablation_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "summary": {
            "single_stream_pass_rate": single,
            "multi_stream_pass_rate": multi,
            "pass_rate_delta": delta,
            "task_level_improvements": improvements,
            "task_level_regressions": regressions,
            "task_count": int(summary.get("total_case_count") or 0),
            "full_body_public_pass_count": int(summary.get("full_body_public_pass_count") or 0),
            "expression_fallback_public_pass_count": int(summary.get("expression_fallback_public_pass_count") or 0),
            "private_sts_off_pass_rate": private_eval.get("sts_off_pass_rate"),
            "private_sts_on_pass_rate": private_eval.get("trained_pass_rate"),
            "private_sts_repair_pass_rate_delta": private_eval.get("sts_repair_pass_rate_delta"),
            "private_sts_repair_task_level_improvements": private_eval.get("sts_repair_task_level_improvements"),
            "private_sts_repair_task_level_regressions": private_eval.get("sts_repair_task_level_regressions"),
            "private_concept_residual_counts": private_eval.get("concept_residual_counts", {}),
            "private_concept_family_pass_rates": private_eval.get("concept_family_pass_rates", {}),
            "external_inference_calls": 0,
        },
        "task_trace_summary": task_rows,
        "gates": gates,
        "score_semantics": "STS-on vs single-stream comparison over the existing public calibration report; not standalone promotion evidence.",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def task_level_trace_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"attempts": 0, "passes": 0, "residuals": defaultdict(int), "modes": defaultdict(int)})
    for row in rows:
        if row.get("event") != "real_code_candidate_test":
            continue
        task = str(row.get("task_id") or row.get("source_task_id") or "unknown")
        item = by_task[task]
        item["attempts"] += 1
        if row.get("passed") is True:
            item["passes"] += 1
        residual = str(row.get("residual_class") or "")
        if residual:
            item["residuals"][residual] += 1
        mode = str(row.get("mode") or "unknown")
        item["modes"][mode] += 1
    out = []
    for task, item in sorted(by_task.items())[:32]:
        out.append(
            {
                "task_hash": short_hash(task),
                "attempts": item["attempts"],
                "passes": item["passes"],
                "residuals": dict(item["residuals"]),
                "modes": dict(item["modes"]),
            }
        )
    return out


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    return "\n".join(
        [
            "# STS Repair Ablation",
            "",
            f"State: **{report.get('trigger_state')}**",
            "",
            f"- Single-stream: {s.get('single_stream_pass_rate')}",
            f"- STS/multi-stream: {s.get('multi_stream_pass_rate')}",
            f"- Delta: {s.get('pass_rate_delta')}",
            f"- Improvements/regressions: {s.get('task_level_improvements')} / {s.get('task_level_regressions')}",
            f"- Private concept STS off/on: {s.get('private_sts_off_pass_rate')} / {s.get('private_sts_on_pass_rate')}",
            f"- Private concept STS delta: {s.get('private_sts_repair_pass_rate_delta')}",
            "",
        ]
    )


def short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
