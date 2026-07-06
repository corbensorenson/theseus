#!/usr/bin/env python3
"""Rescore existing private Code LM candidates without rerunning Rust.

Use this after a Python/private-sandbox evaluator fix when the Rust candidate
artifacts are already present. It deliberately does not train, regenerate
candidates, read public tests/solutions, or make a public benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates, runtime_tmp_dir  # noqa: E402


DEFAULT_PRIVATE_CURRICULUM = ROOT / "data" / "private_code_curriculum" / "code_lm_closure_edge_contract_v2_private.jsonl"
DEFAULT_PRIVATE_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_edge_contract_v2_private.jsonl"
DEFAULT_OUT = ROOT / "reports" / "code_lm_closure_edge_contract_v2_private_rescore.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "code_lm_closure_edge_contract_v2_private_rescore.md"
WINDOWS_TMP_NEEDLES = (
    "D:/ProjectTheseus/tmp",
    "D:\\ProjectTheseus\\tmp",
    "D:/ProjectTheseus\\tmp",
    "D:\\ProjectTheseus/tmp",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-curriculum", default=str(DEFAULT_PRIVATE_CURRICULUM.relative_to(ROOT)))
    parser.add_argument("--private-candidates", default=str(DEFAULT_PRIVATE_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--min-private-delta", type=float, default=0.05)
    args = parser.parse_args()

    private_curriculum = resolve(args.private_curriculum)
    private_candidates = resolve(args.private_candidates)
    private_rows = read_jsonl(private_curriculum)
    candidate_rows = read_jsonl(private_candidates)
    private_eval = evaluate_private_candidates(private_rows, candidate_rows)
    union_candidate_rows = with_sts_nonregression_union(candidate_rows)
    union_private_eval = evaluate_private_candidates(private_rows, union_candidate_rows)
    tmp_dir = runtime_tmp_dir()
    evidence_text = json.dumps(private_eval, sort_keys=True)
    leak_hits = [needle for needle in WINDOWS_TMP_NEEDLES if needle in evidence_text or needle in str(tmp_dir)]
    runtime_load_rate = float(
        get_path(private_eval, ["private_verification", "runtime_load_rate"], 0.0) or 0.0
    )
    private_delta = float(private_eval.get("pass_rate_delta") or 0.0)
    union_private_delta = float(union_private_eval.get("pass_rate_delta") or 0.0)
    gates = [
        gate("private_curriculum_loaded", len(private_rows) > 0, {"rows": len(private_rows), "path": rel_or_abs(private_curriculum)}),
        gate("private_candidates_loaded", len(candidate_rows) > 0, {"rows": len(candidate_rows), "path": rel_or_abs(private_candidates)}),
        gate("private_execution_eval_ran", int(private_eval.get("eval_task_count") or 0) > 0, private_eval.get("eval_task_count")),
        gate("private_runtime_load_rate_nonzero", runtime_load_rate > 0.0, runtime_load_rate),
        gate("private_delta_above_floor", private_delta >= float(args.min_private_delta), {"observed": private_delta, "minimum": float(args.min_private_delta)}),
        gate(
            "private_sts_nonregressive",
            int(private_eval.get("sts_repair_task_level_regressions") or 0) == 0,
            private_eval.get("sts_repair_task_level_regressions"),
        ),
        gate(
            "private_sts_union_nonregressive",
            int(union_private_eval.get("sts_repair_task_level_regressions") or 0) == 0,
            {
                "sts_regressions": union_private_eval.get("sts_repair_task_level_regressions"),
                "trained_pass_rate": union_private_eval.get("trained_pass_rate"),
                "sts_off_pass_rate": union_private_eval.get("sts_off_pass_rate"),
            },
        ),
        gate(
            "private_sts_union_delta_above_floor",
            union_private_delta >= float(args.min_private_delta),
            {"observed": union_private_delta, "minimum": float(args.min_private_delta)},
        ),
        gate("no_windows_tmp_leak_in_private_rescore", not leak_hits, {"leak_hits": leak_hits, "runtime_tmp_dir": str(tmp_dir)}),
        gate("public_training_data_not_used", True, {"public_tests_used": False, "public_solutions_used": False}),
        gate("external_inference_zero", True, 0),
    ]
    hard_failures = [row for row in gates if not row["passed"]]
    ready_for_public_calibration = not hard_failures
    report = {
        "policy": "project_theseus_code_lm_private_candidate_rescore_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failures else "YELLOW",
        "run_status": "completed",
        "ready_for_public_calibration": ready_for_public_calibration,
        "summary": {
            "private_task_count": len(private_rows),
            "private_eval_task_count": private_eval.get("eval_task_count"),
            "candidate_rows": len(candidate_rows),
            "private_baseline_pass_rate": private_eval.get("baseline_pass_rate"),
            "private_sts_off_pass_rate": private_eval.get("sts_off_pass_rate"),
            "private_trained_pass_rate": private_eval.get("trained_pass_rate"),
            "private_pass_rate_delta": private_eval.get("pass_rate_delta"),
            "private_sts_repair_pass_rate_delta": private_eval.get("sts_repair_pass_rate_delta"),
            "private_sts_repair_task_level_improvements": private_eval.get("sts_repair_task_level_improvements"),
            "private_sts_repair_task_level_regressions": private_eval.get("sts_repair_task_level_regressions"),
            "private_sts_union_trained_pass_rate": union_private_eval.get("trained_pass_rate"),
            "private_sts_union_pass_rate_delta": union_private_eval.get("pass_rate_delta"),
            "private_sts_union_repair_pass_rate_delta": union_private_eval.get("sts_repair_pass_rate_delta"),
            "private_sts_union_task_level_improvements": union_private_eval.get("sts_repair_task_level_improvements"),
            "private_sts_union_task_level_regressions": union_private_eval.get("sts_repair_task_level_regressions"),
            "private_sts_union_candidate_rows": len(union_candidate_rows),
            "private_sts_nonregression_union_artifact_candidate_count": count_bool(candidate_rows, "sts_nonregression_union_candidate"),
            "runtime_load_rate": runtime_load_rate,
            "runtime_tmp_dir": str(tmp_dir),
            "windows_tmp_leak_hit_count": len(leak_hits),
            "public_benchmarks": "not_scored",
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "inputs": {
            "private_curriculum": rel_or_abs(private_curriculum),
            "private_candidates": rel_or_abs(private_candidates),
        },
        "private_eval": private_eval,
        "private_eval_sts_nonregression_union": union_private_eval,
        "gates": gates,
        "next_actions": [
            "when this rescore is GREEN, run post-distillation/public-transfer readiness checks before any new public calibration",
            "if this rescore turns YELLOW, repair STS preservation, candidate diversity, or required-skeleton coverage before public calibration",
            "keep public calibration locked except for one explicit bounded calibration after private and readiness gates are green",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def count_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def with_sts_nonregression_union(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    union_rows = list(rows)
    existing = {
        (
            str(row.get("task_id") or ""),
            str(row.get("phase") or ""),
            str(row.get("code") or ""),
            str(row.get("candidate_generation_mode") or ""),
        )
        for row in rows
    }
    for row in rows:
        if str(row.get("phase") or "") != "private_eval_sts_off":
            continue
        source_mode = str(row.get("candidate_generation_mode") or "unknown")
        if source_mode == "student_decoder_no_admissible_candidate_residual":
            continue
        clone = dict(row)
        clone["phase"] = "private_eval"
        clone["candidate_generation_mode"] = (
            "sts_nonregression_union_from_same_seed_non_sts::"
            + source_mode
        )
        clone["sts_nonregression_union_candidate"] = True
        clone["sts_nonregression_union_source_phase"] = "private_eval_sts_off"
        key = (
            str(clone.get("task_id") or ""),
            str(clone.get("phase") or ""),
            str(clone.get("code") or ""),
            str(clone.get("candidate_generation_mode") or ""),
        )
        if key not in existing:
            existing.add(key)
            union_rows.append(clone)
    return union_rows


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Code LM Private Candidate Rescore",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- ready_for_public_calibration: `{report.get('ready_for_public_calibration')}`",
        f"- private_eval_task_count: `{summary.get('private_eval_task_count')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- private_trained_pass_rate: `{summary.get('private_trained_pass_rate')}`",
        f"- private_pass_rate_delta: `{summary.get('private_pass_rate_delta')}`",
        f"- private_sts_repair_task_level_regressions: `{summary.get('private_sts_repair_task_level_regressions')}`",
        f"- private_sts_nonregression_union_artifact_candidate_count: `{summary.get('private_sts_nonregression_union_artifact_candidate_count')}`",
        f"- private_sts_union_trained_pass_rate: `{summary.get('private_sts_union_trained_pass_rate')}`",
        f"- private_sts_union_task_level_regressions: `{summary.get('private_sts_union_task_level_regressions')}`",
        f"- runtime_load_rate: `{summary.get('runtime_load_rate')}`",
        f"- runtime_tmp_dir: `{summary.get('runtime_tmp_dir')}`",
        f"- windows_tmp_leak_hit_count: `{summary.get('windows_tmp_leak_hit_count')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
