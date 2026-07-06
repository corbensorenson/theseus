#!/usr/bin/env python3
"""Audit whether reranking existing neural-seed candidates can close residuals.

This diagnostic reads existing private token-decoder candidate manifests and
semantic-plan audits. It does not generate candidates, train on public data,
call a teacher, or rerun public calibration.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTI = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_multiseed.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_candidate_ranker_boundary_audit.json"
DEFAULT_MD = ROOT / "reports" / "neural_seed_candidate_ranker_boundary_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-report", default=str(DEFAULT_MULTI.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(resolve(args.multiseed_report), started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def build_report(multiseed_path: Path, *, started: float) -> dict[str, Any]:
    multiseed = read_json(multiseed_path)
    seed_rows = list(multiseed.get("seed_rows") or [])
    event_rows: list[dict[str, Any]] = []
    by_arm = defaultdict(lambda: Counter())
    both_fail_task_ids: set[str] = set()
    non_exhausted_failures: list[dict[str, Any]] = []

    for seed_row in seed_rows:
        seed = int(seed_row.get("seed") or 0)
        candidate_path = resolve(str(seed_row.get("candidate_manifest") or ""))
        audit_path = resolve(str(seed_row.get("semantic_plan_audit") or ""))
        candidates = read_jsonl(candidate_path)
        audit = read_json(audit_path)
        counts_by_key: Counter[tuple[str, str, str]] = Counter()
        for row in candidates:
            counts_by_key[
                (
                    str(row.get("substrate_arm") or ""),
                    str(row.get("task_id") or ""),
                    str(row.get("phase") or ""),
                )
            ] += 1

        for task in audit.get("task_rows") or []:
            task_id = str(task.get("task_id") or "")
            gap_status = str(task.get("gap_status") or "")
            if gap_status == "both_fail":
                both_fail_task_ids.add(task_id)
            for arm in ["symliquid_style", "transformer_control"]:
                event = dict_or_empty(get_path(task, ["arms", arm, "private_eval"], {}))
                candidate_count = counts_by_key[(arm, task_id, "private_eval")]
                passed = bool(event.get("passed"))
                selected_rank = int(event.get("selected_rank") or 0)
                exhausted = (not passed) and candidate_count > 0 and selected_rank >= candidate_count
                row = {
                    "seed": seed,
                    "arm": arm,
                    "task_id": task_id,
                    "family": task.get("family"),
                    "gap_status": gap_status,
                    "passed": passed,
                    "candidate_count": candidate_count,
                    "selected_rank": selected_rank,
                    "exhausted_all_candidates": exhausted,
                    "selected_plan": event.get("selected_plan"),
                    "top_plan": event.get("top_plan"),
                    "wrong_answer_shape": event.get("wrong_answer_shape"),
                }
                event_rows.append(row)
                by_arm[arm]["events"] += 1
                by_arm[arm]["passes"] += int(passed)
                by_arm[arm]["failures"] += int(not passed)
                by_arm[arm]["exhausted_failures"] += int(exhausted)
                if (not passed) and not exhausted:
                    non_exhausted_failures.append(row)

    total_failures = sum(1 for row in event_rows if not row["passed"])
    exhausted_failures = sum(1 for row in event_rows if row["exhausted_all_candidates"])
    ranker_can_close_existing_both_fails = any(
        (not row["passed"]) and (not row["exhausted_all_candidates"])
        for row in event_rows
        if row["gap_status"] == "both_fail"
    )
    hard_gates = [
        gate("multiseed_report_loaded", bool(seed_rows), {"path": rel(multiseed_path), "seed_rows": len(seed_rows)}, "hard"),
        gate("candidate_and_audit_rows_loaded", bool(event_rows), {"event_rows": len(event_rows)}, "hard"),
        gate(
            "all_current_failures_exhaust_existing_candidates",
            total_failures == exhausted_failures,
            {"failures": total_failures, "exhausted_failures": exhausted_failures},
            "hard",
        ),
        gate(
            "reranker_has_no_existing_both_fail_score_headroom",
            not ranker_can_close_existing_both_fails,
            {"ranker_can_close_existing_both_fails": ranker_can_close_existing_both_fails},
            "hard",
        ),
        gate("external_inference_zero", True, 0, "hard"),
        gate(
            "teacher_public_promotion_locked",
            True,
            {"teacher_used": False, "public_training_rows": 0, "model_promotion_allowed": False},
            "hard",
        ),
    ]
    trigger = "GREEN" if all(row["passed"] for row in hard_gates if row["severity"] == "hard") else "RED"
    summary = {
        "seed_count": len(seed_rows),
        "event_count": len(event_rows),
        "unique_both_fail_task_count": len(both_fail_task_ids),
        "failure_events": total_failures,
        "exhausted_failure_events": exhausted_failures,
        "ranker_can_reduce_current_both_fail_count": ranker_can_close_existing_both_fails,
        "required_next_work": "new_learned_nonfallback_candidate_generation_not_candidate_reranking",
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
    }
    return {
        "policy": "project_theseus_neural_seed_candidate_ranker_boundary_audit_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "source_multiseed_report": rel(multiseed_path),
        "summary": summary,
        "by_arm": {
            arm: {
                "events": counts["events"],
                "passes": counts["passes"],
                "failures": counts["failures"],
                "exhausted_failures": counts["exhausted_failures"],
                "pass_rate": ratio(counts["passes"], counts["events"]),
            }
            for arm, counts in sorted(by_arm.items())
        },
        "non_exhausted_failures": non_exhausted_failures[:20],
        "gates": hard_gates,
        "score_semantics": (
            "Private diagnostic over existing token-decoder candidate manifests and semantic-plan audits. "
            "It only checks whether candidate ordering has score headroom. It does not train, generate "
            "new candidates, run public calibration, call a teacher, use public data, unlock promotion, "
            "or use eval solutions as generation features."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Candidate Ranker Boundary Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- seed_count: `{summary.get('seed_count')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- unique_both_fail_task_count: `{summary.get('unique_both_fail_task_count')}`",
        f"- failure_events: `{summary.get('failure_events')}`",
        f"- exhausted_failure_events: `{summary.get('exhausted_failure_events')}`",
        f"- ranker_can_reduce_current_both_fail_count: `{summary.get('ranker_can_reduce_current_both_fail_count')}`",
        f"- required_next_work: `{summary.get('required_next_work')}`",
        "",
        "## By Arm",
        "",
    ]
    for arm, row in dict_or_empty(report.get("by_arm")).items():
        lines.extend(
            [
                f"### {arm}",
                f"- events: `{row.get('events')}`",
                f"- passes: `{row.get('passes')}`",
                f"- failures: `{row.get('failures')}`",
                f"- exhausted_failures: `{row.get('exhausted_failures')}`",
                f"- pass_rate: `{row.get('pass_rate')}`",
                "",
            ]
        )
    lines.append("## Gates")
    lines.append("")
    for gate_row in report.get("gates") or []:
        lines.append(f"- `{gate_row.get('name')}`: `{gate_row.get('passed')}`")
    lines.append("")
    lines.append(str(report.get("score_semantics") or ""))
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
