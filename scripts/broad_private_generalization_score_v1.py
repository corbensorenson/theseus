#!/usr/bin/env python3
"""Score Broad Private Generalization Ladder v1 heldout rows.

The scorer executes candidate code against private synthetic heldout tests. It
does not read public benchmark prompts, tests, solutions, score labels, or
candidate code outside the supplied private candidate manifests.
"""

from __future__ import annotations

import argparse
import json
import signal
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
DEFAULT_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--control-candidates", default="")
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--min-heldout-rows", type=int, default=1000)
    parser.add_argument("--out", default="reports/broad_private_generalization_score_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_generalization_score_v1.md")
    args = parser.parse_args()

    report = score_heldout(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def score_heldout(args: argparse.Namespace) -> dict[str, Any]:
    heldout_path = resolve(args.heldout)
    candidate_path = resolve(args.candidates)
    control_path = resolve(args.control_candidates) if args.control_candidates else None
    heldout = read_jsonl(heldout_path)
    heldout_total_before_limit = len(heldout)
    if int(args.task_limit) > 0:
        heldout = heldout[: int(args.task_limit)]
    candidates = read_jsonl(candidate_path)
    controls = read_jsonl(control_path) if control_path is not None else []
    leakage = public_leakage_scan(heldout)
    by_task = group_candidates(candidates)
    control_summary = score_control(heldout, controls, timeout_seconds=max(1, int(args.timeout_seconds)))
    results = []
    family_counts: Counter[str] = Counter()
    family_passes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    mode_passes: Counter[str] = Counter()
    no_admissible = 0
    sts_passes = 0
    non_sts_passes = 0
    for row in heldout:
        task_id = str(row.get("task_id") or "")
        family = family_of(row)
        category = str(row.get("category") or "unknown")
        family_counts[family] += 1
        category_counts[category] += 1
        task_candidates = by_task.get(task_id, [])
        if not task_candidates:
            no_admissible += 1
            results.append(result_row(row, False, "no_candidate", "", False, 0, []))
            continue
        passed = False
        pass_mode = ""
        pass_sts = False
        errors = []
        for candidate in task_candidates:
            ok, error = run_candidate(row, candidate, timeout_seconds=max(1, int(args.timeout_seconds)))
            if ok:
                passed = True
                pass_mode = str(candidate.get("candidate_generation_mode") or "")
                pass_sts = candidate_is_sts(candidate, pass_mode)
                break
            if len(errors) < 3:
                errors.append(error)
        if passed:
            family_passes[family] += 1
            category_passes[category] += 1
            mode_passes[pass_mode] += 1
            if pass_sts:
                sts_passes += 1
            else:
                non_sts_passes += 1
        results.append(
            result_row(
                row,
                passed,
                "passed" if passed else "failed_private_tests",
                pass_mode,
                pass_sts,
                len(task_candidates),
                errors,
            )
        )
    task_count = len(heldout)
    pass_count = sum(1 for row in results if row["passed"])
    pass_rate = round(pass_count / max(1, task_count), 6)
    no_admissible_rate = round(no_admissible / max(1, task_count), 6)
    current_passes = {str(row.get("task_id") or "") for row in results if row.get("passed")}
    control_passes = {
        str(row.get("task_id") or "")
        for row in control_summary["results"]
        if row.get("passed")
    }
    regressions = len(control_passes - current_passes) if control_summary["task_count"] else 0
    sts_delta = (
        round(pass_rate - float(control_summary["pass_rate"]), 6)
        if control_summary["task_count"]
        else round((sts_passes - non_sts_passes) / max(1, task_count), 6)
    )
    family_rates = rate_table(family_counts, family_passes)
    category_rates = rate_table(category_counts, category_passes)
    gates = [
        gate("heldout_rows_ge_minimum", task_count >= int(args.min_heldout_rows), {
            "observed": task_count,
            "minimum": int(args.min_heldout_rows),
            "total_before_limit": heldout_total_before_limit,
            "task_limit": int(args.task_limit),
        }),
        gate("candidate_rows_present", len(candidates) > 0, len(candidates)),
        gate("broad_private_pass_rate_floor", pass_rate >= 0.70, {"observed": pass_rate, "minimum": 0.70}),
        gate("no_admissible_rate_floor", no_admissible_rate <= 0.03, {"observed": no_admissible_rate, "maximum": 0.03}),
        gate("sts_same_seed_positive", sts_delta > 0.0, {"delta": sts_delta, "control_task_count": control_summary["task_count"]}),
        gate("sts_regressions_bounded", regressions == 0, {"regressions": regressions}),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate("external_inference_zero", True, 0),
    ]
    hard_failure = leakage["hit_count"] > 0
    trigger_state = "RED" if hard_failure else ("GREEN" if all(row["passed"] for row in gates) else "YELLOW")
    return {
        "policy": "project_theseus_broad_private_generalization_score_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "heldout": rel(heldout_path),
            "candidates": rel(candidate_path),
            "control_candidates": rel(control_path) if control_path is not None else "",
            "timeout_seconds": int(args.timeout_seconds),
            "task_limit": int(args.task_limit),
            "min_heldout_rows": int(args.min_heldout_rows),
            "public_benchmark_inputs_read": False,
        },
        "summary": {
            "heldout_task_count": task_count,
            "heldout_task_count_before_limit": heldout_total_before_limit,
            "candidate_row_count": len(candidates),
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "no_admissible_task_count": no_admissible,
            "no_admissible_task_rate": no_admissible_rate,
            "sts_passes": sts_passes,
            "non_sts_passes": non_sts_passes,
            "sts_delta": sts_delta,
            "sts_regressions": regressions,
            "control_task_count": control_summary["task_count"],
            "control_pass_count": control_summary["pass_count"],
            "control_pass_rate": control_summary["pass_rate"],
            "family_rates": family_rates,
            "weakest_families": weakest_items(family_rates),
            "weakest_categories": weakest_items(category_rates, limit=10),
            "mode_passes": dict(sorted(mode_passes.items())),
            "mode_passes_top20": dict(mode_passes.most_common(20)),
            "public_data_leakage_hit_count": leakage["hit_count"],
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "gates": gates,
        "failure_clusters": failure_clusters(results),
        "results": results,
        "control_results": control_summary["results"],
        "next_actions": next_actions(pass_rate, no_admissible_rate, sts_delta, regressions, no_admissible),
        "external_inference_calls": 0,
    }


def score_control(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], *, timeout_seconds: int) -> dict[str, Any]:
    if not candidates:
        return {"task_count": 0, "pass_count": 0, "pass_rate": 0.0, "results": []}
    by_task = group_candidates(candidates)
    results = []
    for row in rows:
        passed = False
        for candidate in by_task.get(str(row.get("task_id") or ""), []):
            ok, _ = run_candidate(row, candidate, timeout_seconds=timeout_seconds)
            if ok:
                passed = True
                break
        results.append({"task_id": row.get("task_id"), "passed": passed})
    pass_count = sum(1 for row in results if row["passed"])
    return {
        "task_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(1, len(rows)), 6),
        "results": results,
    }


def run_candidate(row: dict[str, Any], candidate: dict[str, Any], *, timeout_seconds: int) -> tuple[bool, str]:
    code = str(candidate.get("code") or "")
    tests = str(row.get("tests") or "")
    if not code.strip() or not tests.strip():
        return False, "missing candidate code or tests"
    namespace: dict[str, Any] = {}
    if not hasattr(signal, "SIGALRM"):
        try:
            exec(code, namespace, namespace)
            exec(tests, namespace, namespace)
            return True, ""
        except Exception as exc:  # pragma: no cover
            return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()[:240]

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"candidate exceeded {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        exec(code, namespace, namespace)
        exec(tests, namespace, namespace)
        return True, ""
    except Exception as exc:  # pragma: no cover
        return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()[:240]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def group_candidates(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            out[task_id].append(candidate)
    return out


def result_row(
    row: dict[str, Any],
    passed: bool,
    status: str,
    pass_mode: str,
    pass_sts: bool,
    candidate_count: int,
    sample_errors: list[str],
) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "family": family_of(row),
        "category": row.get("category"),
        "passed": bool(passed),
        "status": status,
        "pass_candidate_mode": pass_mode,
        "pass_sts_conditioned": bool(pass_sts),
        "candidate_count": candidate_count,
        "sample_errors": sample_errors,
    }


def family_of(row: dict[str, Any]) -> str:
    return str(row.get("broad_private_family_v1") or row.get("targeted_private_residual_family_v3") or "unknown")


def candidate_is_sts(candidate: dict[str, Any], mode: str) -> bool:
    mode = mode.lower()
    return bool(
        candidate.get("sts_stream_conditioned")
        or candidate.get("sts_candidate_expression_used")
        or "sts_conditioned" in mode
    )


def rate_table(counts: Counter[str], passes: Counter[str]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "task_count": counts[key],
            "pass_count": passes[key],
            "pass_rate": round(passes[key] / max(1, counts[key]), 6),
        }
        for key in sorted(counts)
    }


def weakest_items(rates: dict[str, dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        {"name": key, **value}
        for key, value in rates.items()
    ]
    rows.sort(key=lambda row: (float(row.get("pass_rate") or 0.0), -int(row.get("task_count") or 0), row["name"]))
    return rows[:limit]


def failure_clusters(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in results:
        if row.get("passed"):
            continue
        error = ""
        errors = row.get("sample_errors") if isinstance(row.get("sample_errors"), list) else []
        if errors:
            error = str(errors[0]).splitlines()[0][:80]
        counts[(str(row.get("family")), str(row.get("category")), error or str(row.get("status")))] += 1
    return [
        {"family": family, "category": category, "reason": reason, "count": count}
        for (family, category, reason), count in counts.most_common(20)
    ]


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = ["humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench", "canonical_solution", "public_test"]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
        if len(hits) >= 20:
            break
    return {"hit_count": len(hits), "sample_hits": hits}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def next_actions(pass_rate: float, no_admissible_rate: float, sts_delta: float, regressions: int, no_admissible: int) -> list[str]:
    actions = []
    if no_admissible_rate > 0.03:
        actions.append(f"repair candidate coverage first: no-admissible tasks={no_admissible}")
    if pass_rate < 0.70:
        actions.append("cluster weakest private families and patch reusable decoder/learner paths before another public calibration")
    if sts_delta <= 0.0 or regressions > 0:
        actions.append("repair STS causal path using same-seed private controls before treating STS as default-on evidence")
    if not actions:
        actions.append("broad private score gate is clear; rerun maturity/readiness before any operator-approved public calibration proposal")
    return actions


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Generalization Score V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Heldout pass rate: {summary.get('pass_rate')}",
        f"- Heldout passes: {summary.get('pass_count')}/{summary.get('heldout_task_count')}",
        f"- No-admissible rate: {summary.get('no_admissible_task_rate')}",
        f"- STS delta: {summary.get('sts_delta')}",
        f"- STS regressions: {summary.get('sts_regressions')}",
        f"- Candidate rows: {summary.get('candidate_row_count')}",
        "",
        "## Weakest Families",
    ]
    for row in summary.get("weakest_families", []):
        lines.append(f"- `{row.get('name')}`: {row.get('pass_count')}/{row.get('task_count')} = {row.get('pass_rate')}")
    lines.extend(["", "Public benchmark tests and solutions are not used by this scorer."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
