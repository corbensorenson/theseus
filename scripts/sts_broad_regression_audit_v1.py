#!/usr/bin/env python3
"""Audit STS-on versus STS-off regressions in the broad survival lane.

The broad comparator currently trains a body-template selector from visible
task features. This audit recomputes the private verifier outcomes by task,
arm, and STS view so harmful feature views can be gated instead of treated as
promotion evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import (  # noqa: E402
    bounded_private_verification_workers,
    evaluate_private_task,
    runtime_tmp_dir,
)
from theseus_archive_resolver import read_jsonl_follow_pointer  # noqa: E402


DEFAULT_CONFIG = ROOT / "reports" / "broad_capability_survival_lane_v1_comparator_config.json"
DEFAULT_EVAL = ROOT / "reports" / "broad_capability_survival_lane_v1_eval.jsonl"
DEFAULT_CANDIDATES = ROOT / "reports" / "broad_capability_survival_lane_v1_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "sts_broad_regression_audit_v1.json"
DEFAULT_MD = ROOT / "reports" / "sts_broad_regression_audit_v1.md"
DEFAULT_POLICY = ROOT / "configs" / "sts_broad_survival_policy_v1.json"

PHASE_BY_VIEW = {
    "sts_off": "private_eval_sts_off",
    "sts_on": "private_eval",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--eval", default=rel(DEFAULT_EVAL))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--policy-out", default=rel(DEFAULT_POLICY))
    parser.add_argument("--candidate-timeout-seconds", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(max(1, int(args.candidate_timeout_seconds)))
    try:
        report = build_report(args, started=started)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    policy = report.get("recommended_policy") if isinstance(report.get("recommended_policy"), dict) else {}
    write_json(resolve(args.policy_out), policy)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    config = read_json(resolve(args.config))
    eval_rows = [row for row in read_jsonl(resolve(args.eval)) if str(row.get("split") or "eval") == "eval"]
    candidates = read_jsonl(resolve(args.candidates))
    candidates_by_arm = group_candidates(candidates)
    text_views = config.get("text_views") if isinstance(config.get("text_views"), dict) else {}
    sts_off_fields = [str(item) for item in text_views.get("sts_off", []) or []]
    sts_on_fields = [str(item) for item in text_views.get("sts_on", []) or []]
    extra_sts_fields = [field for field in sts_on_fields if field not in set(sts_off_fields)]

    arm_reports: dict[str, Any] = {}
    task_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_sts_broad_audit_", dir=runtime_tmp_dir()) as tmp:
        root = Path(tmp)
        for arm_id, by_task_phase in sorted(candidates_by_arm.items()):
            arm_task_results = evaluate_arm(root, eval_rows, by_task_phase)
            arm_task_rows = [
                task_audit_row(task_result, by_task_phase=by_task_phase, extra_sts_fields=extra_sts_fields, arm_id=arm_id)
                for task_result in arm_task_results
            ]
            arm_reports[arm_id] = summarize_arm(arm_task_rows)
            task_rows.extend(arm_task_rows)

    transformer = arm_reports.get("transformer_control", {})
    symliquid = arm_reports.get("symliquid_style", {})
    root_cause = infer_root_cause(arm_reports, extra_sts_fields, config)
    policy = recommended_policy(
        arm_reports=arm_reports,
        root_cause=root_cause,
        sts_off_fields=sts_off_fields,
        sts_on_fields=sts_on_fields,
        extra_sts_fields=extra_sts_fields,
    )
    gates = build_gates(eval_rows, candidates, arm_reports, policy)
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and policy.get("action") != "keep_sts_on":
        trigger_state = "YELLOW"

    return {
        "policy": "project_theseus_sts_broad_regression_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "config": rel(resolve(args.config)),
            "eval": rel(resolve(args.eval)),
            "candidates": rel(resolve(args.candidates)),
        },
        "summary": {
            "eval_task_count": len(eval_rows),
            "candidate_rows": len(candidates),
            "arms": sorted(arm_reports),
            "sts_off_fields": sts_off_fields,
            "sts_on_fields": sts_on_fields,
            "extra_sts_fields": extra_sts_fields,
            "transformer_control": compact_arm(transformer),
            "symliquid_style": compact_arm(symliquid),
            "root_cause": root_cause,
            "recommended_action": policy.get("action"),
            "policy_out": rel(resolve(args.policy_out)),
            "external_inference_calls": 0,
            "fallback_return_count": fallback_count(candidates),
        },
        "arms": arm_reports,
        "task_deltas": task_rows[:256],
        "recommended_policy": policy,
        "gates": gates,
        "score_semantics": (
            "Private broad survival-lane STS audit only. It recomputes verifier outcomes from existing "
            "private eval rows and candidate rows. It does not train on public benchmark data, use teacher "
            "outputs, call external inference, expose eval tests to generation, or award fallback credit."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    grouped: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        arm = str(row.get("substrate_arm") or "unknown")
        task_id = str(row.get("task_id") or "")
        phase = str(row.get("phase") or "")
        grouped[arm][(task_id, phase)].append(row)
    for by_task_phase in grouped.values():
        for key, items in by_task_phase.items():
            items.sort(key=lambda row: (int(row.get("rank") or 0), str(row.get("candidate_sha256") or "")))
            by_task_phase[key] = items
    return {arm: dict(by_task_phase) for arm, by_task_phase in grouped.items()}


def evaluate_arm(
    root: Path,
    eval_rows: list[dict[str, Any]],
    by_task_phase: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    worker_count = bounded_private_verification_workers(len(eval_rows))
    if worker_count <= 1:
        return [evaluate_private_task(root, task, by_task_phase) for task in eval_rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(evaluate_private_task, root, task, by_task_phase) for task in eval_rows]
        return [future.result() for future in futures]


def task_audit_row(
    task_result: dict[str, Any],
    *,
    by_task_phase: dict[tuple[str, str], list[dict[str, Any]]],
    extra_sts_fields: list[str],
    arm_id: str,
) -> dict[str, Any]:
    task = task_result["task"]
    task_id = str(task.get("task_id") or "")
    off_result = task_result["sts_off"]
    on_result = task_result["trained"]
    off_candidates = by_task_phase.get((task_id, PHASE_BY_VIEW["sts_off"]), [])
    on_candidates = by_task_phase.get((task_id, PHASE_BY_VIEW["sts_on"]), [])
    off_pass_idx = first_passing_attempt_index(off_result)
    on_pass_idx = first_passing_attempt_index(on_result)
    return {
        "arm_id": arm_id,
        "task_id": task_id,
        "category": task.get("category"),
        "family": task.get("broad_private_family_v1") or task.get("targeted_private_residual_family_v3") or task.get("category"),
        "sts_off_pass_if_any": bool(off_result.get("passed")),
        "sts_on_pass_if_any": bool(on_result.get("passed")),
        "sts_off_top1_pass": top1_passed(off_result),
        "sts_on_top1_pass": top1_passed(on_result),
        "sts_regression": bool(off_result.get("passed")) and not bool(on_result.get("passed")),
        "sts_improvement": bool(on_result.get("passed")) and not bool(off_result.get("passed")),
        "sts_off_first_passing_rank": off_pass_idx,
        "sts_on_first_passing_rank": on_pass_idx,
        "sts_off_top_template_id": candidate_template(off_candidates, 1),
        "sts_on_top_template_id": candidate_template(on_candidates, 1),
        "sts_off_first_passing_template_id": candidate_template(off_candidates, off_pass_idx),
        "sts_on_first_passing_template_id": candidate_template(on_candidates, on_pass_idx),
        "top_template_changed": candidate_template(off_candidates, 1) != candidate_template(on_candidates, 1),
        "sts_off_stage": off_result.get("verification_stage"),
        "sts_on_stage": on_result.get("verification_stage"),
        "sts_off_reward": off_result.get("verification_reward"),
        "sts_on_reward": on_result.get("verification_reward"),
        "extra_sts_field_lengths": {field: len(flatten(get_dotted(task, field))) for field in extra_sts_fields},
        "extra_sts_field_present_count": sum(1 for field in extra_sts_fields if get_dotted(task, field) is not None),
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    off_pass = sum(1 for row in rows if row["sts_off_pass_if_any"])
    on_pass = sum(1 for row in rows if row["sts_on_pass_if_any"])
    off_top1 = sum(1 for row in rows if row["sts_off_top1_pass"])
    on_top1 = sum(1 for row in rows if row["sts_on_top1_pass"])
    regressions = [row for row in rows if row["sts_regression"]]
    improvements = [row for row in rows if row["sts_improvement"]]
    family_total: Counter[str] = Counter(str(row["family"]) for row in rows)
    family_regressions: Counter[str] = Counter(str(row["family"]) for row in regressions)
    family_improvements: Counter[str] = Counter(str(row["family"]) for row in improvements)
    changed_top = sum(1 for row in rows if row["top_template_changed"])
    regression_changed_top = sum(1 for row in regressions if row["top_template_changed"])
    no_admissible_on = sum(1 for row in rows if not row["sts_on_pass_if_any"])
    no_admissible_off = sum(1 for row in rows if not row["sts_off_pass_if_any"])
    return {
        "task_count": total,
        "sts_off_pass_count": off_pass,
        "sts_off_pass_rate": ratio(off_pass, total),
        "sts_on_pass_count": on_pass,
        "sts_on_pass_rate": ratio(on_pass, total),
        "sts_delta": round(ratio(on_pass, total) - ratio(off_pass, total), 6),
        "sts_off_top1_pass_rate": ratio(off_top1, total),
        "sts_on_top1_pass_rate": ratio(on_top1, total),
        "sts_regression_count": len(regressions),
        "sts_improvement_count": len(improvements),
        "no_admissible_sts_on_count": no_admissible_on,
        "no_admissible_sts_off_count": no_admissible_off,
        "top_template_change_rate": ratio(changed_top, total),
        "regression_top_template_change_rate": ratio(regression_changed_top, len(regressions)),
        "family_regression_counts": dict(family_regressions.most_common()),
        "family_improvement_counts": dict(family_improvements.most_common()),
        "family_counts": dict(family_total.most_common()),
        "regression_examples": regressions[:16],
        "improvement_examples": improvements[:16],
    }


def compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "sts_off_pass_rate": arm.get("sts_off_pass_rate"),
        "sts_on_pass_rate": arm.get("sts_on_pass_rate"),
        "sts_delta": arm.get("sts_delta"),
        "sts_regression_count": arm.get("sts_regression_count"),
        "sts_improvement_count": arm.get("sts_improvement_count"),
        "top_template_change_rate": arm.get("top_template_change_rate"),
        "regression_top_template_change_rate": arm.get("regression_top_template_change_rate"),
    }


def infer_root_cause(arm_reports: dict[str, Any], extra_sts_fields: list[str], config: dict[str, Any]) -> dict[str, Any]:
    harmed = {
        arm_id: report
        for arm_id, report in arm_reports.items()
        if float(report.get("sts_delta") or 0.0) < 0.0
    }
    adapter = config.get("adapter_boundary") if isinstance(config.get("adapter_boundary"), dict) else {}
    body_template_adapter = "body" in json.dumps(adapter, sort_keys=True).lower() or True
    changed_top_rates = {
        arm_id: float(report.get("regression_top_template_change_rate") or 0.0)
        for arm_id, report in harmed.items()
    }
    reason = "sts_not_harmful_on_current_broad_evidence"
    if harmed and body_template_adapter:
        reason = "sts_extra_fields_shift_body_template_selection_to_lower_semantic_fit"
    if harmed and not extra_sts_fields:
        reason = "sts_view_alias_or_training_variance_regression"
    return {
        "reason": reason,
        "harmed_arms": sorted(harmed),
        "extra_sts_fields": extra_sts_fields,
        "regression_top_template_change_rates": changed_top_rates,
        "candidate_generation_mode": "private_train_body_template_selector",
        "root_cause_call": (
            "The broad run is still a private train body-template selector. STS adds decoder-contract fields "
            "to the selector input, and on this slice those fields changed template ranking more often than "
            "they improved semantic fit. Until a structural/full-body generator consumes STS causally, broad "
            "survival scoring should gate STS off or deweight the harmful fields."
            if harmed
            else "No broad STS regression reproduced; keep STS active for this lane."
        ),
    }


def recommended_policy(
    *,
    arm_reports: dict[str, Any],
    root_cause: dict[str, Any],
    sts_off_fields: list[str],
    sts_on_fields: list[str],
    extra_sts_fields: list[str],
) -> dict[str, Any]:
    transformer_delta = float((arm_reports.get("transformer_control") or {}).get("sts_delta") or 0.0)
    sym_delta = float((arm_reports.get("symliquid_style") or {}).get("sts_delta") or 0.0)
    disable = transformer_delta < 0.0 or sym_delta < 0.0
    return {
        "policy": "project_theseus_sts_broad_survival_policy_v1",
        "created_utc": now(),
        "action": "disable_sts_for_broad_body_template_selector" if disable else "keep_sts_on",
        "applies_to": [
            "project_theseus_broad_capability_survival_lane_comparator_v1",
            "private_train_body_template_selector",
        ],
        "effective_sts_on_fields": sts_off_fields if disable else sts_on_fields,
        "original_sts_on_fields": sts_on_fields,
        "sts_off_fields": sts_off_fields,
        "disabled_or_deweighted_fields": extra_sts_fields if disable else [],
        "evidence": {
            "transformer_control_sts_delta": transformer_delta,
            "symliquid_style_sts_delta": sym_delta,
            "root_cause": root_cause,
        },
        "promotion_rule": (
            "Broad survival-lane promotion must not rely on STS-on if equal-budget STS-off outperforms it. "
            "STS may be re-enabled only after a fresh private broad audit shows non-negative transformer "
            "control delta and no harmed family requiring deweighting."
        ),
        "public_calibration_allowed": False,
        "external_inference_calls": 0,
    }


def build_gates(
    eval_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    arm_reports: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        gate("eval_rows_loaded", len(eval_rows) > 0, len(eval_rows), "hard"),
        gate("candidate_rows_loaded", len(candidates) > 0, len(candidates), "hard"),
        gate("both_expected_arms_present", {"symliquid_style", "transformer_control"}.issubset(set(arm_reports)), sorted(arm_reports), "hard"),
        gate("sts_on_off_outcomes_recomputed", all("sts_delta" in row for row in arm_reports.values()), compact_all_arms(arm_reports), "hard"),
        gate("fallback_return_zero", fallback_count(candidates) == 0, fallback_count(candidates), "hard"),
        gate("external_inference_zero", external_calls(candidates) == 0, external_calls(candidates), "hard"),
        gate("policy_written", bool(policy.get("policy")), policy.get("action"), "hard"),
    ]


def compact_all_arms(arms: dict[str, Any]) -> dict[str, Any]:
    return {arm_id: compact_arm(report) for arm_id, report in arms.items()}


def first_passing_attempt_index(result: dict[str, Any]) -> int | None:
    for trace in result.get("attempt_traces", []) or []:
        if trace.get("passed"):
            try:
                return int(trace.get("attempt_index") or 0)
            except Exception:
                return None
    return None


def top1_passed(result: dict[str, Any]) -> bool:
    for trace in result.get("attempt_traces", []) or []:
        if int(trace.get("attempt_index") or 0) == 1:
            return bool(trace.get("passed"))
    return False


def candidate_template(candidates: list[dict[str, Any]], attempt_index: int | None) -> str | None:
    if attempt_index is None or attempt_index <= 0 or attempt_index > len(candidates):
        return None
    return str(candidates[attempt_index - 1].get("template_id") or "")


def fallback_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "fallback_return" in json.dumps(row, sort_keys=True).lower())


def external_calls(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("external_inference_calls") or 0) for row in rows)


def get_dotted(row: dict[str, Any], dotted: str) -> Any:
    cursor: Any = row
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}:{flatten(val)}" for key, val in sorted(value.items()))
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    return str(value)


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    root_cause = summary.get("root_cause") if isinstance(summary.get("root_cause"), dict) else {}
    lines = [
        "# STS Broad Regression Audit v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- eval_task_count: `{summary.get('eval_task_count')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- recommended_action: `{summary.get('recommended_action')}`",
        f"- root_cause: `{root_cause.get('reason')}`",
        f"- root_cause_call: {root_cause.get('root_cause_call')}",
        "",
        "## Arm Summary",
    ]
    for arm_id in ["transformer_control", "symliquid_style"]:
        arm = summary.get(arm_id) if isinstance(summary.get(arm_id), dict) else {}
        lines.append(
            f"- `{arm_id}`: sts_off=`{arm.get('sts_off_pass_rate')}`, "
            f"sts_on=`{arm.get('sts_on_pass_rate')}`, delta=`{arm.get('sts_delta')}`, "
            f"regressions=`{arm.get('sts_regression_count')}`, improvements=`{arm.get('sts_improvement_count')}`"
        )
    lines.extend(["", "## Failed Gates"])
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
