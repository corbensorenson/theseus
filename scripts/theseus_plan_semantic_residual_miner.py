#!/usr/bin/env python3
"""Mine shared plan-semantic residuals from the larger private token gate."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTI = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_multiseed.json"
DEFAULT_CONFIG = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_config.json"
DEFAULT_BASELINE_AUDIT = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_semantic_plan_gap_audit_seed_23.json"
DEFAULT_REJECTED_RENDERER_PROBE = ROOT / "reports" / "neural_seed_token_decoder_contract_aware_probe_seed23_semantic_plan_gap_audit.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_plan_semantic_residual_miner.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "theseus_plan_semantic_residual_miner.md"
TARGET_SELECTED_PLANS = {
    "THRESHOLD_LABELS",
    "TOP_K_FREQUENT",
    "ROOM_CAPABILITY_SUMMARY",
    "GROUP_RECORDS_BY_FIELD",
    "LONGEST_EVEN_RUN",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-report", default=str(DEFAULT_MULTI.relative_to(ROOT)))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--baseline-audit", default=str(DEFAULT_BASELINE_AUDIT.relative_to(ROOT)))
    parser.add_argument("--rejected-renderer-probe-audit", default=str(DEFAULT_REJECTED_RENDERER_PROBE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--max-examples-per-bucket", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    multiseed = read_json(resolve(args.multiseed_report))
    config = read_json(resolve(args.config))
    baseline_audit = read_json(resolve(args.baseline_audit))
    rejected_renderer_probe = read_json(resolve(args.rejected_renderer_probe_audit))
    eval_rows = read_jsonl(resolve(str(get_path(config, ["data", "eval_jsonl"], ""))))
    eval_by_id = {str(row.get("task_id") or ""): row for row in eval_rows}

    seed_rows = [row for row in multiseed.get("seed_rows", []) or [] if isinstance(row, dict)]
    examples: list[dict[str, Any]] = []
    both_fail_counts: Counter[str] = Counter()
    expected_buckets: Counter[str] = Counter()
    misleading_label_counts: Counter[str] = Counter()
    candidate_coverage: Counter[str] = Counter()
    route_strategy_counts: Counter[str] = Counter()
    top_plan_counts: Counter[str] = Counter()
    unique_task_ids: set[str] = set()

    for seed_row in seed_rows:
        seed = int(seed_row.get("seed") or 0)
        audit = read_json(resolve(str(seed_row.get("semantic_plan_audit") or "")))
        candidates = read_jsonl(resolve(str(seed_row.get("candidate_manifest") or "")))
        candidates_by_key = group_candidates(candidates)
        for task_row in audit.get("task_rows", []) or []:
            if not isinstance(task_row, dict) or task_row.get("gap_status") != "both_fail":
                continue
            task_id = str(task_row.get("task_id") or "")
            task = eval_by_id.get(task_id, {})
            unique_task_ids.add(task_id)
            expected_plan = str(task_row.get("expected_plan_diagnostic_only") or "")
            expected_shape = str(task_row.get("expected_return_shape_diagnostic_only") or "")
            family = str(task_row.get("family") or "")
            expected_bucket = f"{expected_plan}:{expected_shape}:{family}"
            expected_buckets[expected_bucket] += 1
            for arm in ["symliquid_style", "transformer_control"]:
                event = dict_or_empty(get_path(task_row, ["arms", arm, "private_eval"], {}))
                wrong_shape = str(event.get("wrong_answer_shape") or "")
                selected_mismatch = parse_plan_mismatch(wrong_shape)
                selected_plan = selected_mismatch[1] if selected_mismatch else str(event.get("selected_plan") or "")
                if selected_plan in TARGET_SELECTED_PLANS:
                    misleading_label_counts[f"{expected_plan}->{selected_plan}"] += 1
                route_strategy = str(event.get("selected_learned_internal_semantic_route_strategy") or "")
                if route_strategy:
                    route_strategy_counts[route_strategy] += 1
                top_plan = str(event.get("top_plan") or "")
                if top_plan:
                    top_plan_counts[top_plan] += 1
                arm_candidates = candidates_by_key.get((arm, task_id, "private_eval"), [])
                coverage = summarize_candidates(arm_candidates, expected_plan)
                if coverage.get("expected_plan_present"):
                    candidate_coverage["expected_plan_present"] += 1
                if coverage.get("expected_plan_top_rank"):
                    candidate_coverage["expected_plan_top_rank"] += 1
                if coverage.get("expected_plan_contract_route_present"):
                    candidate_coverage["expected_plan_contract_route_present"] += 1
                if coverage.get("expected_plan_visible_text_route_present"):
                    candidate_coverage["expected_plan_visible_text_route_present"] += 1
                if len(examples) < int(args.max_examples_per_bucket) * 8:
                    examples.append(
                        {
                            "seed": seed,
                            "task_id": task_id,
                            "arm": arm,
                            "expected_bucket": expected_bucket,
                            "expected_plan_diagnostic_only": expected_plan,
                            "expected_return_shape_diagnostic_only": expected_shape,
                            "family": family,
                            "wrong_answer_shape": wrong_shape,
                            "top_plan": top_plan,
                            "selected_rank_when_no_pass": event.get("selected_rank"),
                            "selected_plan_when_no_pass": selected_plan,
                            "selected_route_strategy_when_no_pass": route_strategy,
                            "candidate_coverage": coverage,
                            "visible_contract": visible_contract_summary(task),
                        }
                    )
            both_fail_counts[expected_bucket] += 1

    gates = [
        gate("multiseed_report_loaded_green", multiseed.get("trigger_state") == "GREEN", multiseed.get("trigger_state"), "hard"),
        gate("seed_audits_loaded", bool(seed_rows), {"seed_rows": len(seed_rows)}, "hard"),
        gate("both_fail_rows_mined", bool(expected_buckets), dict(expected_buckets.most_common(12)), "hard"),
        gate(
            "renderer_template_probe_rejected_as_capability_evidence",
            True,
            {
                "rejected_probe_path": rel(resolve(args.rejected_renderer_probe_audit)),
                "probe_present": bool(rejected_renderer_probe),
            },
            "hard",
        ),
        gate("public_calibration_not_run", True, False, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("teacher_used_false", True, False, "hard"),
        gate("promotion_locked", True, False, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    return {
        "policy": "project_theseus_plan_semantic_residual_miner_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failed else "RED",
        "source_multiseed_report": rel(resolve(args.multiseed_report)),
        "completed_seed_count": len(seed_rows),
        "unique_both_fail_task_count": len(unique_task_ids),
        "seed_arm_both_fail_event_count": sum(expected_buckets.values()) * 2,
        "expected_plan_shape_family_counts": dict(expected_buckets.most_common()),
        "target_selected_plan_label_counts": dict(misleading_label_counts.most_common()),
        "candidate_coverage_counts": dict(candidate_coverage),
        "top_plan_counts": dict(top_plan_counts.most_common(16)),
        "selected_route_strategy_when_no_pass_counts": dict(route_strategy_counts.most_common(16)),
        "rejected_renderer_template_probe": summarize_rejected_renderer_probe(
            baseline_audit,
            rejected_renderer_probe,
            rel(resolve(args.baseline_audit)),
            rel(resolve(args.rejected_renderer_probe_audit)),
        ),
        "examples": examples,
        "diagnosis": {
            "label_caveat": (
                "For both-fail rows the semantic audit records the last attempted candidate when no candidate "
                "passes. The target residual labels are therefore often misleading selected-plan labels, not the "
                "expected plan. The stronger signal is candidate coverage plus expected plan/shape/family buckets."
            ),
            "main_wall": (
                "Expected coarse plans are usually present in the candidate set, often through contract routes, "
                "but generic LIST_APPEND, DICT_GROUP_APPEND, and GENERIC_RETURN bodies are too weak for contract "
                "families that need parsing, numeric transformation, grouping, or shaped returns."
            ),
            "rejected_shortcut": (
                "A focused seed-23 probe showed that executable generic renderers keyed by decoder_contract "
                "fields can nearly close the current private gap. That is useful as a ceiling diagnostic, but it "
                "must not be counted as capability evidence because the renderer is doing too much of the work."
            ),
            "safe_next_patch": (
                "Improve the learned/ranked semantic-slot path instead: better plan/routing loss, richer slot "
                "targets, or grammar/AST-valid non-fallback statement generation. Do not add task-id branches, "
                "fallback returns, canned family bodies, public data, teacher data, or held-out solution bodies."
            ),
        },
        "gates": gates,
        "score_semantics": (
            "Residual mining only. It reads existing private diagnostic audits, candidate manifests, and visible "
            "private decoder contracts. It does not train, generate candidates, run public calibration, call a "
            "teacher, use public data, unlock promotion, or feed private eval solutions into generation."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def group_candidates(candidates: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(str(row.get("substrate_arm") or ""), str(row.get("task_id") or ""), str(row.get("phase") or ""))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("rank") or 9999))
    return grouped


def summarize_candidates(candidates: list[dict[str, Any]], expected_plan: str) -> dict[str, Any]:
    rows = []
    expected_rows = []
    for row in candidates:
        decode = dict_or_empty(row.get("body_structure_decode"))
        plan = str(decode.get("semantic_plan") or "")
        route = dict_or_empty(row.get("learned_internal_semantic_route"))
        entry = {
            "rank": row.get("rank"),
            "plan": plan,
            "rank_score": row.get("rank_score"),
            "route_strategy": route.get("strategy") or "",
            "predicted_return_shape": decode.get("predicted_return_shape") or "",
            "semantic_slot_count": decode.get("semantic_slot_count"),
        }
        rows.append(entry)
        if plan == expected_plan:
            expected_rows.append(entry)
    route_strategies = {str(row.get("route_strategy") or "") for row in expected_rows}
    return {
        "candidate_count": len(candidates),
        "expected_plan_present": bool(expected_rows),
        "expected_plan_top_rank": bool(rows and rows[0].get("plan") == expected_plan),
        "expected_plan_first_rank": expected_rows[0].get("rank") if expected_rows else None,
        "expected_plan_route_strategies": sorted(strategy for strategy in route_strategies if strategy),
        "expected_plan_contract_route_present": any(
            str(row.get("route_strategy") or "").startswith("contract_") for row in expected_rows
        ),
        "expected_plan_visible_text_route_present": any(
            str(row.get("route_strategy") or "") == "visible_text_prototype_memory" for row in expected_rows
        ),
        "top_candidates": rows[:8],
    }


def visible_contract_summary(task: dict[str, Any]) -> dict[str, Any]:
    contract = dict_or_empty(task.get("decoder_contract"))
    return_contract = dict_or_empty(contract.get("return_contract"))
    generation_plan = dict_or_empty(contract.get("generation_plan"))
    return {
        "type_family": contract.get("type_family"),
        "argument_roles": contract.get("argument_roles"),
        "return_shape": return_contract.get("shape") or contract.get("return_shape"),
        "must_preserve_container_shape": return_contract.get("must_preserve_container_shape"),
        "required_constructs": contract.get("required_constructs") or [],
        "skeleton_bias": generation_plan.get("skeleton_bias") or [],
        "metamorphic_properties": task.get("metamorphic_properties") or [],
        "visible_arg_count_hint": contract.get("visible_arg_count_hint"),
    }


def summarize_rejected_renderer_probe(
    baseline_audit: dict[str, Any],
    rejected_probe: dict[str, Any],
    baseline_path: str,
    probe_path: str,
) -> dict[str, Any]:
    baseline_summary = dict_or_empty(baseline_audit.get("summary"))
    probe_summary = dict_or_empty(rejected_probe.get("summary"))
    if not baseline_summary and not probe_summary:
        return {
            "present": False,
            "admissibility": "not_run",
            "reason": "No rejected renderer-template probe artifact was found.",
        }
    return {
        "present": bool(probe_summary),
        "admissibility": "rejected_as_capability_evidence",
        "baseline_audit": baseline_path,
        "probe_audit": probe_path,
        "baseline_gap_counts": baseline_summary.get("gap_counts"),
        "probe_gap_counts": probe_summary.get("gap_counts"),
        "baseline_symliquid_private_eval_pass_rate": baseline_summary.get("symliquid_private_eval_pass_rate"),
        "probe_symliquid_private_eval_pass_rate": probe_summary.get("symliquid_private_eval_pass_rate"),
        "baseline_transformer_private_eval_pass_rate": baseline_summary.get("transformer_private_eval_pass_rate"),
        "probe_transformer_private_eval_pass_rate": probe_summary.get("transformer_private_eval_pass_rate"),
        "delta_summary": {
            "both_pass_delta": subtract_int(
                get_path(probe_summary, ["gap_counts", "both_pass"]),
                get_path(baseline_summary, ["gap_counts", "both_pass"]),
            ),
            "both_fail_delta": subtract_int(
                get_path(probe_summary, ["gap_counts", "both_fail"]),
                get_path(baseline_summary, ["gap_counts", "both_fail"]),
            ),
        },
        "reason": (
            "The probe used executable contract-family renderer branches for generic semantic plans. "
            "It demonstrates headroom in contract-aware body construction but is too close to a canned "
            "renderer/body-template shortcut to count as learned proposer progress."
        ),
    }


def subtract_int(left: Any, right: Any) -> int | None:
    try:
        if left is None or right is None:
            return None
        return int(left) - int(right)
    except Exception:
        return None


def parse_plan_mismatch(value: str) -> tuple[str, str] | None:
    match = re.match(r"plan_mismatch:([^>]+)->(.+)$", value)
    if not match:
        return None
    return match.group(1), match.group(2)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return default
            cur = cur[part]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.is_dir():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.is_dir():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Theseus Plan-Semantic Residual Miner",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- completed_seed_count: `{report.get('completed_seed_count')}`",
        f"- unique_both_fail_task_count: `{report.get('unique_both_fail_task_count')}`",
        f"- seed_arm_both_fail_event_count: `{report.get('seed_arm_both_fail_event_count')}`",
        "",
        "## Expected Plan / Shape / Family Buckets",
        "",
    ]
    for key, count in dict_or_empty(report.get("expected_plan_shape_family_counts")).items():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Misleading Target Selected-Plan Labels", ""])
    for key, count in dict_or_empty(report.get("target_selected_plan_label_counts")).items():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Candidate Coverage", ""])
    for key, count in dict_or_empty(report.get("candidate_coverage_counts")).items():
        lines.append(f"- `{key}`: `{count}`")
    rejected_probe = dict_or_empty(report.get("rejected_renderer_template_probe"))
    if rejected_probe:
        lines.extend(["", "## Rejected Renderer-Template Probe", ""])
        lines.append(f"- admissibility: `{rejected_probe.get('admissibility')}`")
        lines.append(f"- baseline_gap_counts: `{rejected_probe.get('baseline_gap_counts')}`")
        lines.append(f"- probe_gap_counts: `{rejected_probe.get('probe_gap_counts')}`")
        lines.append(f"- delta_summary: `{rejected_probe.get('delta_summary')}`")
        lines.append(f"- reason: {rejected_probe.get('reason')}")
    lines.extend(["", "## Diagnosis", ""])
    for key, text in dict_or_empty(report.get("diagnosis")).items():
        lines.append(f"- `{key}`: {text}")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []) or []:
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
