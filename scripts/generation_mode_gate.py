#!/usr/bin/env python3
"""Generation-mode accounting gate for Theseus.

This gate separates raw proposed work from accepted and verified output. It is
designed to stop a common failure mode in this repo: optimizing fanout/decode
speed while verifier-passing output stays flat or regresses.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generation_architecture_contracts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "generation_mode_registry.json"
DEFAULT_REPORT = ROOT / "reports" / "generation_mode_registry.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "generation_mode_registry.md"

REQUIRED_MODE_FIELDS = {
    "id",
    "family",
    "status",
    "route",
    "execution_backend",
    "report_refs",
    "accounting_contract",
    "promotion_rules",
    "non_claims",
}

REQUIRED_ACCOUNTING_KEYS = {
    "proposed_tokens_or_spans_source",
    "accepted_tokens_or_spans_source",
    "task_pass_count_source",
    "verifier_wall_time_ms_source",
    "runtime_memory_mib_source",
}

PROMOTION_STATUSES = {"promotion_candidate", "default", "production", "promoted"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    mode_reports = [audit_mode(record) for record in list_dicts(config.get("mode_records"))]
    by_id = {row["id"]: row for row in mode_reports}
    comparison_reports = [audit_comparison(row, by_id) for row in list_dicts(config.get("comparisons"))]
    boundary_gates = audit_boundaries(dict_value(config.get("hard_boundaries")))
    architecture_contracts = generation_architecture_contracts.run_reference_suite(
        generation_architecture_contracts.load_contract()
    )

    hard_gaps = [
        gap
        for mode in mode_reports
        for gap in mode["hard_gaps"]
    ] + [
        gap
        for comparison in comparison_reports
        for gap in comparison["hard_gaps"]
    ] + [
        gate for gate in boundary_gates if gate["severity"] == "hard" and not gate["passed"]
    ]
    if architecture_contracts["trigger_state"] != "GREEN":
        hard_gaps.append({"mode_id": "generation_architecture_contracts", "kind": "architecture_contract_suite_not_green", "severity": "hard", "evidence": architecture_contracts})
    warnings = [
        warning
        for mode in mode_reports
        for warning in mode["warnings"]
    ] + [
        warning
        for comparison in comparison_reports
        for warning in comparison["warnings"]
    ]

    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    promotable = [row for row in comparison_reports if row["promotable"]]
    summary = {
        "config": rel(config_path),
        "mode_count": len(mode_reports),
        "comparison_count": len(comparison_reports),
        "promotable_comparison_count": len(promotable),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "modes_with_missing_report_refs": sum(1 for row in mode_reports if row["missing_report_refs"]),
        "modes_with_task_pass_evidence": sum(1 for row in mode_reports if row["metrics"]["task_pass_count"] > 0),
        "modes_with_fallback_burden": sum(1 for row in mode_reports if row["metrics"]["repair_or_fallback_count"] > 0),
        "mean_accepted_span_per_second": rounded_mean([row["metrics"]["accepted_span_per_second"] for row in mode_reports if row["metrics"]["accepted_span_per_second"] is not None]),
        "mean_useful_solution_per_second": rounded_mean([row["metrics"]["useful_solution_per_second"] for row in mode_reports if row["metrics"]["useful_solution_per_second"] is not None]),
        "architecture_mode_count": architecture_contracts["summary"]["mode_count"],
        "architecture_included_mode_count": architecture_contracts["summary"]["included_mode_count"],
        "architecture_retired_first_campaign_mode_count": architecture_contracts["summary"]["retired_first_campaign_mode_count"],
        "architecture_mutation_case_count": architecture_contracts["summary"]["mutation_case_count"],
        "architecture_mutation_passed_count": architecture_contracts["summary"]["mutation_passed_count"],
        "architecture_mtp_mlx_canary_passed": architecture_contracts["summary"]["mtp_canary_passed"],
        "architecture_optimizer_exposure_steps": architecture_contracts["summary"]["optimizer_exposure_steps"],
    }
    return {
        "policy": "project_theseus_generation_mode_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "boundary_gates": boundary_gates,
        "mode_records": mode_reports,
        "comparisons": comparison_reports,
        "architecture_contracts": architecture_contracts,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "speed_claim": "Speed claims must report accepted spans per second and useful verified solutions per second separately.",
            "promotion": "A faster mode is not promotable if verifier pass, integrity, context adequacy, no-cheat counters, or fallback burden regresses.",
            "kv_claim": "VCM descriptor reuse is not native model KV/prefix-cache parity unless a model-runtime lifecycle test passes.",
            "learned_generation": "Generation-mode acceleration does not bypass candidate-integrity or learned-generation claim rules.",
            "first_campaign": "AR is the frozen base; MTP is the sole included checkpoint-shaping auxiliary at zero initial weight; speculative decoding is post-hoc and disabled; Medusa, EAGLE, LayerSkip, and sketch-first/LLaDA are retired from the first campaign with explicit re-entry conditions.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_boundaries(boundaries: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("public_benchmark_training_forbidden", boundaries.get("public_benchmark_training_allowed") is False, "hard", boundaries.get("public_benchmark_training_allowed")),
        gate("runtime_external_inference_forbidden", boundaries.get("runtime_external_inference_allowed") is False, "hard", boundaries.get("runtime_external_inference_allowed")),
        gate("fallback_template_router_tool_credit_forbidden", boundaries.get("fallback_template_router_tool_credit_as_learned_generation") is False, "hard", boundaries.get("fallback_template_router_tool_credit_as_learned_generation")),
        gate("raw_throughput_only_promotion_forbidden", boundaries.get("promote_on_raw_throughput_only") is False, "hard", boundaries.get("promote_on_raw_throughput_only")),
        gate("mixed_metric_overclaim_forbidden", boundaries.get("mix_raw_accepted_and_success_metrics") is False, "hard", boundaries.get("mix_raw_accepted_and_success_metrics")),
    ]


def audit_mode(record: dict[str, Any]) -> dict[str, Any]:
    mode_id = str(record.get("id") or "<missing-id>")
    missing = sorted(REQUIRED_MODE_FIELDS - set(record))
    accounting = dict_value(record.get("accounting_contract"))
    missing_accounting = sorted(REQUIRED_ACCOUNTING_KEYS - set(accounting))
    refs = [str(x) for x in list_values(record.get("report_refs"))]
    metric_path = str(record.get("metric_path") or "")
    evidence = [read_report_ref(ref, metric_path=metric_path) for ref in refs]
    metrics = combine_metrics(evidence)
    missing_refs = [row["path"] for row in evidence if not row["present"]]
    missing_metric_paths = [row["path"] for row in evidence if row["present"] and not row["metric_path_present"]]

    status = str(record.get("status") or "")
    promotion_status = status in PROMOTION_STATUSES
    no_cheat_failures = no_cheat_failures_for(metrics)
    hard_gaps = []
    warnings = []
    if missing:
        hard_gaps.append(mode_gap(mode_id, "missing_mode_fields", {"fields": missing}))
    if missing_accounting:
        hard_gaps.append(mode_gap(mode_id, "missing_accounting_fields", {"fields": missing_accounting}))
    if no_cheat_failures:
        hard_gaps.append(mode_gap(mode_id, "no_cheat_counter_failure", {"failures": no_cheat_failures}))
    if promotion_status and missing_refs:
        hard_gaps.append(mode_gap(mode_id, "promotion_mode_missing_report_refs", {"missing": missing_refs}))
    if promotion_status and missing_metric_paths:
        hard_gaps.append(mode_gap(mode_id, "promotion_mode_missing_metric_path", {"missing": missing_metric_paths, "metric_path": metric_path}))
    if promotion_status and metrics["task_pass_count"] <= 0:
        hard_gaps.append(mode_gap(mode_id, "promotion_mode_without_task_pass_evidence", metrics))
    if promotion_status and metrics["integrity_mismatch_count"] > 0:
        hard_gaps.append(mode_gap(mode_id, "promotion_mode_with_integrity_mismatches", metrics))
    if promotion_status and metrics["candidate_manifest_equal"] is not True:
        hard_gaps.append(mode_gap(mode_id, "promotion_mode_without_exact_candidate_parity", metrics))
    if missing_refs and not promotion_status:
        warnings.append(mode_gap(mode_id, "missing_report_refs", {"missing": missing_refs}, severity="warning"))
    if missing_metric_paths and not promotion_status:
        warnings.append(mode_gap(mode_id, "missing_metric_path", {"missing": missing_metric_paths, "metric_path": metric_path}, severity="warning"))
    if metrics["runtime_ms"] is None and refs:
        warnings.append(mode_gap(mode_id, "runtime_ms_missing", {"refs": refs}, severity="warning"))
    if not refs and status != "planned":
        warnings.append(mode_gap(mode_id, "non_planned_mode_has_no_reports", {"status": status}, severity="warning"))
    if len(list_values(record.get("non_claims"))) < 1:
        warnings.append(mode_gap(mode_id, "missing_non_claims", {}, severity="warning"))

    return {
        "id": mode_id,
        "family": str(record.get("family") or ""),
        "status": status,
        "route": str(record.get("route") or ""),
        "execution_backend": str(record.get("execution_backend") or ""),
        "metric_path": metric_path,
        "promotion_status": promotion_status,
        "report_refs": evidence,
        "missing_report_refs": missing_refs,
        "metrics": metrics,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "promotion_rules": list_values(record.get("promotion_rules")),
        "non_claims": list_values(record.get("non_claims")),
    }


def audit_comparison(row: dict[str, Any], modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison_id = str(row.get("id") or "<missing-id>")
    baseline_id = str(row.get("baseline_mode_id") or "")
    candidate_id = str(row.get("candidate_mode_id") or "")
    baseline = modes.get(baseline_id)
    candidate = modes.get(candidate_id)
    hard_gaps = []
    warnings = []
    if baseline is None:
        hard_gaps.append(comparison_gap(comparison_id, "baseline_mode_missing", {"baseline_mode_id": baseline_id}))
    if candidate is None:
        hard_gaps.append(comparison_gap(comparison_id, "candidate_mode_missing", {"candidate_mode_id": candidate_id}))
    if baseline is None or candidate is None:
        return empty_comparison(row, hard_gaps, warnings)

    base_metrics = baseline["metrics"]
    cand_metrics = candidate["metrics"]
    base_pass_rate = pass_rate(base_metrics)
    cand_pass_rate = pass_rate(cand_metrics)
    pass_non_regression = cand_pass_rate >= base_pass_rate
    integrity_non_regression = cand_metrics["integrity_mismatch_count"] <= base_metrics["integrity_mismatch_count"]
    fallback_non_regression = cand_metrics["repair_or_fallback_count"] <= base_metrics["repair_or_fallback_count"]
    no_cheat_clean = not no_cheat_failures_for(cand_metrics)
    candidate_parity = cand_metrics["candidate_manifest_equal"] is True
    accepted_speed_lift = metric_gt(cand_metrics["accepted_span_per_second"], base_metrics["accepted_span_per_second"])
    useful_speed_lift = metric_gt(cand_metrics["useful_solution_per_second"], base_metrics["useful_solution_per_second"])
    promotable = all([
        pass_non_regression,
        integrity_non_regression,
        fallback_non_regression,
        no_cheat_clean,
        candidate_parity,
        bool(accepted_speed_lift or useful_speed_lift),
        cand_metrics["task_pass_count"] > 0,
    ])
    if not promotable:
        warnings.append(
            comparison_gap(
                comparison_id,
                "comparison_not_promotable",
                {
                    "pass_non_regression": pass_non_regression,
                    "integrity_non_regression": integrity_non_regression,
                    "fallback_non_regression": fallback_non_regression,
                    "accepted_speed_lift": accepted_speed_lift,
                    "useful_speed_lift": useful_speed_lift,
                    "candidate_task_pass_count": cand_metrics["task_pass_count"],
                    "candidate_manifest_equal": cand_metrics["candidate_manifest_equal"],
                },
                severity="warning",
            )
        )
    return {
        "id": comparison_id,
        "suite": str(row.get("suite") or ""),
        "baseline_mode_id": baseline_id,
        "candidate_mode_id": candidate_id,
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "baseline_pass_rate": base_pass_rate,
        "candidate_pass_rate": cand_pass_rate,
        "accepted_speed_lift": accepted_speed_lift,
        "useful_speed_lift": useful_speed_lift,
        "pass_non_regression": pass_non_regression,
        "integrity_non_regression": integrity_non_regression,
        "fallback_non_regression": fallback_non_regression,
        "candidate_manifest_equal": candidate_parity,
        "promotable": promotable,
        "promotion_decision": str(row.get("promotion_decision") or ""),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def empty_comparison(row: dict[str, Any], hard_gaps: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or "<missing-id>"),
        "suite": str(row.get("suite") or ""),
        "baseline_mode_id": str(row.get("baseline_mode_id") or ""),
        "candidate_mode_id": str(row.get("candidate_mode_id") or ""),
        "baseline_metrics": {},
        "candidate_metrics": {},
        "promotable": False,
        "promotion_decision": str(row.get("promotion_decision") or ""),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def read_report_ref(path_text: str, *, metric_path: str = "") -> dict[str, Any]:
    path = resolve(path_text)
    row: dict[str, Any] = {
        "path": rel(path),
        "present": path.exists(),
        "trigger_state": "",
        "metric_path": metric_path,
        "metric_path_present": not metric_path,
        "metrics": empty_metrics(),
    }
    if not path.exists() or path.suffix != ".json":
        return row
    payload = read_json(path)
    row["trigger_state"] = str(payload.get("trigger_state") or payload.get("status") or "")
    view = nested_dict(payload, metric_path)
    row["metric_path_present"] = view is not None
    if view is not None:
        row["metrics"] = extract_metrics(view, root=payload)
    return row


def extract_metrics(payload: dict[str, Any], *, root: dict[str, Any] | None = None) -> dict[str, Any]:
    root = payload if root is None else root
    summary = dict_value(payload.get("summary"))
    root_summary = dict_value(root.get("summary"))
    root_generation = dict_value(root.get("generation_mode_canary"))
    runtime_ms = numeric(payload.get("generation_runtime_ms") or payload.get("runtime_ms") or payload.get("wall_time_ms") or summary.get("runtime_ms") or summary.get("wall_time_ms"))
    proposed = first_numeric(
        payload.get("candidate_count"),
        summary.get("manifest_candidate_rows"),
        summary.get("generated_candidate_rows"),
        summary.get("private_candidate_count"),
        summary.get("candidate_rows"),
        payload.get("candidate_rows"),
    )
    accepted = first_numeric(
        payload.get("passed_task_count") if "accepted_verified_output_per_second" in payload else None,
        summary.get("candidate_rows"),
        summary.get("generated_candidate_rows"),
        summary.get("private_candidate_count"),
        payload.get("candidate_rows"),
    )
    split_passes = dict_value(summary.get("split_passes"))
    task_pass_count = sum(int(v or 0) for v in split_passes.values()) if split_passes else int(first_numeric(payload.get("passed_task_count"), summary.get("task_pass_count"), summary.get("pass_count"), 0) or 0)
    task_count = int(first_numeric(payload.get("task_count"), summary.get("task_count"), root_summary.get("family_disjoint_eval_task_count"), summary.get("manifest_candidate_rows"), summary.get("candidate_rows"), 0) or 0)
    integrity = dict_value(summary.get("candidate_integrity"))
    fallback_count = int(first_numeric(payload.get("fallback_return_count"), summary.get("fallback_template_router_tool_credit_count"), root_summary.get("fallback_return_count"), 0) or 0)
    if "fallback_or_template" in dict_value(integrity.get("family_counts")):
        fallback_count += int(dict_value(integrity.get("family_counts")).get("fallback_or_template") or 0)
    runtime_seconds = runtime_ms / 1000.0 if runtime_ms and runtime_ms > 0 else None
    return {
        "runtime_ms": runtime_ms,
        "proposed_tokens_or_spans": proposed,
        "accepted_tokens_or_spans": accepted,
        "task_pass_count": task_pass_count,
        "task_count": task_count,
        "integrity_mismatch_count": int(first_numeric(payload.get("integrity_mismatch_count"), integrity.get("integrity_mismatch_count"), 0) or 0),
        "repair_or_fallback_count": fallback_count,
        "public_training_rows": int(first_numeric(payload.get("public_training_rows_written"), summary.get("public_training_rows"), payload.get("public_training_rows"), root_summary.get("public_training_rows"), root_generation.get("public_training_rows_written"), 0) or 0),
        "external_inference_calls": int(first_numeric(summary.get("external_inference_calls"), payload.get("external_inference_calls"), root_summary.get("external_inference_calls"), root_generation.get("external_inference_calls"), 0) or 0),
        "runtime_memory_mib": first_numeric(summary.get("runtime_memory_mib"), summary.get("peak_memory_mib")),
        "verifier_wall_time_ms": first_numeric(summary.get("verifier_wall_time_ms")),
        "accepted_span_per_second": first_numeric(payload.get("accepted_verified_output_per_second"), round((accepted / runtime_seconds), 6) if accepted is not None and runtime_seconds else None),
        "useful_solution_per_second": round((task_pass_count / runtime_seconds), 6) if runtime_seconds else None,
        "candidate_manifest_equal": root_generation.get("candidate_manifest_equal") if root_generation else payload.get("candidate_manifest_equal"),
    }


def combine_metrics(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row["metrics"] for row in evidence if row.get("present")]
    if not present:
        return empty_metrics()
    combined = empty_metrics()
    combined["runtime_ms"] = sum_or_none(row.get("runtime_ms") for row in present)
    for key in [
        "proposed_tokens_or_spans",
        "accepted_tokens_or_spans",
        "task_pass_count",
        "task_count",
        "integrity_mismatch_count",
        "repair_or_fallback_count",
        "public_training_rows",
        "external_inference_calls",
    ]:
        combined[key] = sum(int(row.get(key) or 0) for row in present)
    combined["runtime_memory_mib"] = max_or_none(row.get("runtime_memory_mib") for row in present)
    combined["verifier_wall_time_ms"] = sum_or_none(row.get("verifier_wall_time_ms") for row in present)
    parity_values = [row.get("candidate_manifest_equal") for row in present if row.get("candidate_manifest_equal") is not None]
    combined["candidate_manifest_equal"] = all(value is True for value in parity_values) if parity_values else None
    seconds = combined["runtime_ms"] / 1000.0 if combined["runtime_ms"] and combined["runtime_ms"] > 0 else None
    combined["accepted_span_per_second"] = round(combined["accepted_tokens_or_spans"] / seconds, 6) if seconds else None
    combined["useful_solution_per_second"] = round(combined["task_pass_count"] / seconds, 6) if seconds else None
    return combined


def empty_metrics() -> dict[str, Any]:
    return {
        "runtime_ms": None,
        "proposed_tokens_or_spans": 0,
        "accepted_tokens_or_spans": 0,
        "task_pass_count": 0,
        "task_count": 0,
        "integrity_mismatch_count": 0,
        "repair_or_fallback_count": 0,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "runtime_memory_mib": None,
        "verifier_wall_time_ms": None,
        "accepted_span_per_second": None,
        "useful_solution_per_second": None,
        "candidate_manifest_equal": None,
    }


def nested_dict(payload: dict[str, Any], dotted_path: str) -> dict[str, Any] | None:
    current: Any = payload
    if not dotted_path:
        return payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, dict) else None


def no_cheat_failures_for(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for key in ["public_training_rows", "external_inference_calls"]:
        if int(metrics.get(key) or 0) != 0:
            failures.append({"field": key, "value": metrics.get(key)})
    return failures


def pass_rate(metrics: dict[str, Any]) -> float:
    count = int(metrics.get("task_count") or 0)
    if count <= 0:
        return 0.0
    return round(int(metrics.get("task_pass_count") or 0) / count, 6)


def metric_gt(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return float(left) > float(right)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Generation Mode Registry",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- modes: `{report['summary']['mode_count']}`",
        f"- comparisons: `{report['summary']['comparison_count']}`",
        f"- promotable comparisons: `{report['summary']['promotable_comparison_count']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        "",
        "## Modes",
        "",
        "| id | status | accepted/s | useful/s | pass | fallback | integrity mismatches |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in report["mode_records"]:
        metrics = mode["metrics"]
        lines.append(
            f"| {mode['id']} | {mode['status']} | {metrics['accepted_span_per_second']} | "
            f"{metrics['useful_solution_per_second']} | {metrics['task_pass_count']}/{metrics['task_count']} | "
            f"{metrics['repair_or_fallback_count']} | {metrics['integrity_mismatch_count']} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for comparison in report["comparisons"]:
        lines.append(
            f"- `{comparison['id']}` promotable=`{comparison['promotable']}` "
            f"accepted_speed_lift=`{comparison.get('accepted_speed_lift')}` useful_speed_lift=`{comparison.get('useful_speed_lift')}` "
            f"pass_non_regression=`{comparison.get('pass_non_regression')}` decision=`{comparison.get('promotion_decision')}`"
        )
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- `{warning['id']}` `{warning['kind']}`: `{json.dumps(warning['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Rules", ""])
    for key, value in report["rules"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def mode_gap(mode_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": mode_id, "kind": kind, "severity": severity, "evidence": evidence}


def comparison_gap(comparison_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": comparison_id, "kind": kind, "severity": severity, "evidence": evidence}


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"id": name, "kind": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = numeric(value)
        if parsed is not None:
            return parsed
    return None


def sum_or_none(values: Any) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    return round(sum(parsed), 6) if parsed else None


def max_or_none(values: Any) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    return round(max(parsed), 6) if parsed else None


def rounded_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
