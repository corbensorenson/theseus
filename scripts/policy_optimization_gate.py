#!/usr/bin/env python3
"""Govern policy optimization records for Theseus.

This is a ledger/audit gate, not a trainer. It makes generator, ranker,
router, VCM, verifier, and generation-mode updates explicit before they can be
treated as default behavior. The central rule is conservative: preference or
reward updates can be useful, but they do not become capability evidence unless
they preserve no-cheat boundaries and improve behavioral verifier evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "policy_optimization_program.json"
DEFAULT_REPORT = ROOT / "reports" / "policy_optimization_program.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "policy_optimization_program.md"

REQUIRED_RECORD_FIELDS = {
    "id",
    "target_layer",
    "policy_family",
    "feedback_source",
    "update_constraint",
    "drift_bound",
    "evaluation_refs",
    "governance_gate",
    "rollback_plan",
    "residuals",
    "reward_hacking_probes",
    "authority_boundary",
    "default_allowed",
    "status",
    "non_claims",
}

REQUIRED_FEEDBACK_FIELDS = {"kind", "allowed_fields", "forbidden_fields"}
REQUIRED_UPDATE_CONSTRAINT_FIELDS = {
    "method",
    "allowed_training_rows",
    "promotion_metric",
    "loss_only_promotion_allowed",
    "deterministic_renderer_credit_allowed",
}
REQUIRED_AUTHORITY_FIELDS = {
    "public_training_rows_allowed",
    "runtime_external_inference_allowed",
    "authority_expansion_allowed",
}

DEFAULT_ELIGIBLE_STATUSES = {"default", "promoted", "production"}
NON_DEFAULT_STATUSES = {"candidate", "shadow", "guarded", "queued", "quarantined", "retained"}


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
    records = list_dicts(config.get("policy_update_records"))
    required_probes = set(str(x) for x in list_values(config.get("required_reward_hacking_probes")))
    hard_boundaries = dict_value(config.get("hard_boundaries"))
    record_reports = [audit_record(record, required_probes) for record in records]
    boundary_gates = audit_hard_boundaries(hard_boundaries)

    hard_failures = [
        gap
        for record_report in record_reports
        for gap in record_report["hard_gaps"]
    ] + [gate for gate in boundary_gates if gate["severity"] == "hard" and not gate["passed"]]
    warnings = [
        warning
        for record_report in record_reports
        for warning in record_report["warnings"]
    ] + [gate for gate in boundary_gates if gate["severity"] == "warning" and not gate["passed"]]

    default_records = [row for row in record_reports if row["default_allowed"]]
    behavior_lift_defaults = [
        row
        for row in default_records
        if row["behavior_evidence"]["has_behavior_lift"] and row["reward_probe_coverage_ratio"] == 1.0
    ]
    trigger_state = "GREEN"
    if hard_failures:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    summary = {
        "config": rel(config_path),
        "record_count": len(record_reports),
        "default_allowed_count": len(default_records),
        "default_with_behavior_lift_count": len(behavior_lift_defaults),
        "hard_gap_count": len(hard_failures),
        "warning_count": len(warnings),
        "reward_probe_catalog_count": len(required_probes),
        "mean_reward_probe_coverage": rounded_mean([row["reward_probe_coverage_ratio"] for row in record_reports]),
        "records_with_missing_evidence_refs": sum(1 for row in record_reports if row["missing_evaluation_refs"]),
        "records_with_behavior_lift": sum(1 for row in record_reports if row["behavior_evidence"]["has_behavior_lift"]),
        "records_with_loss_only_risk": sum(1 for row in record_reports if row["loss_only_risk"]),
        "records_with_authority_expansion_risk": sum(1 for row in record_reports if row["authority_expansion_risk"]),
    }
    return {
        "policy": "project_theseus_policy_optimization_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "hard_boundaries": hard_boundaries,
        "boundary_gates": boundary_gates,
        "required_reward_hacking_probes": sorted(required_probes),
        "records": record_reports,
        "hard_gaps": hard_failures,
        "warnings": warnings,
        "rules": {
            "default_policy": "A policy update may become default only with clean authority boundaries, full reward-hacking probe coverage, and behavioral verifier/accepted-output lift.",
            "loss_boundary": "LM loss, selector score, or reward value is not capability evidence unless behavioral verification improves.",
            "claim_boundary": "Policy updates do not support learned-generation, public-transfer, substrate-win, or runtime-serving claims unless those claims have independent evidence.",
            "public_boundary": "Public benchmark artifacts are calibration-only and never become training rows.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_hard_boundaries(boundaries: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate(
            "public_benchmark_training_forbidden",
            boundaries.get("public_benchmark_training_allowed") is False,
            "hard",
            boundaries.get("public_benchmark_training_allowed"),
        ),
        gate(
            "runtime_external_inference_forbidden",
            boundaries.get("runtime_external_inference_allowed") is False,
            "hard",
            boundaries.get("runtime_external_inference_allowed"),
        ),
        gate(
            "authority_expansion_forbidden",
            boundaries.get("authority_expansion_allowed") is False,
            "hard",
            boundaries.get("authority_expansion_allowed"),
        ),
        gate(
            "fallback_template_router_tool_credit_forbidden",
            boundaries.get("fallback_template_router_tool_credit_as_learned_generation") is False,
            "hard",
            boundaries.get("fallback_template_router_tool_credit_as_learned_generation"),
        ),
        gate(
            "loss_only_promotion_forbidden",
            boundaries.get("loss_only_promotion_allowed") is False,
            "hard",
            boundaries.get("loss_only_promotion_allowed"),
        ),
    ]


def audit_record(record: dict[str, Any], required_probes: set[str]) -> dict[str, Any]:
    record_id = str(record.get("id") or "<missing-id>")
    missing_fields = sorted(REQUIRED_RECORD_FIELDS - set(record))
    feedback = dict_value(record.get("feedback_source"))
    update = dict_value(record.get("update_constraint"))
    authority = dict_value(record.get("authority_boundary"))
    missing_feedback = sorted(REQUIRED_FEEDBACK_FIELDS - set(feedback))
    missing_update = sorted(REQUIRED_UPDATE_CONSTRAINT_FIELDS - set(update))
    missing_authority = sorted(REQUIRED_AUTHORITY_FIELDS - set(authority))

    probes = set(str(x) for x in list_values(record.get("reward_hacking_probes")))
    missing_probes = sorted(required_probes - probes)
    reward_probe_coverage_ratio = 1.0 if not required_probes else round(len(required_probes & probes) / len(required_probes), 6)

    evaluation_refs = [str(x) for x in list_values(record.get("evaluation_refs"))]
    evidence = [read_evidence_ref(path) for path in evaluation_refs]
    missing_evaluation_refs = [row["path"] for row in evidence if not row["present"]]
    behavior_evidence = summarize_behavior_evidence(evidence)

    default_allowed = bool(record.get("default_allowed"))
    status = str(record.get("status") or "")
    status_known = status in DEFAULT_ELIGIBLE_STATUSES or status in NON_DEFAULT_STATUSES
    authority_expansion_risk = any(
        bool(authority.get(key))
        for key in REQUIRED_AUTHORITY_FIELDS
    )
    loss_only_risk = bool(update.get("loss_only_promotion_allowed")) or str(update.get("promotion_metric") or "").lower() in {
        "loss",
        "lm_loss",
        "loss_only",
    }
    renderer_credit_risk = bool(update.get("deterministic_renderer_credit_allowed"))
    feedback_forbidden_count = len(list_values(feedback.get("forbidden_fields")))
    non_claims = list_values(record.get("non_claims"))

    hard_gaps = []
    warnings = []
    if missing_fields:
        hard_gaps.append(record_gap(record_id, "missing_record_fields", {"fields": missing_fields}))
    if missing_feedback:
        hard_gaps.append(record_gap(record_id, "missing_feedback_fields", {"fields": missing_feedback}))
    if missing_update:
        hard_gaps.append(record_gap(record_id, "missing_update_constraint_fields", {"fields": missing_update}))
    if missing_authority:
        hard_gaps.append(record_gap(record_id, "missing_authority_boundary_fields", {"fields": missing_authority}))
    if missing_probes:
        hard_gaps.append(record_gap(record_id, "missing_reward_hacking_probes", {"probes": missing_probes}))
    if authority_expansion_risk:
        hard_gaps.append(record_gap(record_id, "authority_expansion_risk", authority))
    if loss_only_risk:
        hard_gaps.append(record_gap(record_id, "loss_only_promotion_risk", update))
    if renderer_credit_risk:
        hard_gaps.append(record_gap(record_id, "deterministic_renderer_credit_risk", update))
    if default_allowed and not behavior_evidence["has_behavior_lift"]:
        hard_gaps.append(record_gap(record_id, "default_without_behavior_lift", behavior_evidence))
    if default_allowed and missing_evaluation_refs:
        hard_gaps.append(record_gap(record_id, "default_with_missing_evidence_refs", {"missing": missing_evaluation_refs}))
    if default_allowed and reward_probe_coverage_ratio < 1.0:
        hard_gaps.append(record_gap(record_id, "default_without_full_reward_probe_coverage", {"coverage": reward_probe_coverage_ratio}))
    if not status_known:
        warnings.append(record_gap(record_id, "unknown_status", {"status": status}, severity="warning"))
    if not evaluation_refs:
        warnings.append(record_gap(record_id, "no_evaluation_refs", {}, severity="warning"))
    if missing_evaluation_refs and not default_allowed:
        warnings.append(record_gap(record_id, "missing_evaluation_refs", {"missing": missing_evaluation_refs}, severity="warning"))
    if feedback_forbidden_count == 0:
        warnings.append(record_gap(record_id, "no_forbidden_feedback_fields", {}, severity="warning"))
    if len(non_claims) < 2:
        warnings.append(record_gap(record_id, "weak_non_claims", {"count": len(non_claims)}, severity="warning"))

    recommendation = "eligible_for_shadow_or_candidate_tracking"
    if hard_gaps:
        recommendation = "fix_hard_gaps_before_use"
    elif default_allowed:
        recommendation = "default_allowed_with_behavior_evidence"
    elif behavior_evidence["has_behavior_lift"] and reward_probe_coverage_ratio == 1.0:
        recommendation = "may_prepare_guarded_default_review"

    return {
        "id": record_id,
        "target_layer": str(record.get("target_layer") or ""),
        "policy_family": str(record.get("policy_family") or ""),
        "status": status,
        "default_allowed": default_allowed,
        "reward_probe_coverage_ratio": reward_probe_coverage_ratio,
        "missing_reward_hacking_probes": missing_probes,
        "evaluation_refs": evidence,
        "missing_evaluation_refs": missing_evaluation_refs,
        "behavior_evidence": behavior_evidence,
        "loss_only_risk": loss_only_risk,
        "authority_expansion_risk": authority_expansion_risk,
        "deterministic_renderer_credit_risk": renderer_credit_risk,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rollback_plan": str(record.get("rollback_plan") or ""),
        "residuals": list_values(record.get("residuals")),
        "non_claims": non_claims,
        "recommendation": recommendation,
    }


def read_evidence_ref(path_text: str) -> dict[str, Any]:
    path = resolve(path_text)
    row: dict[str, Any] = {
        "path": rel(path),
        "present": path.exists(),
        "kind": "missing",
        "trigger_state": "",
        "gate_passed": None,
        "behavior_metrics": {},
    }
    if not path.exists():
        return row
    row["kind"] = "json" if path.suffix == ".json" else path.suffix.lstrip(".") or "file"
    if path.suffix != ".json":
        return row
    payload = read_json(path)
    row["trigger_state"] = str(payload.get("trigger_state") or payload.get("status") or "")
    if "gate_passed" in payload:
        row["gate_passed"] = bool(payload.get("gate_passed"))
    row["behavior_metrics"] = extract_behavior_metrics(payload)
    return row


def extract_behavior_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for section in [payload, dict_value(payload.get("summary")), dict_value(payload.get("adaptation"))]:
        candidates.append(section)
    metrics: dict[str, Any] = {}
    for section in candidates:
        for key in [
            "selected_pass_rate",
            "sts_policy_selected_pass_rate",
            "non_sts_policy_selected_pass_rate",
            "selected_pass_delta_sts_policy_minus_non_sts_policy",
            "heldout_lm_improved",
            "heldout_lm_loss_before",
            "heldout_lm_loss_after",
            "pass_rate",
            "private_verifier_pass_rate",
            "accepted_output_quality",
            "fallback_return_candidate_count",
            "public_training_rows",
            "external_inference_calls",
            "candidate_rows",
            "generated_candidate_rows",
            "integrity_verified_candidate_count",
            "integrity_mismatch_count",
        ]:
            if key in section and key not in metrics:
                metrics[key] = section.get(key)
        if "candidate_integrity" in section:
            integrity = dict_value(section.get("candidate_integrity"))
            for source_key, metric_key in [
                ("integrity_verified_candidate_count", "integrity_verified_candidate_count"),
                ("integrity_mismatch_count", "integrity_mismatch_count"),
                ("generated_candidate_count", "generated_candidate_rows"),
            ]:
                if source_key in integrity and metric_key not in metrics:
                    metrics[metric_key] = integrity.get(source_key)
        if "family_counts" in section:
            family_counts = dict_value(section.get("family_counts"))
            if "fallback_or_template" in family_counts and "fallback_return_candidate_count" not in metrics:
                metrics["fallback_return_candidate_count"] = family_counts.get("fallback_or_template")
        split_pass_metrics = strict_decode_split_pass_metrics(section)
        for key, value in split_pass_metrics.items():
            metrics.setdefault(key, value)
    return metrics


def strict_decode_split_pass_metrics(section: dict[str, Any]) -> dict[str, Any]:
    split_passes = dict_value(section.get("split_passes"))
    split_verifier = dict_value(section.get("split_private_verifier"))
    if not split_passes and not split_verifier:
        return {}
    pass_count = sum(int(value or 0) for value in split_passes.values())
    task_count = 0
    for row in split_verifier.values():
        verifier = dict_value(row)
        task_count += int(verifier.get("eval_task_count") or 0)
    if task_count <= 0:
        task_count = len(split_passes)
    return {
        "private_verifier_pass_count": pass_count,
        "private_verifier_eval_task_count": task_count,
        "private_verifier_pass_rate": round(pass_count / max(1, task_count), 6),
    }


def summarize_behavior_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    has_behavior_lift = False
    has_loss_only_lift = False
    positive_refs: list[str] = []
    loss_refs: list[str] = []
    no_cheat_failures: list[dict[str, Any]] = []
    for row in evidence:
        metrics = dict_value(row.get("behavior_metrics"))
        if not row.get("present"):
            continue
        if metrics.get("public_training_rows") not in (None, 0, "0"):
            no_cheat_failures.append({"path": row["path"], "field": "public_training_rows", "value": metrics.get("public_training_rows")})
        if metrics.get("external_inference_calls") not in (None, 0, "0"):
            no_cheat_failures.append({"path": row["path"], "field": "external_inference_calls", "value": metrics.get("external_inference_calls")})
        delta = numeric(metrics.get("selected_pass_delta_sts_policy_minus_non_sts_policy"))
        if delta is not None and delta > 0:
            has_behavior_lift = True
            positive_refs.append(row["path"])
        selected = numeric(metrics.get("selected_pass_rate") or metrics.get("sts_policy_selected_pass_rate"))
        non_sts = numeric(metrics.get("non_sts_policy_selected_pass_rate"))
        if selected is not None and non_sts is not None and selected > non_sts:
            has_behavior_lift = True
            positive_refs.append(row["path"])
        private_pass = numeric(metrics.get("private_verifier_pass_rate") or metrics.get("pass_rate"))
        if private_pass is not None and private_pass > 0:
            has_behavior_lift = True
            positive_refs.append(row["path"])
        if metrics.get("accepted_output_quality") not in (None, "", 0, 0.0, "0"):
            has_behavior_lift = True
            positive_refs.append(row["path"])
        if metrics.get("heldout_lm_improved") is True:
            has_loss_only_lift = True
            loss_refs.append(row["path"])
    return {
        "has_behavior_lift": has_behavior_lift,
        "has_loss_only_lift": has_loss_only_lift,
        "positive_behavior_refs": sorted(set(positive_refs)),
        "loss_only_refs": sorted(set(loss_refs)),
        "no_cheat_failures": no_cheat_failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Policy Optimization Program",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- records: `{report['summary']['record_count']}`",
        f"- defaults allowed: `{report['summary']['default_allowed_count']}`",
        f"- defaults with behavior lift: `{report['summary']['default_with_behavior_lift_count']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        f"- mean reward-probe coverage: `{report['summary']['mean_reward_probe_coverage']}`",
        "",
        "## Records",
        "",
        "| id | layer | status | default | probe coverage | behavior lift | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["records"]:
        lines.append(
            "| {id} | {layer} | {status} | {default} | {coverage} | {behavior} | {recommendation} |".format(
                id=record["id"],
                layer=record["target_layer"],
                status=record["status"],
                default=record["default_allowed"],
                coverage=record["reward_probe_coverage_ratio"],
                behavior=record["behavior_evidence"]["has_behavior_lift"],
                recommendation=record["recommendation"],
            )
        )
    lines.extend(["", "## Hard Gaps", ""])
    if report["hard_gaps"]:
        for gap in report["hard_gaps"]:
            lines.append(f"- `{gap['record_id']}` `{gap['kind']}`: `{json.dumps(gap['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- `{warning['record_id']}` `{warning['kind']}`: `{json.dumps(warning['evidence'], sort_keys=True)}`")
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


def record_gap(record_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {
        "record_id": record_id,
        "kind": kind,
        "severity": severity,
        "evidence": evidence,
    }


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


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
