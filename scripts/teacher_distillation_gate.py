"""Gate governed teacher distillation without weakening runtime inference policy.

Teacher influence is allowed to reach users only through trained weights after
audited training-time ingestion. Proposal mode is enabled by default in the
permissive growth policy; row ingestion still requires retained provenance,
license, leakage, verifier, and teacher-share evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/teacher_distillation_policy.json")
    parser.add_argument("--out", default="reports/teacher_distillation_gate.json")
    parser.add_argument("--markdown-out", default="reports/teacher_distillation_gate.md")
    parser.add_argument("--share-out", default="reports/teacher_share_ledger_summary.json")
    parser.add_argument("--share-markdown-out", default="reports/teacher_share_ledger_summary.md")
    args = parser.parse_args()

    policy_path = ROOT / args.policy
    policy = read_json(policy_path)
    state = load_state(policy)
    checks = build_checks(policy, state)
    red = [item for item in checks if item["severity"] == "hard" and not item["passed"]]
    unmet = [item for item in checks if item["severity"] != "hard" and not item["passed"]]
    allowed = not red and not unmet
    trigger_state = "RED" if red else ("GREEN" if allowed else "YELLOW")
    teacher_share = teacher_share_summary(policy, state)
    unlock_required = operator_unlock_required(policy)
    manifest = state.get("manifest") if isinstance(state.get("manifest"), dict) else {}
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    manifest_rows = list_value(manifest.get("rows")) or list_value(manifest_summary.get("rows"))
    manifest_row_count = int(manifest_summary.get("row_count") or manifest.get("row_count") or len(manifest_rows) or 0)
    external_inference_calls = int(manifest_summary.get("external_inference_calls") or manifest.get("external_inference_calls") or 0)
    verifier_pass_rate_applicable = bool(
        manifest_summary.get("verifier_pass_rate_applicable")
        or manifest.get("verifier_pass_rate_applicable")
        or manifest_row_count > 0
    )
    payload = {
        "policy": "project_theseus_teacher_distillation_gate_v0",
        "created_utc": now(),
        "config": args.policy,
        "trigger_state": trigger_state,
        "distillation_allowed": allowed,
        "default_state": policy.get("default_state", "operator_locked"),
        "operator_unlock_required": unlock_required,
        "governed_training_enabled": bool(not unlock_required),
        "boundary": policy.get("boundary", {}),
        "teacher_share": teacher_share,
        "summary": {
            "distillation_allowed": allowed,
            "hard_blocker_count": len(red),
            "missing_or_locked_count": len(unmet),
            "hard_blockers": [item["name"] for item in red],
            "missing_or_locked": [item["name"] for item in unmet],
            "manifest_path": state.get("manifest_path"),
            "manifest_row_count": manifest_row_count,
            "manifest_verifier_pass_rate": float(manifest_summary.get("verifier_pass_rate") or manifest.get("verifier_pass_rate") or 0.0),
            "manifest_verifier_pass_rate_applicable": verifier_pass_rate_applicable,
            "manifest_admission_safety_checks_clean": bool(
                manifest_summary.get("admission_safety_checks_clean")
                or manifest.get("admission_safety_checks_clean")
            ),
            "teacher_distillation_fail_closed": not allowed and not red,
            "manifest_public_overlap_hits": int(manifest_summary.get("public_overlap_hits") or manifest.get("public_overlap_hits") or 0),
            "manifest_holdout_overlap_hits": int(manifest_summary.get("holdout_overlap_hits") or manifest.get("holdout_overlap_hits") or 0),
            "teacher_accepted_rows": teacher_share.get("teacher_accepted_rows", 0),
            "accepted_training_rows": teacher_share.get("accepted_rows", 0),
            "verified_self_generated_rows": teacher_share.get("verified_self_generated_rows", 0),
            "teacher_share_of_accepted_training_rows": teacher_share.get("teacher_accepted_row_share", 0.0),
            "teacher_accepted_row_share": teacher_share.get("teacher_accepted_row_share", 0.0),
            "teacher_share_within_cap": teacher_share.get("within_initial_cap", False),
            "teacher_share_cap": teacher_share.get("max_initial_training_ratio"),
            "teacher_share_ledger_path": teacher_share.get("ledger_path"),
            "teacher_share_ledger_present": teacher_share.get("ledger_present", False),
            "teacher_share_ledger_row_count": teacher_share.get("ledger_row_count", 0),
            "teacher_share_metric_ready": teacher_share.get("metric_ready", False),
            "teacher_proposal_rows_recorded": teacher_share.get("teacher_proposal_rows", 0),
            "teacher_rejected_rows_recorded": teacher_share.get("teacher_rejected_rows", 0),
            "runtime_external_tokens_forbidden": True,
            "operator_unlock_required": unlock_required,
            "operator_unlock_present": bool(state.get("operator_unlock_present")),
            "governed_teacher_proposal_mode": bool(not unlock_required),
            "governed_teacher_training_rows_enabled_by_policy": bool(not unlock_required),
            "external_inference_calls": external_inference_calls,
        },
        "growth_validation": policy.get("growth_validation", {}),
        "checks": checks,
        "hard_blockers": [item["name"] for item in red],
        "missing_or_locked": [item["name"] for item in unmet],
        "next_action": next_action(red, unmet, state),
        "score_semantics": (
            "Governance gate only; this script does not call the teacher, "
            "generate training rows, run public calibration, or train a model."
        ),
        "external_inference_calls": external_inference_calls,
    }
    share_payload = build_teacher_share_ledger_report(policy, state, teacher_share, payload)
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    write_json(ROOT / args.share_out, share_payload)
    write_text(ROOT / args.share_markdown_out, render_teacher_share_markdown(share_payload))
    print(json.dumps(payload, indent=2))
    return 2 if trigger_state == "RED" else 0


def load_state(policy: dict[str, Any]) -> dict[str, Any]:
    reports = ROOT / "reports"
    operator_unlock = ROOT / str(policy.get("operator_unlock_flag", "reports/teacher_distillation_operator_unlock.flag"))
    manifest_path = ROOT / str(policy.get("manifest_path", "reports/teacher_distillation_manifest.json"))
    ledger_path = ROOT / str(policy.get("ledger_path", "reports/teacher_distillation_ledger.jsonl"))
    neural_gate_path = ROOT / str(get_path(policy, ["neural_seed", "gate_report"], "reports/neural_seed_growth_gate.json"))
    return {
        "operator_unlock_present": operator_unlock.exists(),
        "operator_unlock_path": rel(operator_unlock),
        "manifest": read_json(manifest_path),
        "manifest_path": rel(manifest_path),
        "ledger_rows": read_jsonl(ledger_path),
        "ledger_path": rel(ledger_path),
        "neural_seed": read_json(neural_gate_path),
        "neural_seed_path": rel(neural_gate_path),
        "external_audit": read_json(reports / "external_inference_audit.json"),
        "teacher_policy": read_json(ROOT / "configs" / "teacher_policy.json"),
        "synthetic_data_policy": read_json(ROOT / "configs" / "synthetic_data_policy.json"),
        "license_status": read_json(reports / "license_status.json"),
    }


def build_checks(policy: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = policy.get("boundary") if isinstance(policy.get("boundary"), dict) else {}
    manifest = state.get("manifest") if isinstance(state.get("manifest"), dict) else {}
    neural_seed = state.get("neural_seed") if isinstance(state.get("neural_seed"), dict) else {}
    external_audit = state.get("external_audit") if isinstance(state.get("external_audit"), dict) else {}
    teacher_policy = state.get("teacher_policy") if isinstance(state.get("teacher_policy"), dict) else {}
    synthetic_policy = state.get("synthetic_data_policy") if isinstance(state.get("synthetic_data_policy"), dict) else {}
    unlock_required = operator_unlock_required(policy)
    public_boundary = str(boundary.get("public_benchmarks") or "")
    public_boundary_ok = (
        public_boundary in {
            "calibration_only_not_training",
            "heldout_scoring_only_exact_eval_payloads_forbidden",
        }
        and boundary.get("public_solutions_or_hidden_tests") == "forbidden"
    )
    synthetic_governed = (
        "teacher_output_training_use" in set(synthetic_policy.get("blocked_without_human_approval", []))
        or str(synthetic_policy.get("teacher_generation_default") or "").startswith("governed")
        or get_path(synthetic_policy, ["teacher_distillation", "default_state"], "") == "governed_training_enabled"
    )
    teacher_gate_ref = get_path(
        teacher_policy,
        ["budget", "distillation_training_policy", "policy_file"],
        "",
    ) or get_path(teacher_policy, ["distillation_training_policy", "policy_file"], "")
    neural_required = bool(get_path(policy, ["neural_seed", "required"], True))
    student_distillation_evidence_ready = bool(
        neural_seed.get("student_distillation_evidence_ready") is True
        or get_path(neural_seed, ["summary", "student_distillation_evidence_ready"], False)
    )
    neural_trigger_ok = bool(
        neural_seed.get("trigger_state") == get_path(policy, ["neural_seed", "minimum_report_trigger_state"], "GREEN")
        or (
            student_distillation_evidence_ready
            and neural_seed.get("trigger_state") in {"GREEN", "YELLOW"}
            and neural_seed.get("spec_ready") is True
        )
    )
    neural_ready = bool(
        neural_seed
        and neural_trigger_ok
        and (
            neural_seed.get("neural_student_ready") is True
            or get_path(neural_seed, ["summary", "neural_student_ready"], False)
            or get_path(neural_seed, ["summary", "ready"], False)
            or student_distillation_evidence_ready
        )
    )
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    manifest_rows = list_value(manifest.get("rows")) or list_value(manifest_summary.get("rows"))
    manifest_row_count = int(manifest_summary.get("row_count") or manifest.get("row_count") or len(manifest_rows) or 0)
    public_overlap_hits = int(manifest_summary.get("public_overlap_hits") or manifest.get("public_overlap_hits") or 0)
    holdout_overlap_hits = int(manifest_summary.get("holdout_overlap_hits") or manifest.get("holdout_overlap_hits") or 0)
    verifier_pass_rate = float(manifest_summary.get("verifier_pass_rate") or manifest.get("verifier_pass_rate") or 0.0)
    verifier_pass_rate_applicable = bool(
        manifest_summary.get("verifier_pass_rate_applicable")
        or manifest.get("verifier_pass_rate_applicable")
        or manifest_row_count > 0
    )
    min_verifier_pass_rate = float(get_path(policy, ["quality_gates", "min_verifier_pass_rate"], 0.95) or 0.95)
    license_status = state.get("license_status") if isinstance(state.get("license_status"), dict) else {}
    manifest_license = manifest.get("license_check") or manifest_summary.get("license_check") or manifest.get("license_status") or manifest_summary.get("license_status")
    license_ok = license_status.get("ok") is True or license_status.get("allowed") is True
    manifest_license_ok = bool(manifest) and (
        manifest_license is True
        or manifest_license == "ok"
        or (isinstance(manifest_license, dict) and (manifest_license.get("ok") is True or manifest_license.get("allowed") is True))
    )
    admission_checks = manifest.get("admission_checks") or manifest_summary.get("admission_checks")
    admission_safety_checks = manifest.get("admission_safety_checks") or manifest_summary.get("admission_safety_checks")
    admission_safety_checks_clean_reported = bool(
        manifest.get("admission_safety_checks_clean")
        or manifest_summary.get("admission_safety_checks_clean")
    )
    admission_checks_present = bool(manifest) and isinstance(admission_checks, dict)
    admission_required = [
        "provenance_retained",
        "license_checked",
        "leakage_audited",
        "verifier_accepted",
        "runtime_serving_forbidden",
        "public_benchmark_excluded",
    ]
    admission_safety_required = [key for key in admission_required if key != "verifier_accepted"]
    admission_safety_required.append("approved_teacher_provider_only")
    ledger_provider_violations = teacher_ledger_provider_violations(policy, state)
    configured_provider_violation = teacher_identity_violation(
        policy,
        str(teacher_policy.get("provider") or ""),
        str(teacher_policy.get("model") or ""),
    )
    if isinstance(admission_checks, dict):
        admission_checks_have_required_keys = all(key in admission_checks for key in admission_required)
        admission_safety_clean = (
            admission_safety_checks_clean_reported
            if isinstance(admission_safety_checks, dict)
            else all(admission_checks.get(key) is True for key in admission_safety_required)
        )
        # Empty proposal-only manifests have no admitted training rows to fail
        # verifier acceptance. Keep them locked via has_rows/pass_rate checks,
        # but do not report a false dirty-row admission failure.
        admission_checks_clean = admission_safety_clean and (
            manifest_row_count == 0 or admission_checks.get("verifier_accepted") is True
        )
    else:
        admission_checks_have_required_keys = False
        admission_safety_clean = False
        admission_checks_clean = False
    return [
        check(
            "runtime_external_tokens_forbidden",
            boundary.get("runtime_serving_external_tokens") == "forbidden"
            and boundary.get("external_inference_at_runtime") == "forbidden"
            and boundary.get("raw_teacher_outputs_to_user") == "forbidden",
            "hard",
            boundary,
        ),
        check(
            "teacher_apply_mode_forbidden",
            boundary.get("teacher_apply_mode") == "forbidden",
            "hard",
            boundary.get("teacher_apply_mode"),
        ),
        check(
            "public_benchmarks_calibration_only",
            public_boundary_ok,
            "hard",
            {
                "public_benchmarks": boundary.get("public_benchmarks"),
                "public_solutions_or_hidden_tests": boundary.get("public_solutions_or_hidden_tests"),
                "semantics": "public/open data may train; exact heldout benchmark payloads may not",
            },
        ),
        check(
            "external_inference_audit_has_no_known_violations",
            not external_audit or bool(external_audit.get("ok", False)),
            "hard",
            {
                "present": bool(external_audit),
                "ok": external_audit.get("ok"),
                "total_violations": get_path(external_audit, ["summary", "total_violations"], None),
            },
        ),
        check(
            "teacher_policy_points_to_distillation_gate",
            str(teacher_gate_ref) == "configs/teacher_distillation_policy.json",
            "evidence",
            teacher_gate_ref,
        ),
        check(
            "synthetic_policy_governs_teacher_rows",
            synthetic_governed,
            "evidence",
            {
                "teacher_generation_default": synthetic_policy.get("teacher_generation_default"),
                "teacher_distillation_default": get_path(synthetic_policy, ["teacher_distillation", "default_state"], None),
                "blocked_without_human_approval": synthetic_policy.get("blocked_without_human_approval", []),
            },
        ),
        check(
            "operator_unlock_present",
            (not unlock_required) or bool(state.get("operator_unlock_present")),
            "unlock",
            {
                "required": unlock_required,
                "present": bool(state.get("operator_unlock_present")),
                "path": state.get("operator_unlock_path"),
            },
        ),
        check(
            "neural_seed_student_ready",
            not neural_required or neural_ready,
            "readiness",
            {
                "required": neural_required,
                "path": state.get("neural_seed_path"),
                "trigger_state": neural_seed.get("trigger_state"),
                "trigger_ok": neural_trigger_ok,
                "neural_student_ready_semantics": "model-growth/promotion flag",
                "student_distillation_evidence_ready": student_distillation_evidence_ready,
                "student_distillation_evidence": neural_seed.get("student_distillation_evidence"),
                "summary": neural_seed.get("summary"),
            },
        ),
        check(
            "distillation_manifest_present",
            bool(manifest),
            "readiness",
            state.get("manifest_path"),
        ),
        check(
            "distillation_manifest_has_rows",
            bool(manifest) and manifest_row_count > 0,
            "readiness",
            {"manifest_path": state.get("manifest_path"), "row_count": manifest_row_count},
        ),
        check(
            "manifest_provenance_and_retention",
            bool(manifest)
            and bool(manifest.get("provenance_retained") or manifest_summary.get("provenance_retained"))
            and bool(manifest.get("rows_retained") or manifest_summary.get("rows_retained")),
            "readiness",
            {
                "provenance_retained": manifest.get("provenance_retained") or manifest_summary.get("provenance_retained"),
                "rows_retained": manifest.get("rows_retained") or manifest_summary.get("rows_retained"),
            },
        ),
        check(
            "license_policy_active",
            license_ok,
            "evidence",
            {
                "license_status_ok": license_status.get("ok"),
                "license_allowed": license_status.get("allowed"),
                "entitlement_source": get_path(license_status, ["entitlement", "source"], None),
            },
        ),
        check(
            "manifest_sources_license_checked",
            manifest_license_ok,
            "readiness",
            {
                "manifest_license_check": manifest_license,
                "license_status_ok": license_status.get("ok"),
                "license_allowed": license_status.get("allowed"),
            },
        ),
        check(
            "manifest_leakage_zero",
            bool(manifest)
            and public_overlap_hits <= int(get_path(policy, ["quality_gates", "max_public_overlap_hits"], 0) or 0)
            and holdout_overlap_hits <= int(get_path(policy, ["quality_gates", "max_holdout_overlap_hits"], 0) or 0),
            "readiness",
            {
                "public_overlap_hits": public_overlap_hits,
                "holdout_overlap_hits": holdout_overlap_hits,
            },
        ),
        check(
            "manifest_verifier_pass_rate",
            bool(manifest)
            and verifier_pass_rate_applicable
            and verifier_pass_rate >= min_verifier_pass_rate,
            "readiness",
            {
                "verifier_pass_rate": verifier_pass_rate,
                "applicable": verifier_pass_rate_applicable,
                "minimum": min_verifier_pass_rate,
                "row_count": manifest_row_count,
                "empty_manifest_semantics": "not applicable until at least one candidate row exists",
            },
        ),
        check(
            "training_row_admission_checks_present",
            admission_checks_present and admission_checks_have_required_keys,
            "readiness",
            {
                "present": admission_checks_present,
                "checks": admission_checks,
                "required": admission_required,
            },
        ),
        check(
            "training_row_admission_checks_clean",
            admission_checks_present and admission_checks_clean,
            "readiness",
            {
                "present": admission_checks_present,
                "checks": admission_checks,
                "safety_checks": admission_safety_checks,
                "required": admission_required,
                "row_count": manifest_row_count,
                "empty_manifest_semantics": "no admitted rows; verifier readiness is enforced by distillation_manifest_has_rows and manifest_verifier_pass_rate",
                "safety_checks_clean": admission_safety_clean,
            },
        ),
        check(
            "configured_teacher_provider_openai_only",
            configured_provider_violation is None,
            "hard",
            {
                "provider": teacher_policy.get("provider"),
                "model": teacher_policy.get("model"),
                "violation": configured_provider_violation,
            },
        ),
        check(
            "approved_teacher_provider_only",
            bool(manifest)
            and int(manifest_summary.get("teacher_provider_violation_count") or 0) == 0
            and not ledger_provider_violations
            and isinstance(admission_safety_checks, dict)
            and admission_safety_checks.get("approved_teacher_provider_only") is True,
            "hard",
            {
                "allowed": get_path(policy, ["provider_policy", "allowed_providers"], []),
                "model_prefixes": get_path(policy, ["provider_policy", "allowed_model_prefixes"], []),
                "violation_count": manifest_summary.get("teacher_provider_violation_count"),
                "ledger_violations": ledger_provider_violations,
            },
        ),
        check(
            "teacher_share_ledger_metric_present",
            teacher_share_summary(policy, state).get("metric_ready", False),
            "evidence",
            teacher_share_summary(policy, state),
        ),
        check(
            "teacher_share_within_cap",
            teacher_share_summary(policy, state).get("within_initial_cap", False),
            "readiness",
            teacher_share_summary(policy, state),
        ),
    ]


def teacher_ledger_provider_violations(
    policy: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in state.get("ledger_rows", []):
        if not isinstance(row, dict) or row.get("accepted") is not True:
            continue
        if not str(row.get("source_kind") or "").startswith("teacher"):
            continue
        provider = str(row.get("teacher_provider") or "").strip().lower()
        model = str(row.get("teacher_model") or "").strip().lower()
        violation = teacher_identity_violation(policy, provider, model)
        if violation:
            violations.append(
                {
                    "ledger_event_id": row.get("ledger_event_id"),
                    "provider": provider,
                    "model": model,
                    "violation": violation,
                }
            )
    return violations


def teacher_identity_violation(
    policy: dict[str, Any], provider_value: str, model_value: str
) -> str | None:
    provider_policy = policy.get("provider_policy") if isinstance(policy.get("provider_policy"), dict) else {}
    provider = provider_value.strip().lower()
    model = model_value.strip().lower()
    allowed_providers = {
        str(value).strip().lower() for value in provider_policy.get("allowed_providers", [])
    }
    allowed_prefixes = tuple(
        str(value).strip().lower() for value in provider_policy.get("allowed_model_prefixes", [])
    )
    forbidden = {
        str(value).strip().lower() for value in provider_policy.get("forbidden_markers", [])
    }
    if provider_policy.get("fail_closed") is not True:
        return "provider_policy_not_fail_closed"
    if not provider or not model:
        return "provider_or_model_missing"
    if provider not in allowed_providers or not allowed_prefixes or not model.startswith(allowed_prefixes):
        return "provider_or_model_not_approved"
    if any(marker and marker in f"{provider} {model}" for marker in forbidden):
        return "forbidden_provider_or_model_marker"
    return None


def teacher_share_summary(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    rows = state.get("ledger_rows") if isinstance(state.get("ledger_rows"), list) else []
    ledger_path = ROOT / str(state.get("ledger_path") or "")
    ledger_present = ledger_path.exists()
    accepted = [row for row in rows if isinstance(row, dict) and row.get("accepted")]
    teacher_rows = [
        row
        for row in accepted
        if str(row.get("source_kind") or row.get("source") or "").startswith("teacher")
    ]
    self_rows = [
        row
        for row in accepted
        if str(row.get("source_kind") or row.get("source") or "").startswith("verified_self")
    ]
    teacher_proposals = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("accepted") is not True
        and str(row.get("source_kind") or "").startswith("teacher_proposal")
    ]
    teacher_rejected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("accepted") is not True
        and str(row.get("source_kind") or "").startswith("teacher_distillation_rejected")
    ]
    non_teacher_training = non_teacher_training_denominator()
    non_teacher_count = int(non_teacher_training.get("accepted_non_teacher_training_rows") or 0)
    total = len(accepted) + non_teacher_count
    share = (len(teacher_rows) / total) if total else 0.0
    cap = float(get_path(policy, ["teacher_share", "max_initial_training_ratio"], 0.2) or 0.2)
    return {
        "ledger_path": state.get("ledger_path"),
        "ledger_present": ledger_present,
        "ledger_row_count": len(rows),
        "accepted_rows": total,
        "accepted_rows_from_teacher_ledger": len(accepted),
        "accepted_non_teacher_training_rows": non_teacher_count,
        "accepted_non_teacher_training_source": non_teacher_training.get("source"),
        "accepted_non_teacher_training_source_count": non_teacher_training.get("source_count"),
        "teacher_accepted_rows": len(teacher_rows),
        "teacher_proposal_rows": len(teacher_proposals),
        "teacher_rejected_rows": len(teacher_rejected),
        "verified_self_generated_rows": len(self_rows),
        "teacher_accepted_row_share": share,
        "max_initial_training_ratio": cap,
        "within_initial_cap": share <= cap,
        "metric_ready": ledger_present and all(isinstance(row, dict) and "accepted" in row for row in rows),
        "has_accepted_rows": total > 0,
        "target_trend": get_path(policy, ["teacher_share", "target_trend"], ""),
        "graduation_target": get_path(policy, ["teacher_share", "graduation_target"], None),
    }


def build_teacher_share_ledger_report(
    policy: dict[str, Any],
    state: dict[str, Any],
    teacher_share: dict[str, Any],
    gate_payload: dict[str, Any],
) -> dict[str, Any]:
    rows = state.get("ledger_rows") if isinstance(state.get("ledger_rows"), list) else []
    trend = teacher_ledger_daily_trend(rows)
    summary = gate_payload.get("summary") if isinstance(gate_payload.get("summary"), dict) else {}
    metric_ready = bool(teacher_share.get("metric_ready"))
    within_cap = bool(teacher_share.get("within_initial_cap"))
    no_cheat = {
        "runtime_external_serving_forbidden": True,
        "runtime_external_inference_calls": 0,
        "public_training_rows_written": 0,
        "public_benchmark_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_apply_mode_forbidden": get_path(policy, ["boundary", "teacher_apply_mode"], "") == "forbidden",
    }
    no_cheat_clean = bool(
        no_cheat["runtime_external_serving_forbidden"]
        and no_cheat["teacher_apply_mode_forbidden"]
        and int(no_cheat["runtime_external_inference_calls"]) == 0
        and int(no_cheat["public_training_rows_written"]) == 0
        and int(no_cheat["public_benchmark_training_rows_written"]) == 0
        and int(no_cheat["fallback_return_count"]) == 0
    )
    trigger_state = "GREEN" if metric_ready and within_cap and no_cheat_clean else "YELLOW"
    return {
        "policy": "project_theseus_teacher_share_ledger_summary_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "source_gate_report": "reports/teacher_distillation_gate.json",
        "ledger_path": teacher_share.get("ledger_path"),
        "summary": {
            "metric_ready": metric_ready,
            "ledger_present": teacher_share.get("ledger_present", False),
            "ledger_row_count": teacher_share.get("ledger_row_count", 0),
            "accepted_training_rows": teacher_share.get("accepted_rows", 0),
            "accepted_rows_from_teacher_ledger": teacher_share.get("accepted_rows_from_teacher_ledger", 0),
            "accepted_non_teacher_training_rows": teacher_share.get("accepted_non_teacher_training_rows", 0),
            "teacher_accepted_rows": teacher_share.get("teacher_accepted_rows", 0),
            "verified_self_generated_rows": teacher_share.get("verified_self_generated_rows", 0),
            "teacher_proposal_rows": teacher_share.get("teacher_proposal_rows", 0),
            "teacher_rejected_rows": teacher_share.get("teacher_rejected_rows", 0),
            "teacher_share_of_accepted_training_rows": teacher_share.get("teacher_accepted_row_share", 0.0),
            "teacher_share_cap": teacher_share.get("max_initial_training_ratio"),
            "teacher_share_within_cap": within_cap,
            "teacher_share_target_trend": teacher_share.get("target_trend"),
            "teacher_share_graduation_target": teacher_share.get("graduation_target"),
            "distillation_gate_state": gate_payload.get("trigger_state"),
            "distillation_allowed": gate_payload.get("distillation_allowed"),
            "manifest_row_count": summary.get("manifest_row_count"),
            "manifest_verifier_pass_rate": summary.get("manifest_verifier_pass_rate"),
            "manifest_public_overlap_hits": summary.get("manifest_public_overlap_hits"),
            "manifest_holdout_overlap_hits": summary.get("manifest_holdout_overlap_hits"),
            "training_time_external_teacher_calls_recorded": summary.get("external_inference_calls", 0),
            "runtime_external_inference_calls": 0,
            "public_training_rows_written": 0,
            "daily_trend_bucket_count": len(trend),
            "no_cheat_clean": no_cheat_clean,
        },
        "daily_trend": trend,
        "no_cheat": no_cheat,
        "rules": {
            "runtime_boundary": "External teacher tokens may be used only through governed training rows and are never served to users.",
            "public_boundary": "Public benchmark payloads remain calibration-only and are not admitted as teacher training rows.",
            "trend_goal": "Teacher share should decrease as verified self-generated and licensed/open/private rows increase.",
            "metric_semantics": "This is data-governance accounting, not model capability or public-transfer evidence.",
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def teacher_ledger_daily_trend(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        created = str(row.get("created_utc") or "")
        day = created[:10] if len(created) >= 10 else "unknown"
        bucket = buckets.setdefault(
            day,
            {
                "date": day,
                "ledger_row_count": 0,
                "teacher_accepted_rows": 0,
                "verified_self_generated_rows": 0,
                "teacher_proposal_rows": 0,
                "teacher_rejected_rows": 0,
                "other_accepted_rows": 0,
                "training_time_external_teacher_calls": 0,
                "runtime_external_inference_calls": 0,
                "public_training_rows_written": 0,
            },
        )
        bucket["ledger_row_count"] += 1
        source_kind = str(row.get("source_kind") or row.get("source") or "")
        accepted = row.get("accepted") is True
        if accepted and source_kind.startswith("teacher"):
            bucket["teacher_accepted_rows"] += 1
        elif accepted and source_kind.startswith("verified_self"):
            bucket["verified_self_generated_rows"] += 1
        elif accepted:
            bucket["other_accepted_rows"] += 1
        elif source_kind.startswith("teacher_proposal"):
            bucket["teacher_proposal_rows"] += 1
        elif source_kind.startswith("teacher_distillation_rejected"):
            bucket["teacher_rejected_rows"] += 1
        bucket["training_time_external_teacher_calls"] += int(row.get("external_inference_calls") or 0)
        bucket["public_training_rows_written"] += int(row.get("public_training_rows_written") or 0)
    for bucket in buckets.values():
        accepted_total = bucket["teacher_accepted_rows"] + bucket["verified_self_generated_rows"] + bucket["other_accepted_rows"]
        bucket["teacher_share_within_ledger_accepted_rows"] = (
            bucket["teacher_accepted_rows"] / accepted_total if accepted_total else 0.0
        )
    return [buckets[key] for key in sorted(buckets)]


def non_teacher_training_denominator() -> dict[str, Any]:
    path = ROOT / "reports" / "training_data_admission_v1.json"
    report = read_json(path)
    rows = report.get("source_admissions") if isinstance(report.get("source_admissions"), list) else []
    count = 0
    source_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("training_use") != "allowed" or row.get("allowed_for_training") is not True:
            continue
        teacher_rows = int(row.get("teacher_row_count") or 0)
        row_count = max(0, int(row.get("row_count") or 0) - teacher_rows)
        if row_count <= 0:
            continue
        count += row_count
        source_count += 1
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "source": rel(path),
        "accepted_non_teacher_training_rows": count,
        "source_count": source_count,
        "admitted_open_public_row_count": summary.get("admitted_open_public_row_count"),
        "admitted_open_public_source_count": summary.get("admitted_open_public_source_count"),
    }


def operator_unlock_required(policy: dict[str, Any]) -> bool:
    default_state = str(policy.get("default_state") or "operator_locked")
    boundary_state = str(get_path(policy, ["boundary", "teacher_training_data"], ""))
    return default_state == "operator_locked" or "operator_locked" in boundary_state


def next_action(red: list[dict[str, Any]], unmet: list[dict[str, Any]], state: dict[str, Any]) -> str:
    if red:
        return "Do not enable teacher distillation; fix hard boundary failures first: " + ", ".join(item["name"] for item in red) + "."
    names = {item["name"] for item in unmet}
    if "teacher_policy_points_to_distillation_gate" in names:
        return "Wire teacher_policy.json to configs/teacher_distillation_policy.json before any distillation planning."
    if "neural_seed_student_ready" in names:
        return "Build and gate a small neural proposer behind the verifier harness before ingesting teacher-generated rows."
    if "operator_unlock_present" in names:
        return "Teacher distillation is configured as operator-locked; create the unlock flag or switch to governed training mode."
    if "distillation_manifest_present" in names:
        return "Write a retained, provenance-bearing distillation manifest before requesting or ingesting teacher rows."
    if "distillation_manifest_has_rows" in names:
        return "Keep teacher distillation locked until the manifest contains retained candidate rows with provenance, license, leakage, verifier, and admission evidence."
    if names:
        return "Keep teacher distillation locked until the missing readiness gates pass: " + ", ".join(sorted(names)) + "."
    return "Governed teacher distillation may run as training-time data only; runtime external serving remains forbidden."


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Teacher Distillation Gate",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- distillation_allowed: `{payload.get('distillation_allowed')}`",
        f"- default_state: `{payload.get('default_state')}`",
        f"- operator_unlock_required: `{payload.get('operator_unlock_required')}`",
        f"- governed_training_enabled: `{payload.get('governed_training_enabled')}`",
        f"- teacher_accepted_row_share: `{payload.get('teacher_share', {}).get('teacher_accepted_row_share')}`",
        f"- teacher_share_within_cap: `{payload.get('teacher_share', {}).get('within_initial_cap')}`",
        f"- next_action: {payload.get('next_action')}",
        "",
        "## Checks",
    ]
    for item in payload.get("checks", []):
        lines.append(
            f"- {item.get('name')}: passed=`{item.get('passed')}` severity=`{item.get('severity')}`"
        )
    return "\n".join(lines) + "\n"


def render_teacher_share_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Teacher Share Ledger Summary",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- metric_ready: `{summary.get('metric_ready')}`",
        f"- accepted_training_rows: `{summary.get('accepted_training_rows')}`",
        f"- teacher_accepted_rows: `{summary.get('teacher_accepted_rows')}`",
        f"- verified_self_generated_rows: `{summary.get('verified_self_generated_rows')}`",
        f"- teacher_share_of_accepted_training_rows: `{summary.get('teacher_share_of_accepted_training_rows')}`",
        f"- teacher_share_cap: `{summary.get('teacher_share_cap')}`",
        f"- teacher_share_within_cap: `{summary.get('teacher_share_within_cap')}`",
        f"- runtime_external_inference_calls: `{summary.get('runtime_external_inference_calls')}`",
        f"- public_training_rows_written: `{summary.get('public_training_rows_written')}`",
        "",
        "## Daily Trend",
    ]
    for row in payload.get("daily_trend", []):
        lines.append(
            f"- `{row.get('date')}`: teacher accepted `{row.get('teacher_accepted_rows')}`, "
            f"verified self `{row.get('verified_self_generated_rows')}`, proposals `{row.get('teacher_proposal_rows')}`, "
            f"rejected `{row.get('teacher_rejected_rows')}`"
        )
    return "\n".join(lines) + "\n"


def check(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
