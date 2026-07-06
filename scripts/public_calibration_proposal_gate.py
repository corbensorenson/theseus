#!/usr/bin/env python3
"""Audit a public calibration proposal before execution.

This gate does not run public benchmarks. It binds a proposed public
calibration surface to the exact frozen contract, the append-only run registry,
candidate-integrity evidence, and the training-data firewall. A fresh,
unconsumed surface may be approved by this report; an exact consumed surface is
classified and refused without turning that refusal into a hard gate failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs" / "public_benchmark_contract_v1.json"
DEFAULT_REGISTRY = ROOT / "reports" / "public_benchmark_run_registry.jsonl"
DEFAULT_ALIGNMENT = ROOT / "reports" / "public_calibration_alignment_preflight.json"
DEFAULT_TRAINING_ADMISSION = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_CANDIDATE_INTEGRITY = ROOT / "reports" / "candidate_integrity_audit.json"
DEFAULT_READINESS = ROOT / "reports" / "public_transfer_readiness_refresh_v1.json"
DEFAULT_OUT = ROOT / "reports" / "public_calibration_proposal_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "public_calibration_proposal_gate.md"
NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=rel(DEFAULT_CONTRACT))
    parser.add_argument("--run-registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--alignment-preflight", default=rel(DEFAULT_ALIGNMENT))
    parser.add_argument("--training-admission", default=rel(DEFAULT_TRAINING_ADMISSION))
    parser.add_argument("--candidate-integrity", default=rel(DEFAULT_CANDIDATE_INTEGRITY))
    parser.add_argument("--public-readiness", default=rel(DEFAULT_READINESS))
    parser.add_argument("--surface-slug", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    contract_path = resolve(args.contract)
    registry_path = resolve(args.run_registry)
    alignment_path = resolve(args.alignment_preflight)
    admission_path = resolve(args.training_admission)
    integrity_path = resolve(args.candidate_integrity)
    readiness_path = resolve(args.public_readiness)

    contract = read_json(contract_path)
    registry_rows = read_jsonl(registry_path)
    alignment = read_json(alignment_path)
    admission = read_json(admission_path)
    integrity = read_json(integrity_path)
    readiness = read_json(readiness_path)

    stage = object_field(contract.get("stage_1_code_generation_surface"))
    requested_slug = str(args.surface_slug or stage.get("slug") or "").strip()
    registry_state = classify_registry_state(requested_slug, registry_rows, registry_path)
    training_firewall = audit_training_firewall(admission)
    candidate_state = audit_candidate_integrity(integrity)
    alignment_state = audit_alignment(alignment, requested_slug)
    contract_state = audit_contract(contract, contract_path, requested_slug)
    readiness_state = audit_readiness(readiness, requested_slug)

    hard_gates = [
        gate("contract_present", contract_path.exists(), rel(contract_path)),
        gate("contract_policy_valid", contract.get("policy") == "project_theseus_public_benchmark_contract_v1", contract.get("policy")),
        gate("surface_slug_declared", bool(requested_slug), requested_slug),
        gate("run_registry_present", registry_path.exists(), rel(registry_path)),
        gate("candidate_integrity_cited_and_green", candidate_state["ready"], candidate_state),
        gate("training_firewall_cited_and_clean", training_firewall["ready"], training_firewall),
        gate("alignment_preflight_cited_and_clean", alignment_state["ready"], alignment_state),
        gate("contract_surface_matches_alignment", contract_state["ready"], contract_state),
        gate("readiness_report_cited", readiness_state["present"], readiness_state),
    ]
    hard_failed = [row for row in hard_gates if not row["passed"]]
    consumed = registry_state["exact_consumed_surface"]
    execution_allowed = not hard_failed and not consumed
    if execution_allowed:
        decision = "READY_FOR_GOVERNED_FRESH_SURFACE_EXECUTE"
        support_state = "SUPPORTED"
        transition = "fresh public calibration proposal bound to integrity/firewall/registry evidence"
    elif consumed:
        decision = "REFUSED_EXACT_CONSUMED_SURFACE_RERUN"
        support_state = "SUPPORTED_REFUSAL"
        transition = "exact consumed public surface classified and refused before execution"
    else:
        decision = "NOT_READY_MISSING_PROPOSAL_EVIDENCE"
        support_state = "UNSUPPORTED"
        transition = "proposal lacks required integrity/firewall/alignment evidence"

    hard_gaps = [gap("proposal_evidence_missing_or_invalid", {"failed_gates": hard_failed})] if hard_failed else []
    records = build_records(
        requested_slug=requested_slug,
        decision=decision,
        support_state=support_state,
        transition_reason=transition,
        contract_path=contract_path,
        registry_path=registry_path,
        alignment_path=alignment_path,
        admission_path=admission_path,
        integrity_path=integrity_path,
        readiness_path=readiness_path,
        contract_state=contract_state,
        registry_state=registry_state,
        alignment_state=alignment_state,
        training_firewall=training_firewall,
        candidate_state=candidate_state,
        readiness_state=readiness_state,
    )
    return {
        "policy": "project_theseus_public_calibration_proposal_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": {
            "surface_slug": requested_slug,
            "proposal_decision": decision,
            "execution_allowed": execution_allowed,
            "exact_consumed_surface": consumed,
            "registry_row_count": len(registry_rows),
            "matching_registry_row_count": registry_state["matching_registry_row_count"],
            "candidate_integrity_ready": candidate_state["ready"],
            "candidate_integrity_verified_candidate_count": candidate_state["integrity_verified_candidate_count"],
            "training_firewall_ready": training_firewall["ready"],
            "public_benchmark_training_payload_admitted": training_firewall["public_benchmark_training_payload_admitted"],
            "alignment_preflight_ready": alignment_state["ready"],
            "readiness_report_state": readiness_state["trigger_state"],
            "hard_failed_gate_count": len(hard_failed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "hard_gates": hard_gates,
        "hard_gaps": hard_gaps,
        "contract_state": contract_state,
        "run_registry_state": registry_state,
        "alignment_state": alignment_state,
        "training_firewall": training_firewall,
        "candidate_integrity": candidate_state,
        "readiness_state": readiness_state,
        **records,
        **NO_CHEAT,
        "non_claims": [
            "This gate does not run public calibration.",
            "This gate does not train on public benchmark prompts, tests, traces, solutions, or scores.",
            "A READY decision is permission evidence for one governed run, not a score or capability claim.",
            "A REFUSED decision proves exact-surface rerun discipline, not model improvement.",
        ],
    }


def audit_contract(contract: dict[str, Any], path: Path, slug: str) -> dict[str, Any]:
    stage = object_field(contract.get("stage_1_code_generation_surface"))
    rules = object_field(contract.get("global_rules"))
    registry_policy = object_field(rules.get("benchmark_run_registry"))
    execute_requires = list_values(rules.get("execute_requires"))
    stage_slug = str(stage.get("slug") or "")
    return {
        "path": rel(path),
        "content_hash": stable_hash(contract),
        "policy": contract.get("policy"),
        "stage_slug": stage_slug,
        "requested_slug": slug,
        "stage_status": stage.get("status"),
        "cards": list_values(stage.get("cards")),
        "seed": stage.get("seed"),
        "cases_per_card": stage.get("cases_per_card"),
        "total_task_count": stage.get("total_task_count"),
        "case_manifest": stage.get("case_manifest"),
        "run_registry": registry_policy.get("run_registry"),
        "per_surface_max_runs": registry_policy.get("per_surface_max_runs"),
        "time_period_run_cap_enabled": registry_policy.get("time_period_run_cap_enabled"),
        "calendar_throttle_enabled": registry_policy.get("calendar_throttle_enabled"),
        "fresh_surface_execution_policy": registry_policy.get("fresh_surface_execution_policy"),
        "execute_requires": execute_requires,
        "ready": contract.get("policy") == "project_theseus_public_benchmark_contract_v1"
        and bool(slug)
        and stage_slug == slug
        and rules.get("public_benchmarks_are_calibration_only") is True
        and rules.get("train_on_public_prompts_tests_solutions_traces_or_scores") is False
        and rules.get("external_inference_during_scoring") is False,
    }


def classify_registry_state(slug: str, rows: list[dict[str, Any]], registry_path: Path) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if slug
        and (
            str(row.get("surface_slug") or "") == slug
            or str(row.get("run_id") or "") == slug
            or slug in " ".join(str(part) for part in list_values(row.get("command")))
        )
    ]
    consumed_rows = [row for row in matches if row.get("consumed") is True or str(row.get("status") or "") in {"completed", "failed", "consumed"}]
    clean_rows = [row for row in matches if no_cheat_clean(row) and row.get("no_training_on_public_eval_payloads") is not False]
    latest = consumed_rows[-1] if consumed_rows else (matches[-1] if matches else {})
    return {
        "surface_slug": slug,
        "registry_path": rel(registry_path),
        "registry_row_count": len(rows),
        "matching_registry_row_count": len(matches),
        "exact_consumed_surface": bool(consumed_rows),
        "consumed_row_count": len(consumed_rows),
        "clean_matching_row_count": len(clean_rows),
        "latest_matching_run_id": latest.get("run_id"),
        "latest_matching_status": latest.get("status"),
        "latest_matching_consumed": latest.get("consumed"),
        "latest_matching_score_report_path": latest.get("score_report_path"),
        "latest_matching_residual_report_path": latest.get("residual_report_path"),
        "latest_matching_public_task_count": latest.get("public_task_count"),
        "latest_matching_pass_rate": latest.get("real_public_task_pass_rate"),
        "latest_matching_no_training_on_public_eval_payloads": latest.get("no_training_on_public_eval_payloads"),
        "latest_matching_external_inference_calls": latest.get("external_inference_calls"),
        "latest_matching_template_like_candidate_count": latest.get("template_like_candidate_count"),
        "registry_rule": "fresh surfaces may execute immediately when clean; exact consumed surfaces are refused to prevent score fishing",
        "ready": True,
    }


def audit_alignment(alignment: dict[str, Any], slug: str) -> dict[str, Any]:
    summary = object_field(alignment.get("summary"))
    if not summary:
        summary = alignment
    return {
        "path": rel(DEFAULT_ALIGNMENT),
        "content_hash": stable_hash(alignment),
        "policy": alignment.get("policy"),
        "trigger_state": alignment.get("trigger_state"),
        "contract_slug": summary.get("contract_slug"),
        "case_manifest": summary.get("case_manifest"),
        "case_manifest_row_count": summary.get("case_manifest_row_count"),
        "case_manifest_bound_to_command": summary.get("case_manifest_bound_to_command"),
        "candidate_manifest_bound_to_case_manifest": summary.get("candidate_manifest_bound_to_case_manifest"),
        "candidate_manifest_preexists_before_run": summary.get("candidate_manifest_preexists_before_run"),
        "candidate_manifest_generation_deferred_to_execute": summary.get("candidate_manifest_generation_deferred_to_execute"),
        "public_prompts_exported": summary.get("public_prompts_exported"),
        "public_tests_used": summary.get("public_tests_used"),
        "public_solutions_used": summary.get("public_solutions_used"),
        "training_rows_written": int_value(summary.get("training_rows_written")),
        "external_inference_calls": int_value(summary.get("external_inference_calls")),
        "ready": alignment.get("trigger_state") == "GREEN"
        and summary.get("contract_slug") == slug
        and summary.get("case_manifest_bound_to_command") is True
        and summary.get("candidate_manifest_bound_to_case_manifest") is True
        and summary.get("public_prompts_exported") is False
        and summary.get("public_tests_used") is False
        and summary.get("public_solutions_used") is False
        and int_value(summary.get("training_rows_written")) == 0
        and int_value(summary.get("external_inference_calls")) == 0,
    }


def audit_training_firewall(admission: dict[str, Any]) -> dict[str, Any]:
    gates = list_dicts(admission.get("gates"))
    gates_by_name = {str(row.get("name") or ""): row for row in gates}

    def gate_passed(name: str) -> bool:
        return bool(object_field(gates_by_name.get(name)).get("passed"))

    hard_failed = [row for row in gates if row.get("severity") == "hard" and row.get("passed") is not True]
    public_payload_admitted = not gate_passed("public_benchmark_payload_admitted_zero")
    return {
        "path": rel(DEFAULT_TRAINING_ADMISSION),
        "content_hash": stable_hash(admission),
        "policy": admission.get("policy"),
        "trigger_state": admission.get("trigger_state"),
        "hard_failed_gate_count": len(hard_failed),
        "public_benchmark_training_payload_admitted": public_payload_admitted,
        "public_benchmark_quarantine_not_train_allowed": gate_passed("public_benchmark_quarantine_not_train_allowed"),
        "exact_public_fingerprint_overlap_zero_for_training": gate_passed("exact_public_fingerprint_overlap_zero_for_training"),
        "fallback_returns_not_admitted": gate_passed("fallback_returns_not_admitted"),
        "raw_user_text_not_admitted": gate_passed("raw_user_text_not_admitted"),
        "teacher_rows_not_admitted_outside_distillation_gate": gate_passed("teacher_rows_not_admitted_outside_distillation_gate"),
        "teacher_distillation_manifest_public_clean": gate_passed("teacher_distillation_manifest_public_clean"),
        "external_inference_zero": gate_passed("external_inference_zero"),
        "ready": admission.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failed
        and gate_passed("public_benchmark_payload_admitted_zero")
        and gate_passed("public_benchmark_quarantine_not_train_allowed")
        and gate_passed("exact_public_fingerprint_overlap_zero_for_training")
        and gate_passed("fallback_returns_not_admitted")
        and gate_passed("raw_user_text_not_admitted")
        and gate_passed("teacher_rows_not_admitted_outside_distillation_gate")
        and gate_passed("teacher_distillation_manifest_public_clean")
        and gate_passed("external_inference_zero"),
    }


def audit_candidate_integrity(integrity: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(integrity.get("summary"))
    return {
        "path": rel(DEFAULT_CANDIDATE_INTEGRITY),
        "content_hash": stable_hash(integrity),
        "policy": integrity.get("policy"),
        "trigger_state": integrity.get("trigger_state"),
        "candidate_count": int_value(summary.get("candidate_count")),
        "audited_candidate_count": int_value(summary.get("audited_candidate_count")),
        "integrity_verified_candidate_count": int_value(summary.get("integrity_verified_candidate_count")),
        "integrity_mismatch_count": int_value(summary.get("integrity_mismatch_count")),
        "fallback_or_template_count": int_value(object_field(summary.get("family_counts")).get("fallback_or_template")),
        "learned_full_body_token_count": int_value(object_field(summary.get("family_counts")).get("learned_full_body_token")),
        "viea_spine_view_ready": summary.get("viea_spine_view_ready"),
        "viea_spine_view_record_count": int_value(summary.get("viea_spine_view_record_count")),
        "external_inference_calls": int_value(integrity.get("external_inference_calls")),
        "ready": integrity.get("trigger_state") == "GREEN"
        and int_value(summary.get("candidate_count")) > 0
        and int_value(summary.get("integrity_mismatch_count")) == 0
        and int_value(summary.get("integrity_verified_candidate_count")) > 0
        and summary.get("viea_spine_view_ready") is True
        and int_value(integrity.get("external_inference_calls")) == 0,
    }


def audit_readiness(readiness: dict[str, Any], slug: str) -> dict[str, Any]:
    summary = object_field(readiness.get("summary"))
    return {
        "path": rel(DEFAULT_READINESS),
        "content_hash": stable_hash(readiness),
        "policy": readiness.get("policy"),
        "trigger_state": readiness.get("trigger_state"),
        "present": bool(readiness),
        "contract_slug": summary.get("contract_slug"),
        "alignment_preflight_ready": summary.get("alignment_preflight_ready"),
        "hard_failed_gate_count": int_value(summary.get("hard_failed_gate_count")),
        "evidence_failed_gate_count": int_value(summary.get("evidence_failed_gate_count")),
        "latest_public_pass_rate": summary.get("latest_public_pass_rate"),
        "latest_public_task_count": summary.get("latest_public_task_count"),
        "fallback_return_candidate_count": int_value(summary.get("fallback_return_candidate_count")),
        "full_body_public_leakage_count": int_value(summary.get("full_body_public_leakage_count")),
        "ready": bool(readiness)
        and readiness.get("trigger_state") in {"GREEN", "YELLOW"}
        and summary.get("contract_slug") == slug
        and summary.get("alignment_preflight_ready") is True
        and int_value(summary.get("hard_failed_gate_count")) == 0
        and int_value(summary.get("fallback_return_candidate_count")) == 0
        and int_value(summary.get("full_body_public_leakage_count")) == 0,
    }


def build_records(**kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    slug = kwargs["requested_slug"]
    decision = kwargs["decision"]
    support_state = kwargs["support_state"]
    evidence_refs = [
        rel(kwargs["contract_path"]),
        rel(kwargs["registry_path"]),
        rel(kwargs["alignment_path"]),
        rel(kwargs["admission_path"]),
        rel(kwargs["integrity_path"]),
        rel(kwargs["readiness_path"]),
        "reports/public_calibration_proposal_gate.json",
    ]
    record_id = stable_id("public_calibration_proposal", slug, kwargs["registry_state"], kwargs["candidate_state"], kwargs["training_firewall"])
    proposal = {
        "record_type": "public_calibration_proposal",
        "record_id": f"public-calibration-proposal-{record_id}",
        "surface_slug": slug,
        "decision": decision,
        "support_state": support_state,
        "execution_allowed": decision == "READY_FOR_GOVERNED_FRESH_SURFACE_EXECUTE",
        "contract_ref": rel(kwargs["contract_path"]),
        "registry_ref": rel(kwargs["registry_path"]),
        "alignment_ref": rel(kwargs["alignment_path"]),
        "training_firewall_ref": rel(kwargs["admission_path"]),
        "candidate_integrity_ref": rel(kwargs["integrity_path"]),
        "readiness_ref": rel(kwargs["readiness_path"]),
        "required_before_execute": [
            "candidate integrity audit must be GREEN",
            "training-data firewall must prove public benchmark payloads are not training rows",
            "alignment preflight must bind candidate manifest to exact frozen case manifest",
            "run registry must classify exact-surface consumed state",
            "fresh surfaces may execute without calendar budget throttles",
            "exact consumed surfaces must be refused before execution",
        ],
        **NO_CHEAT,
        "non_claims": ["proposal gate only", "not a benchmark score", "not learned generation evidence"],
    }
    registry_record = {
        "record_type": "public_run_registry_row",
        "record_id": f"public-run-registry-state-{record_id}",
        "surface_slug": slug,
        "support_state": "SUPPORTED",
        **kwargs["registry_state"],
        **NO_CHEAT,
        "non_claims": ["registry state only", "not model capability"],
    }
    contamination = {
        "record_type": "contamination_check",
        "record_id": f"contamination-check-{record_id}",
        "surface_slug": slug,
        "support_state": "SUPPORTED" if kwargs["training_firewall"]["ready"] else "UNSUPPORTED",
        "training_firewall": kwargs["training_firewall"],
        "alignment_state": kwargs["alignment_state"],
        "candidate_integrity": kwargs["candidate_state"],
        **NO_CHEAT,
        "non_claims": ["firewall evidence only", "not public calibration execution"],
    }
    failure_taxonomy = {
        "record_type": "failure_taxonomy",
        "record_id": f"public-failure-taxonomy-{record_id}",
        "surface_slug": slug,
        "support_state": "PLANNED",
        "taxonomy": [
            "algorithm_choice",
            "return_shape",
            "io_contract",
            "verifier_mismatch",
            "timeout_runtime",
            "no_admissible",
            "parsing_syntax",
            "selector_ranking_miss",
            "public_surface_refusal",
        ],
        "source": "proposal_time_taxonomy_for_post_run_residual_mining",
        **NO_CHEAT,
        "non_claims": ["taxonomy is a reporting contract, not a result"],
    }
    repair_manifest = {
        "record_type": "private_repair_manifest",
        "record_id": f"private-repair-manifest-{record_id}",
        "surface_slug": slug,
        "support_state": "PLANNED",
        "inputs": ["post-run residual categories only; public prompts/tests/solutions/traces remain excluded from training rows"],
        "allowed_private_repairs": [
            "algorithm planning private residual rows",
            "return-shape private residual rows",
            "verifier-contract private residual rows",
            "selector/ranking private residual rows",
            "direct learned full-body generation private repairs",
        ],
        "forbidden_repairs": [
            "train on public benchmark prompts",
            "train on public tests or hidden tests",
            "train on public solutions or answer templates",
            "count fallback/template/router/tool candidates as learned generation",
        ],
        **NO_CHEAT,
        "non_claims": ["private repair plan only", "not training execution"],
    }
    failure_boundary = {
        "record_type": "failure_boundary",
        "record_id": f"public-calibration-failure-boundary-{record_id}",
        "failure_id": f"failure.public_calibration.{slug}",
        "surface_slug": slug,
        "blocked_reason": "exact_consumed_surface" if kwargs["registry_state"]["exact_consumed_surface"] else "",
        "state": "fail_closed_consumed_surface" if kwargs["registry_state"]["exact_consumed_surface"] else "fresh_surface_policy_ready",
        "terminal": False,
        "structured_non_solved": kwargs["registry_state"]["exact_consumed_surface"],
        "fallback_return_used": False,
        "learned_generation_claim_allowed": False,
        **NO_CHEAT,
        "non_claims": ["failure boundary only", "not capability evidence"],
    }
    claim_records = [
        {
            "record_type": "claim_record",
            "claim_id": f"claim.public_calibration_proposal.{slug}",
            "claim": "Public calibration proposals are now bound to candidate integrity, training firewall, alignment preflight, and exact run-registry state before execution can count as evidence.",
            "support_state": support_state,
            "evidence_refs": evidence_refs,
            **NO_CHEAT,
            "non_claims": ["implementation discipline claim", "not public score", "not learned generation"],
        }
    ]
    artifact_graph_records = [
        {
            "record_type": "artifact_graph_record",
            "artifact_id": f"artifact.public_calibration_proposal.{record_id}",
            "artifact_type": "public_calibration_proposal_gate",
            "parent_job": "public_calibration_proposal_gate",
            "source_refs": evidence_refs[:-1],
            "context_refs": [rel(kwargs["readiness_path"])],
            "context_transaction_refs": [],
            "semantic_certificate_refs": [],
            "tool_refs": [],
            "claim_refs": [claim_records[0]["claim_id"]],
            "test_refs": ["python3 scripts/public_calibration_proposal_gate.py"],
            "audit_events": [
                "contract_loaded",
                "run_registry_scanned",
                "candidate_integrity_cited",
                "training_firewall_cited",
                "alignment_preflight_cited",
                "proposal_decision_emitted",
            ],
            "replay_metadata": {
                "surface_slug": slug,
                "decision": decision,
                "registry_state": kwargs["registry_state"],
            },
            "replay_grade": "metadata_replayable_from_registered_reports",
            "environment_assumptions": ["local report registry and frozen contract are available"],
            "provenance_status": "registered_public_calibration_proposal",
            "replay_limits": ["does not rerun benchmark", "does not inspect public hidden tests"],
            "evidence_gate": {"state": support_state, **NO_CHEAT},
            "residuals": [] if decision == "READY_FOR_GOVERNED_FRESH_SURFACE_EXECUTE" else [decision],
            **NO_CHEAT,
            "non_claims": ["not public calibration execution", "not model capability"],
        }
    ]
    evidence_transition_records = [
        {
            "record_type": "evidence_transition_record",
            "record_id": f"evidence.public_calibration_proposal.{record_id}",
            "artifact_ref": "reports/public_calibration_proposal_gate.json",
            "previous_support_state": "PROSE_OR_SCATTERED_PUBLIC_CALIBRATION_REQUIREMENTS",
            "current_support_state": support_state,
            "transition_reason": kwargs["transition_reason"],
            "evidence_ref": "reports/public_calibration_proposal_gate.json",
            **NO_CHEAT,
            "non_claims": ["proposal evidence transition only"],
        }
    ]
    return {
        "public_calibration_proposal_records": [proposal],
        "public_run_registry_records": [registry_record],
        "contamination_check_records": [contamination],
        "failure_taxonomy_records": [failure_taxonomy],
        "private_repair_manifest_records": [repair_manifest],
        "failure_boundary_records": [failure_boundary],
        "claim_records": claim_records,
        "artifact_graph_records": artifact_graph_records,
        "evidence_transition_records": evidence_transition_records,
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def gap(kind: str, evidence: Any) -> dict[str, Any]:
    return {"kind": kind, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_json_or_scalar(value: Any) -> Any:
    return value


def object_field(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def no_cheat_clean(payload: dict[str, Any]) -> bool:
    return (
        int_value(payload.get("public_training_rows_written")) == 0
        and int_value(payload.get("external_inference_calls")) == 0
        and int_value(payload.get("fallback_return_count")) == 0
    )


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except Exception:
        return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report.get("summary"))
    lines = [
        "# Public Calibration Proposal Gate",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Surface: `{summary.get('surface_slug')}`",
        f"- Decision: `{summary.get('proposal_decision')}`",
        f"- Execution allowed: `{summary.get('execution_allowed')}`",
        f"- Exact consumed surface: `{summary.get('exact_consumed_surface')}`",
        f"- Candidate integrity ready: `{summary.get('candidate_integrity_ready')}`",
        f"- Training firewall ready: `{summary.get('training_firewall_ready')}`",
        f"- Alignment preflight ready: `{summary.get('alignment_preflight_ready')}`",
        f"- Hard failed gates: `{summary.get('hard_failed_gate_count')}`",
        "",
        "This report is a proposal/evidence gate only. It does not run public benchmarks or train on public benchmark payloads.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
