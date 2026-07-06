#!/usr/bin/env python3
"""Proof-carrying contract gate for Circle-derived Theseus fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "proof_carrying_contracts.json"
DEFAULT_REPORT = ROOT / "reports" / "proof_carrying_contract_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "proof_carrying_contract_gate.md"
DEFAULT_CONTRACT_RECORDS = ROOT / "reports" / "proof_carrying_contract_records.jsonl"
DEFAULT_ADOPTION_RECORDS = ROOT / "reports" / "substrate_adoption_records.jsonl"
PROVED_STATUSES = {"proved", "lean_proved"}
FORBIDDEN_MODEL_CLAIMS = [
    "model-quality promotion",
    "runtime promotion",
    "context-length promotion",
    "speed or memory promotion",
    "public-transfer promotion",
    "ASI progress claim",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--contract-records-out", default=rel(DEFAULT_CONTRACT_RECORDS))
    parser.add_argument("--adoption-records-out", default=rel(DEFAULT_ADOPTION_RECORDS))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.contract_records_out), report["contract_records"])
    write_jsonl(resolve(args.adoption_records_out), report["substrate_adoption_records"])
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    inputs = dict_value(config.get("inputs"))
    pack_path = resolve(str(inputs.get("circle_contract_pack") or ""))
    legacy_path = resolve(str(inputs.get("circle_legacy_contracts") or ""))
    theorem_manifest_path = resolve(str(inputs.get("circle_theorem_manifest") or ""))
    pack = read_json(pack_path)
    legacy_pack = read_json(legacy_path)
    theorem_manifest = read_json(theorem_manifest_path)
    theorem_index = build_theorem_index(theorem_manifest)
    lean_status = inspect_lean_status(inputs)
    contract_records = [
        audit_contract(row, theorem_index, config)
        for row in list_dicts(pack.get("contracts"))
    ]
    claim_decisions = [decide_claim(row, config, lean_status, contract_records) for row in list_dicts(config.get("claim_requests"))]
    legacy_reports = audit_legacy_reports(config)
    adoption_records = [audit_adoption_record(row, contract_records, config) for row in list_dicts(config.get("substrate_adoption_records"))]
    pack_gates = audit_pack(pack_path, pack, legacy_pack, theorem_index)
    boundary_gates = audit_policy_boundaries(config)

    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for group in [pack_gates, boundary_gates, lean_status["gates"], legacy_reports["gates"]]:
        for row in group:
            if not row["passed"] and row["severity"] == "hard":
                hard_gaps.append(item_gap(row["id"], "gate_failed", row, row["severity"]))
            elif not row["passed"]:
                warnings.append(item_gap(row["id"], "gate_warning", row, row["severity"]))
    for row in contract_records:
        if row["hard_gap_count"]:
            hard_gaps.extend(row["hard_gaps"])
        if row["warning_count"]:
            warnings.extend(row["warnings"])
    for row in claim_decisions:
        if row["decision"] == "allowed" and row["expected_decision"] == "blocked":
            hard_gaps.append(item_gap(row["id"], "unexpected_claim_allowed", row))
        if row["decision"] == "blocked" and row["expected_decision"] == "allowed":
            hard_gaps.append(item_gap(row["id"], "expected_claim_blocked", row))
    for row in adoption_records:
        if row["hard_gap_count"]:
            hard_gaps.extend(row["hard_gaps"])

    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    summary = {
        "config": rel(config_path),
        "circle_contract_pack": rel(pack_path),
        "legacy_contract_pack": rel(legacy_path),
        "theorem_manifest": rel(theorem_manifest_path),
        "contract_count": len(contract_records),
        "legacy_contract_count": len(list_dicts(legacy_pack.get("contracts"))),
        "theorem_manifest_count": len(theorem_index),
        "contracts_with_theorem_ids": sum(1 for row in contract_records if row["theorem_count"] > 0),
        "contracts_proof_ready": sum(1 for row in contract_records if row["proof_ready"]),
        "contracts_fixture_ready": sum(1 for row in contract_records if row["fixture_ready"]),
        "blocked_claim_count": sum(1 for row in claim_decisions if row["decision"] == "blocked"),
        "allowed_claim_count": sum(1 for row in claim_decisions if row["decision"] == "allowed"),
        "substrate_adoption_record_count": len(adoption_records),
        "proof_contract_receipt_record_count": len(contract_records),
        "schema_shaped_substrate_adoption_record_count": len(adoption_records),
        "production_default_adoption_count": sum(1 for row in adoption_records if row["production_default"]),
        "local_lean_contract_artifact_current": lean_status["artifact_current"],
        "legacy_bridge_reports_present": legacy_reports["summary"]["present_count"],
        "legacy_bridge_reports_clean": legacy_reports["summary"]["clean"],
        "external_inference_calls": legacy_reports["summary"]["external_inference_calls"],
        "training_mutation": legacy_reports["summary"]["training_mutation"],
        "promotion_evidence": legacy_reports["summary"]["promotion_evidence"],
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_proof_carrying_contract_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "pack_gates": pack_gates,
        "policy_boundary_gates": boundary_gates,
        "lean_status": lean_status,
        "legacy_bridge_audit": legacy_reports,
        "contract_records": contract_records,
        "claim_decisions": claim_decisions,
        "substrate_adoption_records": adoption_records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "fixture_boundary": "Circle-derived fixtures can support only their finite theorem/contract boundary.",
            "separate_evidence": "Model quality, speed, memory, transfer, context-length, promotion, and ASI claims require separate Theseus task evidence.",
            "proof_status": "Proof-carrying claims require resolved/proved theorem IDs and a current local Lean contract artifact.",
            "baseline_discipline": "Cyclic/Coil/RoPE adoption records need baselines, negative controls, falsification criteria, residuals, and non-claims.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": legacy_reports["summary"]["external_inference_calls"],
        "fallback_return_count": 0,
    }


def audit_pack(pack_path: Path, pack: dict[str, Any], legacy_pack: dict[str, Any], theorem_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    contracts = list_dicts(pack.get("contracts"))
    legacy_contracts = list_dicts(legacy_pack.get("contracts"))
    return [
        gate("circle_contract_pack_present", pack_path.exists(), "hard", rel(pack_path)),
        gate("circle_contract_pack_status_public_safe_fixture", pack.get("status") == "public_safe_fixture", "hard", pack.get("status")),
        gate("circle_contract_pack_has_claim_boundary", bool(pack.get("claim_boundary")), "hard", pack.get("claim_boundary")),
        gate("circle_contract_pack_has_contracts", bool(contracts), "hard", len(contracts)),
        gate("circle_contract_pack_has_validation_commands", bool(list_values(pack.get("validation_commands"))), "hard", len(list_values(pack.get("validation_commands")))),
        gate("circle_legacy_bridge_present", bool(legacy_contracts), "warning", len(legacy_contracts)),
        gate("theorem_manifest_nonempty", bool(theorem_index), "hard", len(theorem_index)),
    ]


def audit_policy_boundaries(config: dict[str, Any]) -> list[dict[str, Any]]:
    policy = dict_value(config.get("proof_claim_policy"))
    return [
        gate("fixture_requires_consumer_check", policy.get("fixture_use_requires_pack_consumer_check") is True, "hard", policy.get("fixture_use_requires_pack_consumer_check")),
        gate("proof_claim_requires_local_lean_artifact", policy.get("proof_carrying_claim_requires_local_lean_artifact") is True, "hard", policy.get("proof_carrying_claim_requires_local_lean_artifact")),
        gate("proof_claim_requires_manifest_resolution", policy.get("proof_carrying_claim_requires_theorem_manifest_resolution") is True, "hard", policy.get("proof_carrying_claim_requires_theorem_manifest_resolution")),
        gate("proof_placeholder_blocks_claim", policy.get("proof_placeholder_blocks_claim") is True, "hard", policy.get("proof_placeholder_blocks_claim")),
        gate("model_quality_needs_separate_evidence", policy.get("model_quality_claim_requires_separate_theseus_task_evidence") is True, "hard", policy.get("model_quality_claim_requires_separate_theseus_task_evidence")),
        gate("transfer_needs_separate_evidence", policy.get("transfer_claim_requires_separate_theseus_transfer_evidence") is True, "hard", policy.get("transfer_claim_requires_separate_theseus_transfer_evidence")),
        gate("learned_generation_needs_candidate_integrity", policy.get("learned_generation_claim_requires_candidate_integrity_evidence") is True, "hard", policy.get("learned_generation_claim_requires_candidate_integrity_evidence")),
    ]


def audit_contract(contract: dict[str, Any], theorem_index: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    required = set(str(x) for x in list_values(config.get("required_contract_fields")))
    missing = sorted(required - set(contract))
    theorem_ids = [str(x) for x in list_values(contract.get("theorem_ids"))]
    resolved = [tid for tid in theorem_ids if tid in theorem_index]
    unresolved = sorted(set(theorem_ids) - set(resolved))
    unproved = sorted(tid for tid in resolved if str(theorem_index[tid].get("status") or theorem_index[tid].get("canonical_status") or "") not in PROVED_STATUSES)
    proof_status = dict_value(contract.get("proof_status"))
    consumer_check = dict_value(contract.get("consumer_check"))
    not_claimed = text_or_list(contract.get("not_claimed"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if missing:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "missing_required_contract_fields", {"missing": missing}))
    if not theorem_ids:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "missing_theorem_ids", {}))
    if unresolved:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "unresolved_theorem_ids", {"theorem_ids": unresolved[:20], "count": len(unresolved)}))
    if unproved:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "unproved_theorem_ids", {"theorem_ids": unproved[:20], "count": len(unproved)}))
    if proof_status.get("all_theorem_ids_proved") is not True or proof_status.get("all_theorem_ids_resolved") is not True:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "proof_status_not_ready", proof_status))
    if consumer_check.get("ready_for_downstream_fixture_use") is not True:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "consumer_check_not_fixture_ready", consumer_check))
    if not not_claimed:
        hard_gaps.append(item_gap(str(contract.get("id") or ""), "missing_not_claimed_boundary", {}))
    if not list_values(contract.get("ordinary_baselines")):
        warnings.append(item_gap(str(contract.get("id") or ""), "missing_ordinary_baselines", {}, "warning"))
    fixture_ready = not hard_gaps and bool(consumer_check.get("ready_for_downstream_fixture_use"))
    proof_ready = fixture_ready and bool(proof_status.get("all_theorem_ids_proved")) and bool(proof_status.get("all_theorem_ids_resolved"))
    validation_commands = [str(item) for item in list_values(contract.get("validation_commands"))]
    deterministic_fields = [str(item) for item in list_values(consumer_check.get("minimum_fields"))]
    if not deterministic_fields:
        deterministic_fields = sorted(str(key) for key in dict_value(contract.get("fields")).keys())[:64]
    not_claimed = text_or_list(contract.get("not_claimed"))
    return {
        "record_type": "proof_contract_receipt_record",
        "receipt_id": stable_id("proof_contract_receipt", contract.get("id"), contract.get("content_fingerprint")),
        "source_project": "circle_math",
        "contract_family": str(contract.get("kind") or ""),
        "engineering_object": str(contract.get("integration_use") or contract.get("id") or ""),
        "finite_model": str(dict_value(contract.get("fields")).get("certificate_schema_id") or contract.get("kind") or ""),
        "theorem_refs": theorem_ids,
        "proof_status": "local_proved" if proof_ready else "blocked",
        "source_version": "circle_contract_pack:public_safe_fixture",
        "deterministic_fields": deterministic_fields,
        "verifier_command": " && ".join(validation_commands) or "not_run",
        "verifier_result": "pass" if fixture_ready else "fail",
        "resolver_status": "resolved_local" if not unresolved else "missing",
        "replay_status": "replayed" if fixture_ready else "blocked",
        "consumer_gate": {
            "consumer_id": "theseus.proof_carrying_contract_gate",
            "allowed_uses": [
                "source-boundary discussion",
                "structural fixture use",
                "private benchmark design input",
                "deterministic configuration input",
            ],
            "blocked_uses": FORBIDDEN_MODEL_CLAIMS,
            "required_downstream_evidence": [
                "separate Theseus workload",
                "ordinary baseline",
                "negative control",
                "metric and report artifact",
                "candidate integrity evidence for learned-generation use",
            ],
        },
        "failure_behavior": "If theorem resolution, local artifact freshness, replay, or consumer readiness fails, block downstream promotion and keep only structural discussion.",
        "non_claims": not_claimed,
        "evidence_refs": [
            "configs/proof_carrying_contracts.json",
            "reports/proof_carrying_contract_gate.json",
            "../circle math/site/data/generated/circle_ai_contract_pack.json",
            "../circle math/site/data/generated/theorem_manifest.json",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "id": str(contract.get("id") or ""),
        "kind": str(contract.get("kind") or ""),
        "content_fingerprint": str(contract.get("content_fingerprint") or ""),
        "theorem_count": len(theorem_ids),
        "resolved_theorem_count": len(resolved),
        "unresolved_theorem_ids": unresolved,
        "unproved_theorem_ids": unproved,
        "proof_ready": proof_ready,
        "fixture_ready": fixture_ready,
        "consumer_ready": bool(consumer_check.get("ready_for_downstream_fixture_use")),
        "not_claimed_count": len(not_claimed),
        "ordinary_baseline_count": len(list_values(contract.get("ordinary_baselines"))),
        "planner_recommendation_count": len(list_values(contract.get("planner_recommendations"))),
        "validation_command_count": len(list_values(contract.get("validation_commands"))),
        "missing_required_fields": missing,
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "claim_boundary": "fixture/config/proof-boundary only; not model-quality evidence",
    }


def inspect_lean_status(inputs: dict[str, Any]) -> dict[str, Any]:
    source = resolve(str(inputs.get("circle_lean_contract_source") or ""))
    artifact = resolve(str(inputs.get("circle_lean_contract_artifact") or ""))
    source_exists = source.exists()
    artifact_exists = artifact.exists()
    artifact_current = bool(source_exists and artifact_exists and artifact.stat().st_mtime >= source.stat().st_mtime)
    gates = [
        gate("circle_lean_contract_source_present", source_exists, "hard", rel(source)),
        gate("circle_lean_contract_artifact_present", artifact_exists, "hard", rel(artifact)),
        gate("circle_lean_contract_artifact_current", artifact_current, "hard", {
            "source": rel(source),
            "artifact": rel(artifact),
            "source_mtime": source.stat().st_mtime if source_exists else None,
            "artifact_mtime": artifact.stat().st_mtime if artifact_exists else None,
        }),
    ]
    return {
        "source": rel(source),
        "artifact": rel(artifact),
        "source_present": source_exists,
        "artifact_present": artifact_exists,
        "artifact_current": artifact_current,
        "proof_carrying_claims_enabled": artifact_current,
        "gates": gates,
    }


def decide_claim(row: dict[str, Any], config: dict[str, Any], lean_status: dict[str, Any], contracts: list[dict[str, Any]]) -> dict[str, Any]:
    claim_type = str(row.get("claim_type") or "")
    allowed_fixture = claim_type in set(str(x) for x in list_values(config.get("allowed_fixture_claims")))
    forbidden = claim_type in set(str(x) for x in list_values(config.get("forbidden_claims_without_separate_theseus_evidence")))
    evidence_refs = list_values(row.get("separate_theseus_task_evidence_refs"))
    contract_map = {record["id"]: record for record in contracts}
    source = str(row.get("source") or "")
    source_contract_ready = contract_map.get(source, {}).get("fixture_ready", source == "circle_contract_pack")
    reasons = []
    if forbidden and not evidence_refs:
        reasons.append("forbidden_without_separate_theseus_evidence")
    if not allowed_fixture and not forbidden:
        reasons.append("unknown_claim_type")
    if allowed_fixture and not source_contract_ready:
        reasons.append("source_contract_not_fixture_ready")
    if allowed_fixture and not lean_status.get("proof_carrying_claims_enabled"):
        reasons.append("local_lean_artifact_not_current")
    decision = "allowed" if not reasons and allowed_fixture else "blocked"
    return {
        "id": str(row.get("id") or ""),
        "source": source,
        "claim_type": claim_type,
        "requested_support": str(row.get("requested_support") or ""),
        "expected_decision": str(row.get("expected_decision") or ""),
        "decision": decision,
        "reasons": reasons,
        "separate_theseus_task_evidence_refs": evidence_refs,
        "non_claim": decision == "blocked",
    }


def audit_legacy_reports(config: dict[str, Any]) -> dict[str, Any]:
    inputs = dict_value(config.get("inputs"))
    expectations = dict_value(config.get("legacy_bridge_expectations"))
    report_keys = [
        "legacy_consumer_report",
        "legacy_workload_smoke_report",
        "legacy_proxy_benchmark_report",
    ]
    reports = []
    gates = []
    totals = {"external_inference_calls": 0, "training_mutation": False, "promotion_evidence": False}
    for key in report_keys:
        path = resolve(str(inputs.get(key) or ""))
        present = path.exists()
        payload = read_json(path) if present else {}
        totals["external_inference_calls"] += int(payload.get("external_inference_calls") or 0)
        totals["training_mutation"] = totals["training_mutation"] or bool(payload.get("training_mutation"))
        totals["promotion_evidence"] = totals["promotion_evidence"] or bool(payload.get("promotion_evidence"))
        clean = (
            present
            and int(payload.get("external_inference_calls") or 0) == int(expectations.get("external_inference_calls") or 0)
            and bool(payload.get("public_calibration_used")) is bool(expectations.get("public_calibration_used"))
            and bool(payload.get("training_mutation")) is bool(expectations.get("training_mutation"))
            and bool(payload.get("promotion_evidence")) is bool(expectations.get("promotion_evidence"))
            and bool(payload.get("private_data_exported")) is bool(expectations.get("private_data_exported"))
        )
        gates.append(gate(f"{key}_clean", clean, "warning", {"path": rel(path), "present": present}))
        reports.append({
            "id": key,
            "path": rel(path),
            "present": present,
            "trigger_state": payload.get("trigger_state"),
            "external_inference_calls": int(payload.get("external_inference_calls") or 0),
            "public_calibration_used": bool(payload.get("public_calibration_used")),
            "training_mutation": bool(payload.get("training_mutation")),
            "promotion_evidence": bool(payload.get("promotion_evidence")),
            "private_data_exported": bool(payload.get("private_data_exported")),
            "clean_boundary": clean,
        })
    return {
        "summary": {
            "present_count": sum(1 for row in reports if row["present"]),
            "clean": all(row["clean_boundary"] for row in reports),
            "external_inference_calls": totals["external_inference_calls"],
            "training_mutation": totals["training_mutation"],
            "promotion_evidence": totals["promotion_evidence"],
        },
        "reports": reports,
        "gates": gates,
    }


def audit_adoption_record(row: dict[str, Any], contracts: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    contract_ids = set(str(x) for x in list_values(row.get("source_contract_ids")))
    known = {contract["id"] for contract in contracts}
    missing_contracts = sorted(contract_ids - known)
    hard_gaps = []
    required_lists = ["baselines", "negative_controls", "falsification_criteria", "required_future_evidence", "non_claims"]
    for field in required_lists:
        if not list_values(row.get(field)):
            hard_gaps.append(item_gap(str(row.get("id") or ""), f"missing_{field}", {}))
    if missing_contracts:
        hard_gaps.append(item_gap(str(row.get("id") or ""), "unknown_source_contracts", {"missing": missing_contracts}))
    if bool(row.get("production_default")):
        hard_gaps.append(item_gap(str(row.get("id") or ""), "production_default_not_allowed_for_research_fixture", {}))
    known_contracts = sorted(contract_ids & known)
    non_claims = [str(item) for item in list_values(row.get("non_claims"))]
    required_future_evidence = [str(item) for item in list_values(row.get("required_future_evidence"))]
    current_lifecycle = str(row.get("current_lifecycle") or "")
    adoption_state = "structural_only" if known_contracts else "blocked"
    if current_lifecycle in {"exploratory", "research_fixture"} and known_contracts:
        adoption_state = "exploratory"
    return {
        "record_type": "substrate_adoption_record",
        "substrate_id": str(row.get("id") or ""),
        "substrate_kind": str(row.get("substrate_kind") or row.get("id") or "").replace("substrate.", ""),
        "intended_use": str(row.get("candidate_use") or ""),
        "expected_advantage": str(row.get("expected_advantage") or "potential narrow structural advantage only; must be proven against ordinary baselines before adoption"),
        "baseline_refs": [str(item) for item in list_values(row.get("baselines"))],
        "negative_controls": [str(item) for item in list_values(row.get("negative_controls"))],
        "proof_boundary": str(row.get("proof_boundary") or "Circle proof-carrying fixture boundary only; structural receipt does not imply model quality, runtime, memory, context, or transfer improvement."),
        "experiment_requirements": required_future_evidence,
        "consumer_gate": str(row.get("consumer_gate") or "May be used for research planning and structural fixture checks only; routing, compression, model-quality, runtime, and public-transfer claims require separate Theseus evidence."),
        "axis_ledger": [
            {
                "axis": "structure",
                "status": "measured_positive" if known_contracts and not hard_gaps else "blocked",
                "evidence_refs": known_contracts,
                "non_claims": ["structural proof/fixture evidence is not downstream task quality"],
            },
            {
                "axis": "downstream_task_quality",
                "status": "unmeasured",
                "evidence_refs": [],
                "non_claims": ["no Theseus task-quality result is implied"],
            },
            {
                "axis": "runtime_cost",
                "status": "unmeasured",
                "evidence_refs": [],
                "non_claims": ["no speed, memory, or context-length result is implied"],
            },
        ],
        "falsification_condition": "; ".join(str(item) for item in list_values(row.get("falsification_criteria"))) or "missing falsification criteria blocks adoption",
        "adoption_state": adoption_state,
        "residuals": required_future_evidence,
        "evidence_refs": [
            "configs/proof_carrying_contracts.json",
            "reports/proof_carrying_contract_gate.json",
            "reports/proof_carrying_contract_records.jsonl",
        ],
        "non_claims": non_claims,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "id": str(row.get("id") or ""),
        "source_contract_ids": sorted(contract_ids),
        "candidate_use": str(row.get("candidate_use") or ""),
        "current_lifecycle": current_lifecycle,
        "production_default": bool(row.get("production_default")),
        "baseline_count": len(list_values(row.get("baselines"))),
        "negative_control_count": len(list_values(row.get("negative_controls"))),
        "falsification_criteria_count": len(list_values(row.get("falsification_criteria"))),
        "required_future_evidence_count": len(list_values(row.get("required_future_evidence"))),
        "non_claim_count": len(list_values(row.get("non_claims"))),
        "known_source_contracts": known_contracts,
        "missing_source_contracts": missing_contracts,
        "hard_gap_count": len(hard_gaps),
        "hard_gaps": hard_gaps,
    }


def build_theorem_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in list_dicts(manifest.get("theorems")):
        theorem_id = str(row.get("id") or "")
        if theorem_id:
            out[theorem_id] = row
    return out


def gate(gate_id: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {"id": gate_id, "passed": bool(passed), "severity": severity, "detail": detail}


def item_gap(item_id: str, reason: str, detail: Any, severity: str = "hard") -> dict[str, Any]:
    return {"item_id": item_id, "reason": reason, "severity": severity, "detail": detail}


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"][:20],
        "warnings": report["warnings"][:20],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Proof-Carrying Contract Gate",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Circle contract pack: `{summary['circle_contract_pack']}`",
        f"- Contracts: `{summary['contract_count']}`",
        f"- Fixture-ready contracts: `{summary['contracts_fixture_ready']}`",
        f"- Proof-ready contracts: `{summary['contracts_proof_ready']}`",
        f"- Local Lean artifact current: `{summary['local_lean_contract_artifact_current']}`",
        f"- Claim decisions: `{summary['allowed_claim_count']}` allowed, `{summary['blocked_claim_count']}` blocked",
        f"- Substrate adoption records: `{summary['substrate_adoption_record_count']}`",
        f"- Legacy bridge clean: `{summary['legacy_bridge_reports_clean']}`",
        f"- Hard gaps: `{summary['hard_gap_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "",
        "## Boundary",
        "",
        "Circle-derived contracts are fixture/configuration/proof-boundary evidence only. Model-quality, speed, memory, transfer, context-length, promotion, learned-generation, and ASI claims require separate Theseus task evidence.",
        "",
        "## Claim Decisions",
    ]
    for row in report["claim_decisions"]:
        reasons = ", ".join(row["reasons"]) or "none"
        lines.append(f"- `{row['id']}`: `{row['decision']}` ({reasons})")
    lines.extend(["", "## Adoption Records"])
    for row in report["substrate_adoption_records"]:
        lines.append(
            f"- `{row['id']}`: lifecycle `{row['current_lifecycle']}`, "
            f"production_default `{row['production_default']}`, "
            f"baselines `{row['baseline_count']}`, negative controls `{row['negative_control_count']}`"
        )
    if report["hard_gaps"]:
        lines.extend(["", "## Hard Gaps"])
        for gap in report["hard_gaps"][:20]:
            lines.append(f"- `{gap['item_id']}`: {gap['reason']}")
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        for gap in report["warnings"][:20]:
            lines.append(f"- `{gap['item_id']}`: {gap['reason']}")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text_or_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:24]


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
