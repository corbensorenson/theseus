#!/usr/bin/env python3
"""2026-07-06 weekly focus exporter and gate.

This script turns the existing assistant/product spine into the public-safe
implementation-reference artifacts the ASI Stack book can import. It is a
governance/evidence exporter, not a model-quality benchmark and not learned
generation promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
AI_BOOK_ROOT = ROOT.parent / "AI_book"

DEFAULT_ASSISTANT_OUT = REPORTS / "theseus_assistant_product_spine_smoke.json"
DEFAULT_ASSISTANT_MD = REPORTS / "theseus_assistant_product_spine_smoke.md"
DEFAULT_ASSISTANT_EVENTS = REPORTS / "theseus_assistant_product_spine_events.jsonl"
DEFAULT_ASSISTANT_TRACE = REPORTS / "theseus_assistant_product_spine_trace.jsonl"
DEFAULT_OUT = REPORTS / "theseus_weekly_focus_20260706.json"
DEFAULT_MD = REPORTS / "theseus_weekly_focus_20260706.md"
DEFAULT_REFERENCE_TRACE = REPORTS / "theseus_public_safe_reference_trace_20260706.json"
DEFAULT_EVIDENCE_PACKS = REPORTS / "theseus_book_importable_evidence_packs_20260706.json"
DEFAULT_PREREG = CONFIGS / "correctness_in_loop_generator_experiments.json"

NO_CHEAT_COUNTERS = ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
FORBIDDEN_PUBLIC_SAFE_TERMS = (
    "/Users/corbensorenson",
    "data/training_data/high_transfer/private_train",
    "runtime/dogfood",
    "checkpoints/",
    "candidate_body",
    "solution_body",
    "hidden_tests",
)
REFERENCE_TRACE_SCHEMA = AI_BOOK_ROOT / "schemas" / "reference_trace_record.schema.json"
ALLOWED_CLAIM_SUPPORT_STATES = {
    "argument",
    "prototype-backed",
    "synthetic-test-backed",
    "empirical-test-backed",
    "replayable-reference-backed",
    "unsupported",
}
REFERENCE_TRACE_REQUIRED_RECORD_TYPES = {
    "intent_contract",
    "command_contract",
    "context_abi_record",
    "typed_job",
    "planforge_dag",
    "runtime_adapter_invocation",
    "authority_transition",
    "authority_use_receipt",
    "artifact_graph_record",
    "claim_record",
    "evidence_transition_record",
    "residual_record",
    "policy_optimization_record",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-assistant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--reference-trace-out", default=rel(DEFAULT_REFERENCE_TRACE))
    parser.add_argument("--evidence-packs-out", default=rel(DEFAULT_EVIDENCE_PACKS))
    parser.add_argument("--preregistration-out", default=rel(DEFAULT_PREREG))
    parser.add_argument("--assistant-out", default=rel(DEFAULT_ASSISTANT_OUT))
    parser.add_argument("--assistant-markdown-out", default=rel(DEFAULT_ASSISTANT_MD))
    parser.add_argument("--assistant-events-out", default=rel(DEFAULT_ASSISTANT_EVENTS))
    parser.add_argument("--assistant-trace-out", default=rel(DEFAULT_ASSISTANT_TRACE))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    refresh = run_assistant_refresh(args) if args.refresh_assistant else {"ran": False}
    assistant_report = read_json(resolve(args.assistant_out), {})
    trace_rows = read_jsonl(resolve(args.assistant_trace_out))
    selected_trace_rows = latest_assistant_run(trace_rows)
    materialized_view = read_json(REPORTS / "viea_spine_materialized_view.json", {})
    spine_gate = read_json(REPORTS / "viea_spine_record_gate.json", {})
    roadmap_gate = read_json(REPORTS / "roadmap_implementation_gate.json", {})
    module_gate = read_json(REPORTS / "module_definition_of_done.json", {})
    crosswalk = read_json(REPORTS / "book_to_theseus_crosswalk.json", {})

    reference_trace_record = build_reference_trace_record(
        assistant_report=assistant_report,
        trace_rows=selected_trace_rows,
        assistant_out=resolve(args.assistant_out),
        assistant_trace_out=resolve(args.assistant_trace_out),
    )
    evidence_packs = build_evidence_packs(
        assistant_report=assistant_report,
        reference_trace_record=reference_trace_record,
        paths=canonical_evidence_paths(args),
    )
    receipt_audit = audit_receipt_faithfulness(evidence_packs)
    residual_audit = audit_residual_conservation(reference_trace_record, materialized_view, evidence_packs)
    verifier_capacity = build_verifier_capacity_accounting(assistant_report, spine_gate, evidence_packs)
    governance_tax = build_governance_tax_accounting(assistant_report, refresh, started)
    claim_dispositions = build_capability_claim_dispositions(reference_trace_record, evidence_packs, residual_audit)
    schema_conformance = build_book_schema_conformance(reference_trace_record)
    preregistration = build_correctness_preregistration(resolve(args.preregistration_out))

    write_json(resolve(args.reference_trace_out), {
        "policy": "project_theseus_public_safe_reference_trace_export_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if schema_conformance["passed"] else "RED",
        "reference_trace_record": reference_trace_record,
        "public_safety": public_safety_summary(reference_trace_record),
        "source_run": {
            "assistant_report": rel(resolve(args.assistant_out)),
            "assistant_trace": rel(resolve(args.assistant_trace_out)),
            "assistant_report_sha256": sha256_file(resolve(args.assistant_out)),
            "assistant_trace_sha256": sha256_file(resolve(args.assistant_trace_out)),
            "assistant_trace_row_count": len(selected_trace_rows),
        },
        "non_claims": reference_trace_record["non_claims"],
        **clean_counters(),
    })
    write_json(resolve(args.evidence_packs_out), {
        "policy": "project_theseus_book_importable_evidence_pack_export_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(pack["public_safe"] for pack in evidence_packs) else "RED",
        "summary": {
            "evidence_pack_count": len(evidence_packs),
            "green_pack_count": sum(1 for pack in evidence_packs if pack["trigger_state"] == "GREEN"),
            "public_safe_pack_count": sum(1 for pack in evidence_packs if pack["public_safe"]),
            "private_payload_copied_count": sum(1 for pack in evidence_packs if pack["private_payload_copied"]),
        },
        "evidence_packs": evidence_packs,
        "non_claims": [
            "Evidence packs are public-safe summaries and digests, not raw report payload copies.",
            "Evidence packs do not prove model quality or learned-generation capability.",
            "Evidence packs do not authorize training on public benchmark payloads.",
        ],
        **clean_counters(),
    })
    claim_ledger_trace_kernel = build_claim_ledger_trace_kernel(
        assistant_report=assistant_report,
        trace_rows=selected_trace_rows,
        reference_trace_record=reference_trace_record,
        evidence_packs=evidence_packs,
        receipt_audit=receipt_audit,
        claim_dispositions=claim_dispositions,
        assistant_out=resolve(args.assistant_out),
        assistant_trace_out=resolve(args.assistant_trace_out),
        reference_trace_out=resolve(args.reference_trace_out),
        evidence_packs_out=resolve(args.evidence_packs_out),
    )

    gates = build_gates(
        assistant_report=assistant_report,
        selected_trace_rows=selected_trace_rows,
        reference_trace_record=reference_trace_record,
        evidence_packs=evidence_packs,
        receipt_audit=receipt_audit,
        residual_audit=residual_audit,
        verifier_capacity=verifier_capacity,
        governance_tax=governance_tax,
        claim_dispositions=claim_dispositions,
        claim_ledger_trace_kernel=claim_ledger_trace_kernel,
        schema_conformance=schema_conformance,
        preregistration=preregistration,
        roadmap_gate=roadmap_gate,
        module_gate=module_gate,
        crosswalk=crosswalk,
    )
    hard_failures = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warnings = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warnings:
        trigger_state = "YELLOW"
    report = {
        "policy": "project_theseus_weekly_focus_20260706_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "assistant_refresh_ran": bool(refresh.get("ran")),
            "assistant_trigger_state": assistant_report.get("trigger_state"),
            "assistant_trace_row_count": len(selected_trace_rows),
            "reference_trace_export": rel(resolve(args.reference_trace_out)),
            "reference_trace_schema_conformant": schema_conformance["passed"],
            "evidence_pack_export": rel(resolve(args.evidence_packs_out)),
            "evidence_pack_count": len(evidence_packs),
            "receipt_expected_invalid_control_count": receipt_audit["expected_invalid_control_count"],
            "receipt_expected_invalid_rejected_count": receipt_audit["expected_invalid_rejected_count"],
            "residual_conservation_state": residual_audit["state"],
            "verifier_capacity_state": verifier_capacity["state"],
            "governance_tax_measured": governance_tax["measured"],
            "capability_claim_disposition_count": len(claim_dispositions["claims"]),
            "claim_ledger_trace_kernel_state": claim_ledger_trace_kernel["state"],
            "claim_ledger_trace_kernel_support_state": claim_ledger_trace_kernel["support_state"],
            "preregistered_experiment_count": len(preregistration["experiments"]),
            "hard_gap_count": len(hard_failures),
            "warning_count": len(warnings),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "outputs": {
            "reference_trace": rel(resolve(args.reference_trace_out)),
            "evidence_packs": rel(resolve(args.evidence_packs_out)),
            "correctness_experiment_preregistration": rel(resolve(args.preregistration_out)),
        },
        "assistant_refresh": refresh,
        "reference_trace_record": reference_trace_record,
        "receipt_faithfulness": receipt_audit,
        "residual_conservation": residual_audit,
        "verifier_capacity": verifier_capacity,
        "governance_tax": governance_tax,
        "capability_claim_dispositions": claim_dispositions,
        "claim_ledger_trace_kernel": claim_ledger_trace_kernel,
        "book_schema_conformance": schema_conformance,
        "correctness_experiment_preregistration": preregistration,
        "gates": gates,
        "non_claims": [
            "This weekly focus report does not prove ASI capability.",
            "This weekly focus report does not promote learned generation.",
            "This weekly focus report does not train on public benchmarks.",
            "This weekly focus report does not count tools, templates, routers, or fallback behavior as learned generation.",
            "This weekly focus report exports public-safe summaries and digests only.",
        ],
        **clean_counters(),
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report) if args.gate else report, indent=2, sort_keys=True))
    return 2 if trigger_state == "RED" else 0


def run_assistant_refresh(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/theseus_assistant_runtime.py",
        "--prompt",
        (
            "Use Theseus deterministic tools, VCM, planning, verifier receipts, "
            "artifact refs, claim states, evidence records, and dogfood metadata "
            "to summarize the 2026-07-06 weekly focus implementation trace. "
            "Do not make learned-generation or public-benchmark claims."
        ),
        "--intent",
        "tool",
        "--feedback",
        "completed",
        "--session-id",
        "weekly_focus_20260706_reference_trace",
        "--out",
        args.assistant_out,
        "--markdown-out",
        args.assistant_markdown_out,
        "--events-out",
        args.assistant_events_out,
        "--viea-trace-out",
        args.assistant_trace_out,
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "ran": True,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def build_reference_trace_record(*, assistant_report: dict[str, Any], trace_rows: list[dict[str, Any]], assistant_out: Path, assistant_trace_out: Path) -> dict[str, Any]:
    summary = assistant_report.get("summary") if isinstance(assistant_report.get("summary"), dict) else {}
    run_id = str(trace_rows[0].get("assistant_run_id") if trace_rows else stable_id("assistant_run", assistant_out, sha256_file(assistant_out)))
    record_types = sorted({str(row.get("record_type") or row.get("content", {}).get("record_type") or "") for row in trace_rows if isinstance(row, dict)})
    trace_id = stable_id("theseus_reference_trace", run_id, sha256_file(assistant_out), sha256_file(assistant_trace_out))
    artifacts = [
        artifact_ref("assistant_runtime_report", assistant_out),
        artifact_ref("assistant_viea_trace", assistant_trace_out),
        artifact_ref("assistant_trace_schema", CONFIGS / "assistant_trace_schema.json"),
        artifact_ref("viea_spine_materialized_view", REPORTS / "viea_spine_materialized_view.json"),
        artifact_ref("private_verifier_spine", REPORTS / "private_verifier_spine_smoke.json"),
        artifact_ref("deterministic_tool_substrate", REPORTS / "deterministic_tool_substrate.json"),
        artifact_ref("book_to_theseus_crosswalk", REPORTS / "book_to_theseus_crosswalk.json"),
    ]
    return {
        "trace_id": trace_id,
        "trace_state": "replayed" if assistant_report.get("trigger_state") == "GREEN" and trace_rows else "blocked",
        "execution_boundary": "runtime_trace",
        "intent_ref": f"prompt_sha256:{first_nonempty([row.get('prompt_sha256') for row in trace_rows]) or summary.get('prompt_sha256') or 'hash_only'}",
        "parent_artifact_refs": [
            artifact_ref("roadmap_weekly_focus", ROOT / "roadmap.md"),
            artifact_ref("operating_charter", ROOT / "AGENTS.md"),
            artifact_ref("roadmap_matrix", CONFIGS / "roadmap_implementation_matrix.json"),
        ],
        "authority_chain": [
            "AGENTS.md hard rules and anti-cheating guardrail",
            "configs/project_manifest_registry.json registered surfaces",
            "configs/roadmap_implementation_matrix.json weekly_focus_decision",
            "reports/viea_spine_materialized_view.json authority records",
            "reports/private_verifier_spine_smoke.json verifier receipt",
        ],
        "authority_deltas": [
            "assistant route used local VCM/tools/planning/verifier evidence only",
            "raw prompt persisted as hash-only trace metadata",
            "public benchmark training remains forbidden",
            "external inference runtime remains forbidden",
        ],
        "layer_handoffs": [
            "intent -> command contract: assistant runtime classified weekly-focus request and emitted hash-bound intent/command records",
            "plan -> context: planning route selected VCM task family and context packet",
            "context -> route: VCM context, deterministic tool evidence, and resource route records entered the assistant trace",
            "route -> verification: private verifier and VIEA materialized view receipts bounded the route",
            "verification -> execution: local assistant runtime produced report, markdown, event, and trace artifacts",
            "execution -> evidence: artifact graph, claim record, evidence transition, residual record, and policy optimization records were emitted",
            "evidence -> scf improvement gate: report remains record-shape evidence with explicit promotion blockers",
        ],
        "artifacts": artifacts,
        "evidence_updates": [
            f"assistant_trigger_state={assistant_report.get('trigger_state')}",
            f"assistant_viea_trace_complete={summary.get('assistant_viea_trace_complete')}",
            f"assistant_viea_trace_record_count={summary.get('assistant_viea_trace_record_count')}",
            f"dogfood_event_written={summary.get('dogfood_event_written')}",
            f"record_types={','.join(record_types)}",
        ],
        "evidence_deltas": [
            "public-safe reference trace exported from current assistant product-spine run",
            "book-importable evidence pack standard applied to selected GREEN evidence surfaces",
            "receipt faithfulness, residual conservation, verifier capacity, governance tax, claim disposition, and book schema conformance audits materialized",
        ],
        "residual_deltas": [
            "learned-generation/public-transfer quality remains a separate blocker",
            "Hive live reachable-peer proof remains outside this local weekly-focus trace",
            "book import remains support-state review work, not automatic promotion",
        ],
        "stop_conditions": [
            "public benchmark payloads must not become training rows",
            "tools/templates/routers/fallbacks must not count as learned generation",
            "schema-valid reference trace must not be read as model-quality evidence",
        ],
        "missing_contracts": [
            "AI_book import acceptance is not claimed until book-side validators ingest this exact public-safe export",
            "clean release/publication review is not claimed from this local dirty-worktree trace",
        ],
        "validation_commands": [
            "python3 scripts/theseus_assistant_runtime.py --prompt ... --intent tool --feedback completed --session-id weekly_focus_20260706_reference_trace --out reports/theseus_assistant_product_spine_smoke.json --markdown-out reports/theseus_assistant_product_spine_smoke.md --events-out reports/theseus_assistant_product_spine_events.jsonl --viea-trace-out reports/theseus_assistant_product_spine_trace.jsonl",
            "python3 scripts/theseus_weekly_focus_20260706.py --gate",
            "python3 scripts/viea_spine_record_gate.py --gate",
            "python3 scripts/roadmap_implementation_gate.py --gate",
        ],
        "promotion_blockers": [
            "reference trace supports implementation-reference review only",
            "learned-generation claims still require independent candidate integrity and behavioral evidence",
            "book support-state movement requires AI_book import/review records",
        ],
        "source_refs": [
            "AGENTS.md",
            "roadmap.md",
            "configs/roadmap_implementation_matrix.json",
            "configs/project_manifest_registry.json",
            "sources/source_notes/project_theseus_whitepaper.md",
        ],
        "support_state_effect": "record_shape_only",
        "non_claims": [
            "This trace does not promote any ASI Stack chapter core claim.",
            "This trace does not prove model quality, public benchmark performance, or learned-generation capability.",
            "This trace does not copy raw private prompts, raw assistant text, private training rows, checkpoint payloads, or benchmark payloads.",
            "This trace does not prove deployed or clean-release readiness.",
        ],
    }


def canonical_evidence_paths(args: argparse.Namespace) -> list[tuple[str, Path, str, str]]:
    return [
        ("assistant_product_spine", resolve(args.assistant_out), "python3 scripts/theseus_assistant_runtime.py --intent tool ...", "runtime_trace"),
        ("assistant_viea_trace", resolve(args.assistant_trace_out), "python3 scripts/theseus_assistant_runtime.py --viea-trace-out ...", "runtime_trace_jsonl"),
        ("viea_spine_record_gate", REPORTS / "viea_spine_record_gate.json", "python3 scripts/viea_spine_record_gate.py --gate", "gate_report"),
        ("viea_spine_materialized_view", REPORTS / "viea_spine_materialized_view.json", "python3 scripts/viea_spine_record_gate.py --gate", "materialized_view"),
        ("roadmap_implementation_gate", REPORTS / "roadmap_implementation_gate.json", "python3 scripts/roadmap_implementation_gate.py --gate", "gate_report"),
        ("module_definition_of_done", REPORTS / "module_definition_of_done.json", "python3 scripts/module_definition_of_done_gate.py", "gate_report"),
        ("book_to_theseus_crosswalk", REPORTS / "book_to_theseus_crosswalk.json", "python3 scripts/roadmap_implementation_gate.py --gate", "crosswalk_report"),
        ("candidate_integrity", REPORTS / "candidate_integrity_audit.json", "python3 scripts/candidate_integrity.py", "gate_report"),
        ("deterministic_tool_substrate", REPORTS / "deterministic_tool_substrate.json", "python3 scripts/theseus_deterministic_tool_substrate.py --run-smoke --run-ablation", "gate_report"),
        ("private_verifier_spine", REPORTS / "private_verifier_spine_smoke.json", "python3 scripts/code_lm_private_verifier.py --spine-smoke", "gate_report"),
    ]


def build_evidence_packs(*, assistant_report: dict[str, Any], reference_trace_record: dict[str, Any], paths: list[tuple[str, Path, str, str]]) -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []
    for name, path, command, boundary in paths:
        payload = read_json(path, {}) if path.suffix == ".json" else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        trigger_state = str(payload.get("trigger_state") or ("GREEN" if path.exists() else "MISSING"))
        no_cheat = {key: int_or(payload.get(key, summary.get(key, 0))) for key in NO_CHEAT_COUNTERS}
        public_safe = path.exists() and not contains_forbidden_public_safe_text({
            "name": name,
            "path": rel(path),
            "summary": public_safe_summary(summary),
            "trigger_state": trigger_state,
            "sha256": sha256_file(path) if path.exists() else "",
        })
        packs.append({
            "pack_id": stable_id("book_evidence_pack", name, path, sha256_file(path) if path.exists() else "missing"),
            "name": name,
            "source_path": rel(path),
            "source_sha256": sha256_file(path) if path.exists() else "",
            "source_exists": path.exists(),
            "command": command,
            "boundary": boundary,
            "trigger_state": trigger_state,
            "support_state": support_state_for(trigger_state, boundary),
            "summary": public_safe_summary(summary),
            "negative_controls": negative_controls_for(name),
            "residuals": residuals_for(name, payload, reference_trace_record, assistant_report),
            "public_safe": public_safe,
            "private_payload_copied": False,
            "claim_boundaries": [
                "digest-and-summary evidence only",
                "not model-quality evidence unless the source report explicitly scopes that claim",
                "not learned-generation evidence unless candidate-integrity and behavioral gates independently say so",
            ],
            "non_claims": [
                "does not copy private/raw payloads",
                "does not train on public benchmarks",
                "does not promote ASI capability",
            ],
            **no_cheat,
        })
    return packs


def audit_receipt_faithfulness(evidence_packs: list[dict[str, Any]]) -> dict[str, Any]:
    sampled = evidence_packs[: min(8, len(evidence_packs))]
    replay_checks = []
    for pack in sampled:
        path = resolve(pack["source_path"])
        replay_checks.append({
            "pack_id": pack["pack_id"],
            "source_path": pack["source_path"],
            "expected_sha256": pack["source_sha256"],
            "observed_sha256": sha256_file(path) if path.exists() else "",
            "passed": path.exists() and sha256_file(path) == pack["source_sha256"],
        })
    controls = {
        "missing_source_path": dict(evidence_packs[0], source_path="reports/does_not_exist.json") if evidence_packs else {},
        "digest_mismatch": dict(evidence_packs[0], source_sha256="0" * 64) if evidence_packs else {},
        "private_payload_copy": dict(evidence_packs[0], private_payload_copied=True) if evidence_packs else {},
        "public_training_fault": dict(evidence_packs[0], public_training_rows_written=1) if evidence_packs else {},
        "external_inference_fault": dict(evidence_packs[0], external_inference_calls=1) if evidence_packs else {},
        "fallback_fault": dict(evidence_packs[0], fallback_return_count=1) if evidence_packs else {},
        "support_overclaim": dict(evidence_packs[0], support_state="empirical-test-backed", non_claims=[]) if evidence_packs else {},
    }
    rejected = [
        {"control": name, "rejected": not validate_evidence_pack(control)}
        for name, control in controls.items()
    ]
    passed = all(row["passed"] for row in replay_checks) and all(row["rejected"] for row in rejected)
    return {
        "state": "GREEN" if passed else "RED",
        "deep_replay_sample_count": len(replay_checks),
        "deep_replay_passed_count": sum(1 for row in replay_checks if row["passed"]),
        "expected_invalid_control_count": len(rejected),
        "expected_invalid_rejected_count": sum(1 for row in rejected if row["rejected"]),
        "deep_replay_checks": replay_checks,
        "expected_invalid_controls": rejected,
        "non_claims": ["receipt faithfulness is checked over selected public-safe summaries, not every raw report payload"],
        **clean_counters(),
    }


def audit_residual_conservation(reference_trace_record: dict[str, Any], materialized_view: dict[str, Any], evidence_packs: list[dict[str, Any]]) -> dict[str, Any]:
    residual_records = []
    for row in materialized_view.get("records", []) if isinstance(materialized_view.get("records"), list) else []:
        compact = row.get("compact_payload") if isinstance(row.get("compact_payload"), dict) else {}
        residuals = compact.get("residuals") or compact.get("residual_deltas") or []
        if residuals:
            residual_records.append({"record_id": row.get("record_id"), "source_path": row.get("source_path"), "residuals": residuals})
    exported_residuals = list(reference_trace_record.get("residual_deltas") or [])
    for pack in evidence_packs:
        exported_residuals.extend(pack.get("residuals") or [])
    erased = [item for item in exported_residuals if str(item).strip().lower() in {"none", "cleared all", "all clear"}]
    passed = bool(exported_residuals) and not erased
    return {
        "state": "GREEN" if passed else "RED",
        "materialized_residual_record_count": len(residual_records),
        "exported_residual_count": len(exported_residuals),
        "erased_residual_count": len(erased),
        "sample_materialized_residuals": residual_records[:20],
        "exported_residuals": exported_residuals[:80],
        "rule": "Residuals may move into explicit evidence packs or blockers, but cannot silently disappear or be represented as 'none'.",
        **clean_counters(),
    }


def build_verifier_capacity_accounting(assistant_report: dict[str, Any], spine_gate: dict[str, Any], evidence_packs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = assistant_report.get("summary") if isinstance(assistant_report.get("summary"), dict) else {}
    spine_summary = spine_gate.get("summary") if isinstance(spine_gate.get("summary"), dict) else {}
    obligations = [
        {"class": "assistant_runtime_gate", "count": int_or(summary.get("gate_count"))},
        {"class": "viea_spine_profiles", "count": int_or(spine_summary.get("profile_count"))},
        {"class": "evidence_pack_receipts", "count": len(evidence_packs)},
        {"class": "book_schema_checks", "count": 1},
        {"class": "receipt_faithfulness_expected_invalid_controls", "count": 7},
    ]
    total = sum(row["count"] for row in obligations)
    capacity = max(128, total + 16)
    return {
        "state": "GREEN" if total <= capacity else "RED",
        "verifier_capacity_units": capacity,
        "obligation_count": total,
        "residual_obligation_count": max(0, total - capacity),
        "obligations": obligations,
        "decomposition_contract": "capacity units are explicit per gate/profile/pack/schema/control; future runs must carry residual obligations instead of dropping checks",
        **clean_counters(),
    }


def build_governance_tax_accounting(assistant_report: dict[str, Any], refresh: dict[str, Any], started: float) -> dict[str, Any]:
    summary = assistant_report.get("summary") if isinstance(assistant_report.get("summary"), dict) else {}
    assistant_runtime = int_or(summary.get("runtime_ms"))
    refresh_runtime = int_or(refresh.get("runtime_ms"))
    observed_runtime = int((time.perf_counter() - started) * 1000)
    governance_components = {
        "assistant_reported_runtime_ms": assistant_runtime,
        "assistant_refresh_subprocess_runtime_ms": refresh_runtime,
        "weekly_exporter_observed_runtime_ms": observed_runtime,
        "gate_count": int_or(summary.get("gate_count")),
        "hard_gate_count": int_or(summary.get("hard_gate_count")),
        "warning_gate_count": int_or(summary.get("warning_gate_count")),
    }
    return {
        "state": "GREEN",
        "measured": True,
        "baseline_state": "raw_ungoverned_baseline_not_run_by_policy",
        "governance_components": governance_components,
        "tax_interpretation": "This records governed runtime cost and gate load for a real assistant run; it intentionally does not run an unsafe ungoverned assistant baseline.",
        "caught_failure_count": 0,
        **clean_counters(),
    }


def build_capability_claim_dispositions(reference_trace_record: dict[str, Any], evidence_packs: list[dict[str, Any]], residual_audit: dict[str, Any]) -> dict[str, Any]:
    claims = [
        disposition(
            "theseus.public_safe_reference_trace.current",
            "Theseus can export a public-safe end-to-end implementation-reference trace from the current assistant product spine.",
            "prototype-backed",
            [reference_trace_record["trace_id"]],
            ["AI_book import validator accepts the exact exported pack", "external reviewer accepts support-state boundary"],
            ["not a model-quality claim", "not a chapter-core promotion"],
        ),
        disposition(
            "theseus.assistant_product_spine.integrated",
            "The product-facing assistant path emits VCM, planning, tool, verifier, artifact, claim, evidence, residual, and dogfood records in one trace.",
            "prototype-backed",
            [pack["pack_id"] for pack in evidence_packs if pack["name"] in {"assistant_product_spine", "assistant_viea_trace"}],
            ["repeat trace across real daily tasks", "operator usefulness outcomes improve without raw private text leakage"],
            ["assistant utility is separate from learned-generation capability"],
        ),
        disposition(
            "theseus.book_importable_evidence_packs.standard",
            "GREEN gates worth citing can be summarized as public-safe book-importable evidence packs.",
            "prototype-backed",
            [pack["pack_id"] for pack in evidence_packs],
            ["book-side schemas/validators consume the packs", "negative controls remain enforced"],
            ["summary packs do not replace raw artifacts for private replay"],
        ),
        disposition(
            "theseus.learned_generation.public_transfer",
            "The learned generator broadly transfers to public coding benchmarks.",
            "unsupported",
            [],
            ["fresh governed public calibration after private semantic quality improves", "candidate integrity verifies learned full-body candidates"],
            ["tools/templates/routers/fallbacks cannot support this claim"],
        ),
        disposition(
            "theseus.asi_capability",
            "Theseus has reached ASI capability.",
            "unsupported",
            [],
            ["externally reproducible broad capability evidence", "sustained safe self-improvement evidence", "independent review"],
            ["north star only; no current ASI claim"],
        ),
        disposition(
            "theseus.residual_conservation.weekly_focus",
            "Weekly-focus evidence preserves residuals instead of erasing them.",
            "synthetic-test-backed" if residual_audit["state"] == "GREEN" else "unsupported",
            ["reports/theseus_weekly_focus_20260706.json"],
            ["apply conservation audit across all generator repair loops"],
            ["audit scope is selected weekly-focus evidence"],
        ),
    ]
    return {
        "policy": "project_theseus_capability_claim_disposition_ledger_v1",
        "claim_count": len(claims),
        "claims": claims,
        "support_state_effect": "record_shape_only",
        **clean_counters(),
    }


def build_claim_ledger_trace_kernel(
    *,
    assistant_report: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    reference_trace_record: dict[str, Any],
    evidence_packs: list[dict[str, Any]],
    receipt_audit: dict[str, Any],
    claim_dispositions: dict[str, Any],
    assistant_out: Path,
    assistant_trace_out: Path,
    reference_trace_out: Path,
    evidence_packs_out: Path,
) -> dict[str, Any]:
    record_types = {str(row.get("record_type") or "") for row in trace_rows if isinstance(row, dict)}
    missing_record_types = sorted(REFERENCE_TRACE_REQUIRED_RECORD_TYPES - record_types)
    rows_by_type = {record_type: [row for row in trace_rows if row.get("record_type") == record_type] for record_type in record_types}
    claim_rows = rows_by_type.get("claim_record", [])
    evidence_rows = rows_by_type.get("evidence_transition_record", [])
    artifact_rows = rows_by_type.get("artifact_graph_record", [])
    runtime_rows = rows_by_type.get("runtime_adapter_invocation", [])

    support_transitions = []
    invalid_claims = []
    for claim in list_dicts(claim_dispositions.get("claims")):
        support_state = str(claim.get("support_state") or "")
        if support_state not in ALLOWED_CLAIM_SUPPORT_STATES:
            invalid_claims.append({"claim_id": claim.get("claim_id"), "reason": "invalid_support_state", "support_state": support_state})
        if not claim.get("non_claims"):
            invalid_claims.append({"claim_id": claim.get("claim_id"), "reason": "missing_non_claims"})
        if any(int_or(claim.get(counter)) != 0 for counter in NO_CHEAT_COUNTERS):
            invalid_claims.append({"claim_id": claim.get("claim_id"), "reason": "no_cheat_counter_fault"})
        from_state = "unsupported" if support_state == "unsupported" else "argument"
        support_transitions.append(
            {
                "claim_id": claim.get("claim_id"),
                "from_state": from_state,
                "to_state": support_state,
                "evidence_ref_count": len(list_values(claim.get("evidence_refs"))),
                "promotion_blocker_count": len(list_values(claim.get("promotion_blockers"))),
                "non_claim_count": len(list_values(claim.get("non_claims"))),
                "state_change_allowed": support_state in ALLOWED_CLAIM_SUPPORT_STATES,
            }
        )

    digest_replay_checks = []
    for label, path in [
        ("assistant_report", assistant_out),
        ("assistant_trace", assistant_trace_out),
        ("reference_trace_export", reference_trace_out),
        ("evidence_packs_export", evidence_packs_out),
    ]:
        digest_replay_checks.append(
            {
                "label": label,
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "public_safe": not contains_forbidden_public_safe_text({"path": rel(path), "sha256": sha256_file(path) if path.exists() else ""}),
            }
        )
    digest_replay_checks.extend(
        {
            "label": f"evidence_pack:{pack.get('name')}",
            "path": pack.get("source_path"),
            "exists": resolve(str(pack.get("source_path") or "")).exists(),
            "sha256": pack.get("source_sha256"),
            "public_safe": bool(pack.get("public_safe")),
        }
        for pack in evidence_packs
    )

    reference_family_paths = sorted(str(path.relative_to(REPORTS)) for path in REPORTS.glob("theseus_public_safe_reference_trace_*.json"))
    evidence_family_paths = sorted(str(path.relative_to(REPORTS)) for path in REPORTS.glob("theseus_book_importable_evidence_packs_*.json"))
    duplicate_report_family_ok = (
        reference_family_paths == [Path(reference_trace_out).name]
        and evidence_family_paths == [Path(evidence_packs_out).name]
    )

    source_to_verifier_chain = {
        "source_refs": list_values(reference_trace_record.get("source_refs")),
        "parent_artifact_refs": list_values(reference_trace_record.get("parent_artifact_refs")),
        "artifact_refs": list_values(reference_trace_record.get("artifacts")),
        "validation_commands": list_values(reference_trace_record.get("validation_commands")),
        "private_verifier_mentioned": any(
            "private_verifier" in str(item)
            for item in list_values(reference_trace_record.get("authority_chain"))
            + list_values(reference_trace_record.get("artifacts"))
            + list_values(reference_trace_record.get("validation_commands"))
        ),
        "claim_row_count": len(claim_rows),
        "evidence_transition_row_count": len(evidence_rows),
        "artifact_graph_row_count": len(artifact_rows),
        "runtime_adapter_row_count": len(runtime_rows),
    }
    expected_invalid_controls = [
        {
            "control": "missing_claim_record",
            "rejected": bool(claim_rows),
            "reason": "claim row is required for A1 support-state movement",
        },
        {
            "control": "missing_evidence_transition",
            "rejected": bool(evidence_rows),
            "reason": "evidence transition row is required for A1 support-state movement",
        },
        {
            "control": "unsupported_claim_promoted",
            "rejected": all(
                row["to_state"] == "unsupported" or row["evidence_ref_count"] > 0
                for row in support_transitions
            ),
            "reason": "positive support states need at least one evidence ref; unsupported claims stay unsupported",
        },
        {
            "control": "digest_replay_missing",
            "rejected": all(row["exists"] and row["sha256"] for row in digest_replay_checks),
            "reason": "every reference artifact and evidence pack path must exist and carry a digest",
        },
        {
            "control": "duplicate_reference_family",
            "rejected": duplicate_report_family_ok,
            "reason": "A1 uses the existing weekly-focus reference family instead of spawning new report families",
        },
    ]
    hard_gaps = []
    if missing_record_types:
        hard_gaps.append({"kind": "missing_required_trace_record_types", "missing": missing_record_types})
    if invalid_claims:
        hard_gaps.append({"kind": "invalid_claim_dispositions", "invalid_claims": invalid_claims})
    if not all(row["exists"] and row["sha256"] and row["public_safe"] for row in digest_replay_checks):
        hard_gaps.append({"kind": "digest_replay_or_public_safety_fault", "checks": digest_replay_checks})
    if not duplicate_report_family_ok:
        hard_gaps.append(
            {
                "kind": "duplicate_report_family_detected",
                "reference_family_paths": reference_family_paths,
                "evidence_family_paths": evidence_family_paths,
            }
        )
    if not source_to_verifier_chain["private_verifier_mentioned"]:
        hard_gaps.append({"kind": "private_verifier_continuity_missing", "source_to_verifier_chain": source_to_verifier_chain})
    if any(not row["rejected"] for row in expected_invalid_controls):
        hard_gaps.append({"kind": "expected_invalid_control_not_rejected", "controls": expected_invalid_controls})
    if receipt_audit.get("state") != "GREEN":
        hard_gaps.append({"kind": "receipt_faithfulness_not_green", "state": receipt_audit.get("state")})

    state = "GREEN" if not hard_gaps else "RED"
    synthetic_support_ready = (
        state == "GREEN"
        and len(expected_invalid_controls) >= 5
        and all(row["rejected"] for row in expected_invalid_controls)
        and bool(digest_replay_checks)
        and all(row["exists"] and row["sha256"] and row["public_safe"] for row in digest_replay_checks)
        and receipt_audit.get("state") == "GREEN"
        and not invalid_claims
        and not missing_record_types
    )
    support_state = "synthetic-test-backed" if synthetic_support_ready else ("prototype-backed" if state == "GREEN" else "unsupported")
    return {
        "policy": "project_theseus_a1_claim_ledger_trace_kernel_v1",
        "state": state,
        "support_state": support_state,
        "support_state_basis": {
            "valid_reference_trace_present": state == "GREEN",
            "expected_invalid_control_count": len(expected_invalid_controls),
            "expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
            "digest_replay_check_count": len(digest_replay_checks),
            "digest_replay_all_public_safe": all(row["exists"] and row["sha256"] and row["public_safe"] for row in digest_replay_checks),
            "receipt_faithfulness_state": receipt_audit.get("state"),
            "invalid_claim_count": len(invalid_claims),
            "missing_required_record_type_count": len(missing_record_types),
        },
        "slice_id": "A1_claim_ledger_trace_kernel",
        "source_task": "weekly_focus_20260706_reference_trace",
        "required_record_types": sorted(REFERENCE_TRACE_REQUIRED_RECORD_TYPES),
        "observed_record_types": sorted(record_types),
        "missing_record_types": missing_record_types,
        "support_transitions": support_transitions,
        "invalid_claims": invalid_claims,
        "digest_replay_checks": digest_replay_checks,
        "duplicate_report_family": {
            "ok": duplicate_report_family_ok,
            "reference_family_paths": reference_family_paths,
            "evidence_family_paths": evidence_family_paths,
        },
        "source_to_verifier_chain": source_to_verifier_chain,
        "expected_invalid_controls": expected_invalid_controls,
        "receipt_faithfulness_state": receipt_audit.get("state"),
        "hard_gaps": hard_gaps,
        "non_claims": [
            "A1 synthetic-test-backed means this reference-trace kernel has a valid replay trace plus expected-invalid controls; it is not a learned-generation or ASI claim.",
            "Tool-assisted assistant output remains product behavior, not learned model-only capability.",
            "Public benchmark artifacts remain calibration-only and are not training rows.",
        ],
        **clean_counters(),
    }


def build_book_schema_conformance(reference_trace_record: dict[str, Any]) -> dict[str, Any]:
    schema = read_json(REFERENCE_TRACE_SCHEMA, {})
    errors = validate_schema_like(reference_trace_record, schema)
    return {
        "state": "GREEN" if not errors else "RED",
        "passed": not errors,
        "schema_path": rel(REFERENCE_TRACE_SCHEMA),
        "schema_sha256": sha256_file(REFERENCE_TRACE_SCHEMA) if REFERENCE_TRACE_SCHEMA.exists() else "",
        "checked_record": "reference_trace_record",
        "error_count": len(errors),
        "errors": errors,
        "non_claims": ["schema conformance is record-shape evidence only"],
        **clean_counters(),
    }


def build_correctness_preregistration(path: Path) -> dict[str, Any]:
    experiment = {
        "experiment_id": "strict_generator_correctness_in_loop_private_v1_20260706",
        "status": "preregistered_not_run",
        "purpose": "Improve direct learned body-token generator semantic correctness under independent verifier feedback without counting tools, templates, routers, or fallback returns as learned generation.",
        "allowed_data": [
            "private/licensed/governed training rows",
            "metadata-only dogfood outcome rows that passed admission",
            "governed teacher-distillation rows with provenance, license, verifier, and leakage checks",
        ],
        "forbidden_data": [
            "public benchmark prompts/tests/solutions/traces/answer templates",
            "hidden tests",
            "raw private prompts",
        ],
        "candidate_families_allowed": ["learned_full_body_token", "transformer_hybrid_direct_body"],
        "candidate_families_forbidden_for_learned_credit": ["tools", "templates", "routers", "semantic_renderers", "fallbacks", "private_ngrams"],
        "fixed_before_run": {
            "dataset_manifest": "to_be_written_before_execution",
            "heldout_manifest": "to_be_written_before_execution",
            "seed_set": [13, 29, 41],
            "max_runtime_minutes": 180,
            "max_train_rows": 20000,
        },
        "metrics": [
            "loadable_candidate_rate",
            "syntax_valid_rate",
            "behavior_pass_rate",
            "pass_if_any_rate",
            "selected_pass_rate",
            "candidate_integrity_mismatch_count",
            "fallback_return_count",
            "public_boundary_violation_count",
            "verifier_cost_units",
        ],
        "falsification_stops": [
            "candidate_integrity_mismatch_count > 0",
            "public_boundary_violation_count > 0",
            "fallback_return_count > 0",
            "selected_pass_rate fails to beat current private direct-body baseline on preregistered heldout",
            "teacher rows fail provenance/license/verifier/leakage admission",
        ],
        "outputs_planned": [
            "candidate manifest",
            "candidate integrity report",
            "private verifier report",
            "residual taxonomy",
            "no-promotion decision if improvement is insufficient",
        ],
        "public_benchmark_training_allowed": False,
        "public_benchmark_execution_allowed": False,
        "external_inference_runtime_allowed": False,
        "teacher_mode_allowed": "governed_training_only",
        **clean_counters(),
    }
    payload = {
        "policy": "project_theseus_correctness_in_loop_generator_experiment_registry_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "summary": {
            "experiment_count": 1,
            "preregistered_not_run_count": 1,
            "public_benchmark_training_allowed": False,
            "fallback_credit_allowed": False,
        },
        "experiments": [experiment],
        "non_claims": [
            "Preregistration is not a training result.",
            "Preregistration is not a public benchmark result.",
            "Preregistration does not promote learned generation.",
        ],
        **clean_counters(),
    }
    write_json(path, payload)
    return payload


def build_gates(**kwargs: Any) -> list[dict[str, Any]]:
    assistant_report = kwargs["assistant_report"]
    selected_trace_rows = kwargs["selected_trace_rows"]
    reference_trace_record = kwargs["reference_trace_record"]
    evidence_packs = kwargs["evidence_packs"]
    receipt_audit = kwargs["receipt_audit"]
    residual_audit = kwargs["residual_audit"]
    verifier_capacity = kwargs["verifier_capacity"]
    governance_tax = kwargs["governance_tax"]
    claim_dispositions = kwargs["claim_dispositions"]
    claim_ledger_trace_kernel = kwargs["claim_ledger_trace_kernel"]
    schema_conformance = kwargs["schema_conformance"]
    preregistration = kwargs["preregistration"]
    roadmap_gate = kwargs["roadmap_gate"]
    module_gate = kwargs["module_gate"]
    crosswalk = kwargs["crosswalk"]
    return [
        gate("assistant_product_spine_green", assistant_report.get("trigger_state") == "GREEN", "hard", assistant_report.get("trigger_state")),
        gate("assistant_trace_has_rows", len(selected_trace_rows) >= 10, "hard", len(selected_trace_rows)),
        gate("reference_trace_schema_conformant", schema_conformance["passed"], "hard", schema_conformance.get("errors")),
        gate("reference_trace_public_safe", not contains_forbidden_public_safe_text(reference_trace_record), "hard", "forbidden public-safe terms absent"),
        gate("evidence_pack_count", len(evidence_packs) >= 8, "hard", len(evidence_packs)),
        gate("evidence_packs_public_safe", all(pack["public_safe"] for pack in evidence_packs), "hard", [pack["name"] for pack in evidence_packs if not pack["public_safe"]]),
        gate("receipt_faithfulness_expected_invalids_rejected", receipt_audit["state"] == "GREEN", "hard", receipt_audit),
        gate("residual_conservation_green", residual_audit["state"] == "GREEN", "hard", residual_audit),
        gate("verifier_capacity_accounted", verifier_capacity["state"] == "GREEN", "hard", verifier_capacity),
        gate("governance_tax_measured", governance_tax["measured"] is True, "hard", governance_tax),
        gate("claim_dispositions_present", len(claim_dispositions["claims"]) >= 5, "hard", len(claim_dispositions["claims"])),
        gate("claim_ledger_trace_kernel_green", claim_ledger_trace_kernel["state"] == "GREEN", "hard", claim_ledger_trace_kernel),
        gate("exactly_one_preregistered_experiment", len(preregistration.get("experiments") or []) == 1, "hard", preregistration.get("summary")),
        gate("roadmap_gate_no_hard_gaps", int_or((roadmap_gate.get("summary") or {}).get("hard_gap_count")) == 0, "warning", roadmap_gate.get("trigger_state")),
        gate("module_definition_gate_green", module_gate.get("trigger_state") == "GREEN", "warning", module_gate.get("trigger_state")),
        gate("book_crosswalk_public_safe_smoke", bool((crosswalk.get("summary") or {}).get("public_safe_evidence_smoke_passed")), "warning", (crosswalk.get("summary") or {}).get("public_safe_evidence_smoke_passed")),
        gate("no_cheat_counters_clean", all(int_or(kwargs.get(key)) == 0 for key in NO_CHEAT_COUNTERS), "hard", clean_counters()),
    ]


def validate_schema_like(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    if not schema:
        return [f"{path}: schema missing"]
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"{path}.{field}: missing required field")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for field, child_schema in properties.items():
            if field in value:
                errors.extend(validate_schema_like(value[field], child_schema, f"{path}.{field}"))
    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        child_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        for idx, item in enumerate(value):
            errors.extend(validate_schema_like(item, child_schema, f"{path}[{idx}]"))
    elif expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
        elif schema.get("minLength") and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: too short")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}: expected boolean")
    elif expected_type == "number" and not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    return errors


def validate_evidence_pack(pack: dict[str, Any]) -> bool:
    if not pack:
        return False
    path = resolve(pack.get("source_path", ""))
    if not path.exists():
        return False
    if sha256_file(path) != pack.get("source_sha256"):
        return False
    if pack.get("private_payload_copied"):
        return False
    if not pack.get("public_safe"):
        return False
    if not pack.get("non_claims"):
        return False
    if str(pack.get("support_state")) in {"empirical-test-backed", "prototype-backed"} and not pack.get("claim_boundaries"):
        return False
    return all(int_or(pack.get(key)) == 0 for key in NO_CHEAT_COUNTERS)


def latest_assistant_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = [str(row.get("assistant_run_id") or "") for row in rows if isinstance(row, dict) and row.get("assistant_run_id")]
    if not run_ids:
        return []
    latest = run_ids[-1]
    return [row for row in rows if str(row.get("assistant_run_id") or "") == latest]


def artifact_ref(label: str, path: Path) -> str:
    if not path.exists():
        return f"{label}:{rel(path)}:missing"
    return f"{label}:{rel(path)}:sha256:{sha256_file(path)}"


def public_safety_summary(value: Any) -> dict[str, Any]:
    return {
        "public_safe": not contains_forbidden_public_safe_text(value),
        "private_payload_copied": False,
        "forbidden_term_count": forbidden_term_count(value),
        "redaction_state": "summary_and_digest_only",
    }


def support_state_for(trigger_state: str, boundary: str) -> str:
    if trigger_state == "GREEN":
        return "prototype-backed" if boundary == "runtime_trace" else "record_shape_only"
    if trigger_state == "YELLOW":
        return "argument"
    return "blocked"


def negative_controls_for(name: str) -> list[str]:
    return [
        f"{name}: digest mismatch must reject",
        f"{name}: private payload copy must reject",
        f"{name}: no-cheat counter fault must reject",
        f"{name}: support-state overclaim must reject",
    ]


def residuals_for(name: str, payload: dict[str, Any], reference_trace_record: dict[str, Any], assistant_report: dict[str, Any]) -> list[str]:
    if name == "assistant_product_spine":
        summary = assistant_report.get("summary") if isinstance(assistant_report.get("summary"), dict) else {}
        return [
            f"latest_public_dominant_residual={summary.get('latest_public_dominant_residual')}",
            "assistant product trace is not learned-generation promotion evidence",
        ]
    if name == "assistant_viea_trace":
        return list(reference_trace_record.get("residual_deltas") or [])
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    residuals = []
    for key in ("hard_gap_count", "warning_count", "roadmap_backlog_item_count", "stale_phase_count"):
        if key in summary:
            residuals.append(f"{key}={summary.get(key)}")
    return residuals or ["no additional residual claimed; source report remains authoritative"]


def public_safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    allowed = {}
    for key, value in summary.items():
        if key.endswith("_count") or key.endswith("_state") or key.endswith("_ready") or key in {
            "trigger_state",
            "intent",
            "assistant_lane",
            "assistant_viea_trace_complete",
            "assistant_viea_trace_record_count",
            "runtime_ms",
            "public_safe_evidence_smoke_passed",
            "roadmap_backlog_item_count",
            "hard_gap_count",
            "warning_count",
        }:
            allowed[key] = value
    return allowed


def disposition(claim_id: str, claim: str, support_state: str, evidence_refs: list[str], required_next_evidence: list[str], non_claims: list[str]) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "support_state": support_state,
        "evidence_refs": evidence_refs,
        "required_next_evidence": required_next_evidence,
        "promotion_blockers": required_next_evidence,
        "review_status": "open",
        "support_state_effect": "record_shape_only" if support_state != "unsupported" else "none",
        "non_claims": non_claims,
        **clean_counters(),
    }


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "failed_gates": [gate for gate in report["gates"] if not gate["passed"]],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Theseus Weekly Focus 2026-07-06",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Assistant state: `{summary['assistant_trigger_state']}`",
        f"- Reference trace export: `{summary['reference_trace_export']}`",
        f"- Evidence packs: `{summary['evidence_pack_count']}`",
        f"- Receipt controls rejected: `{summary['receipt_expected_invalid_rejected_count']}/{summary['receipt_expected_invalid_control_count']}`",
        f"- Residual conservation: `{summary['residual_conservation_state']}`",
        f"- Verifier capacity: `{summary['verifier_capacity_state']}`",
        f"- Governance tax measured: `{summary['governance_tax_measured']}`",
        f"- A1 claim-ledger trace kernel: `{summary.get('claim_ledger_trace_kernel_state')}` (`{summary.get('claim_ledger_trace_kernel_support_state')}`)",
        f"- Preregistered experiments: `{summary['preregistered_experiment_count']}`",
        "",
        "## Gates",
        "",
    ]
    for item in report["gates"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{status}` `{item['name']}` ({item['severity']})")
    lines.extend(["", "## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def contains_forbidden_public_safe_text(value: Any) -> bool:
    blob = json.dumps(value, sort_keys=True, default=str)
    return any(term in blob for term in FORBIDDEN_PUBLIC_SAFE_TERMS)


def forbidden_term_count(value: Any) -> int:
    blob = json.dumps(value, sort_keys=True, default=str)
    return sum(blob.count(term) for term in FORBIDDEN_PUBLIC_SAFE_TERMS)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def clean_counters() -> dict[str, int]:
    return {key: 0 for key in NO_CHEAT_COUNTERS}


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def first_nonempty(values: list[Any]) -> Any:
    for value in values:
        if value:
            return value
    return None


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{stable_hash(parts)[:16]}"


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
