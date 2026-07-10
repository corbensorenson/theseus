#!/usr/bin/env python3
"""Admission audit for broad Theseus training sources.

This layer answers a narrow question before any survival-lane training run:
which local files are allowed to contribute training pressure, which files are
heldout/eval-only, and which public benchmark surfaces must remain quarantine
or calibration-only. It is deliberately local-only and metadata-first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vcm_consumer_abi
import training_data_lineage_audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "training_data_admission_v1.md"
DEFAULT_MANIFEST = ROOT / "data" / "training_sources" / "training_data_admission_v1.json"
DEFAULT_GROWTH_POLICY = ROOT / "configs" / "permissive_growth_policy.json"
DEFAULT_VCM_CONTEXT_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"

ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "mpl-2.0",
    "cc0-1.0",
    "unlicense",
    "public-domain",
    "odc-by",
    "cc-by-4.0",
    "project-internal",
    "project_generated",
    "private_generated",
}

PUBLIC_BENCHMARK_TOKENS = {
    "mbpp",
    "evalplus",
    "human_eval",
    "humaneval",
    "bigcodebench",
    "livecodebench",
    "swe_bench",
    "swe-bench",
    "terminal_bench",
    "webarena",
    "mmlu",
    "gpqa",
    "gsm8k",
    "ruler",
    "babilong",
    "longbench",
    "longmemeval",
    "needlebench",
    "infinitebench",
    "blimp",
}

PUBLIC_FLAG_KEYS = {
    "public_benchmark",
    "public_benchmark_row",
    "public_benchmark_solutions_included",
    "public_prompts_included",
    "public_score_labels_included",
    "public_tests_included",
    "public_solutions_used",
    "public_tests_used",
    "public_prompts_used",
    "public_benchmark_answers_used",
}

TEXT_FINGERPRINT_KEYS = {
    "prompt",
    "tests",
    "test",
    "canonical_solution",
    "solution",
    "solution_body",
    "signature",
    "buggy",
    "case_id",
    "task_id",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--manifest-out", default=rel(DEFAULT_MANIFEST))
    parser.add_argument("--candidate-ledger-out", default=rel(training_data_lineage_audit.DEFAULT_LEDGER))
    parser.add_argument("--max-candidate-receipts", type=int, default=0)
    parser.add_argument("--max-public-contamination-texts", type=int, default=20000)
    parser.add_argument("--growth-policy", default=rel(DEFAULT_GROWTH_POLICY))
    parser.add_argument("--vcm-context-governor", default=rel(DEFAULT_VCM_CONTEXT_GOVERNOR))
    parser.add_argument("--sample-rows-per-source", type=int, default=256)
    parser.add_argument("--max-public-fingerprints", type=int, default=20000)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    lineage = training_data_lineage_audit.build_lineage_bundle(
        report,
        ledger_path=resolve(args.candidate_ledger_out),
        max_rows=max(0, int(args.max_candidate_receipts)),
        max_public_texts=max(100, int(args.max_public_contamination_texts)),
    )
    report["candidate_lineage"] = lineage
    report["viea_training_data_context_records"].extend(lineage.get("viea_data_governance_records", []))
    report["summary"].update({
        "candidate_receipt_count": (lineage.get("summary") or {}).get("candidate_receipt_count", 0),
        "candidate_admitted_count": (lineage.get("summary") or {}).get("admitted_candidate_count", 0),
        "candidate_quarantined_count": (lineage.get("summary") or {}).get("quarantined_candidate_count", 0),
        "candidate_rejected_count": (lineage.get("summary") or {}).get("rejected_candidate_count", 0),
        "candidate_lineage_trigger_state": lineage.get("trigger_state"),
        "candidate_hash_filter_ready": (lineage.get("summary") or {}).get("admitted_hash_filter_ready", False),
    })
    report["gates"].append(gate(
        "candidate_level_lineage_admission_ready",
        lineage.get("trigger_state") in {"GREEN", "YELLOW"}
        and bool((lineage.get("summary") or {}).get("admitted_hash_filter_ready")),
        lineage.get("summary"),
        "hard",
    ))
    if lineage.get("trigger_state") == "RED":
        report["trigger_state"] = "RED"
    elif lineage.get("trigger_state") == "YELLOW" and report.get("trigger_state") == "GREEN":
        report["trigger_state"] = "YELLOW"
    write_json(resolve(args.out), report)
    write_json(resolve(args.manifest_out), manifest_payload(report))
    write_text(resolve(args.markdown_out), render_markdown(report))
    report_summary = report.get("summary") or {}
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "summary": {
            key: report_summary.get(key) for key in (
                "local_source_count",
                "allowed_training_source_count",
                "admitted_open_public_source_count",
                "admitted_open_public_row_count",
                "candidate_receipt_count",
                "candidate_admitted_count",
                "candidate_quarantined_count",
                "candidate_rejected_count",
                "candidate_lineage_trigger_state",
                "candidate_hash_filter_ready",
                "public_benchmark_payload_admitted",
                "teacher_distillation_manifest_rows",
                "teacher_rows_admitted_outside_distillation_gate",
                "vcm_context_governor_ready",
                "external_inference_calls",
            )
        },
        "failed_hard_gates": [
            row.get("name") for row in report.get("gates", [])
            if row.get("severity") == "hard" and row.get("passed") is not True
        ],
        "candidate_lineage": {
            "trigger_state": lineage.get("trigger_state"),
            "summary": lineage.get("summary"),
            "warnings": lineage.get("warnings"),
        },
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    growth_policy = read_json(resolve(args.growth_policy))
    vcm_receipt = vcm_context_governor_receipt(resolve(args.vcm_context_governor))
    public_open_policy = growth_policy.get("public_open_training_data") if isinstance(growth_policy.get("public_open_training_data"), dict) else {}
    public_open_training_allowed = public_open_policy.get("default") == "allowed_when_governed"
    public_fingerprints = load_public_fingerprints(limit=max(1, args.max_public_fingerprints))
    local_sources = discover_local_sources()
    source_rows = [
        audit_local_source(
            path,
            public_fingerprints=public_fingerprints,
            sample_limit=max(1, args.sample_rows_per_source),
            public_open_training_allowed=public_open_training_allowed,
        )
        for path in local_sources
    ]
    open_code_candidates = audit_open_code_pantry(public_open_training_allowed=public_open_training_allowed)
    teacher_distillation = audit_teacher_distillation_gate()
    public_benchmark_sources = [row for row in source_rows if row["source_kind"] == "public_benchmark_quarantine"]
    train_allowed = [row for row in source_rows if row["allowed_for_training"]]
    rejected = [row for row in source_rows if not row["allowed_for_training"] and row["training_use"] == "rejected"]
    heldout = [row for row in source_rows if row["training_use"] == "heldout_eval_only"]
    quarantined = [row for row in source_rows if row["training_use"] == "quarantine"]

    capability_counts: Counter[str] = Counter()
    for row in train_allowed:
        capability_counts.update(row["source_family_tags"])

    gates = [
        gate("local_sources_discovered", len(source_rows) > 0, len(source_rows), "hard"),
        gate("training_sources_admitted", len(train_allowed) > 0, len(train_allowed), "hard"),
        gate(
            "public_benchmark_payload_admitted_zero",
            not any(row["public_benchmark_payload_detected"] for row in train_allowed),
            [row["source_id"] for row in train_allowed if row["public_benchmark_payload_detected"]],
            "hard",
        ),
        gate(
            "public_benchmark_quarantine_not_train_allowed",
            not any(row["allowed_for_training"] for row in public_benchmark_sources),
            [row["source_id"] for row in public_benchmark_sources if row["allowed_for_training"]],
            "hard",
        ),
        gate(
            "exact_public_fingerprint_overlap_zero_for_training",
            not any(row["benchmark_overlap_check"]["exact_overlap_count"] > 0 for row in train_allowed),
            overlap_evidence(train_allowed),
            "hard",
        ),
        gate(
            "fallback_returns_not_admitted",
            not any(row["fallback_return_count"] > 0 for row in train_allowed),
            [row["source_id"] for row in train_allowed if row["fallback_return_count"] > 0],
            "hard",
        ),
        gate(
            "raw_user_text_not_admitted",
            not any(row["raw_user_text_count"] > 0 for row in train_allowed),
            [row["source_id"] for row in train_allowed if row["raw_user_text_count"] > 0],
            "hard",
        ),
        gate(
            "external_inference_zero",
            sum(int(row["external_inference_calls"]) for row in source_rows) == 0,
            sum(int(row["external_inference_calls"]) for row in source_rows),
            "hard",
        ),
        gate(
            "teacher_rows_not_admitted_outside_distillation_gate",
            not any(row["teacher_row_count"] > 0 for row in train_allowed),
            teacher_row_evidence(train_allowed),
            "hard",
        ),
        gate(
            "teacher_distillation_manifest_rows_require_gate",
            teacher_distillation["accepted_manifest_row_count"] == 0 or teacher_distillation["distillation_allowed"],
            teacher_distillation,
            "hard",
        ),
        gate(
            "teacher_distillation_manifest_public_clean",
            teacher_distillation["public_overlap_hits"] == 0
            and teacher_distillation["public_training_rows_written"] == 0
            and teacher_distillation["runtime_serving_forbidden"] is True,
            teacher_distillation,
            "hard",
        ),
        gate(
            "vcm_context_governor_ready_for_training_admission",
            bool(vcm_receipt.get("ready")),
            vcm_receipt,
            "hard",
        ),
        gate(
            "open_code_corpora_manifested_pending_import",
            len(open_code_candidates) > 0,
            len(open_code_candidates),
            "warning",
        ),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] != "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and (warning_failed or quarantined):
        trigger_state = "YELLOW"

    return {
        "policy": "project_theseus_training_data_admission_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": (
            "Admit only provenance-clean, decontaminated local/private training sources for the "
            "broad capability survival lane; keep public benchmarks calibration-only."
        ),
        "summary": {
            "local_source_count": len(source_rows),
            "allowed_training_source_count": len(train_allowed),
            "heldout_eval_only_source_count": len(heldout),
            "rejected_source_count": len(rejected),
            "quarantined_source_count": len(quarantined),
            "public_benchmark_quarantine_source_count": len(public_benchmark_sources),
            "open_code_candidate_count": len(open_code_candidates),
            "public_open_training_allowed": public_open_training_allowed,
            "admitted_open_public_source_count": sum(1 for row in train_allowed if row["source_kind"] == "open_public_training_rows"),
            "admitted_open_public_row_count": sum(int(row["row_count"]) for row in train_allowed if row["source_kind"] == "open_public_training_rows"),
            "public_benchmark_training_allowed": False,
            "public_benchmark_payload_admitted": False,
            "teacher_rows_admitted_outside_distillation_gate": False,
            "teacher_row_count_in_admitted_sources": sum(int(row["teacher_row_count"]) for row in train_allowed),
            "teacher_distillation_gate_allowed": teacher_distillation["distillation_allowed"],
            "teacher_distillation_manifest_ready_for_distillation": teacher_distillation["manifest_ready_for_distillation"],
            "teacher_distillation_manifest_rows": teacher_distillation["accepted_manifest_row_count"],
            "teacher_distillation_ledger_rows": teacher_distillation["ledger_row_count"],
            "teacher_distillation_proposal_ledger_rows": teacher_distillation["proposal_ledger_row_count"],
            "teacher_distillation_rejected_ledger_rows": teacher_distillation["rejected_ledger_row_count"],
            "teacher_distillation_public_overlap_hits": teacher_distillation["public_overlap_hits"],
            "teacher_distillation_verifier_pass_rate_applicable": teacher_distillation["verifier_pass_rate_applicable"],
            "teacher_distillation_admission_safety_clean": teacher_distillation["admission_safety_checks_clean"],
            "public_fingerprint_count": len(public_fingerprints),
            "capability_tag_counts": dict(sorted(capability_counts.items())),
            "external_inference_calls": 0,
            "teacher_used": False,
            "vcm_context_governor_ready": bool(vcm_receipt.get("ready")),
            "vcm_context_governor_state": vcm_receipt.get("trigger_state"),
            "vcm_context_governor_receipt_id": vcm_receipt.get("receipt_id"),
            "vcm_context_resolver_status": vcm_receipt.get("context_resolver_status"),
            "vcm_context_resolver_passed_count": vcm_receipt.get("context_resolver_passed_count"),
            "vcm_context_resolver_request_count": vcm_receipt.get("context_resolver_request_count"),
        },
        "hard_invariants": [
            "Public benchmark prompts, tests, hidden tests, solutions, traces, and answer templates are calibration-only.",
            "Fallback returns and template/loop-closure credit are not admissible training evidence.",
            "Dogfood rows are admissible only when raw user text is absent and outcome metadata is redacted.",
            "Teacher-generated rows are admissible only through the governed teacher distillation manifest and gate.",
            "Open public corpora are admissible when local hash, provenance, license, and decontamination evidence exist.",
            "Heldout private eval files are never training sources.",
        ],
        "source_admissions": source_rows,
        "vcm_context_governor_receipt": vcm_receipt,
        "viea_training_data_context_records": training_data_vcm_records(vcm_receipt),
        "teacher_distillation_admission": teacher_distillation,
        "open_code_public_corpus_candidates": open_code_candidates,
        "growth_policy": rel(resolve(args.growth_policy)),
        "gates": gates,
        "score_semantics": (
            "Metadata admission only. This script does not train, fetch data, call a teacher, run public "
            "calibration, write model artifacts, or serve external tokens."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def discover_local_sources() -> list[Path]:
    candidates: list[Path] = []
    roots = [
        ROOT / "data" / "training_data" / "high_transfer" / "private_train",
        ROOT / "data" / "training_data" / "high_transfer" / "private_eval",
        ROOT / "data" / "private_code_curriculum",
        ROOT / "data" / "sts_learning",
        ROOT / "data" / "training_sources",
        ROOT / "data" / "training_data" / "open_conversation_pantry" / "private_train",
        ROOT / "data" / "training_data" / "open_conversation_pantry" / "samples",
        ROOT / "data" / "training_data" / "open_conversation_pantry" / "sts_streams",
        ROOT / "data" / "public_code_benchmark_manifests",
        ROOT / "data" / "old_project_benchmarks" / "cases",
        ROOT / "data" / "public_benchmarks" / "vcm_memory_quarantine",
    ]
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.jsonl", "*.json"):
            candidates.extend(path for path in root.rglob(suffix) if path.is_file())
    for path in [ROOT / "data" / "public_blimp_train.jsonl", ROOT / "data" / "public_blimp_eval.jsonl"]:
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates), key=lambda path: rel(path))


def audit_local_source(
    path: Path,
    *,
    public_fingerprints: set[str],
    sample_limit: int,
    public_open_training_allowed: bool,
) -> dict[str, Any]:
    rel_path = rel(path)
    rows = read_sample_rows(path, limit=sample_limit)
    row_count = count_rows(path)
    file_sha = sha256_file(path)
    path_lower = rel_path.lower()
    public_path = is_public_benchmark_path(path)
    heldout_eval = "/private_eval/" in f"/{rel_path}"
    private_train_path = "/private_train/" in f"/{rel_path}"
    dogfood_path = "dogfood" in path_lower
    trace_path = "trace_fabric" in path_lower
    private_code_curriculum_path = "/private_code_curriculum/" in f"/{rel_path}"
    open_public_path = "/data/training_data/open_conversation_pantry/" in f"/{rel_path}"

    flag_counts = Counter()
    license_counts = Counter()
    family_tags = Counter()
    exact_overlap_count = 0
    fallback_count = 0
    raw_user_text_count = 0
    external_calls = 0
    teacher_rows = 0
    sampled_fingerprints = []

    for row in rows:
        flat = flatten_dict(row)
        for key in PUBLIC_FLAG_KEYS:
            if truthy(flat.get(key)):
                flag_counts[key] += 1
        license_value = normalize_license(first_present(flat, ["license_spdx", "license", "spdx_license"]))
        if license_value:
            license_counts[license_value] += 1
        for tag in extract_family_tags(row, rel_path):
            family_tags[tag] += 1
        fallback_count += int(contains_fallback_return(row))
        raw_user_text_count += int(truthy(flat.get("raw_user_text_included")) or "raw_user_text" in flat)
        external_calls += int_or(flat.get("external_inference_calls"))
        teacher_rows += int(truthy(flat.get("teacher_generated")) or truthy(flat.get("teacher_used")))
        fingerprints = row_fingerprints(row)
        if len(sampled_fingerprints) < 12:
            sampled_fingerprints.extend(sorted(fingerprints)[: 12 - len(sampled_fingerprints)])
        exact_overlap_count += len(fingerprints.intersection(public_fingerprints))

    suspicious_public_name = path_has_public_benchmark_token(path) and not (
        private_train_path and rows_claim_public_safe_private(rows)
    )
    public_payload_detected = public_path or bool(flag_counts) or suspicious_public_name
    license_status = license_decision(license_counts, private_train_path=private_train_path, dogfood_path=dogfood_path)
    provenance_status = provenance_decision(rows, rel_path)
    training_use = decide_training_use(
        public_path=public_path,
        public_payload_detected=public_payload_detected,
        heldout_eval=heldout_eval,
        private_train_path=private_train_path,
        dogfood_path=dogfood_path,
        trace_path=trace_path,
        private_code_curriculum_path=private_code_curriculum_path,
        open_public_path=open_public_path,
        public_open_training_allowed=public_open_training_allowed,
        exact_overlap_count=exact_overlap_count,
        fallback_count=fallback_count,
        raw_user_text_count=raw_user_text_count,
        external_calls=external_calls,
        license_status=license_status,
        provenance_status=provenance_status,
    )
    allowed = training_use == "allowed"
    rejection_reasons = rejection_reasons_for(
        public_path=public_path,
        public_payload_detected=public_payload_detected,
        heldout_eval=heldout_eval,
        exact_overlap_count=exact_overlap_count,
        fallback_count=fallback_count,
        raw_user_text_count=raw_user_text_count,
        external_calls=external_calls,
        license_status=license_status,
        provenance_status=provenance_status,
        training_use=training_use,
    )
    source_kind = source_kind_for(
        path,
        training_use,
        public_path=public_path,
        dogfood_path=dogfood_path,
        trace_path=trace_path,
        open_public_path=open_public_path,
    )
    return {
        "source_id": stable_id(rel_path),
        "path": rel_path,
        "source_kind": source_kind,
        "training_use": training_use,
        "allowed_for_training": allowed,
        "row_count": row_count,
        "sampled_row_count": len(rows),
        "sha256": file_sha,
        "split_hash": stable_hash({"path": rel_path, "sha256": file_sha, "row_count": row_count, "training_use": training_use}),
        "license_status": license_status,
        "license_counts": dict(sorted(license_counts.items())),
        "provenance_status": provenance_status,
        "source_family_tags": sorted(tag for tag, count in family_tags.items() if count > 0),
        "public_flag_counts": dict(sorted(flag_counts.items())),
        "public_benchmark_payload_detected": public_payload_detected,
        "public_open_source_detected": open_public_path,
        "public_open_training_allowed_by_policy": public_open_training_allowed,
        "benchmark_overlap_check": {
            "method": "exact_hash_of_public_manifest_text_fields_vs_sampled_source_text_fields",
            "public_fingerprint_count": len(public_fingerprints),
            "sampled_fingerprint_count": len(set(sampled_fingerprints)),
            "exact_overlap_count": exact_overlap_count,
        },
        "fallback_return_count": fallback_count,
        "raw_user_text_count": raw_user_text_count,
        "teacher_row_count": teacher_rows,
        "external_inference_calls": external_calls,
        "rejection_reasons": rejection_reasons,
        "opt_out_removal_notes": opt_out_notes(rel_path, training_use),
    }


def audit_open_code_pantry(*, public_open_training_allowed: bool) -> list[dict[str, Any]]:
    path = ROOT / "configs" / "open_code_training_pantry_expanded.json"
    payload = read_json(path)
    rows = []
    repos = payload.get("repos") if isinstance(payload.get("repos"), list) else []
    for row in repos:
        if not isinstance(row, dict):
            continue
        repo = str(row.get("repo") or "").strip()
        if not repo:
            continue
        rows.append({
            "source_id": stable_id(f"open_code:{repo}"),
            "repo": repo,
            "why": row.get("why"),
            "source_kind": "public_code_corpus_candidate",
            "training_use": (
                "eligible_after_local_import_hash_license_and_decontamination"
                if public_open_training_allowed
                else "pending_policy_enablement"
            ),
            "allowed_for_training": False,
            "license_status": "pending_spdx_verification",
            "provenance_status": "manifested_candidate_only",
            "benchmark_overlap_check": "not_run_until_local_source_import",
            "required_before_training": [
                "local source archive or clone path",
                "verified SPDX license in allowlist",
                "source hash manifest",
                "benchmark-name and public-prompt decontamination scan",
            ],
        })
    return rows


def audit_teacher_distillation_gate() -> dict[str, Any]:
    manifest_path = ROOT / "reports" / "teacher_distillation_manifest.json"
    gate_path = ROOT / "reports" / "teacher_distillation_gate.json"
    ledger_path = ROOT / "reports" / "teacher_distillation_ledger.jsonl"
    manifest = read_json(manifest_path)
    gate_report = read_json(gate_path)
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    ledger_rows = read_jsonl(ledger_path)
    admission_checks = manifest.get("admission_checks") or manifest_summary.get("admission_checks") or {}
    admission_safety_checks = manifest.get("admission_safety_checks") or manifest_summary.get("admission_safety_checks") or {}
    gate_summary = gate_report.get("summary") if isinstance(gate_report.get("summary"), dict) else {}
    accepted_ledger_rows = [
        row
        for row in ledger_rows
        if isinstance(row, dict)
        and row.get("accepted") is True
        and str(row.get("source_kind") or "").startswith("teacher")
    ]
    return {
        "manifest_path": rel(manifest_path),
        "manifest_present": bool(manifest),
        "gate_path": rel(gate_path),
        "gate_trigger_state": gate_report.get("trigger_state"),
        "distillation_allowed": bool(gate_report.get("distillation_allowed") or gate_summary.get("distillation_allowed")),
        "manifest_ready_for_distillation": bool(manifest_summary.get("manifest_ready_for_distillation")),
        "accepted_manifest_row_count": int(manifest_summary.get("row_count") or len(rows) or 0),
        "accepted_ledger_row_count": len(accepted_ledger_rows),
        "proposal_ledger_row_count": int(manifest_summary.get("proposal_ledger_row_count") or 0),
        "rejected_ledger_row_count": int(manifest_summary.get("rejected_ledger_row_count") or 0),
        "ledger_path": rel(ledger_path),
        "ledger_present": ledger_path.exists(),
        "ledger_row_count": len(ledger_rows),
        "public_overlap_hits": int(manifest_summary.get("public_overlap_hits") or manifest.get("public_overlap_hits") or 0),
        "holdout_overlap_hits": int(manifest_summary.get("holdout_overlap_hits") or manifest.get("holdout_overlap_hits") or 0),
        "verifier_pass_rate_applicable": bool(
            manifest_summary.get("verifier_pass_rate_applicable")
            or manifest.get("verifier_pass_rate_applicable")
        ),
        "admission_safety_checks_clean": bool(
            manifest_summary.get("admission_safety_checks_clean")
            or manifest.get("admission_safety_checks_clean")
        ),
        "public_training_rows_written": int(
            manifest_summary.get("public_training_rows_written")
            or manifest.get("public_training_rows_written")
            or 0
        ),
        "runtime_serving_forbidden": admission_checks.get("runtime_serving_forbidden") is True
        or get_path(manifest, ["boundary", "runtime_serving_external_tokens"]) == "forbidden",
        "admission_checks": admission_checks,
        "admission_safety_checks": admission_safety_checks,
        "score_semantics": (
            "Teacher rows are trainable only through the governed teacher distillation gate. "
            "Proposal and rejected ledger rows are evidence, not training rows."
        ),
    }


def decide_training_use(
    *,
    public_path: bool,
    public_payload_detected: bool,
    heldout_eval: bool,
    private_train_path: bool,
    dogfood_path: bool,
    trace_path: bool,
    private_code_curriculum_path: bool,
    open_public_path: bool,
    public_open_training_allowed: bool,
    exact_overlap_count: int,
    fallback_count: int,
    raw_user_text_count: int,
    external_calls: int,
    license_status: str,
    provenance_status: str,
) -> str:
    if public_path:
        return "rejected"
    if heldout_eval:
        return "heldout_eval_only"
    if public_payload_detected:
        return "quarantine"
    if exact_overlap_count > 0 or fallback_count > 0 or raw_user_text_count > 0 or external_calls > 0:
        return "quarantine"
    if license_status.startswith("blocked") or provenance_status.startswith("blocked"):
        return "quarantine"
    if public_open_training_allowed and open_public_path:
        return "allowed"
    if private_train_path or dogfood_path or trace_path or private_code_curriculum_path:
        return "allowed"
    return "quarantine"


def rejection_reasons_for(**kwargs: Any) -> list[str]:
    reasons = []
    if kwargs["public_path"]:
        reasons.append("public_benchmark_or_quarantine_path")
    if kwargs["public_payload_detected"]:
        reasons.append("public_benchmark_payload_or_name_detected")
    if kwargs["heldout_eval"]:
        reasons.append("private_heldout_eval_only")
    if kwargs["exact_overlap_count"] > 0:
        reasons.append("exact_public_fingerprint_overlap")
    if kwargs["fallback_count"] > 0:
        reasons.append("fallback_return_detected")
    if kwargs["raw_user_text_count"] > 0:
        reasons.append("raw_user_text_detected")
    if kwargs["external_calls"] > 0:
        reasons.append("external_inference_detected")
    if kwargs["license_status"].startswith("blocked"):
        reasons.append(kwargs["license_status"])
    if kwargs["provenance_status"].startswith("blocked"):
        reasons.append(kwargs["provenance_status"])
    if kwargs["training_use"] == "allowed":
        return []
    return reasons or [kwargs["training_use"]]


def source_kind_for(
    path: Path,
    training_use: str,
    *,
    public_path: bool,
    dogfood_path: bool,
    trace_path: bool,
    open_public_path: bool,
) -> str:
    rel_path = rel(path)
    if public_path:
        return "public_benchmark_quarantine"
    if training_use == "heldout_eval_only":
        return "private_heldout_eval"
    if dogfood_path:
        return "dogfood_metadata"
    if open_public_path and training_use == "allowed":
        return "open_public_training_rows"
    if trace_path:
        return "trace_metadata_or_materialized_private"
    if "/private_train/" in f"/{rel_path}":
        return "private_training_rows"
    if "/private_code_curriculum/" in f"/{rel_path}":
        return "private_code_curriculum"
    return "local_training_source_candidate"


def load_public_fingerprints(*, limit: int) -> set[str]:
    fingerprints: set[str] = set()
    public_roots = [
        ROOT / "data" / "public_code_benchmark_manifests",
        ROOT / "data" / "old_project_benchmarks" / "cases",
        ROOT / "data" / "public_benchmarks" / "vcm_memory_quarantine",
    ]
    for root in public_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            for row in read_sample_rows(path, limit=limit):
                fingerprints.update(row_fingerprints(row))
                if len(fingerprints) >= limit:
                    return set(list(fingerprints)[:limit])
    for path in [ROOT / "data" / "public_blimp_train.jsonl", ROOT / "data" / "public_blimp_eval.jsonl"]:
        if path.exists():
            for row in read_sample_rows(path, limit=limit):
                fingerprints.update(row_fingerprints(row))
                if len(fingerprints) >= limit:
                    return set(list(fingerprints)[:limit])
    return fingerprints


def read_sample_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            rows = []
            for key in ["rows", "items", "sources", "capsules", "repos"]:
                value = data.get(key)
                if isinstance(value, list):
                    rows.extend(item for item in value if isinstance(item, dict))
            return rows[:limit] if rows else [data]
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"_raw_text": line[:2000]}
            if isinstance(value, dict):
                rows.append(value)
    return rows


def row_fingerprints(row: dict[str, Any]) -> set[str]:
    flat = flatten_dict(row)
    out = set()
    for key, value in flat.items():
        short_key = key.split(".")[-1]
        if short_key not in TEXT_FINGERPRINT_KEYS:
            continue
        text = normalize_text(value)
        if len(text) < 8:
            continue
        out.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return out


def extract_family_tags(row: dict[str, Any], rel_path: str) -> set[str]:
    flat = flatten_dict(row)
    tags: set[str] = set()
    path_tokens = tokenize(rel_path)
    tags.update(path_tokens)
    for key in [
        "category",
        "residual_concept",
        "concept_residual_label",
        "broad_private_family_v1",
        "targeted_private_residual_family_v3",
        "assistant_lane",
        "source_id",
        "card_id",
    ]:
        value = flat.get(key)
        if value:
            tags.update(tokenize(str(value)))
    raw_tags = flat.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            tags.update(tokenize(str(tag)))
    return {tag for tag in tags if len(tag) > 2}


def rows_claim_public_safe_private(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    checked = 0
    for row in rows[:32]:
        flat = flatten_dict(row)
        checked += 1
        public_false = all(not truthy(flat.get(key)) for key in PUBLIC_FLAG_KEYS)
        provenance_text = json.dumps(flat, sort_keys=True).lower()
        if not public_false:
            return False
        if "public_safe" not in provenance_text and "private" not in provenance_text:
            return False
    return checked > 0


def license_decision(counts: Counter[str], *, private_train_path: bool, dogfood_path: bool) -> str:
    if not counts:
        if private_train_path or dogfood_path:
            return "project_internal_private_generated"
        return "blocked_missing_license"
    blocked = [lic for lic in counts if lic and lic not in ALLOWED_LICENSES]
    if blocked:
        return "blocked_license_" + "_".join(sorted(blocked)[:3])
    return "allowed_" + "_".join(sorted(counts)[:3])


def provenance_decision(rows: list[dict[str, Any]], rel_path: str) -> str:
    if not rows:
        return "metadata_only_or_empty"
    good = 0
    blocked = 0
    for row in rows[:64]:
        flat = flatten_dict(row)
        if any(truthy(flat.get(key)) for key in PUBLIC_FLAG_KEYS):
            blocked += 1
            continue
        provenance = flat.get("provenance")
        if isinstance(provenance, dict) or flat.get("source_id") or flat.get("policy"):
            good += 1
    if blocked:
        return "blocked_public_flags"
    if good:
        return "provenance_tags_present"
    if "/private_train/" in f"/{rel_path}" or "/private_code_curriculum/" in f"/{rel_path}":
        return "project_internal_path_provenance"
    return "metadata_only"


def is_public_benchmark_path(path: Path) -> bool:
    rel_path = rel(path).lower()
    return any(
        marker in rel_path
        for marker in [
            "data/public_code_benchmark_manifests/",
            "data/public_benchmarks/",
            "data/old_project_benchmarks/",
            "data/public_blimp_",
        ]
    )


def path_has_public_benchmark_token(path: Path) -> bool:
    tokens = tokenize(rel(path))
    return any(token in PUBLIC_BENCHMARK_TOKENS for token in tokens)


def contains_fallback_return(row: dict[str, Any]) -> bool:
    text = json.dumps(row, sort_keys=True).lower()
    bad_markers = [
        "fallback_return",
        "expression_memory_fallback",
        "placeholder_scaffold_body",
        "return none  # fallback",
        "return 0  # fallback",
    ]
    return any(marker in text for marker in bad_markers)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".json":
        data = read_json(path)
        for key in ["rows", "items", "sources", "capsules", "repos"]:
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, list):
                return len(value)
        return 1 if data else 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_dict(child, child_key))
            out[str(key)] = child
    else:
        out[prefix] = value
    return out


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return row[key]
    return None


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def normalize_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return " ".join(text.strip().lower().split())


def tokenize(value: str) -> set[str]:
    text = value.lower()
    for ch in "/\\._:-0123456789":
        text = text.replace(ch, " ")
    return {part for part in text.split() if part}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def overlap_evidence(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        row["source_id"]: int(row["benchmark_overlap_check"]["exact_overlap_count"])
        for row in rows
        if int(row["benchmark_overlap_check"]["exact_overlap_count"]) > 0
    }


def teacher_row_evidence(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        row["source_id"]: int(row["teacher_row_count"])
        for row in rows
        if int(row["teacher_row_count"]) > 0
    }


def opt_out_notes(path: str, training_use: str) -> str:
    if training_use == "allowed":
        return f"Remove or quarantine {path}, then rerun scripts/training_data_admission_v1.py before the next sampler."
    if training_use == "heldout_eval_only":
        return f"{path} is heldout-only; do not point a trainer at it."
    return f"{path} is not train-admitted; keep as quarantine/calibration metadata unless a future admission report changes state."


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def vcm_context_governor_receipt(path: Path) -> dict[str, Any]:
    packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="training_data_admission_v1",
        purpose="training_admission",
        read_set=[rel(path), "data/training_data", "data/training_sources"],
        write_set=["reports/training_data_admission_v1.json", "data/training_sources/training_data_admission_v1.json"],
        authority_ceiling=["local_training_metadata_read", "governed_context_read"],
        permitted_uses=["training_source_metadata_admission", "lineage_accounting", "audit_replay"],
        governor_path=path,
        taint_labels=["training_metadata", "raw_text_not_staged"],
        deletion_obligations=["exclude_raw_user_text", "exclude_public_benchmark_payloads", "propagate_source_revocation"],
        audit_refs=["scripts/training_data_admission_v1.py"],
    )
    governor = packet["governor_receipt"]
    summary = governor.get("summary") if isinstance(governor.get("summary"), dict) else {}
    receipt_payload = {
        **governor,
        "path": rel(path),
        "hard_gap_count": int_or(summary.get("hard_gap_count")),
        "context_resolver_status": summary.get("context_resolver_status"),
        "context_resolver_passed_count": int_or(summary.get("context_resolver_passed_count")),
        "context_resolver_request_count": int_or(summary.get("context_resolver_request_count")),
        "context_resolver_materialized_count": int_or(summary.get("context_resolver_materialized_count")),
        "context_resolver_typed_fault_count": int_or(summary.get("context_resolver_typed_fault_count")),
        "context_resolver_viea_record_count": int_or(summary.get("context_resolver_viea_record_count")),
    }
    return {
        **receipt_payload,
        "record_type": "vcm_context_governor_receipt",
        "ready": bool(packet.get("ready")),
        "consumer_abi": packet,
        "required_for": "training_data_admission",
        "required_escalation": "refresh_vcm_context_governor_before_training_admission" if not packet.get("ready") else "none",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "This receipt proves context-governor readiness for admission metadata only; it is not training, public calibration, or learned-generation evidence.",
    }


def training_data_vcm_records(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    ready = bool(receipt.get("ready"))
    support_state = "SUPPORTED" if ready else "BLOCKED"
    receipt_id = str(receipt.get("receipt_id") or "missing_vcm_receipt")
    run_id = stable_id("training_data_admission_context:" + receipt_id)
    common = {
        "task_kind": "training_data_admission",
        "target": "training_data_admission_v1",
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    abi_packet = receipt.get("consumer_abi") if isinstance(receipt.get("consumer_abi"), dict) else {}
    abi_records = abi_packet.get("records") if isinstance(abi_packet.get("records"), list) else []
    return list(abi_records) + [
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("training_data_admission_authority:" + run_id),
            "authority_scope": "local_training_metadata_read,local_context_governor_read",
            "allowed_effects": ["scan_local_training_metadata", "emit_admission_manifest", "emit_context_receipt"],
            "denied_effects": ["public_benchmark_training", "raw_user_text_training", "runtime_external_inference", "fallback_return_admission"],
        },
        {
            **common,
            "record_type": "context_transaction",
            "record_id": stable_id("training_data_admission_context_transaction:" + run_id),
            "transaction_id": stable_id("training_data_admission_context_transaction:" + run_id),
            "operation": "training_source_admission_context_check",
            "snapshot_id": receipt.get("created_utc") or "",
            "mounts": ["vcm_context_governor", "local_training_source_metadata"],
            "read_set": [receipt.get("path") or "reports/vcm_context_governor.json", "data/training_data", "data/training_sources"],
            "write_set": ["reports/training_data_admission_v1.json", "data/training_sources/training_data_admission_v1.json"],
            "branch_policy": "fail_closed_if_context_governor_not_ready",
            "taint_labels": ["training_metadata", "public_benchmark_quarantine_checked", "raw_text_not_staged"],
            "deletion_obligations": ["exclude_raw_user_text", "exclude_public_benchmark_payloads"],
            "declassification_refs": [],
            "derivative_refs": [receipt_id],
            "contradiction_refs": [],
            "materialization_state": "materialized" if ready else "blocked",
            "closure_state": "closed" if ready else "typed_fault",
            "faults": [] if ready else ["vcm_context_governor_not_ready"],
            "audit_refs": [receipt.get("path") or "reports/vcm_context_governor.json", "reports/training_data_admission_v1.json"],
            "replay_boundary": "metadata_hashes_only_no_payload_training",
            "non_claims": [
                "training data admission scans metadata and hashes, not public benchmark payload training rows",
                "VCM context readiness is not a model capability claim",
            ],
            "evidence_ref": "reports/training_data_admission_v1.json",
            "content_hash": stable_hash(receipt),
        },
        {
            **common,
            "record_type": "context_adequacy",
            "record_id": stable_id("training_data_admission_context_adequacy:" + run_id),
            "adequacy_id": stable_id("training_data_admission_context_adequacy:" + run_id),
            "state": "governed_sufficient_for_training_admission" if ready else "fault_missing_or_ungoverned_context",
            "adequacy_state": "governed_sufficient_for_training_admission" if ready else "fault_missing_or_ungoverned_context",
            "context_transaction_id": stable_id("training_data_admission_context_transaction:" + run_id),
            "evidence_ref": "reports/training_data_admission_v1.json",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("training_data_admission_failure_boundary:" + run_id),
            "failure_id": stable_id("training_data_admission_vcm_fault:" + run_id),
            "blocked_reason": "none" if ready else "vcm_context_governor_not_ready",
            "terminal": ready,
            "structured_non_solved": not ready,
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": stable_id("training_data_admission_artifact:" + run_id),
            "artifact_id": stable_id("training_data_admission_artifact:" + run_id),
            "artifact_ref": "reports/training_data_admission_v1.json",
            "evidence_ref": receipt.get("path") or "reports/vcm_context_governor.json",
            "content_hash": stable_hash(receipt),
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": stable_id("training_data_admission_claim:" + run_id),
            "claim_id": stable_id("training_data_admission_vcm_context_ready:" + run_id),
            "evidence_ref": "reports/training_data_admission_v1.json",
            "learned_generation_claim_allowed": False,
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("training_data_admission_evidence:" + run_id),
            "previous_support_state": "UNREVIEWED",
            "current_support_state": support_state,
            "evidence_ref": "reports/training_data_admission_v1.json",
        },
    ]


def manifest_payload(report: dict[str, Any]) -> dict[str, Any]:
    full_receipt = report.get("vcm_context_governor_receipt")
    full_receipt = dict(full_receipt) if isinstance(full_receipt, dict) else {}
    receipt_fields = (
        "record_type", "receipt_id", "consumer_id", "path", "content_hash", "created_utc",
        "trigger_state", "ready", "hard_gap_count", "context_resolver_status",
        "context_resolver_passed_count", "context_resolver_request_count",
        "context_resolver_materialized_count", "context_resolver_typed_fault_count",
        "context_resolver_viea_record_count", "required_for", "required_escalation",
        "public_training_rows_written", "external_inference_calls", "fallback_return_count",
        "raw_prompt_stored", "raw_private_text_stored", "non_claim",
    )
    receipt = {key: full_receipt.get(key) for key in receipt_fields if key in full_receipt}
    packet = full_receipt.get("consumer_abi")
    if isinstance(packet, dict):
        receipt["consumer_abi"] = vcm_consumer_abi.compact_consumer_packet(packet)
    lineage = report.get("candidate_lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    compact_lineage = {
        "policy": lineage.get("policy"),
        "trigger_state": lineage.get("trigger_state"),
        "summary": lineage.get("summary"),
        "candidate_receipt_ledger": lineage.get("candidate_receipt_ledger"),
        "non_claims": lineage.get("non_claims"),
    }
    return {
        "policy": "project_theseus_training_data_admission_manifest_v1",
        "created_utc": report.get("created_utc"),
        "source_report": rel(DEFAULT_OUT),
        "summary": report.get("summary"),
        "train_admitted_sources": [
            row
            for row in report.get("source_admissions", [])
            if isinstance(row, dict) and row.get("allowed_for_training")
        ],
        "open_code_public_corpus_candidates": report.get("open_code_public_corpus_candidates", []),
        "public_open_training_allowed": bool((report.get("summary") or {}).get("public_open_training_allowed")),
        "public_benchmark_training_allowed": False,
        "vcm_context_governor_receipt": receipt,
        "candidate_lineage": compact_lineage,
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Training Data Admission v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- local sources audited: `{summary.get('local_source_count')}`",
        f"- training sources admitted: `{summary.get('allowed_training_source_count')}`",
        f"- public/open training allowed: `{summary.get('public_open_training_allowed')}`",
        f"- admitted public/open sources: `{summary.get('admitted_open_public_source_count')}`",
        f"- admitted public/open rows: `{summary.get('admitted_open_public_row_count')}`",
        f"- heldout eval only: `{summary.get('heldout_eval_only_source_count')}`",
        f"- rejected: `{summary.get('rejected_source_count')}`",
        f"- quarantined: `{summary.get('quarantined_source_count')}`",
        f"- public benchmark payload admitted: `{summary.get('public_benchmark_payload_admitted')}`",
        f"- public fingerprints loaded: `{summary.get('public_fingerprint_count')}`",
        f"- VCM governor ready: `{summary.get('vcm_context_governor_ready')}` resolver `{summary.get('vcm_context_resolver_status')}` `{summary.get('vcm_context_resolver_passed_count')}/{summary.get('vcm_context_resolver_request_count')}`",
        f"- external inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Capability Tags",
    ]
    for tag, count in sorted((summary.get("capability_tag_counts") or {}).items(), key=lambda item: (-int(item[1]), item[0]))[:30]:
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Failed Gates"])
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    lines.extend(["", "## Hard Invariants"])
    for item in report.get("hard_invariants", []):
        lines.append(f"- {item}")
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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
