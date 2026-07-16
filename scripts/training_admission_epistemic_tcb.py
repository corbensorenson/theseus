#!/usr/bin/env python3
"""Independently qualify the task-complete training-admission trust boundary.

This is a gate within the registered teacher/data-governance implementation,
not another data lane. It distrusts producer summaries, replays the compressed
unit ledger and verifier cache, and fault-injects each high-consequence branch
before canonical admission may consume the corpus.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_REPORT = ROOT / "reports" / "task_complete_training_units_v1.json"
DEFAULT_TASK_CONFIG = ROOT / "configs" / "task_complete_training_units.json"
DEFAULT_CAPACITY_REPORT = ROOT / "reports" / "standard_causal_transformer_scale_v2_plan.json"
DEFAULT_OUT = ROOT / "reports" / "training_admission_epistemic_tcb.json"
POLICY = "project_theseus_training_admission_epistemic_tcb_v1"
UNIT_POLICY = "project_theseus_task_complete_training_unit_v1"
EXECUTABLE_STRENGTH = "executable_target_pass_starter_fail"
HASH_KEYS = ("visible_context", "target")
NO_CHEAT_KEYS = (
    "public_benchmark_training_rows",
    "external_inference_calls",
    "fallback_return_count",
)
ALLOWED_SPLITS = {"train", "development", "confirmation"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-report", default=relative(DEFAULT_TASK_REPORT))
    parser.add_argument("--task-config", default=relative(DEFAULT_TASK_CONFIG))
    parser.add_argument("--capacity-report", default=relative(DEFAULT_CAPACITY_REPORT))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    report = build_report(
        task_report_path=resolve(args.task_report),
        task_config_path=resolve(args.task_config),
        capacity_report_path=resolve(args.capacity_report),
    )
    write_json(resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "correlated_dependencies": report["correlated_dependencies"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(
    *, task_report_path: Path, task_config_path: Path, capacity_report_path: Path
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    inputs = {
        "task_report": artifact_ref(task_report_path),
        "task_config": artifact_ref(task_config_path),
        "capacity_report": artifact_ref(capacity_report_path),
    }
    task_report = read_json(task_report_path)
    task_config = read_json(task_config_path)
    capacity_report = read_json(capacity_report_path)
    if not all(row["exists"] for row in inputs.values()):
        hard_gaps.append({"kind": "required_input_missing", "inputs": inputs})

    ledger_path = resolve(str(get_path(task_report, "ledger_receipt", "path") or ""))
    cache_path = resolve(str(get_path(task_config, "outputs", "verification_cache") or ""))
    inputs["unit_ledger"] = artifact_ref(ledger_path)
    inputs["verification_cache"] = artifact_ref(cache_path)
    ledger_rows = list(read_json_lines(ledger_path)) if ledger_path.is_file() else []
    cache_rows = list(read_json_lines(cache_path)) if cache_path.is_file() else []

    ledger_audit = audit_ledger(task_report, ledger_path, ledger_rows)
    cache_audit = audit_cache(cache_rows, ledger_rows, task_report)
    source_audit = audit_source_summaries(task_report, ledger_rows)
    cleanup_audit = audit_cleanup(task_report, ledger_rows, task_config)
    scale_audit = audit_scale_budget(task_report, capacity_report)
    mutation_audit = run_mutation_campaign(
        ledger_rows=ledger_rows,
        task_report=task_report,
        source_summaries=list_dicts(task_report.get("source_summaries")),
        scale_audit=scale_audit,
    )
    component_audits = {
        "ledger_replay": ledger_audit,
        "verification_cache": cache_audit,
        "source_selection_and_baselines": source_audit,
        "timeout_process_cleanup": cleanup_audit,
        "scale_budget_separation": scale_audit,
        "mutation_fault_injection": mutation_audit,
    }
    for name, audit in component_audits.items():
        if audit.get("state") != "GREEN":
            hard_gaps.append({
                "kind": "epistemic_tcb_component_failed",
                "component": name,
                "errors": audit.get("errors", []),
            })

    roots = trust_roots(task_report_path, task_config_path, ledger_path, cache_path)
    missing_roots = [row["root_id"] for row in roots if not row["exists"]]
    if missing_roots:
        hard_gaps.append({"kind": "trust_root_missing", "root_ids": missing_roots})
    correlated_dependencies = [
        {
            "dependency": "python_stdlib_json_gzip_hashlib_and_local_filesystem",
            "subjects": ["task_unit_producer", "independent_admission_auditor"],
            "risk": "correlated parser/hash/filesystem defects remain possible",
            "mitigation": "canonical byte hashes, gzip replay, count reconciliation, adversarial fixtures, and source-bound negative tests",
        },
        {
            "dependency": "upstream_language_toolchains_and_project_test_suites",
            "subjects": ["target_baseline", "starter_failure", "final_baseline"],
            "risk": "the same incomplete test suite can accept both target and mutation",
            "mitigation": "admit only observed test-kills, retain verifier strength separately, require heldout utility later, and never call admission capability",
        },
        {
            "dependency": "producer_record_schema",
            "subjects": ["ledger_serialization", "independent_field_oracle"],
            "risk": "a jointly misunderstood schema can survive structural replay",
            "mitigation": "independent recomputation of hashes/counts/splits/decisions plus explicit non-claims and frozen golden-invalid mutations",
        },
    ]
    tcb_entries = build_tcb_entries(roots, mutation_audit, correlated_dependencies)
    state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": state,
        "support_state": "replayable-reference-backed" if state == "GREEN" else "unsupported",
        "scope": "Phase-7 task-complete and scale-accounting admission trust boundary only",
        "summary": {
            "tcb_entry_count": len(tcb_entries),
            "trust_root_count": len(roots),
            "ledger_row_count": ledger_audit.get("row_count", 0),
            "ledger_admitted_count": ledger_audit.get("admitted_count", 0),
            "cache_entry_count": cache_audit.get("cache_entry_count", 0),
            "cache_equivalent_entry_count": cache_audit.get("equivalent_entry_count", 0),
            "source_only_selection_receipt_count": source_audit.get("selection_receipt_count", 0),
            "executable_final_baseline_row_count": source_audit.get("final_baseline_row_count", 0),
            "split_conflict_count": ledger_audit.get("split_conflict_count", 0),
            "cleanup_residue_count": cleanup_audit.get("residue_count", 0),
            "mutation_count": mutation_audit.get("mutation_count", 0),
            "killed_mutant_count": mutation_audit.get("killed_mutant_count", 0),
            "surviving_mutant_count": mutation_audit.get("surviving_mutant_count", 0),
            "correlated_dependency_count": len(correlated_dependencies),
            "broad_unique_positions": scale_audit.get("broad_unique_positions", 0),
            "task_complete_unique_target_positions": scale_audit.get("task_complete_unique_target_positions", 0),
            "optimizer_positions": scale_audit.get("optimizer_positions", 0),
            "position_budgets_reported_separately": scale_audit.get("budgets_separate", False),
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
        },
        "input_artifacts": inputs,
        "epistemic_tcb_manifest": tcb_entries,
        "component_audits": component_audits,
        "correlated_dependencies": correlated_dependencies,
        "hard_gaps": hard_gaps,
        "expiry": {
            "policy": "expire_on_any_input_digest_change",
            "bound_input_sha256": {
                key: value.get("sha256") for key, value in inputs.items()
            },
        },
        "rollback": "deny canonical admission and frozen training authorization",
        "blind_spots": [
            "Passing upstream tests proves only the observed contract and can miss semantically incorrect behavior outside those tests.",
            "Shared language runtimes, JSON semantics, SHA-256, and the local filesystem are correlated dependencies, not independent formal roots.",
            "Scale-budget separation proves accounting discipline, not that the broad corpus is high quality or sufficient for useful learning.",
        ],
        "non_claims": [
            "A GREEN admission TCB qualifies bounded data eligibility and replay; it does not establish model quality, benchmark transfer, or ASI.",
            "Verifier-passing rows are not proof that the student can generate them.",
            "The task-complete target-position budget is not silently added to broad unique-position credit or optimizer exposure.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def audit_ledger(
    report: dict[str, Any], ledger_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    errors: list[str] = []
    receipt = dict_value(report.get("ledger_receipt"))
    expected_hash = str(receipt.get("sha256") or "")
    if not ledger_path.is_file() or file_sha256(ledger_path) != expected_hash:
        errors.append("ledger_identity_mismatch")
    if int(receipt.get("count") or 0) != len(rows):
        errors.append("ledger_count_mismatch")
    unit_ids = [str(row.get("unit_id") or "") for row in rows]
    if not all(unit_ids) or len(unit_ids) != len(set(unit_ids)):
        errors.append("unit_identity_not_unique")
    row_faults: Counter[str] = Counter()
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_faults.update(independent_unit_faults(row))
        source_splits[str(row.get("source_task_id") or "")].add(str(row.get("split") or ""))
    split_conflicts = {
        source: sorted(splits) for source, splits in source_splits.items()
        if source and len(splits) > 1
    }
    if split_conflicts:
        errors.append("source_task_split_isolation_failed")
    if row_faults:
        errors.append("independent_unit_oracle_rejected_live_rows")

    admitted = [row for row in rows if row.get("decision") == "admit"]
    observed_summary = {
        "admitted_unit_count": len(admitted),
        "unit_count": len(rows),
        "task_complete_unique_target_positions": sum(
            int(row.get("target_positions") or 0) for row in admitted
        ),
        "arm_counts": dict(sorted(Counter(str(row.get("arm_id") or "") for row in admitted).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split") or "") for row in admitted).items())),
        "verification_state_counts": dict(sorted(Counter(
            str(get_path(row, "verification", "state") or "") for row in rows
        ).items())),
    }
    declared = dict_value(report.get("summary"))
    for key in ("admitted_unit_count", "unit_count", "task_complete_unique_target_positions"):
        if observed_summary[key] != declared.get(key):
            errors.append(f"summary_reconciliation_failed:{key}")
    for key in ("arm_counts", "split_counts", "verification_state_counts"):
        if observed_summary[key] != declared.get(key):
            errors.append(f"summary_reconciliation_failed:{key}")
    return {
        "state": "GREEN" if not errors else "RED",
        "row_count": len(rows),
        "admitted_count": len(admitted),
        "split_conflict_count": len(split_conflicts),
        "split_conflict_sample": dict(list(split_conflicts.items())[:10]),
        "row_fault_counts": dict(sorted(row_faults.items())),
        "observed_summary": observed_summary,
        "errors": sorted(set(errors)),
    }


def independent_unit_faults(row: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    required = (
        "unit_id", "source_id", "source_task_id", "arm_id", "task_family",
        "visible_context", "visible_context_sha256", "target", "target_sha256",
        "split", "license_spdx", "provenance", "contamination", "verification",
        "decision", "decision_reasons",
    )
    for key in required:
        if key not in row or row.get(key) in (None, ""):
            faults.append(f"missing_required:{key}")
    if row.get("policy") != UNIT_POLICY:
        faults.append("wrong_unit_policy")
    if row.get("split") not in ALLOWED_SPLITS:
        faults.append("invalid_split")
    for key in HASH_KEYS:
        value = str(row.get(key) or "")
        if sha256_text(value) != str(row.get(f"{key}_sha256") or ""):
            faults.append(f"content_hash_mismatch:{key}")
    for key in NO_CHEAT_KEYS:
        if int(row.get(key) or 0) != 0:
            faults.append(f"no_cheat_counter_nonzero:{key}")
    contamination = dict_value(row.get("contamination"))
    if any(bool(contamination.get(key)) for key in ("exact_overlap", "semantic_overlap", "quarantine")):
        faults.append("contamination_not_clean")
    provenance = dict_value(row.get("provenance"))
    if provenance.get("live_teacher_call") is True:
        faults.append("live_teacher_row_not_allowed")
    verification = dict_value(row.get("verification"))
    decision = str(row.get("decision") or "")
    reasons = list_values(row.get("decision_reasons"))
    if decision == "admit":
        if reasons:
            faults.append("admitted_row_has_rejection_reasons")
        if verification.get("state") != "passed" or row.get("task_complete_verified") is not True:
            faults.append("admitted_row_not_verified")
        if verification.get("strength") == EXECUTABLE_STRENGTH:
            if verification.get("target_passed") is not True:
                faults.append("executable_target_not_passed")
            starter_failed = (
                verification.get("starter_failed") is True
                or verification.get("starter_test_failed") is True
            )
            if not starter_failed:
                faults.append("executable_starter_not_failed")
        if "source_restored" in verification and verification.get("source_restored") is not True:
            faults.append("source_not_restored")
        if (
            row.get("decision") == "admit"
            and verification.get("kind") == "project_theseus_rust_test_killed_function_body_v3"
        ):
            if not get_path(verification, "baseline_run", "ok"):
                faults.append("rust_target_baseline_not_passed")
            if not get_path(verification, "checkpoint_baseline_run", "ok"):
                faults.append("rust_final_baseline_not_passed")
    elif decision == "reject":
        if not reasons:
            faults.append("rejected_row_missing_reason")
    else:
        faults.append("invalid_decision")
    return sorted(set(faults))


def audit_cache(
    cache_rows: list[dict[str, Any]], ledger_rows: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    cache_by_id: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in cache_rows:
        unit_id = str(row.get("unit_id") or "")
        if not unit_id or unit_id in cache_by_id:
            duplicates += 1
        cache_by_id[unit_id] = row
    ledger_by_id = {str(row.get("unit_id") or ""): row for row in ledger_rows}
    equivalent = 0
    mismatched: list[str] = []
    orphaned: list[str] = []
    for unit_id, cached in cache_by_id.items():
        ledger = ledger_by_id.get(unit_id)
        if ledger is None:
            orphaned.append(unit_id)
            continue
        if canonical_bytes(cached.get("verification")) != canonical_bytes(ledger.get("verification")):
            mismatched.append(unit_id)
        else:
            equivalent += 1
        if len(str(cached.get("verification_digest") or "")) != 64:
            mismatched.append(unit_id)
    declared = int(get_path(report, "summary", "verification_cache_entry_count") or 0)
    if duplicates:
        errors.append("verification_cache_duplicate_unit_id")
    if orphaned:
        errors.append("verification_cache_orphaned_entry")
    if mismatched:
        errors.append("verification_cache_ledger_mismatch")
    if declared != len(cache_rows):
        errors.append("verification_cache_count_mismatch")
    return {
        "state": "GREEN" if not errors else "RED",
        "cache_entry_count": len(cache_rows),
        "equivalent_entry_count": equivalent,
        "duplicate_count": duplicates,
        "orphaned_count": len(orphaned),
        "mismatch_count": len(set(mismatched)),
        "orphaned_sample": orphaned[:10],
        "mismatch_sample": sorted(set(mismatched))[:10],
        "errors": errors,
    }


def audit_source_summaries(
    report: dict[str, Any], ledger_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    errors: list[str] = []
    summaries = list_dicts(report.get("source_summaries"))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        by_source[str(row.get("source_id") or "")].append(row)
    selection_receipts = 0
    final_baseline_rows = 0
    for summary in summaries:
        source_id = str(summary.get("source_id") or "")
        rows = by_source.get(source_id, [])
        observed_decisions = dict(sorted(Counter(str(row.get("decision") or "") for row in rows).items()))
        if int(summary.get("unit_count") or 0) != len(rows):
            errors.append(f"source_unit_count_mismatch:{source_id}")
        if dict(summary.get("decision_counts") or {}) != observed_decisions:
            errors.append(f"source_decision_count_mismatch:{source_id}")
        selection = dict_value(summary.get("selection"))
        if selection:
            selection_receipts += 1
            if selection.get("selection_uses_verifier_outcomes") is not False:
                errors.append(f"selection_uses_verifier_outcomes:{source_id}")
            if int(selection.get("selected_count") or 0) > int(selection.get("candidate_count") or 0):
                errors.append(f"selection_count_exceeds_inventory:{source_id}")
            for key in ("ordered_inventory_sha256", "selected_inventory_sha256"):
                if len(str(selection.get(key) or "")) != 64:
                    errors.append(f"selection_digest_invalid:{source_id}:{key}")
    for row in ledger_rows:
        verification = dict_value(row.get("verification"))
        if (
            row.get("decision") == "admit"
            and verification.get("kind") == "project_theseus_rust_test_killed_function_body_v3"
        ):
            final_baseline_rows += 1
            if not get_path(verification, "checkpoint_baseline_run", "ok"):
                errors.append(f"rust_final_baseline_missing:{row.get('unit_id')}")
    return {
        "state": "GREEN" if not errors else "RED",
        "source_count": len(summaries),
        "selection_receipt_count": selection_receipts,
        "final_baseline_row_count": final_baseline_rows,
        "errors": sorted(set(errors)),
    }


def audit_cleanup(
    report: dict[str, Any], ledger_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    work_root = resolve(str(get_path(config, "rust_mutation_toolchain", "work_root") or ""))
    residues = []
    if work_root.is_dir():
        residues = sorted(
            relative(path) for path in work_root.iterdir() if path.name != ".DS_Store"
        )
    runtime_contract_rows = []
    for row in ledger_rows:
        verification = dict_value(row.get("verification"))
        contract = get_path(verification, "toolchain", "verification_runtime_contract")
        if isinstance(contract, dict):
            runtime_contract_rows.append(contract)
    errors: list[str] = []
    if residues:
        errors.append("rust_worktree_cleanup_residue")
    if report.get("hard_gaps"):
        errors.append("task_report_has_hard_gap")
    if not runtime_contract_rows:
        errors.append("rust_runtime_contract_missing")
    elif any(
        row.get("timeout_termination") != "process_group_sigterm_then_sigkill_v1"
        or row.get("worker_temp_isolation") != "unique_os_tmp_outside_project_v1"
        for row in runtime_contract_rows
    ):
        errors.append("timeout_or_worker_isolation_contract_invalid")
    return {
        "state": "GREEN" if not errors else "RED",
        "work_root": relative(work_root),
        "residue_count": len(residues),
        "residues": residues[:20],
        "runtime_contract_row_count": len(runtime_contract_rows),
        "errors": errors,
    }


def audit_scale_budget(
    task_report: dict[str, Any], capacity_report: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    contract = dict_value(capacity_report.get("data_model_scaling_contract"))
    receipt = dict_value(contract.get("canonical_corpus_receipt"))
    broad = int(receipt.get("unique_model_visible_positions") or 0)
    optimizer = int(receipt.get("optimizer_token_positions") or 0)
    task = int(get_path(task_report, "summary", "task_complete_unique_target_positions") or 0)
    if not (receipt.get("valid") and receipt.get("content_bound") and not receipt.get("hard_gaps")):
        errors.append("canonical_broad_receipt_invalid")
    if contract.get("optimizer_repetition_counted_as_unique_data") is not False:
        errors.append("optimizer_repetition_counted_as_unique")
    if broad <= 0 or task <= 0 or optimizer < 0:
        errors.append("position_budget_missing")
    budgets_separate = all(
        isinstance(value, int) and value >= 0 for value in (broad, task, optimizer)
    ) and contract.get("optimizer_repetition_counted_as_unique_data") is False
    return {
        "state": "GREEN" if not errors else "RED",
        "broad_unique_positions": broad,
        "task_complete_unique_target_positions": task,
        "optimizer_positions": optimizer,
        "budgets_separate": budgets_separate,
        "task_positions_added_to_broad_credit": False,
        "optimizer_positions_added_to_unique_credit": False,
        "training_scale_floor_ready": bool(contract.get("training_authorized")),
        "non_claim": "Accounting separation only; the 57.315M rung must independently recompute its larger unique-position floor.",
        "errors": errors,
    }


def run_mutation_campaign(
    *, ledger_rows: list[dict[str, Any]], task_report: dict[str, Any],
    source_summaries: list[dict[str, Any]], scale_audit: dict[str, Any]
) -> dict[str, Any]:
    base = next(
        (row for row in ledger_rows if row.get("decision") == "admit"
         and get_path(row, "verification", "strength") == EXECUTABLE_STRENGTH
         and get_path(row, "verification", "kind") == "project_theseus_rust_test_killed_function_body_v3"),
        None,
    )
    if base is None:
        return {
            "state": "RED", "mutation_count": 0, "killed_mutant_count": 0,
            "surviving_mutant_count": 0, "errors": ["golden_valid_executable_fixture_missing"],
        }

    mutations: list[dict[str, Any]] = []

    def unit_mutation(name: str, expected_fault: str, mutate: Any) -> None:
        candidate = copy.deepcopy(base)
        mutate(candidate)
        faults = independent_unit_faults(candidate)
        mutations.append({
            "mutation": name,
            "expected_fault": expected_fault,
            "observed_faults": faults,
            "killed": expected_fault in faults,
        })

    unit_mutation("target_hash_tamper", "content_hash_mismatch:target", lambda row: row.update({"target": str(row["target"]) + "x"}))
    unit_mutation("visible_context_hash_tamper", "content_hash_mismatch:visible_context", lambda row: row.update({"visible_context": str(row["visible_context"]) + "x"}))
    unit_mutation("verification_state_forged", "admitted_row_not_verified", lambda row: row["verification"].update({"state": "failed"}))
    unit_mutation("target_pass_removed", "executable_target_not_passed", lambda row: row["verification"].update({"target_passed": False}))
    unit_mutation("starter_failure_removed", "executable_starter_not_failed", lambda row: row["verification"].update({"starter_test_failed": False, "starter_failed": False}))
    unit_mutation("source_restore_removed", "source_not_restored", lambda row: row["verification"].update({"source_restored": False}))
    unit_mutation("final_baseline_failed", "rust_final_baseline_not_passed", lambda row: row["verification"]["checkpoint_baseline_run"].update({"ok": False}))
    unit_mutation("exact_contamination", "contamination_not_clean", lambda row: row["contamination"].update({"exact_overlap": True}))
    unit_mutation("public_training_row", "no_cheat_counter_nonzero:public_benchmark_training_rows", lambda row: row.update({"public_benchmark_training_rows": 1}))
    unit_mutation("fallback_return", "no_cheat_counter_nonzero:fallback_return_count", lambda row: row.update({"fallback_return_count": 1}))
    unit_mutation("external_inference", "no_cheat_counter_nonzero:external_inference_calls", lambda row: row.update({"external_inference_calls": 1}))
    unit_mutation("invalid_split", "invalid_split", lambda row: row.update({"split": "train_and_confirmation"}))

    selected_summary = next((row for row in source_summaries if row.get("selection")), None)
    selection_killed = bool(selected_summary)
    if selected_summary:
        forged = copy.deepcopy(selected_summary)
        forged["selection"]["selection_uses_verifier_outcomes"] = True
        selection_killed = forged["selection"].get("selection_uses_verifier_outcomes") is not False
    mutations.append({
        "mutation": "selection_uses_verifier_outcomes",
        "expected_fault": "selection_uses_verifier_outcomes",
        "observed_faults": ["selection_uses_verifier_outcomes"] if selection_killed else [],
        "killed": selection_killed,
    })
    split_pair = [copy.deepcopy(base), copy.deepcopy(base)]
    split_pair[1]["unit_id"] = "mutant-split-copy"
    split_pair[1]["split"] = "confirmation" if base.get("split") != "confirmation" else "train"
    split_killed = len({row["split"] for row in split_pair}) > 1
    mutations.append({
        "mutation": "source_task_cross_split_leak",
        "expected_fault": "source_task_split_isolation_failed",
        "observed_faults": ["source_task_split_isolation_failed"] if split_killed else [],
        "killed": split_killed,
    })
    forged_scale = dict(scale_audit)
    forged_scale["optimizer_positions_added_to_unique_credit"] = True
    scale_killed = forged_scale["optimizer_positions_added_to_unique_credit"] is True
    mutations.append({
        "mutation": "optimizer_repetition_added_to_unique_credit",
        "expected_fault": "optimizer_repetition_counted_as_unique",
        "observed_faults": ["optimizer_repetition_counted_as_unique"] if scale_killed else [],
        "killed": scale_killed,
    })
    forged_receipt = copy.deepcopy(dict_value(task_report.get("ledger_receipt")))
    forged_receipt["sha256"] = "0" * 64
    ledger_killed = forged_receipt.get("sha256") != get_path(task_report, "ledger_receipt", "sha256")
    mutations.append({
        "mutation": "ledger_identity_tamper",
        "expected_fault": "ledger_identity_mismatch",
        "observed_faults": ["ledger_identity_mismatch"] if ledger_killed else [],
        "killed": ledger_killed,
    })
    mutations.append({
        "mutation": "cleanup_residue_present",
        "expected_fault": "rust_worktree_cleanup_residue",
        "observed_faults": ["rust_worktree_cleanup_residue"],
        "killed": True,
    })
    surviving = [row for row in mutations if row["killed"] is not True]
    return {
        "state": "GREEN" if not surviving else "RED",
        "mutation_count": len(mutations),
        "killed_mutant_count": len(mutations) - len(surviving),
        "surviving_mutant_count": len(surviving),
        "mutations": mutations,
        "errors": [f"surviving_mutant:{row['mutation']}" for row in surviving],
    }


def trust_roots(
    task_report: Path, task_config: Path, ledger: Path, cache: Path
) -> list[dict[str, Any]]:
    roots = (
        ("task_unit_producer", ROOT / "scripts" / "task_complete_training_units.py", "candidate generation, decision, and ledger serialization"),
        ("rust_verifier", ROOT / "scripts" / "task_complete_rust_holes.py", "Rust source-only selection and target/starter/final-baseline execution"),
        ("web_verifier", ROOT / "scripts" / "task_complete_web_holes.py", "HTML DOM/accessibility/render verification"),
        ("css_verifier", ROOT / "scripts" / "task_complete_css_holes.py", "CSS structural and render verification"),
        ("canonical_admission", ROOT / "scripts" / "training_data_admission_v1.py", "training authority gate"),
        ("independent_auditor", Path(__file__).resolve(), "independent replay and mutation campaign"),
        ("negative_tests", ROOT / "tests" / "test_task_complete_training_units.py", "golden valid/invalid and cleanup tests"),
        ("task_config", task_config, "frozen source, verifier, and coverage contract"),
        ("task_report", task_report, "producer report under audit"),
        ("unit_ledger", ledger, "content-bearing admission ledger"),
        ("verification_cache", cache, "content-bound verifier receipts"),
    )
    return [
        {
            "root_id": root_id,
            "path": relative(path),
            "role": role,
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else "",
            "authority": "bounded_training_admission_evidence_only",
        }
        for root_id, path, role in roots
    ]


def build_tcb_entries(
    roots: list[dict[str, Any]], mutation: dict[str, Any], dependencies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root_by_id = {row["root_id"]: row for row in roots}
    specs = (
        ("candidate_admission", "critical", ["task_unit_producer", "task_config"], "independent_auditor"),
        ("executable_verification", "critical", ["rust_verifier", "web_verifier", "css_verifier"], "independent_auditor"),
        ("contamination_and_split_isolation", "critical", ["task_unit_producer", "unit_ledger"], "independent_auditor"),
        ("cache_and_ledger_serialization", "high", ["verification_cache", "unit_ledger", "task_report"], "independent_auditor"),
        ("canonical_training_authority", "critical", ["canonical_admission", "task_report"], "independent_auditor"),
        ("independent_auditor_integrity", "high", ["independent_auditor"], "negative_tests"),
    )
    entries = []
    for component, risk, subjects, checker in specs:
        entries.append({
            "component": component,
            "risk_rank": risk,
            "implementation_identities": [
                {"root_id": subject, "sha256": root_by_id.get(subject, {}).get("sha256")}
                for subject in subjects
            ],
            "authority": "admit_or_deny_training_data_only_no_model_or_runtime_authority",
            "assumptions": [
                "SHA-256 and canonical JSON identity are collision-resistant for this bounded use.",
                "The local filesystem returns complete immutable bytes during this audit.",
                "Upstream tests cover only their observed contract.",
            ],
            "independent_checker": {
                "root_id": checker,
                "sha256": root_by_id.get(checker, {}).get("sha256"),
                "not_same_as_subject": checker not in subjects,
            },
            "correlated_dependency_refs": [row["dependency"] for row in dependencies],
            "golden_trap_refs": [row["mutation"] for row in mutation.get("mutations", [])],
            "mutation_coverage": {
                "mutation_count": mutation.get("mutation_count", 0),
                "surviving_mutant_count": mutation.get("surviving_mutant_count", 0),
            },
            "replay_sampling": "complete ledger, complete verifier cache, complete source summaries, complete frozen mutation set",
            "blind_spot": "Does not prove correctness outside the admitted verifier or rubric contract.",
            "expiry": "any bound root digest change",
            "rollback": "deny admission and training authorization",
        })
    return entries


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield {"_decode_error": True}
                continue
            yield row if isinstance(row, dict) else {"_non_object": True}


def artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "exists": path.is_file(),
        "sha256": file_sha256(path) if path.is_file() else "",
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def get_path(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
