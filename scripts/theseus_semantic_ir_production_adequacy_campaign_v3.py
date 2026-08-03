#!/usr/bin/env python3
"""Resume the sealed adequacy panel from preserved candidates 1 through 3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_assistant_route_integrity_v2 as route_integrity_v2
import theseus_p4_cognitive_compilation as p4
import theseus_p4_cognitive_compilation_repaired as p4r
import theseus_semantic_ir_production_adequacy_campaign as base_campaign
import theseus_semantic_ir_production_adequacy_campaign_v2 as campaign_v2
import theseus_semantic_ir_production_adequacy_runtime as production


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign_v3.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v3.json"
DEFAULT_JOURNAL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_v3_journal.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_candidates_v3"
CONFIG_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_v3"
AUDIT_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_audit_v3"
MODEL_CONTEXT_TOKENS = 262_144
HOST_WATCHDOG_SECONDS = 600
PRESERVED_CANDIDATE_COUNT = 3
PRESERVED_MODEL_CALLS = 6
NEW_MODEL_CALLS = 30
InfrastructureBoundary = campaign_v2.InfrastructureBoundary
persistent_adequacy_session = campaign_v2.persistent_adequacy_session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--journal", default=p2a.rel(DEFAULT_JOURNAL))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = (
        audit_config(config_path)
        if args.audit_only
        else run_campaign(config_path, p2a.resolve(args.out), p2a.resolve(args.journal))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_config(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "PROSPECTIVELY_BOUND_AFTER_V3_POOL_SEAL_BEFORE_RESUME":
        faults.append("config_not_prospectively_bound")
    bindings = (
        ("adequacy_preregistration", "adequacy_preregistration_sha256"),
        ("sealed_resume_pool", "sealed_resume_pool_sha256"),
        ("adequacy_runtime", "adequacy_runtime_sha256"),
        ("candidate_runner", "candidate_runner_sha256"),
        ("independent_scorer", "independent_scorer_sha256"),
        ("base_evaluator_owner", "base_evaluator_owner_sha256"),
        ("replacement_02_evaluator_owner", "replacement_02_evaluator_owner_sha256"),
        ("replacement_04_evaluator_owner", "replacement_04_evaluator_owner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
        ("preserved_candidate_run", "preserved_candidate_run_sha256"),
        ("interruption_receipt", "interruption_receipt_sha256"),
        ("adequacy_backend", "adequacy_backend_sha256"),
    )
    for path_key, digest_key in bindings:
        owner = p2a.resolve(str(value.get(path_key) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(value.get(digest_key) or ""):
            faults.append(f"binding_invalid:{path_key}")
    pool_path = p2a.resolve(str(value.get("sealed_resume_pool") or ""))
    pool = p2a.read_json(pool_path) if pool_path.is_file() else {}
    if (
        pool.get("trigger_state") != "GREEN"
        or pool.get("state") != "SEALED_V3_REPLACEMENT_DENOMINATOR_BEFORE_RESUME"
        or int(pool.get("task_count") or 0) != 18
        or int(pool.get("sealed_packet_count") or 0) != 18
        or int(pool.get("preserved_candidate_count") or 0) != PRESERVED_CANDIDATE_COUNT
        or int(pool.get("preserved_denominator_model_calls") or 0) != PRESERVED_MODEL_CALLS
        or pool.get("resume_generation_indices") != list(range(4, 19))
        or any(int(count or 0) != 0 for count in p2a.mapping(pool.get("counters")).values())
    ):
        faults.append("sealed_resume_pool_invalid")
    pool_rows = p2a.dicts(pool.get("rows"))
    for row in pool_rows:
        index = int(row.get("index") or 0)
        task = p2a.resolve(str(row.get("task_manifest") or ""))
        packet = p2a.resolve(str(row.get("candidate_packet") or ""))
        if not task.is_file() or p2a.sha256_file(task) != row.get("task_manifest_sha256"):
            faults.append(f"task_binding_invalid:{index:02d}")
        if not packet.is_file() or p2a.sha256_file(packet) != row.get("candidate_packet_sha256"):
            faults.append(f"packet_binding_invalid:{index:02d}")
        if task.is_file() and p4.audit_task(task).get("trigger_state") != "GREEN":
            faults.append(f"task_audit_red:{index:02d}")
        prompt = str(p2a.read_json(packet).get("serialized_prompt") or "") if packet.is_file() else ""
        if p2a.sha256_text(prompt) != row.get("serialized_prompt_sha256"):
            faults.append(f"prompt_binding_invalid:{index:02d}")
        if len(prompt.encode("utf-8")) >= MODEL_CONTEXT_TOKENS:
            faults.append(f"prompt_context_residual_invalid:{index:02d}")
    preserved_path = p2a.resolve(str(value.get("preserved_candidate_run") or ""))
    preserved = p2a.read_json(preserved_path) if preserved_path.is_file() else {}
    preserved_rows = p2a.dicts(preserved.get("rows"))
    if (
        preserved.get("state") != "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED"
        or preserved.get("hidden_evaluation_opened") is not False
        or len(preserved_rows) != PRESERVED_CANDIDATE_COUNT
        or [int(row.get("index") or 0) for row in preserved_rows] != [1, 2, 3]
        or len(pool_rows) != 18
    ):
        faults.append("preserved_candidate_run_invalid")
    else:
        for row, pool_row in zip(preserved_rows, pool_rows[:3], strict=True):
            if campaign_v2.audit_candidate_custody(row, pool_row):
                faults.append(f"preserved_candidate_custody_invalid:{int(row.get('index') or 0):02d}")
    completion = p2a.mapping(value.get("generation_completion"))
    if completion.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if completion.get("normal_completion") != ["parser_complete", "model_eos"]:
        faults.append("normal_completion_invalid")
    if completion.get("physical_context_boundary_hit_invalidates_observation") is not True:
        faults.append("physical_context_boundary_disposition_invalid")
    if completion.get("host_watchdog_activation_invalidates_observation") is not True:
        faults.append("watchdog_disposition_invalid")
    if int(completion.get("host_watchdog_seconds") or 0) != HOST_WATCHDOG_SECONDS:
        faults.append("watchdog_value_invalid")
    if completion.get("backend_telemetry_precedes_route_consequence_classification") is not True:
        faults.append("watchdog_classifier_precedence_invalid")
    design = p2a.mapping(value.get("adequacy_design"))
    if (
        int(design.get("task_count") or 0) != 18
        or int(design.get("preserved_model_calls") or 0) != PRESERVED_MODEL_CALLS
        or int(design.get("new_model_calls") or 0) != NEW_MODEL_CALLS
        or int(design.get("total_model_calls") or 0) != 36
        or int(design.get("minimum_successes") or 0) != 13
        or int(design.get("minimum_successes_per_stratum") or 0) != 2
        or design.get("hidden_evaluation_after_all_candidate_seals") is not True
    ):
        faults.append("adequacy_design_invalid")
    authority = p2a.mapping(value.get("authority"))
    if authority.get("new_local_model_calls_authorized_after_green_audit") != NEW_MODEL_CALLS:
        faults.append("local_model_authority_invalid")
    for key in (
        "external_inference_authorized",
        "teacher_calls_authorized",
        "training_rows_authorized",
        "serving_authorized",
        "D1_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
    ):
        if authority.get(key) is not False:
            faults.append(f"cross_stage_authority_present:{key}")
    base = p2a.read_json(p2a.resolve(str(value.get("base_local_instrument") or "")))
    binding = p2a.mapping(base.get("runtime_binding"))
    frozen = p2a.mapping(base.get("frozen_model"))
    contract = route_integrity_v2.load_model_contract(
        str(binding.get("worker_config") or ""),
        str(binding.get("runtime_preflight") or ""),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
    )
    identity = p2a.mapping(contract.get("identity"))
    if contract.get("ready") is not True:
        faults.append("frozen_model_contract_not_ready")
    if identity.get("identity_sha256") != frozen.get("identity_sha256"):
        faults.append("frozen_model_identity_mismatch")
    if identity.get("decoder_sha256") != frozen.get("decoder_sha256"):
        faults.append("frozen_decoder_mismatch")
    return {
        "policy": AUDIT_POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "sealed_task_count": len(pool_rows),
        "preserved_candidate_count": len(preserved_rows),
        "resume_generation_indices": list(range(4, 19)),
        "host_watchdog_seconds": HOST_WATCHDOG_SECONDS,
        "watchdog_classifier_precedence": "backend_telemetry_before_route_consequence",
        "frozen_model_contract": contract,
        "candidate_generation_opened": False,
        "hidden_evaluation_opened": False,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def run_campaign(
    config_path: Path,
    out_path: Path,
    journal_path: Path,
    *,
    session_factory: Callable[..., Any] = persistent_adequacy_session,
) -> dict[str, Any]:
    started = time.perf_counter()
    audit = audit_config(config_path)
    if audit.get("trigger_state") != "GREEN":
        return campaign_report(config_path, audit, [], ["campaign_config_audit_red"], started, terminal=False)
    config = p2a.read_json(config_path)
    pool = p2a.read_json(p2a.resolve(str(config.get("sealed_resume_pool") or "")))
    pool_rows = p2a.dicts(pool.get("rows"))
    preserved = p2a.read_json(p2a.resolve(str(config.get("preserved_candidate_run") or "")))
    preserved_rows = [dict(row) for row in p2a.dicts(preserved.get("rows"))]
    rows, existing_fault, already_complete = load_resume_rows(
        out_path, journal_path, config_path, pool_rows, preserved_rows
    )
    if existing_fault:
        return campaign_report(config_path, audit, rows, [existing_fault], started, terminal=False)
    if already_complete:
        return p2a.read_json(out_path)
    base = p2a.read_json(p2a.resolve(str(config.get("base_local_instrument") or "")))
    binding = p2a.mapping(base.get("runtime_binding"))
    frozen = p2a.mapping(base.get("frozen_model"))
    session = session_factory(
        worker_config_path=p2a.resolve(str(binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(binding.get("runtime_preflight") or "")),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
        session_id="semantic-ir-production-adequacy-panel-v3",
        completion_predicate=production.complete,
    )
    if not session.ready:
        return campaign_report(
            config_path, audit, rows, ["persistent_backend_not_ready:" + ",".join(session.faults)], started, terminal=False
        )
    infrastructure_faults: list[str] = []
    runtime_config = str(base.get("runtime_config") or "")
    with base_campaign.assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for pool_row in pool_rows[len(rows):]:
            index = int(pool_row.get("index") or 0)
            try:
                row = run_task(
                    pool_row,
                    runtime_config,
                    journal_path,
                    config_path,
                    [int(existing.get("index") or 0) for existing in rows],
                )
            except InfrastructureBoundary as exc:
                infrastructure_faults.append(exc.fault)
                break
            rows.append(row)
            if row.get("infrastructure_faults"):
                infrastructure_faults.extend(p2a.strings(row.get("infrastructure_faults")))
            p2a.write_json(
                out_path,
                campaign_report(config_path, audit, rows, infrastructure_faults, started, terminal=False),
            )
            write_journal(
                journal_path,
                config_path,
                state="TASK_SEALED" if not infrastructure_faults else "TASK_INFRASTRUCTURE_INVALID",
                task_index=index,
                call_number=2,
                completed_candidate_indices=[int(row.get("index") or 0) for row in rows],
                fault=infrastructure_faults[0] if infrastructure_faults else "",
            )
            if infrastructure_faults:
                break
    terminal = len(rows) == 18 and not infrastructure_faults
    report = campaign_report(config_path, audit, rows, infrastructure_faults, started, terminal=terminal)
    if terminal:
        write_journal(
            journal_path,
            config_path,
            state="CANDIDATE_DENOMINATOR_SEALED",
            task_index=18,
            call_number=2,
            completed_candidate_indices=list(range(1, 19)),
            fault="",
        )
    return report


def run_task(
    pool_row: dict[str, Any],
    runtime_config: str,
    journal_path: Path,
    config_path: Path,
    completed_candidate_indices: list[int],
) -> dict[str, Any]:
    index = int(pool_row.get("index") or 0)
    original_runtime_call = p2a.runtime_call

    def guarded_runtime_call(
        arm: str,
        task_id: str,
        call_number: int,
        prompt: str,
        maximum: int,
        runtime_config_value: str,
    ) -> dict[str, Any]:
        namespaced = "semantic_ir_adequacy_04r1" if index == 4 else task_id
        write_journal(
            journal_path,
            config_path,
            state="MODEL_CALL_IN_FLIGHT",
            task_index=index,
            call_number=call_number,
            completed_candidate_indices=completed_candidate_indices,
            fault="",
        )
        result = original_runtime_call(arm, namespaced, call_number, prompt, maximum, runtime_config_value)
        telemetry = p4r.termination_telemetry(
            [{"arm_id": namespaced, "runtime_calls": [result["receipt"]]}]
        )
        route = p2a.mapping(p2a.mapping(result.get("runtime_report")).get("route_integrity"))
        abnormal = completion_fault(index, call_number, telemetry, route)
        if abnormal:
            write_journal(
                journal_path,
                config_path,
                state="MODEL_CALL_INFRASTRUCTURE_INVALID",
                task_index=index,
                call_number=call_number,
                completed_candidate_indices=completed_candidate_indices,
                fault=abnormal,
            )
            raise InfrastructureBoundary(abnormal)
        write_journal(
            journal_path,
            config_path,
            state="MODEL_CALL_COMPLETED_NOT_YET_CANDIDATE_SEALED",
            task_index=index,
            call_number=call_number,
            completed_candidate_indices=completed_candidate_indices,
            fault="",
        )
        return result

    p2a.runtime_call = guarded_runtime_call
    try:
        return base_campaign.run_task(pool_row, runtime_config)
    finally:
        p2a.runtime_call = original_runtime_call


def completion_fault(
    index: int, call_number: int, telemetry: list[dict[str, Any]], route: dict[str, Any]
) -> str:
    suffix = f"task_{index:02d}:call_{call_number}"
    if len(telemetry) != 1:
        return f"completion_telemetry_missing:{suffix}"
    receipt = telemetry[0]
    if receipt.get("termination_reason") == "host_safety_wall_time" or receipt.get(
        "host_safety_wall_time_hit"
    ) is True:
        return f"host_watchdog_infrastructure_invalid:{suffix}"
    if receipt.get("termination_reason") not in {"parser_complete", "model_eos"}:
        return f"completion_custody_red:{suffix}"
    if receipt.get("safety_ceiling_hit") is True:
        return f"safety_boundary_infrastructure_invalid:{suffix}"
    if route.get("ready") is not True or route.get("release_allowed") is not True:
        return f"route_integrity_red:{suffix}"
    return ""


def load_resume_rows(
    out_path: Path,
    journal_path: Path,
    config_path: Path,
    pool_rows: list[dict[str, Any]],
    preserved_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, bool]:
    if not journal_path.is_file():
        return preserved_rows, "", False
    journal = p2a.read_json(journal_path)
    if journal.get("config_sha256") != p2a.sha256_file(config_path):
        return preserved_rows, "resume_journal_config_mismatch", False
    if not out_path.is_file():
        return preserved_rows, "resume_candidate_report_missing", False
    prior = p2a.read_json(out_path)
    prior_rows = p2a.dicts(prior.get("rows"))
    if prior.get("policy") != POLICY or prior.get("hidden_evaluation_opened") is not False:
        return preserved_rows, "resume_candidate_report_invalid", False
    if [int(row.get("index") or 0) for row in prior_rows] != list(range(1, len(prior_rows) + 1)):
        return preserved_rows, "resume_candidate_indices_invalid", False
    for row, pool_row in zip(prior_rows, pool_rows, strict=False):
        if campaign_v2.audit_candidate_custody(row, pool_row):
            return preserved_rows, "resume_candidate_custody_invalid", False
    if journal.get("state") == "CANDIDATE_DENOMINATOR_SEALED":
        if prior.get("trigger_state") == "GREEN" and len(prior_rows) == 18:
            return prior_rows, "", True
        return preserved_rows, "completed_journal_candidate_report_invalid", False
    if journal.get("state") != "TASK_SEALED":
        return prior_rows or preserved_rows, "resume_journal_has_consumed_unsealed_surface", False
    if int(journal.get("task_index") or 0) != len(prior_rows):
        return preserved_rows, "resume_journal_task_index_mismatch", False
    if journal.get("completed_candidate_indices") != list(range(1, len(prior_rows) + 1)):
        return preserved_rows, "resume_journal_candidate_indices_mismatch", False
    if prior.get("trigger_state") != "YELLOW" or prior.get("state") != "RUNNING_CANDIDATE_GENERATION":
        return preserved_rows, "resume_running_candidate_report_invalid", False
    return prior_rows, "", False


def write_journal(
    path: Path,
    config_path: Path,
    *,
    state: str,
    task_index: int,
    call_number: int,
    completed_candidate_indices: list[int],
    fault: str,
) -> None:
    p2a.write_json(
        path,
        {
            "policy": "project_theseus_semantic_ir_production_adequacy_campaign_journal_v3",
            "created_utc": p2a.now(),
            "state": state,
            "config_sha256": p2a.sha256_file(config_path),
            "task_index": task_index,
            "call_number": call_number,
            "completed_candidate_indices": completed_candidate_indices,
            "fault": fault,
            "hidden_evaluation_opened": False,
        },
    )


def campaign_report(
    config_path: Path,
    audit: dict[str, Any],
    rows: list[dict[str, Any]],
    infrastructure_faults: list[str],
    started: float,
    *,
    terminal: bool,
) -> dict[str, Any]:
    calls = sum(len(p2a.dicts(row.get("runtime_calls"))) for row in rows)
    state = (
        "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION"
        if terminal
        else "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED"
        if infrastructure_faults
        else "RUNNING_CANDIDATE_GENERATION"
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if terminal else "RED" if infrastructure_faults else "YELLOW",
        "state": state,
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "config_audit": audit,
        "preserved_candidate_count": min(PRESERVED_CANDIDATE_COUNT, len(rows)),
        "new_candidate_count": max(0, len(rows) - PRESERVED_CANDIDATE_COUNT),
        "completed_task_count": len(rows),
        "expected_task_count": 18,
        "rows": rows,
        "faults": sorted(set(infrastructure_faults)),
        "hidden_evaluation_opened": False,
        "counters": {**zero_counters(), "candidate_or_control_calls": calls, "local_model_calls": calls},
        "maximum_inference": (
            "Candidate generation and syntax-visible mechanics only. Host-watchdog activation "
            "or abnormal completion is invalid infrastructure, not capability evidence. Hidden "
            "evaluation, implementation adequacy, subsystem effects, D1, D2, training value, "
            "serving, and book support remain unestablished until independent scoring."
        ),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


zero_counters = campaign_v2.zero_counters


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "completed_task_count": report.get("completed_task_count"),
        "preserved_candidate_count": report.get("preserved_candidate_count"),
        "new_candidate_count": report.get("new_candidate_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
