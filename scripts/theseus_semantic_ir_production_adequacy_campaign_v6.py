#!/usr/bin/env python3
"""Run the prospectively sealed fresh-v6 compact Semantic-IR denominator."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_assistant_route_integrity_v2 as route_integrity_v2
import theseus_p4_cognitive_compilation as p4
import theseus_p4_cognitive_compilation_repaired as p4r
import theseus_p4v2r2_cognitive_compilation as local_completion
import theseus_semantic_ir_production_adequacy_backend_v2 as backend_v2
import theseus_semantic_ir_production_adequacy_campaign as base_campaign
import theseus_semantic_ir_production_adequacy_campaign_v2 as campaign_v2
import theseus_semantic_ir_production_adequacy_runtime_v5 as runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign_v6.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v6.json"
DEFAULT_JOURNAL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_v6_journal.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_candidates_v6"
CONFIG_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_v6"
AUDIT_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_audit_v6"
JOURNAL_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_journal_v6"
MODEL_CONTEXT_TOKENS = 262_144
HOST_WATCHDOG_SECONDS = 600
MODEL_CALLS = 36
InfrastructureBoundary = campaign_v2.InfrastructureBoundary


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
    if (
        value.get("policy") != CONFIG_POLICY
        or value.get("state")
        != "PROSPECTIVELY_BOUND_FRESH_V6_BEFORE_CANDIDATE_GENERATION"
    ):
        faults.append("config_identity_invalid")
    for path_key, digest_key in (
        ("adequacy_preregistration", "adequacy_preregistration_sha256"),
        ("sealed_task_pool", "sealed_task_pool_sha256"),
        ("compact_statement_runtime", "compact_statement_runtime_sha256"),
        ("adequacy_backend", "adequacy_backend_sha256"),
        ("candidate_runner", "candidate_runner_sha256"),
        ("independent_scorer", "independent_scorer_sha256"),
        ("fresh_evaluator_owner", "fresh_evaluator_owner_sha256"),
        ("base_evaluator_owner", "base_evaluator_owner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
    ):
        owner = p2a.resolve(str(value.get(path_key) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(value.get(digest_key) or ""):
            faults.append(f"binding_invalid:{path_key}")
    pool_path = p2a.resolve(str(value.get("sealed_task_pool") or ""))
    pool = p2a.read_json(pool_path) if pool_path.is_file() else {}
    if (
        pool.get("trigger_state") != "GREEN"
        or pool.get("state")
        != "SEALED_FRESH_V6_UNIFORM_COMPACT_DENOMINATOR_BEFORE_CANDIDATE_GENERATION"
        or int(pool.get("task_count") or 0) != 18
        or int(pool.get("sealed_packet_count") or 0) != 18
        or int(pool.get("repository_count") or 0) != 18
        or any(int(count or 0) != 3 for count in p2a.mapping(pool.get("stratum_counts")).values())
        or any(int(count or 0) != 0 for count in p2a.mapping(pool.get("counters")).values())
    ):
        faults.append("sealed_task_pool_invalid")
    pool_rows = p2a.dicts(pool.get("rows"))
    if [int(row.get("index") or 0) for row in pool_rows] != list(range(1, 19)):
        faults.append("sealed_task_indices_invalid")
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
        prompt_tokens = int(row.get("exact_prompt_tokens") or 0)
        residual = int(row.get("exact_context_residual_tokens") or 0)
        if prompt_tokens <= 0 or residual != MODEL_CONTEXT_TOKENS - prompt_tokens or residual <= 0:
            faults.append(f"exact_prompt_context_residual_invalid:{index:02d}")
        compact = p2a.mapping(row.get("compact_integrity_abi"))
        if compact.get("handle_bits") != 128 or compact.get("inventory_complete") is not True:
            faults.append(f"compact_statement_abi_invalid:{index:02d}")
    completion = p2a.mapping(value.get("generation_completion"))
    if (
        completion.get("project_selected_quality_token_cap") is not None
        or completion.get("normal_completion") != ["parser_complete", "model_eos"]
        or completion.get("physical_context_boundary_hit_invalidates_observation") is not True
        or completion.get("host_watchdog_activation_invalidates_observation") is not True
        or int(completion.get("host_watchdog_seconds") or 0) != HOST_WATCHDOG_SECONDS
        or completion.get("backend_telemetry_precedes_route_consequence_classification") is not True
    ):
        faults.append("generation_completion_policy_invalid")
    design = p2a.mapping(value.get("adequacy_design"))
    if (
        int(design.get("task_count") or 0) != 18
        or int(design.get("model_calls_per_task") or 0) != 2
        or int(design.get("new_model_calls") or 0) != MODEL_CALLS
        or int(design.get("total_model_calls") or 0) != MODEL_CALLS
        or int(design.get("minimum_successes") or 0) != 13
        or int(design.get("minimum_successes_per_stratum") or 0) != 2
        or design.get("hidden_evaluation_after_all_candidate_seals") is not True
    ):
        faults.append("adequacy_design_invalid")
    authority = p2a.mapping(value.get("authority"))
    if authority.get("new_local_model_calls_authorized_after_green_audit") != MODEL_CALLS:
        faults.append("local_model_authority_invalid")
    for key in (
        "external_inference_authorized", "teacher_calls_authorized",
        "training_rows_authorized", "serving_authorized", "D1_authorized",
        "D2_authorized", "book_support_promotion_authorized",
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
    if (
        contract.get("ready") is not True
        or identity.get("identity_sha256") != frozen.get("identity_sha256")
        or identity.get("decoder_sha256") != frozen.get("decoder_sha256")
    ):
        faults.append("frozen_model_contract_invalid")
    return {
        "policy": AUDIT_POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "sealed_task_count": len(pool_rows),
        "generation_indices": list(range(1, 19)),
        "host_watchdog_seconds": HOST_WATCHDOG_SECONDS,
        "watchdog_classifier_precedence": "backend_telemetry_before_route_consequence",
        "frozen_model_contract": contract,
        "candidate_generation_opened": False,
        "hidden_evaluation_opened": False,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def persistent_adequacy_session(**kwargs: Any) -> Any:
    completion_predicate = kwargs.get("completion_predicate")

    def model_factory(card: dict[str, Any], snapshot: Path, maximum: int) -> Any:
        return backend_v2.DiagnosticAdequacyLocalMlxChatModel(
            card, snapshot, maximum,
            completion_predicate=completion_predicate,
            maximum_wall_seconds=HOST_WATCHDOG_SECONDS,
        )

    kwargs["model_factory"] = model_factory
    session = local_completion.persistent_v2_session(**kwargs)
    original_generate_report = session.generate_report

    def generate_report(**request: Any) -> dict[str, Any]:
        report = original_generate_report(**request)
        model = getattr(session, "model", None)
        return backend_v2.attach_invalid_observation_diagnostic(report, model) if model is not None else report

    session.generate_report = generate_report
    return session


def run_campaign(
    config_path: Path, out_path: Path, journal_path: Path, *,
    session_factory: Callable[..., Any] = persistent_adequacy_session,
) -> dict[str, Any]:
    started = time.perf_counter()
    audit = audit_config(config_path)
    if audit.get("trigger_state") != "GREEN":
        return campaign_report(config_path, audit, [], ["campaign_config_audit_red"], started, terminal=False)
    config = p2a.read_json(config_path)
    pool = p2a.read_json(p2a.resolve(str(config.get("sealed_task_pool") or "")))
    pool_rows = p2a.dicts(pool.get("rows"))
    rows, resume_fault, already_complete = load_resume_rows(out_path, journal_path, config_path, pool_rows)
    if resume_fault:
        return campaign_report(config_path, audit, rows, [resume_fault], started, terminal=False)
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
        session_id="semantic-ir-production-adequacy-panel-v6",
        completion_predicate=runtime.complete,
    )
    if not session.ready:
        fault = "persistent_backend_not_ready:" + ",".join(session.faults)
        return campaign_report(config_path, audit, rows, [fault], started, terminal=False)
    faults: list[str] = []
    runtime_config = str(base.get("runtime_config") or "")
    with base_campaign.assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for pool_row in pool_rows[len(rows):]:
            index = int(pool_row.get("index") or 0)
            try:
                row = run_task(pool_row, runtime_config, journal_path, config_path, [int(existing.get("index") or 0) for existing in rows])
            except InfrastructureBoundary as exc:
                faults.append(exc.fault)
                break
            rows.append(row)
            faults.extend(p2a.strings(row.get("infrastructure_faults")))
            p2a.write_json(out_path, campaign_report(config_path, audit, rows, faults, started, terminal=False))
            write_journal(
                journal_path, config_path,
                state="TASK_SEALED" if not faults else "TASK_INFRASTRUCTURE_INVALID",
                task_index=index, call_number=2,
                completed_candidate_indices=[int(item.get("index") or 0) for item in rows],
                fault=faults[0] if faults else "",
            )
            if faults:
                break
    terminal = len(rows) == 18 and not faults
    report = campaign_report(config_path, audit, rows, faults, started, terminal=terminal)
    if terminal:
        write_journal(journal_path, config_path, state="CANDIDATE_DENOMINATOR_SEALED", task_index=18, call_number=2, completed_candidate_indices=list(range(1, 19)), fault="")
    return report


def run_task(
    pool_row: dict[str, Any], runtime_config: str, journal_path: Path,
    config_path: Path, completed_candidate_indices: list[int],
) -> dict[str, Any]:
    index = int(pool_row.get("index") or 0)
    task_path = p2a.resolve(str(pool_row.get("task_manifest") or ""))
    packet_path = p2a.resolve(str(pool_row.get("candidate_packet") or ""))
    task = p2a.read_json(task_path)
    packet = p2a.read_json(packet_path)
    prompt = str(packet.get("serialized_prompt") or "")
    original_runtime_call = p2a.runtime_call
    namespace = f"semantic_ir_adequacy_fresh_v6_{index:02d}"

    def guarded_call(arm: str, task_id: str, call_number: int, call_prompt: str, maximum: int, runtime_config_value: str) -> dict[str, Any]:
        write_journal(journal_path, config_path, state="MODEL_CALL_IN_FLIGHT", task_index=index, call_number=call_number, completed_candidate_indices=completed_candidate_indices, fault="")
        result = original_runtime_call(arm, namespace, call_number, call_prompt, maximum, runtime_config_value)
        telemetry = p4r.termination_telemetry([{"arm_id": namespace, "runtime_calls": [result["receipt"]]}])
        route = p2a.mapping(p2a.mapping(result.get("runtime_report")).get("route_integrity"))
        abnormal = completion_fault(index, call_number, telemetry, route)
        if abnormal:
            write_journal(journal_path, config_path, state="MODEL_CALL_INFRASTRUCTURE_INVALID", task_index=index, call_number=call_number, completed_candidate_indices=completed_candidate_indices, fault=abnormal)
            raise InfrastructureBoundary(abnormal)
        write_journal(journal_path, config_path, state="MODEL_CALL_COMPLETED_NOT_YET_CANDIDATE_SEALED", task_index=index, call_number=call_number, completed_candidate_indices=completed_candidate_indices, fault="")
        return result

    p2a.runtime_call = guarded_call
    try:
        with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-v6-run-{index:02d}-") as directory:
            root = Path(directory) / "source"
            p2a.extract_source_archive(p2a.resolve(str(task.get("source_archive") or "")), root, str(task.get("source_archive_root") or ""))
            first = p2a.runtime_call(p2a.route_integrity.DIRECT_MODE, namespace, 1, prompt, MODEL_CONTEXT_TOKENS, runtime_config)
            first_candidate = runtime.parse(first["assistant_text"], task, root)
            first_verification = verify_candidate(task, p2a.dicts(first_candidate.get("actions")))
            repair_prompt = runtime.render_repair_prompt(prompt, first["assistant_text"], p2a.strings(first_candidate.get("faults")), first_verification)
            second = p2a.runtime_call(p2a.route_integrity.DIRECT_MODE, namespace, 2, repair_prompt, MODEL_CONTEXT_TOKENS, runtime_config)
            final_candidate = runtime.parse(second["assistant_text"], task, root)
            actions = p2a.dicts(final_candidate.get("actions"))
            final_verification = verify_candidate(task, actions)
    finally:
        p2a.runtime_call = original_runtime_call
    calls = (first, second)
    routes = [p2a.mapping(p2a.mapping(call.get("runtime_report")).get("route_integrity")) for call in calls]
    telemetry = p4r.termination_telemetry([{"arm_id": namespace, "runtime_calls": [first["receipt"], second["receipt"]]}])
    faults: list[str] = []
    if not all(receipt.get("ready") is True and receipt.get("release_allowed") is True for receipt in routes):
        faults.append(f"route_integrity_red:task_{index:02d}")
    if len(telemetry) != 2 or any(row.get("termination_reason") not in {"parser_complete", "model_eos"} or row.get("safety_ceiling_hit") is True for row in telemetry):
        faults.append(f"completion_custody_red:task_{index:02d}")
    changed = p2a.strings(final_verification.get("changed_paths"))
    authorized = bool(changed) and set(changed).issubset(set(p2a.strings(task.get("allowed_effect_paths"))))
    mechanics = bool(actions) and not final_candidate.get("faults") and not final_verification.get("apply_faults") and p2a.mapping(final_verification.get("visible_verifier")).get("passed") is True
    payload = {
        "actions": actions, "changed_paths": changed,
        "final_inventory_sha256": final_verification.get("final_inventory_sha256"),
        "semantic_receipt": final_candidate.get("semantic_receipt"),
    }
    seal = {
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(prompt),
        "first_output_sha256": p2a.sha256_text(first["assistant_text"]),
        "final_output_sha256": p2a.sha256_text(second["assistant_text"]),
        "candidate_payload_sha256": p2a.stable_hash(payload),
        "sealed_before_hidden_evaluation": True,
    }
    return {
        "index": index, "opaque_task_id": task.get("opaque_task_id"),
        "task_manifest": p2a.rel(task_path), "candidate_packet": p2a.rel(packet_path),
        "candidate_mechanics_ready_for_hidden_evaluation": not faults and mechanics and authorized,
        "parse_faults": p2a.strings(final_candidate.get("faults")),
        "apply_faults": p2a.strings(final_verification.get("apply_faults")),
        "visible_verifier_passed": p2a.mapping(final_verification.get("visible_verifier")).get("passed") is True,
        "authorized_effect": authorized, "candidate_payload": payload, "candidate_seal": seal,
        "model_outputs": {"first": first["assistant_text"], "final": second["assistant_text"]},
        "runtime_calls": [first["receipt"], second["receipt"]],
        "route_integrity_rounds": routes, "termination_telemetry": telemetry,
        "infrastructure_faults": sorted(set(faults)), "hidden_evaluator_executions": 0,
    }


def verify_candidate(task: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-adequacy-v6-verify-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(p2a.resolve(str(task.get("source_archive") or "")), root, str(task.get("source_archive_root") or ""))
        baseline = p2a.inventory(root)
        apply_faults = runtime.apply_actions(root, actions) if actions else []
        inventory = p2a.inventory(root)
        changed = p2a.changed_paths(baseline, inventory) if not apply_faults else []
        visible = p2a.run_visible_verifier(root, task) if actions and not apply_faults else {}
        return {"apply_faults": apply_faults, "changed_paths": changed, "visible_verifier": visible, "final_inventory_sha256": p2a.stable_hash(inventory)}


def completion_fault(index: int, call_number: int, telemetry: list[dict[str, Any]], route: dict[str, Any]) -> str:
    suffix = f"task_{index:02d}:call_{call_number}"
    if len(telemetry) != 1:
        return f"completion_telemetry_missing:{suffix}"
    receipt = telemetry[0]
    if receipt.get("termination_reason") == "host_safety_wall_time" or receipt.get("host_safety_wall_time_hit") is True:
        return f"host_watchdog_infrastructure_invalid:{suffix}"
    if receipt.get("termination_reason") not in {"parser_complete", "model_eos"}:
        return f"completion_custody_red:{suffix}"
    if receipt.get("safety_ceiling_hit") is True:
        return f"safety_boundary_infrastructure_invalid:{suffix}"
    if route.get("ready") is not True or route.get("release_allowed") is not True:
        return f"route_integrity_red:{suffix}"
    return ""


def load_resume_rows(out_path: Path, journal_path: Path, config_path: Path, pool_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, bool]:
    if not journal_path.is_file():
        return [], "", False
    journal = p2a.read_json(journal_path)
    if journal.get("policy") != JOURNAL_POLICY or journal.get("config_sha256") != p2a.sha256_file(config_path):
        return [], "resume_journal_config_mismatch", False
    if not out_path.is_file():
        return [], "resume_candidate_report_missing", False
    prior = p2a.read_json(out_path)
    rows = p2a.dicts(prior.get("rows"))
    if prior.get("policy") != POLICY or prior.get("hidden_evaluation_opened") is not False:
        return [], "resume_candidate_report_invalid", False
    if [int(row.get("index") or 0) for row in rows] != list(range(1, len(rows) + 1)):
        return [], "resume_candidate_indices_invalid", False
    for row, pool_row in zip(rows, pool_rows, strict=False):
        if campaign_v2.audit_candidate_custody(row, pool_row):
            return [], "resume_candidate_custody_invalid", False
    if journal.get("state") == "CANDIDATE_DENOMINATOR_SEALED":
        return (rows, "", True) if prior.get("trigger_state") == "GREEN" and len(rows) == 18 else ([], "completed_journal_candidate_report_invalid", False)
    if journal.get("state") != "TASK_SEALED":
        return rows, "resume_journal_has_consumed_unsealed_surface", False
    if int(journal.get("task_index") or 0) != len(rows) or journal.get("completed_candidate_indices") != list(range(1, len(rows) + 1)):
        return [], "resume_journal_task_identity_invalid", False
    if prior.get("trigger_state") != "YELLOW" or prior.get("state") != "RUNNING_CANDIDATE_GENERATION":
        return [], "resume_running_candidate_report_invalid", False
    return rows, "", False


def write_journal(path: Path, config_path: Path, *, state: str, task_index: int, call_number: int, completed_candidate_indices: list[int], fault: str) -> None:
    p2a.write_json(path, {
        "policy": JOURNAL_POLICY, "created_utc": p2a.now(), "state": state,
        "config_sha256": p2a.sha256_file(config_path), "task_index": task_index,
        "call_number": call_number, "completed_candidate_indices": completed_candidate_indices,
        "fault": fault, "hidden_evaluation_opened": False,
    })


def campaign_report(config_path: Path, audit: dict[str, Any], rows: list[dict[str, Any]], infrastructure_faults: list[str], started: float, *, terminal: bool) -> dict[str, Any]:
    calls = sum(len(p2a.dicts(row.get("runtime_calls"))) for row in rows)
    state = "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION" if terminal else "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED" if infrastructure_faults else "RUNNING_CANDIDATE_GENERATION"
    return {
        "policy": POLICY, "created_utc": p2a.now(),
        "trigger_state": "GREEN" if terminal else "RED" if infrastructure_faults else "YELLOW",
        "state": state, "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "config_audit": audit, "preserved_candidate_count": 0, "new_candidate_count": len(rows),
        "completed_task_count": len(rows), "expected_task_count": 18, "rows": rows,
        "faults": sorted(set(infrastructure_faults)), "hidden_evaluation_opened": False,
        "counters": {**zero_counters(), "candidate_or_control_calls": calls, "local_model_calls": calls},
        "maximum_inference": "Candidate generation and syntax-visible mechanics only. Host-watchdog activation or abnormal completion is invalid infrastructure, not capability evidence. Hidden evaluation, implementation adequacy, subsystem effects, D1, D2, training value, serving, and book support remain unestablished until independent scoring.",
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


zero_counters = campaign_v2.zero_counters


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("policy", "trigger_state", "state", "completed_task_count", "new_candidate_count", "faults", "counters")}


if __name__ == "__main__":
    raise SystemExit(main())
