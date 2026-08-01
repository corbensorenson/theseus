#!/usr/bin/env python3
"""P4 successor with completion-based generation and explicit truncation custody."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_route_integrity as route_integrity  # noqa: E402
import theseus_assistant_runtime as assistant_runtime  # noqa: E402
import theseus_generation_completion as completion  # noqa: E402
import theseus_local_inference_backend as local_backend  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402


POLICY = "project_theseus_p4_cognitive_compilation_repaired_run_v1"
INSTRUMENT_POLICY = "project_theseus_p4_cognitive_compilation_repaired_instrument_v1"
MODEL_CONTEXT_TOKENS = 262144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument",
        default="configs/theseus_p4_cognitive_compilation_repaired_instrument.json",
    )
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--out",
        default="reports/theseus_p4_cognitive_compilation_repaired_run.json",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    instrument_path = p2a.resolve(args.instrument)
    report = (
        audit_instrument(instrument_path)
        if args.audit_only
        else run_experiment(instrument_path, p2a.resolve(args.task))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "denominators": report.get("denominators"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_instrument(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != INSTRUMENT_POLICY:
        faults.append("instrument_policy_invalid")
    if value.get("state") != "PROSPECTIVELY_BOUND_BEFORE_REPAIRED_TASK_POOL":
        faults.append("instrument_not_prospectively_bound")
    disposition_path = p2a.resolve(str(value.get("predecessor_disposition") or ""))
    if p2a.sha256_file(disposition_path) != str(value.get("predecessor_disposition_sha256") or ""):
        faults.append("predecessor_disposition_digest_mismatch")
    disposition = p2a.read_json(disposition_path)
    if disposition.get("campaign_state") != "P4_V1_INTERRUPTED_BUDGET_POLICY_REJECTED":
        faults.append("predecessor_disposition_state_invalid")
    completion_path = p2a.resolve(str(value.get("generation_completion_policy") or ""))
    if p2a.sha256_file(completion_path) != str(value.get("generation_completion_policy_sha256") or ""):
        faults.append("completion_policy_digest_mismatch")
    completion_policy = p2a.read_json(completion_path)
    if completion_policy.get("state") != "ACTIVE_FOR_ALL_NEW_GENERATION_INSTRUMENTS":
        faults.append("completion_policy_not_active")
    base_path = p2a.resolve(str(value.get("base_local_instrument") or ""))
    if p2a.sha256_file(base_path) != str(value.get("base_local_instrument_sha256") or ""):
        faults.append("base_local_instrument_digest_mismatch")
    base = p2a.read_json(base_path)
    runtime = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(runtime.get("local_inference"))
    frozen = p2a.mapping(base.get("frozen_model"))
    contract = route_integrity.load_model_contract(
        str(binding.get("worker_config") or ""),
        str(binding.get("runtime_preflight") or ""),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
    )
    identity = p2a.mapping(contract.get("identity"))
    if contract.get("ready") is not True:
        faults.append("completion_model_contract_not_ready")
    if identity.get("identity_sha256") != frozen.get("identity_sha256"):
        faults.append("completion_model_identity_mismatch")
    if identity.get("decoder_sha256") != frozen.get("decoder_sha256"):
        faults.append("completion_decoder_identity_mismatch")
    if frozen.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(frozen.get("model_declared_context_window_tokens") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("model_context_window_binding_invalid")
    harness = p2a.mapping(value.get("harness"))
    for name in ("candidate_runner", "blind_evaluator", "local_backend"):
        owner = p2a.resolve(str(harness.get(name) or ""))
        if p2a.sha256_file(owner) != str(harness.get(f"{name}_sha256") or ""):
            faults.append(f"{name}_digest_mismatch")
    matched = p2a.mapping(value.get("matched_arm_contract"))
    if tuple(p2a.strings(matched.get("arms"))) != p4.ARMS:
        faults.append("arm_set_invalid")
    for key in (
        "same_frozen_weights", "same_information", "same_completion_policy",
        "same_model_context_residual_rule", "same_two_model_calls",
        "same_verifier_and_effect_sandbox",
    ):
        if matched.get(key) is not True:
            faults.append(f"matched_contract_false:{key}")
    if p2a.mapping(value.get("generation_budget")).get("project_selected_quality_token_cap") is not None:
        faults.append("instrument_quality_token_cap_present")
    if p2a.mapping(value.get("generation_budget")).get("ceiling_hit_invalidates_observation") is not True:
        faults.append("ceiling_hit_disposition_invalid")
    mechanics = p4.mechanics_audit()
    if mechanics.get("trigger_state") != "GREEN":
        faults.append("semantic_ir_mechanics_red")
    return {
        "policy": "project_theseus_p4_cognitive_compilation_repaired_instrument_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": p2a.sha256_file(path),
        "completion_model_contract": contract,
        "semantic_ir_mechanics": mechanics,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": p2a.zero_counters(),
    }


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    started = time.perf_counter()
    instrument_audit = audit_instrument(instrument_path)
    task_audit = p4.audit_task(task_path)
    if instrument_audit.get("trigger_state") != "GREEN" or task_audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY,
            "created_utc": p2a.now(),
            "trigger_state": "RED",
            "faults": ["instrument_or_task_audit_red"],
            "instrument_audit": instrument_audit,
            "task_audit": task_audit,
            "counters": p2a.zero_counters(),
        }
    instrument = p2a.read_json(instrument_path)
    task = p2a.read_json(task_path)
    base = p2a.read_json(p2a.resolve(str(instrument.get("base_local_instrument") or "")))
    runtime = assistant_runtime.load_runtime_config(p2a.resolve(str(base.get("runtime_config") or "")))
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(runtime.get("local_inference"))
    frozen = p2a.mapping(base.get("frozen_model"))
    session = session_factory(
        worker_config_path=p2a.resolve(str(binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(binding.get("runtime_preflight") or "")),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
        session_id=f"p4r-{p2a.safe_slug(str(task.get('opaque_task_id') or 'task'))}",
        completion_predicate=completion.candidate_envelope_complete,
    )
    if not session.ready:
        return {
            "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED",
            "faults": ["persistent_backend_not_ready", *list(session.faults)],
            "instrument_audit": instrument_audit, "task_audit": task_audit,
            "counters": p2a.zero_counters(),
        }
    attempts: list[dict[str, Any]] = []
    order = p4.arm_order(int(task.get("campaign_index") or 0))
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for arm in order:
            attempts.append(p4.run_arm(arm, instrument, task, session))
    telemetry = termination_telemetry(attempts)
    static_control = p4.run_deterministic_compiler_control(task)
    identities = {str(row.get("model_identity_sha256") or "") for row in attempts}
    termination_valid = (
        len(telemetry) == 6
        and all(row.get("termination_reason") in {"parser_complete", "model_eos"} for row in telemetry)
        and not any(row.get("safety_ceiling_hit") is True for row in telemetry)
    )
    matched = {
        "ready": (
            len(identities) == 1 and "" not in identities
            and session.model_load_count == 1 and session.inference_calls == 6
            and all(len(p2a.dicts(row.get("runtime_calls"))) == 2 for row in attempts)
            and all(
                all(p2a.mapping(route).get("ready") is True for route in p2a.dicts(row.get("route_integrity_rounds")))
                for row in attempts
            )
            and termination_valid
        ),
        "same_model_identity": len(identities) == 1 and "" not in identities,
        "persistent_model_load_count": session.model_load_count,
        "persistent_inference_calls": session.inference_calls,
        "two_calls_per_learned_arm": all(
            len(p2a.dicts(row.get("runtime_calls"))) == 2 for row in attempts
        ),
        "completion_telemetry_valid": termination_valid,
        "safety_ceiling_hits": sum(int(row.get("safety_ceiling_hit") is True) for row in telemetry),
    }
    faults = [] if matched["ready"] else ["matched_set_or_completion_custody_invalid"]
    parseable = sum(int(row.get("parseable_candidate") is True) for row in attempts)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else ("GREEN" if parseable else "YELLOW"),
        "faults": faults,
        "scope": "Repaired P4 cognitive-compilation development only; no D1, D2, serving, training, or book-support claim",
        "instrument_sha256": p2a.sha256_file(instrument_path),
        "task_sha256": p2a.sha256_file(task_path),
        "instrument_audit": instrument_audit,
        "task_audit": task_audit,
        "actual_arm_order": list(order),
        "attempts": attempts,
        "deterministic_compiler_control": static_control,
        "generation_termination_telemetry": telemetry,
        "matched_set": matched,
        "denominators": {
            "tasks": 1,
            "learned_arms": 3,
            "model_calls": session.inference_calls,
            "model_loads": session.model_load_count,
            "parseable_candidates": parseable,
            "deterministic_compiler_candidates": int(static_control.get("parseable_candidate") is True),
            "project_selected_quality_token_cap": None,
        },
        "counters": p2a.zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def termination_telemetry(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        for receipt in p2a.dicts(attempt.get("runtime_calls")):
            runtime_path = p2a.resolve(str(receipt.get("report_path") or ""))
            runtime = p2a.read_json(runtime_path)
            backend_path = p2a.resolve(
                str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
            )
            backend = p2a.read_json(backend_path)
            metrics = p2a.mapping(backend.get("metrics"))
            rows.append({
                "arm_id": attempt.get("arm_id"),
                "call_number": receipt.get("call_number"),
                "runtime_report": p2a.rel(runtime_path),
                "runtime_report_sha256": p2a.sha256_file(runtime_path),
                "backend_report": p2a.rel(backend_path),
                "backend_report_sha256": p2a.sha256_file(backend_path),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "generated_tokens": metrics.get("generated_tokens"),
                "model_context_window_tokens": metrics.get("model_context_window_tokens"),
                "effective_maximum_tokens": metrics.get("effective_maximum_tokens"),
                "backend_finish_reason": metrics.get("backend_finish_reason"),
                "termination_reason": metrics.get("termination_reason"),
                "completion_predicate_enabled": metrics.get("completion_predicate_enabled"),
                "safety_ceiling_hit": metrics.get("safety_ceiling_hit"),
            })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
