#!/usr/bin/env python3
"""Run the sealed local Semantic-IR implementation-adequacy panel.

Generation receives only the exact sealed candidate packet.  This owner never
imports or invokes the hidden evaluator.  It persists a candidate seal after
each task so the independent scorer can evaluate only after generation custody
has closed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_assistant_runtime as assistant_runtime
import theseus_assistant_route_integrity_v2 as route_integrity_v2
import theseus_p4_cognitive_compilation as p4
import theseus_p4_cognitive_compilation_repaired as p4r
import theseus_p4v2r2_cognitive_compilation as local_completion
import theseus_semantic_ir_production_adequacy_runtime as production


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_candidates_v1"
CONFIG_POLICY = "project_theseus_semantic_ir_production_adequacy_campaign_v1"
MODEL_CONTEXT_TOKENS = 262_144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = (
        audit_config(config_path)
        if args.audit_only
        else run_campaign(config_path, p2a.resolve(args.out))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "completed_task_count": report.get("completed_task_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_config(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "PROSPECTIVELY_BOUND_AFTER_POOL_SEAL_BEFORE_MODEL_EXPOSURE":
        faults.append("config_not_prospectively_bound")
    bindings = (
        ("adequacy_preregistration", "adequacy_preregistration_sha256"),
        ("sealed_task_pool", "sealed_task_pool_sha256"),
        ("adequacy_runtime", "adequacy_runtime_sha256"),
        ("candidate_runner", "candidate_runner_sha256"),
        ("independent_scorer", "independent_scorer_sha256"),
        ("evaluator_owner", "evaluator_owner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
    )
    for path_key, digest_key in bindings:
        owner = p2a.resolve(str(value.get(path_key) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(
            value.get(digest_key) or ""
        ):
            faults.append(f"binding_invalid:{path_key}")

    pool_path = p2a.resolve(str(value.get("sealed_task_pool") or ""))
    pool = p2a.read_json(pool_path) if pool_path.is_file() else {}
    if (
        pool.get("trigger_state") != "GREEN"
        or pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION"
        or int(pool.get("task_count") or 0) != 18
        or int(pool.get("sealed_packet_count") or 0) != 18
        or any(int(count or 0) != 0 for count in p2a.mapping(pool.get("counters")).values())
    ):
        faults.append("sealed_task_pool_invalid")
    task_faults: list[str] = []
    for row in p2a.dicts(pool.get("rows")):
        task = p2a.resolve(str(row.get("task_manifest") or ""))
        packet = p2a.resolve(str(row.get("candidate_packet") or ""))
        if p2a.sha256_file(task) != str(row.get("task_manifest_sha256") or ""):
            task_faults.append(f"task_binding_invalid:{row.get('index')}")
        if p2a.sha256_file(packet) != str(row.get("candidate_packet_sha256") or ""):
            task_faults.append(f"packet_binding_invalid:{row.get('index')}")
        if p4.audit_task(task).get("trigger_state") != "GREEN":
            task_faults.append(f"task_audit_red:{row.get('index')}")
        packet_value = p2a.read_json(packet)
        prompt = str(packet_value.get("serialized_prompt") or "")
        if p2a.sha256_text(prompt) != str(row.get("serialized_prompt_sha256") or ""):
            task_faults.append(f"prompt_binding_invalid:{row.get('index')}")
        if len(prompt.encode("utf-8")) >= MODEL_CONTEXT_TOKENS:
            task_faults.append(f"prompt_physical_context_residual_invalid:{row.get('index')}")
    faults.extend(task_faults)

    completion = p2a.mapping(value.get("generation_completion"))
    if completion.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if completion.get("normal_completion") != ["parser_complete", "model_eos"]:
        faults.append("normal_completion_invalid")
    if completion.get("physical_context_boundary_hit_invalidates_observation") is not True:
        faults.append("physical_context_boundary_disposition_invalid")
    design = p2a.mapping(value.get("adequacy_design"))
    if (
        int(design.get("task_count") or 0) != 18
        or int(design.get("minimum_successes") or 0) != 13
        or int(design.get("minimum_successes_per_stratum") or 0) != 2
        or int(design.get("model_calls_per_task") or 0) != 2
        or design.get("hidden_evaluation_after_candidate_seal") is not True
    ):
        faults.append("adequacy_design_invalid")
    authority = p2a.mapping(value.get("authority"))
    if authority.get("local_model_calls_authorized_after_green_audit") is not True:
        faults.append("local_model_authority_missing")
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
        required_snapshot_manifest_sha256=str(
            frozen.get("snapshot_manifest_sha256") or ""
        ),
    )
    identity = p2a.mapping(contract.get("identity"))
    if contract.get("ready") is not True:
        faults.append("frozen_model_contract_not_ready")
    if identity.get("identity_sha256") != frozen.get("identity_sha256"):
        faults.append("frozen_model_identity_mismatch")
    if identity.get("decoder_sha256") != frozen.get("decoder_sha256"):
        faults.append("frozen_decoder_mismatch")
    return {
        "policy": "project_theseus_semantic_ir_production_adequacy_campaign_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "sealed_task_count": len(p2a.dicts(pool.get("rows"))),
        "frozen_model_contract": contract,
        "candidate_generation_opened": False,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def run_campaign(
    config_path: Path,
    out_path: Path,
    *,
    session_factory: Callable[..., Any] = local_completion.persistent_v2_session,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_audit = audit_config(config_path)
    if config_audit.get("trigger_state") != "GREEN":
        return blocked_report(
            config_path, config_audit, [], ["campaign_config_audit_red"], started
        )
    config = p2a.read_json(config_path)
    pool = p2a.read_json(p2a.resolve(str(config.get("sealed_task_pool") or "")))
    base = p2a.read_json(p2a.resolve(str(config.get("base_local_instrument") or "")))
    binding = p2a.mapping(base.get("runtime_binding"))
    frozen = p2a.mapping(base.get("frozen_model"))
    session = session_factory(
        worker_config_path=p2a.resolve(str(binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(binding.get("runtime_preflight") or "")),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(
            frozen.get("snapshot_manifest_sha256") or ""
        ),
        session_id="semantic-ir-production-adequacy-panel",
        completion_predicate=production.complete,
    )
    if not session.ready:
        return blocked_report(
            config_path,
            config_audit,
            [],
            ["persistent_backend_not_ready:" + ",".join(session.faults)],
            started,
        )

    rows: list[dict[str, Any]] = []
    infrastructure_faults: list[str] = []
    runtime_config = str(base.get("runtime_config") or "")
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for pool_row in p2a.dicts(pool.get("rows")):
            row = run_task(pool_row, runtime_config)
            rows.append(row)
            if row.get("infrastructure_faults"):
                infrastructure_faults.extend(p2a.strings(row.get("infrastructure_faults")))
            running = campaign_report(
                config_path,
                config_audit,
                rows,
                infrastructure_faults,
                started,
                terminal=False,
            )
            p2a.write_json(out_path, running)
            if infrastructure_faults:
                break
    return campaign_report(
        config_path,
        config_audit,
        rows,
        infrastructure_faults,
        started,
        terminal=len(rows) == 18 and not infrastructure_faults,
    )


def run_task(pool_row: dict[str, Any], runtime_config: str) -> dict[str, Any]:
    index = int(pool_row.get("index") or 0)
    task_path = p2a.resolve(str(pool_row.get("task_manifest") or ""))
    packet_path = p2a.resolve(str(pool_row.get("candidate_packet") or ""))
    task = p2a.read_json(task_path)
    packet = p2a.read_json(packet_path)
    prompt = str(packet.get("serialized_prompt") or "")
    with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-run-{index:02d}-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        first = p2a.runtime_call(
            p2a.route_integrity.DIRECT_MODE,
            f"semantic_ir_adequacy_{index:02d}",
            1,
            prompt,
            MODEL_CONTEXT_TOKENS,
            runtime_config,
        )
        first_candidate = production.parse(first["assistant_text"], task, root)
        first_verification = verify_candidate(task, p2a.dicts(first_candidate.get("actions")))
        repair_prompt = production.render_repair_prompt(
            prompt,
            first["assistant_text"],
            p2a.strings(first_candidate.get("faults")),
            first_verification,
        )
        second = p2a.runtime_call(
            p2a.route_integrity.DIRECT_MODE,
            f"semantic_ir_adequacy_{index:02d}",
            2,
            repair_prompt,
            MODEL_CONTEXT_TOKENS,
            runtime_config,
        )
        final_candidate = production.parse(second["assistant_text"], task, root)
        actions = p2a.dicts(final_candidate.get("actions"))
        final_verification = verify_candidate(task, actions)

    route_receipts = [
        p2a.mapping(p2a.mapping(call.get("runtime_report")).get("route_integrity"))
        for call in (first, second)
    ]
    telemetry = p4r.termination_telemetry([{
        "arm_id": f"semantic_ir_adequacy_{index:02d}",
        "runtime_calls": [first["receipt"], second["receipt"]],
    }])
    infrastructure_faults: list[str] = []
    if not all(
        receipt.get("ready") is True and receipt.get("release_allowed") is True
        for receipt in route_receipts
    ):
        infrastructure_faults.append(f"route_integrity_red:task_{index:02d}")
    if (
        len(telemetry) != 2
        or any(
            row.get("termination_reason") not in {"parser_complete", "model_eos"}
            or row.get("safety_ceiling_hit") is True
            for row in telemetry
        )
    ):
        infrastructure_faults.append(f"completion_custody_red:task_{index:02d}")
    changed = p2a.strings(final_verification.get("changed_paths"))
    authorized = bool(changed) and set(changed).issubset(
        set(p2a.strings(task.get("allowed_effect_paths")))
    )
    parse_apply_visible = bool(actions) and not final_candidate.get("faults") and not final_verification.get("apply_faults") and p2a.mapping(
        final_verification.get("visible_verifier")
    ).get("passed") is True
    candidate_payload = {
        "actions": actions,
        "changed_paths": changed,
        "final_inventory_sha256": final_verification.get("final_inventory_sha256"),
        "semantic_receipt": final_candidate.get("semantic_receipt"),
    }
    candidate_seal = {
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(prompt),
        "first_output_sha256": p2a.sha256_text(first["assistant_text"]),
        "final_output_sha256": p2a.sha256_text(second["assistant_text"]),
        "candidate_payload_sha256": p2a.stable_hash(candidate_payload),
        "sealed_before_hidden_evaluation": True,
    }
    return {
        "index": index,
        "opaque_task_id": task.get("opaque_task_id"),
        "task_manifest": p2a.rel(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_mechanics_ready_for_hidden_evaluation": (
            not infrastructure_faults and parse_apply_visible and authorized
        ),
        "parse_faults": p2a.strings(final_candidate.get("faults")),
        "apply_faults": p2a.strings(final_verification.get("apply_faults")),
        "visible_verifier_passed": p2a.mapping(
            final_verification.get("visible_verifier")
        ).get("passed") is True,
        "authorized_effect": authorized,
        "candidate_payload": candidate_payload,
        "candidate_seal": candidate_seal,
        "model_outputs": {
            "first": first["assistant_text"],
            "final": second["assistant_text"],
        },
        "runtime_calls": [first["receipt"], second["receipt"]],
        "route_integrity_rounds": route_receipts,
        "termination_telemetry": telemetry,
        "infrastructure_faults": infrastructure_faults,
        "hidden_evaluator_executions": 0,
    }


def verify_candidate(task: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-adequacy-verify-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(root)
        apply_faults = production.apply_actions(root, actions) if actions else []
        changed = p2a.changed_paths(baseline, p2a.inventory(root)) if not apply_faults else []
        visible = p2a.run_visible_verifier(root, task) if actions and not apply_faults else {}
        return {
            "apply_faults": apply_faults,
            "changed_paths": changed,
            "visible_verifier": visible,
            "final_inventory_sha256": p2a.stable_hash(p2a.inventory(root)),
        }


def campaign_report(
    config_path: Path,
    config_audit: dict[str, Any],
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
        else "BLOCKED_INFRASTRUCTURE_RESUMABLE"
        if infrastructure_faults
        else "RUNNING_CANDIDATE_GENERATION"
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if terminal else "RED" if infrastructure_faults else "YELLOW",
        "state": state,
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "config_audit": config_audit,
        "completed_task_count": len(rows),
        "expected_task_count": 18,
        "rows": rows,
        "faults": sorted(set(infrastructure_faults)),
        "hidden_evaluation_opened": False,
        "counters": {
            **zero_counters(),
            "candidate_or_control_calls": calls,
            "local_model_calls": calls,
        },
        "maximum_inference": (
            "Candidate generation and syntax-visible mechanics only. Hidden evaluator results, "
            "implementation adequacy, subsystem effects, D1, D2, training value, serving, and "
            "book support remain unestablished until independent scoring."
        ),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def blocked_report(
    config_path: Path,
    config_audit: dict[str, Any],
    rows: list[dict[str, Any]],
    faults: list[str],
    started: float,
) -> dict[str, Any]:
    return campaign_report(
        config_path, config_audit, rows, faults, started, terminal=False
    )


def zero_counters() -> dict[str, int]:
    return {
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "hidden_evaluator_executions": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
