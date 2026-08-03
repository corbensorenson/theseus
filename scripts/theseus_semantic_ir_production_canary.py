#!/usr/bin/env python3
"""Run one frozen-model, source-disjoint Semantic-IR production canary.

This is a mechanics observation, not a matched subsystem-effect experiment.
It uses the real local route, parser-complete/EOS termination, deterministic
lowerer, disposable effect sandbox, and visible verifier.  No hidden evaluator,
external inference, training admission, D1, or D2 is opened.
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
import theseus_p4_cognitive_compilation as p4
import theseus_p4_cognitive_compilation_repaired as p4r
import theseus_p4s_cognitive_compilation as p4s
import theseus_p4v2r2_cognitive_compilation as local_completion
import theseus_semantic_ir_production as production


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production.json"
DEFAULT_TASK = ROOT / "configs" / "theseus_semantic_ir_production_non_claim_task_01.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_non_claim_canary.json"
POLICY = "project_theseus_semantic_ir_production_non_claim_canary_v1"
CONFIG_POLICY = "project_theseus_semantic_ir_production_v1"
MODEL_CONTEXT_TOKENS = 262144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--task", default=p2a.rel(DEFAULT_TASK))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = (
        audit_config(config_path)
        if args.audit_only
        else run_canary(config_path, p2a.resolve(args.task))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "mechanics": report.get("mechanics"),
        "counters": report.get("counters"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_config(path: Path) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "PRODUCTION_MECHANICS_REPAIR_ACTIVE":
        faults.append("config_state_invalid")
    for path_key, hash_key in (
        ("production_owner", "production_owner_sha256"),
        ("conformance_report", "conformance_report_sha256"),
        ("non_claim_task", "non_claim_task_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
        ("candidate_runner", "candidate_runner_sha256"),
    ):
        owner = p2a.resolve(str(value.get(path_key) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(
            value.get(hash_key) or ""
        ):
            faults.append(f"binding_invalid:{path_key}")
    conformance_path = p2a.resolve(str(value.get("conformance_report") or ""))
    conformance = p2a.read_json(conformance_path) if conformance_path.is_file() else {}
    if conformance.get("trigger_state") != "GREEN":
        faults.append("production_conformance_not_green")
    completion = p2a.mapping(value.get("generation_completion"))
    if completion.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if completion.get("normal_completion") != ["parser_complete", "model_eos"]:
        faults.append("normal_completion_invalid")
    authority = p2a.mapping(value.get("authority"))
    if any(authority.get(key) is not False for key in (
        "external_inference_authorized",
        "teacher_calls_authorized",
        "training_rows_authorized",
        "serving_authorized",
        "hidden_evaluator_authorized",
        "D1_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
    )):
        faults.append("cross_stage_authority_present")
    return {
        "policy": "project_theseus_semantic_ir_production_config_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "counters": zero_counters(),
    }


def run_canary(
    config_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = local_completion.persistent_v2_session,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_audit = audit_config(config_path)
    task_audit = p4.audit_task(task_path)
    if config_audit.get("trigger_state") != "GREEN" or task_audit.get("trigger_state") != "GREEN":
        return red("config_or_task_audit_red", config_audit, task_audit, started)

    config = p2a.read_json(config_path)
    task = p2a.read_json(task_path)
    base = p2a.read_json(p2a.resolve(str(config.get("base_local_instrument") or "")))
    runtime_config = str(base.get("runtime_config") or "")
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
        session_id="semantic-production-non-claim-01-attempt1",
        completion_predicate=production.complete,
    )
    if not session.ready:
        return red(
            "persistent_backend_not_ready:" + ",".join(session.faults),
            config_audit,
            task_audit,
            started,
        )

    with tempfile.TemporaryDirectory(prefix="theseus-semantic-production-canary-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(root)
        symbols = p4s.semantic_scope_symbol_table(root, task)
        common = render_common_context(root, task, symbols)
        first_prompt = production.render_prompt(task, common)
        task_id = "semantic_production_non_claim_01_attempt1"
        call_receipts: list[dict[str, Any]] = []
        route_receipts: list[dict[str, Any]] = []
        with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
            first = p2a.runtime_call(
                p2a.route_integrity.DIRECT_MODE,
                task_id,
                1,
                first_prompt,
                MODEL_CONTEXT_TOKENS,
                runtime_config,
            )
            call_receipts.append(first["receipt"])
            route_receipts.append(
                p2a.mapping(first.get("runtime_report")).get("route_integrity")
            )
            first_candidate = production.parse(
                first["assistant_text"], task, root
            )
            first_verification = p4.verify_provisional(
                root, baseline, task, first_candidate
            )
            repair_prompt = production.render_repair_prompt(
                first_prompt,
                first["assistant_text"],
                p2a.strings(first_candidate.get("faults")),
                first_verification,
            )
            second = p2a.runtime_call(
                p2a.route_integrity.DIRECT_MODE,
                task_id,
                2,
                repair_prompt,
                MODEL_CONTEXT_TOKENS,
                runtime_config,
            )
            call_receipts.append(second["receipt"])
            route_receipts.append(
                p2a.mapping(second.get("runtime_report")).get("route_integrity")
            )

        final_candidate = production.parse(second["assistant_text"], task, root)
        final_verification = p4.verify_provisional(
            root, baseline, task, final_candidate
        )
        actions = p2a.dicts(final_candidate.get("actions"))
        changed = p2a.strings(final_verification.get("changed_paths"))
        authorized = bool(changed) and set(changed).issubset(
            set(p2a.strings(task.get("allowed_effect_paths")))
        )
        visible_passed = p2a.mapping(
            final_verification.get("visible_verifier")
        ).get("passed") is True
        parse_lower_apply = bool(actions) and not final_candidate.get("faults") and not final_verification.get("apply_faults")
        telemetry = p4r.termination_telemetry([{
            "arm_id": "typed_semantic_ir_production_non_claim",
            "runtime_calls": call_receipts,
        }])
        routes_green = all(
            p2a.mapping(receipt).get("ready") is True
            and p2a.mapping(receipt).get("release_allowed") is True
            for receipt in route_receipts
        )
        termination_green = (
            len(telemetry) == 2
            and all(
                row.get("termination_reason") in {"parser_complete", "model_eos"}
                and row.get("safety_ceiling_hit") is not True
                for row in telemetry
            )
        )
        infrastructure_faults = []
        if not routes_green:
            infrastructure_faults.append("route_integrity_red")
        if not termination_green:
            infrastructure_faults.append("completion_custody_red")
        mechanics_green = parse_lower_apply and authorized and visible_passed
        trigger = "RED" if infrastructure_faults else "GREEN" if mechanics_green else "YELLOW"
        return {
            "policy": POLICY,
            "created_utc": p2a.now(),
            "trigger_state": trigger,
            "faults": infrastructure_faults,
            "scope": "Source-disjoint model-produced non-claim production mechanics only.",
            "config_sha256": p2a.sha256_file(config_path),
            "task_sha256": p2a.sha256_file(task_path),
            "config_audit": config_audit,
            "task_audit": task_audit,
            "model": {
                "repo_id": frozen.get("repo_id"),
                "revision": frozen.get("revision"),
                "identity_sha256": session.identity.get("identity_sha256"),
                "model_loads": session.model_load_count,
            },
            "mechanics": {
                "first_parse_faults": p2a.strings(first_candidate.get("faults")),
                "first_apply_faults": p2a.strings(first_verification.get("apply_faults")),
                "final_parse_faults": p2a.strings(final_candidate.get("faults")),
                "final_apply_faults": p2a.strings(final_verification.get("apply_faults")),
                "final_actions": len(actions),
                "changed_paths": changed,
                "authorized_effects": authorized,
                "visible_verifier_passed": visible_passed,
                "parse_lower_apply_passed": parse_lower_apply,
                "route_integrity_passed": routes_green,
                "natural_completion_passed": termination_green,
                "first_artifact_sha256": p2a.sha256_text(first["assistant_text"]),
                "final_artifact_sha256": p2a.sha256_text(second["assistant_text"]),
                "repair_prompt_contains_complete_first_artifact": first["assistant_text"] in repair_prompt,
            },
            "termination_telemetry": telemetry,
            "runtime_receipts": call_receipts,
            "counters": {
                **zero_counters(),
                "local_model_calls": session.inference_calls,
            },
            "project_selected_quality_token_cap": None,
            "maximum_inference": (
                "This observation decides only whether the frozen local model can traverse "
                "the exact repaired Semantic-IR production path on one project-authored, "
                "source-disjoint non-claim task. It cannot establish a subsystem treatment "
                "effect, task-distribution competence, D1, D2, serving, training, or book support."
            ),
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def render_common_context(
    root: Path, task: dict[str, Any], symbols: dict[str, Any]
) -> str:
    original = p4.semantic_symbol_table
    try:
        p4.semantic_symbol_table = lambda _root, _task: symbols
        return p4.render_common_context(root, task)
    finally:
        p4.semantic_symbol_table = original


def zero_counters() -> dict[str, int]:
    return {
        "local_model_calls": 0,
        "hidden_evaluator_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def red(
    fault: str,
    config_audit: dict[str, Any],
    task_audit: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED",
        "faults": [fault],
        "config_audit": config_audit,
        "task_audit": task_audit,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
