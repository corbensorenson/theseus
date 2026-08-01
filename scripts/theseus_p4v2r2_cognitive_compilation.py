#!/usr/bin/env python3
"""Prospective fresh P4 runner for TMax with Semantic-IR v2r2."""

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
import theseus_assistant_runtime as assistant_runtime  # noqa: E402
import theseus_generation_completion as completion  # noqa: E402
import theseus_local_inference_backend as backend_v1  # noqa: E402
import theseus_local_inference_backend_v2 as backend_v2  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_cognitive_compilation_repaired as p4r  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402
import theseus_semantic_ir_v2r2 as ir_v2r2  # noqa: E402


POLICY = "project_theseus_p4v2r2_cognitive_compilation_run_v1"
INSTRUMENT_POLICY = "project_theseus_p4v2r2_cognitive_compilation_instrument_v1"
MODEL_CONTEXT_TOKENS = 262144
SEMANTIC_PROMPT_MARKER = "[P4V2R2_LABELED_SEMANTIC_IR_TREATMENT]"
BASE_CANDIDATE_ENVELOPE_COMPLETE = completion.candidate_envelope_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument",
        default="configs/theseus_p4v2r2_cognitive_compilation_instrument.json",
    )
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--out", default="reports/theseus_p4v2r2_cognitive_compilation_run.json"
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
    print(
        json.dumps(
            {
                "policy": report.get("policy"),
                "trigger_state": report.get("trigger_state"),
                "faults": report.get("faults"),
                "denominators": report.get("denominators"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_instrument(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != INSTRUMENT_POLICY:
        faults.append("instrument_policy_invalid")
    if value.get("state") not in {
        "PROSPECTIVELY_BOUND_BEFORE_FRESH_P4V2R2_TASK_ACQUISITION",
        "PROSPECTIVELY_RESEALED_AFTER_ROUTE_IMPLEMENTATION_FAILURE",
    }:
        faults.append("instrument_not_prospectively_bound")
    if value.get("runtime_attempt_namespace") not in {
        "p4v2r2_attempt1",
        "p4v2r2r1_attempt1",
    }:
        faults.append("runtime_attempt_namespace_invalid")

    for name, digest_name in (
        ("p4s_terminal_disposition", "p4s_terminal_disposition_sha256"),
        ("v2r2_mechanics_binding", "v2r2_mechanics_binding_sha256"),
        ("generation_completion_policy", "generation_completion_policy_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
    ):
        owner = p2a.resolve(str(value.get(name) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(
            value.get(digest_name) or ""
        ):
            faults.append(f"{name}_digest_mismatch")

    disposition = p2a.read_json(
        p2a.resolve(str(value.get("p4s_terminal_disposition") or ""))
    )
    if disposition.get("scientific_status") != "INCONCLUSIVE_IMPLEMENTATION":
        faults.append("p4s_terminal_disposition_invalid")
    if p2a.mapping(disposition.get("next_stage")).get("D1_eligible") is not False:
        faults.append("p4s_D1_boundary_invalid")

    mechanics = p2a.read_json(
        p2a.resolve(str(value.get("v2r2_mechanics_binding") or ""))
    )
    if mechanics.get("state") != "PROSPECTIVE_LIST_NORMALIZATION_MECHANICS_GREEN":
        faults.append("v2r2_mechanics_not_green")
    invariants = p2a.mapping(mechanics.get("hard_invariants"))
    required_invariants = (
        "identifier_order_changed",
        "duplicates_removed",
        "replacement_source_touched",
        "path_node_operation_source_digest_touched",
        "p4s_scores_recomputed",
        "p4s_disposition_modified",
    )
    if any(invariants.get(key) is not False for key in required_invariants):
        faults.append("v2r2_noninterference_invalid")
    if int(invariants.get("identifier_values_invented") or 0) != 0:
        faults.append("v2r2_identifier_invention_invalid")
    if invariants.get("ambiguous_or_executable_surface_rejected") is not True:
        faults.append("v2r2_ambiguous_surface_guard_missing")

    base_path = p2a.resolve(str(value.get("base_local_instrument") or ""))
    base = p2a.read_json(base_path)
    runtime = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(
        runtime.get("local_inference")
    )
    frozen = p2a.mapping(base.get("frozen_model"))
    contract = backend_v2.route_integrity.load_model_contract(
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
        faults.append("model_contract_not_ready")
    if identity.get("identity_sha256") != frozen.get("identity_sha256"):
        faults.append("model_identity_mismatch")
    if identity.get("decoder_sha256") != frozen.get("decoder_sha256"):
        faults.append("decoder_identity_mismatch")
    if frozen.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(frozen.get("model_declared_context_window_tokens") or 0) != (
        MODEL_CONTEXT_TOKENS
    ):
        faults.append("model_context_window_binding_invalid")

    harness = p2a.mapping(value.get("harness"))
    for name in (
        "candidate_runner",
        "blind_evaluator",
        "local_backend",
        "semantic_parser",
    ):
        owner = p2a.resolve(str(harness.get(name) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(
            harness.get(f"{name}_sha256") or ""
        ):
            faults.append(f"{name}_digest_mismatch")

    matched = p2a.mapping(value.get("matched_arm_contract"))
    if tuple(p2a.strings(matched.get("arms"))) != p4.ARMS:
        faults.append("arm_set_invalid")
    for key in (
        "same_frozen_weights",
        "same_information",
        "same_completion_policy",
        "same_model_context_residual_rule",
        "same_two_model_calls",
        "same_verifier_and_effect_sandbox",
        "same_semantic_scope_symbol_table",
    ):
        if matched.get(key) is not True:
            faults.append(f"matched_contract_false:{key}")
    generation = p2a.mapping(value.get("generation_budget"))
    if generation.get("project_selected_quality_token_cap") is not None:
        faults.append("instrument_quality_token_cap_present")
    if generation.get("ceiling_hit_invalidates_observation") is not True:
        faults.append("ceiling_hit_disposition_invalid")
    budgets = p2a.mapping(value.get("budgets"))
    if int(budgets.get("task_count") or 0) != 10:
        faults.append("task_count_invalid")
    if int(budgets.get("model_calls_per_learned_arm") or 0) != 2:
        faults.append("learned_arm_call_count_invalid")
    if int(budgets.get("maximum_generation_tokens_per_call") or 0) != (
        MODEL_CONTEXT_TOKENS
    ):
        faults.append("model_context_transport_invalid")
    fresh = p2a.mapping(value.get("fresh_task_pool_contract"))
    if fresh.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    if fresh.get("task_count") != 10 or fresh.get("distinct_repository_count") != 10:
        faults.append("fresh_task_pool_shape_invalid")
    if fresh.get("source_disjoint_from_all_consumed_P2_through_P4S") is not True:
        faults.append("source_disjointness_not_required")

    return {
        "policy": "project_theseus_p4v2r2_instrument_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": p2a.sha256_file(path),
        "completion_model_contract": contract,
        "v2r2_mechanics": {
            "state": mechanics.get("state"),
            "binding_sha256": p2a.sha256_file(
                p2a.resolve(str(value.get("v2r2_mechanics_binding") or ""))
            ),
        },
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": p2a.zero_counters(),
    }


def persistent_v2_session(**kwargs: Any) -> backend_v1.PersistentLocalInferenceSession:
    """Reuse one model load while applying the v2 exact-context decoder."""

    completion_predicate = kwargs.get("completion_predicate")
    supplied_factory = kwargs.pop("model_factory", None)

    def model_factory(card: dict[str, Any], snapshot: Path, maximum: int) -> Any:
        if supplied_factory is not None:
            return supplied_factory(card, snapshot, maximum)
        return backend_v2.LocalMlxChatModel(
            card,
            snapshot,
            maximum,
            completion_predicate=completion_predicate,
        )

    session = backend_v1.PersistentLocalInferenceSession(
        **kwargs, model_factory=model_factory
    )
    contract = backend_v2.route_integrity.load_model_contract(
        kwargs["worker_config_path"],
        kwargs["runtime_preflight_path"],
        maximum_tokens=int(kwargs["maximum_tokens"]),
        required_repo_id=str(kwargs.get("required_repo_id") or ""),
        required_revision=str(kwargs.get("required_revision") or ""),
        required_snapshot_manifest_sha256=str(
            kwargs.get("required_snapshot_manifest_sha256") or ""
        ),
    )
    session.contract = contract
    session.identity = dict(contract.get("identity") or {})
    session.faults = sorted(set(session.faults + list(contract.get("faults") or [])))
    original_generate_report = session.generate_report

    def generate_report(**request: Any) -> dict[str, Any]:
        report = original_generate_report(**request)
        report["policy"] = backend_v2.BACKEND_POLICY
        response = p2a.mapping(report.get("response"))
        evidence = p2a.mapping(response.get("evidence"))
        evidence["backend_policy"] = backend_v2.BACKEND_POLICY
        response["evidence"] = evidence
        report["response"] = response
        metrics = p2a.mapping(report.get("metrics"))
        if metrics.get("physical_context_boundary_hit") is True:
            report["faults"] = sorted(
                set(
                    p2a.strings(report.get("faults"))
                    + ["instrument_inadequate_generation_boundary_hit"]
                )
            )
            report["trigger_state"] = "RED"
            response["answer"] = ""
        return report

    session.generate_report = generate_report
    return session


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = persistent_v2_session,
) -> dict[str, Any]:
    original_audit = p4r.audit_instrument
    original_symbol_table = p4.semantic_symbol_table
    original_render = p4.render_arm_prompt
    original_parse = p4.parse_arm_output
    original_final = p4.render_final_prompt
    original_run_arm = p4.run_arm
    original_complete = p4r.completion.candidate_envelope_complete
    original_runtime_call = p4.p2a.runtime_call
    execution_instrument = p2a.read_json(instrument_path)
    namespace = str(execution_instrument.get("runtime_attempt_namespace") or "")

    def bound_run_arm(
        arm: str, instrument: dict[str, Any], task: dict[str, Any], session: Any
    ) -> dict[str, Any]:
        row = original_run_arm(arm, instrument, task, session)
        if arm == p4.SEMANTIC and row.get("parseable_candidate") is True:
            candidate = p2a.mapping(row.get("candidate"))
            candidate["protocol"] = "theseus_semantic_ir_v2r2_labeled"
            row["candidate"] = candidate
        return row

    p4r.audit_instrument = audit_instrument
    p4.semantic_symbol_table = p4s.semantic_scope_symbol_table
    p4.render_arm_prompt = render_arm_prompt
    p4.parse_arm_output = parse_arm_output
    p4.render_final_prompt = render_final_prompt
    p4.run_arm = bound_run_arm
    p4r.completion.candidate_envelope_complete = candidate_envelope_complete
    namespaced_runtime_call = p4s.bind_runtime_attempt_namespace(
        original_runtime_call, namespace
    )

    def route_guarded_runtime_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = namespaced_runtime_call(*args, **kwargs)
        runtime_report = p2a.mapping(result.get("runtime_report"))
        if runtime_report:
            route_receipt = p2a.mapping(runtime_report.get("route_integrity"))
            if (
                runtime_report.get("trigger_state") != "GREEN"
                or route_receipt.get("ready") is not True
                or route_receipt.get("release_allowed") is not True
            ):
                raise RuntimeError("p4v2r2_route_integrity_release_failed")
        return result

    p4.p2a.runtime_call = route_guarded_runtime_call
    try:
        report = p4r.run_experiment(
            instrument_path, task_path, session_factory=session_factory
        )
    finally:
        p4r.audit_instrument = original_audit
        p4.semantic_symbol_table = original_symbol_table
        p4.render_arm_prompt = original_render
        p4.parse_arm_output = original_parse
        p4.render_final_prompt = original_final
        p4.run_arm = original_run_arm
        p4r.completion.candidate_envelope_complete = original_complete
        p4.p2a.runtime_call = original_runtime_call
    report["policy"] = POLICY
    report["scope"] = (
        "Fresh source-disjoint P4 cognitive-compilation development only; no D1, "
        "D2, serving, training, or automatic book-support authority."
    )
    report["semantic_treatment"] = {
        "transport": "theseus_semantic_ir_v2r2_labeled",
        "semantic_unit_policy": "complete functions and module/class assignments",
        "mechanics_binding": "configs/theseus_semantic_ir_v2r2_mechanics.json",
    }
    return report


def candidate_envelope_complete(text: str) -> bool:
    return BASE_CANDIDATE_ENVELOPE_COMPLETE(text) or ir_v2r2.complete(text)


def render_arm_prompt(
    arm: str, task: dict[str, Any], common: str, protocol: dict[str, Any]
) -> str:
    if arm != p4.SEMANTIC:
        return p4s.ORIGINAL_RENDER_ARM_PROMPT(arm, task, common, protocol)
    return (
        f"{SEMANTIC_PROMPT_MARKER}\n"
        f"Implement this repository task: {task.get('natural_request')}\n\n"
        "Use the information-matched obligations, dependencies, semantic-scope symbol "
        "table, and repository context below. Choose the least-sufficient complete "
        "FunctionDef, AsyncFunctionDef, or module/class Assign nodes. Return only one "
        "complete labeled Semantic IR artifact. Copy SOURCE, PATH, NODE, and NODE_SHA "
        "identities exactly. Use dependency-closed obligation groups, stable unique UNIT "
        "IDs, explicit LOSS, and replacement source scoped exactly to each selected node.\n\n"
        + common
        + "\n\nOUTPUT ONLY THIS LABELED SHAPE; repeat UNIT blocks only when independently necessary:\n"
        f"{ir_v2r2.HEADER}\n"
        "SOURCE <copy exact semantic source digest>\n"
        "ALL_OBLIGATIONS <copy all obligation ids in exact order>\n"
        "UNIT U1\n"
        "OBLIGATIONS <dependency-closed obligation ids covered by U1>\n"
        "OP REPLACE\n"
        "PATH <copy selected path>\n"
        "NODE <copy selected semantic-scope node id>\n"
        "NODE_SHA <copy selected node sha256>\n"
        "<<<\n"
        "<complete replacement source for only the selected semantic-scope node>\n"
        ">>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )


def parse_arm_output(
    arm: str, text: str, task: dict[str, Any], root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    if arm == p4.SEMANTIC:
        return ir_v2r2.parse(text, task, root)
    return p4s.ORIGINAL_PARSE_ARM_OUTPUT(arm, text, task, root, protocol)


def render_final_prompt(
    original: str,
    first_output: str,
    first_candidate: dict[str, Any],
    verification: dict[str, Any],
    implicated: set[str],
) -> str:
    if not original.startswith(SEMANTIC_PROMPT_MARKER):
        return p4s.ORIGINAL_RENDER_FINAL_PROMPT(
            original, first_output, first_candidate, verification, implicated
        )
    feedback = {
        "parse_or_lower_faults": p2a.strings(first_candidate.get("faults")),
        "apply_faults": p2a.strings(verification.get("apply_faults")),
        "visible_verifier_returncode": p2a.mapping(
            verification.get("visible_verifier")
        ).get("returncode"),
        "visible_verifier_stdout_tail": str(
            p2a.mapping(verification.get("visible_verifier")).get("stdout_tail") or ""
        )[-1000:],
        "visible_verifier_stderr_tail": str(
            p2a.mapping(verification.get("visible_verifier")).get("stderr_tail") or ""
        )[-1000:],
        "dependency_local_repair_obligation_ids": sorted(implicated),
    }
    return (
        original
        + "\n\n[PROVISIONAL_LABELED_SEMANTIC_IR]\n"
        + str(first_output or "")[-16000:]
        + "\n\n[ACTUAL_TARGET_VALIDATION_FEEDBACK]\n"
        + json.dumps(feedback, sort_keys=True)
        + "\n\nReturn only one complete corrected labeled Semantic IR artifact against the "
        "ORIGINAL snapshot. Repair only units whose obligation references are within "
        "the dependency-local repair set; copy every unrelated unit byte-for-byte. "
        "Do not emit a delta, plan, JSON, Markdown, or commentary."
    )


if __name__ == "__main__":
    raise SystemExit(main())
