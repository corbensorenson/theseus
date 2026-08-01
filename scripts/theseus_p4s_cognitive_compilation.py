#!/usr/bin/env python3
"""Fresh P4 decision runner with mechanics-qualified labeled Semantic IR."""

from __future__ import annotations

import argparse
import ast
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
import theseus_local_inference_backend as local_backend  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_cognitive_compilation_repaired as p4r  # noqa: E402
import theseus_semantic_ir_v2r1 as ir_v2r1  # noqa: E402


POLICY = "project_theseus_p4s_cognitive_compilation_run_v1"
INSTRUMENT_POLICY = "project_theseus_p4s_cognitive_compilation_instrument_v1"
MODEL_CONTEXT_TOKENS = 262144
SEMANTIC_PROMPT_MARKER = "[P4S_LABELED_SEMANTIC_IR_TREATMENT]"
SEMANTIC_SCOPE_NODE_TYPES = {
    "FunctionDef", "AsyncFunctionDef", "Assign", "AnnAssign",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument", default="configs/theseus_p4s_cognitive_compilation_instrument.json"
    )
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--out", default="reports/theseus_p4s_cognitive_compilation_run.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    instrument_path = p2a.resolve(args.instrument)
    report = (
        audit_instrument(instrument_path)
        if args.audit_only else run_experiment(instrument_path, p2a.resolve(args.task))
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
    if value.get("state") != "PROSPECTIVELY_BOUND_BEFORE_FRESH_P4S_TASK_ACQUISITION":
        faults.append("instrument_not_prospectively_bound")
    for name, digest_name in (
        ("p4r_terminal_disposition", "p4r_terminal_disposition_sha256"),
        ("mechanics_qualification", "mechanics_qualification_sha256"),
        ("generation_completion_policy", "generation_completion_policy_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
    ):
        owner = p2a.resolve(str(value.get(name) or ""))
        if p2a.sha256_file(owner) != str(value.get(digest_name) or ""):
            faults.append(f"{name}_digest_mismatch")
    disposition = p2a.read_json(
        p2a.resolve(str(value.get("p4r_terminal_disposition") or ""))
    )
    if disposition.get("scientific_status") != "INCONCLUSIVE_IMPLEMENTATION":
        faults.append("p4r_terminal_disposition_invalid")
    mechanics = p2a.read_json(
        p2a.resolve(str(value.get("mechanics_qualification") or ""))
    )
    if mechanics.get("state") != "INTERVENTION_AND_DEPENDENCY_LOCAL_REPAIR_MECHANICS_GREEN":
        faults.append("mechanics_qualification_not_green")
    for key in (
        "first_verified", "injected_feedback_verified", "final_verified",
        "dependency_local_repairs",
    ):
        if mechanics.get(key) != "2/2":
            faults.append(f"mechanics_floor_invalid:{key}")
    if int(mechanics.get("safety_ceiling_hits") or 0) != 0:
        faults.append("mechanics_context_boundary_hit")
    base_path = p2a.resolve(str(value.get("base_local_instrument") or ""))
    base = p2a.read_json(base_path)
    runtime = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(
        runtime.get("local_inference")
    )
    frozen = p2a.mapping(base.get("frozen_model"))
    contract = p4r.route_integrity.load_model_contract(
        str(binding.get("worker_config") or ""),
        str(binding.get("runtime_preflight") or ""),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
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
    if int(frozen.get("model_declared_context_window_tokens") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("model_context_window_binding_invalid")
    harness = p2a.mapping(value.get("harness"))
    for name in ("candidate_runner", "blind_evaluator", "local_backend", "semantic_parser"):
        owner = p2a.resolve(str(harness.get(name) or ""))
        if p2a.sha256_file(owner) != str(harness.get(f"{name}_sha256") or ""):
            faults.append(f"{name}_digest_mismatch")
    matched = p2a.mapping(value.get("matched_arm_contract"))
    if tuple(p2a.strings(matched.get("arms"))) != p4.ARMS:
        faults.append("arm_set_invalid")
    for key in (
        "same_frozen_weights", "same_information", "same_completion_policy",
        "same_model_context_residual_rule", "same_two_model_calls",
        "same_verifier_and_effect_sandbox", "same_semantic_scope_symbol_table",
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
    if int(budgets.get("maximum_generation_tokens_per_call") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("legacy_context_transport_invalid")
    fresh = p2a.mapping(value.get("fresh_task_pool_contract"))
    if fresh.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    if fresh.get("task_count") != 10 or fresh.get("distinct_repository_count") != 10:
        faults.append("fresh_task_pool_shape_invalid")
    if fresh.get("source_disjoint_from_all_consumed_P2_P3_P4_P4R") is not True:
        faults.append("source_disjointness_not_required")
    return {
        "policy": "project_theseus_p4s_cognitive_compilation_instrument_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": p2a.sha256_file(path),
        "completion_model_contract": contract,
        "mechanics_qualification": {
            "state": mechanics.get("state"),
            "report_sha256": p2a.sha256_file(
                p2a.resolve(str(value.get("mechanics_qualification") or ""))
            ),
        },
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": p2a.zero_counters(),
    }


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    original_audit = p4r.audit_instrument
    original_symbol_table = p4.semantic_symbol_table
    original_render = p4.render_arm_prompt
    original_parse = p4.parse_arm_output
    original_final = p4.render_final_prompt
    original_run_arm = p4.run_arm
    original_complete = p4r.completion.candidate_envelope_complete

    def bound_run_arm(
        arm: str, instrument: dict[str, Any], task: dict[str, Any], session: Any
    ) -> dict[str, Any]:
        row = original_run_arm(arm, instrument, task, session)
        if arm == p4.SEMANTIC and row.get("parseable_candidate") is True:
            candidate = p2a.mapping(row.get("candidate"))
            candidate["protocol"] = "theseus_semantic_ir_v2r1_labeled"
            row["candidate"] = candidate
        return row

    p4r.audit_instrument = audit_instrument
    p4.semantic_symbol_table = semantic_scope_symbol_table
    p4.render_arm_prompt = render_arm_prompt
    p4.parse_arm_output = parse_arm_output
    p4.render_final_prompt = render_final_prompt
    p4.run_arm = bound_run_arm
    p4r.completion.candidate_envelope_complete = candidate_envelope_complete
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
    report["policy"] = POLICY
    report["scope"] = (
        "Fresh source-disjoint P4 cognitive-compilation decision development only; "
        "no D1, D2, serving, training, or automatic book-support authority."
    )
    report["semantic_treatment"] = {
        "transport": "theseus_semantic_ir_v2r1_labeled",
        "semantic_unit_policy": "complete functions and module/class assignments",
        "mechanics_qualification": "reports/theseus_p4r_intervention_locality_canary_r1.json",
    }
    return report


def candidate_envelope_complete(text: str) -> bool:
    return completion.candidate_envelope_complete(text) or ir_v2r1.complete(text)


def semantic_scope_symbol_table(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    table = ORIGINAL_SEMANTIC_SYMBOL_TABLE(root, task)
    eligible: dict[str, set[tuple[str, int, int]]] = {}
    for path in p2a.strings(task.get("allowed_effect_paths")):
        source_path = p2a.checked_source_path(root, path)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            node_type = type(node).__name__
            if node_type not in SEMANTIC_SCOPE_NODE_TYPES or not hasattr(node, "lineno"):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and not isinstance(
                parents.get(node), (ast.Module, ast.ClassDef)
            ):
                continue
            eligible.setdefault(path, set()).add((
                node_type,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno) or node.lineno),
            ))
    nodes = [
        row for row in p2a.dicts(table.get("nodes"))
        if (
            str(row.get("node_type") or ""),
            int(row.get("start_line") or 0),
            int(row.get("end_line") or 0),
        ) in eligible.get(str(row.get("path") or ""), set())
    ]
    return {**table, "nodes": nodes, "semantic_unit_policy": "p4s_complete_scope_v1"}


ORIGINAL_SEMANTIC_SYMBOL_TABLE = p4.semantic_symbol_table
ORIGINAL_RENDER_ARM_PROMPT = p4.render_arm_prompt
ORIGINAL_PARSE_ARM_OUTPUT = p4.parse_arm_output
ORIGINAL_RENDER_FINAL_PROMPT = p4.render_final_prompt


def render_arm_prompt(
    arm: str, task: dict[str, Any], common: str, protocol: dict[str, Any]
) -> str:
    if arm != p4.SEMANTIC:
        return ORIGINAL_RENDER_ARM_PROMPT(arm, task, common, protocol)
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
        f"{ir_v2r1.HEADER}\n"
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
        return ir_v2r1.parse(text, task, root)
    return ORIGINAL_PARSE_ARM_OUTPUT(arm, text, task, root, protocol)


def render_final_prompt(
    original: str,
    first_output: str,
    first_candidate: dict[str, Any],
    verification: dict[str, Any],
    implicated: set[str],
) -> str:
    if not original.startswith(SEMANTIC_PROMPT_MARKER):
        return ORIGINAL_RENDER_FINAL_PROMPT(
            original, first_output, first_candidate, verification, implicated
        )
    feedback = {
        "parse_or_lower_faults": p2a.strings(first_candidate.get("faults")),
        "apply_faults": p2a.strings(verification.get("apply_faults")),
        "visible_verifier_returncode": p2a.mapping(
            verification.get("visible_verifier")
        ).get("returncode"),
        "visible_verifier_stdout_tail": str(p2a.mapping(
            verification.get("visible_verifier")
        ).get("stdout_tail") or "")[-1000:],
        "visible_verifier_stderr_tail": str(p2a.mapping(
            verification.get("visible_verifier")
        ).get("stderr_tail") or "")[-1000:],
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
