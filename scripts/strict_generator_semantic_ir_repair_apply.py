#!/usr/bin/env python3
"""Apply localized semantic-IR repair obligations to strict-generator outputs.

This is a Phase 13 consumer for ``strict_generator_semantic_ir_repair_bridge``.
It is intentionally not part of learned candidate generation. It reads already
generated private candidates, applies a tiny set of local semantic-IR patch
obligations, then runs the private verifier on the repaired candidates.

The generated model keeps zero learned-generation credit for repaired rows.
Repairs are labeled as deterministic semantic-IR/GVR repair evidence only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import semantic_ir
import verifier_guided_search
import gvr_state_machine
from candidate_integrity import recompute_candidate_integrity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_REPAIR_BRIDGE = ROOT / "reports" / "strict_generator_semantic_ir_repair_bridge.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "strict_generator_mlx_decode_eval_initializer_operation_delay_canary_v9_20260706_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_semantic_ir_repair_apply_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "strict_generator_semantic_ir_repair_apply_v1.md"
DEFAULT_CANDIDATES_OUT = ROOT / "reports" / "strict_generator_semantic_ir_repair_apply_v1_candidates.jsonl"

NO_CHEAT = {
    "public_training_rows": 0,
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
    "fallback_template_router_tool_credit_count": 0,
    "candidate_generation_credit": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--repair-bridge", default=rel(DEFAULT_REPAIR_BRIDGE))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--candidates-out", default=rel(DEFAULT_CANDIDATES_OUT))
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report, repaired = build_report(args, started)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.candidates_out), repaired)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = resolve(args.config)
    bridge_path = resolve(args.repair_bridge)
    candidates_path = resolve(args.candidates)
    config = read_json(config_path)
    bridge = read_json(bridge_path)
    source_candidates = read_jsonl(candidates_path)
    selected = candidate_rows_for_repair(source_candidates, limit=max(1, int(args.max_candidates or 1)))
    source_selected: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    gvr_contract = gvr_state_machine.run_reference_fixture()
    for row in selected:
        source_row = copy.deepcopy(row)
        source_row["semantic_ir"] = semantic_ir.candidate_receipt(
            str(source_row.get("code") or ""),
            learned_prefix_tokens=list(
                dict_or_empty(source_row.get("body_structure_decode")).get("learned_plan_prefix_tokens") or []
            ),
            residual_lineage=[f"candidate:{str(source_row.get('candidate_sha256') or semantic_ir.stable_hash(source_row.get('code') or ''))}"],
        )
        source_selected.append(source_row)
        repaired_row, reason = apply_local_repair(row)
        if repaired_row is None:
            skipped.append(skip_record(row, reason))
            continue
        repaired.append(repaired_row)

    hard_gates = [
        gate("config_present", bool(config), "hard", rel(config_path)),
        gate("repair_bridge_present", bool(bridge), "hard", rel(bridge_path)),
        gate(
            "repair_bridge_policy",
            bridge.get("policy") == "project_theseus_strict_generator_semantic_ir_repair_bridge_v1",
            "hard",
            bridge.get("policy"),
        ),
        gate("source_candidates_present", bool(source_candidates), "hard", rel(candidates_path)),
        gate("no_public_or_external_credit", no_cheat_clean(source_candidates + repaired), "hard", no_cheat_counts(source_candidates + repaired)),
        gate("repaired_candidates_are_noncredit", repaired_candidates_noncredit(repaired), "hard", len(repaired)),
        gate("gvr_state_machine_contract_ready", gvr_contract.get("trigger_state") == "GREEN", "hard", gvr_contract.get("summary")),
    ]
    source_verifier_report: dict[str, Any] = {}
    verifier_report: dict[str, Any] = {}
    if args.execute and not [row for row in hard_gates if row["severity"] == "hard" and not row["passed"]]:
        private_rows = private_rows_for_candidates(config, source_selected + repaired)
        source_verifier_report = run_private_verifier(private_rows, source_selected)
        verifier_report = run_private_verifier(private_rows, repaired)
    elif not args.execute:
        verifier_report = {
            "execute": False,
            "score_semantics": "Dry run only; private verifier was not executed.",
        }
        source_verifier_report = dict(verifier_report)
    verifier_comparison = compare_verifier_reports(source_verifier_report, verifier_report)
    search_receipts = verifier_guided_search_receipts(
        source_selected,
        repaired,
        source_verifier_report,
        verifier_report,
        execute=bool(args.execute),
    )
    semantic_ir_ready = bool(repaired) and all(
        dict_or_empty(row.get("semantic_ir")).get("state") == "READY"
        and dict_or_empty(row.get("semantic_ir")).get("roundtrip_ast_equal") is True
        for row in repaired
    )
    localized_patch_count = sum(
        1
        for row in repaired
        if list(dict_or_empty(row.get("semantic_ir_repair_apply")).get("changed_atom_ids") or [])
    )
    hard_gates.extend(
        [
            gate("typed_semantic_ir_roundtrip_ready", semantic_ir_ready, "hard", len(repaired)),
            gate("localized_atom_scope_present", localized_patch_count == len(repaired) if repaired else False, "hard", localized_patch_count),
            gate(
                "verifier_guided_search_replay_valid",
                all(row.get("replay", {}).get("passed") is True for row in search_receipts)
                if args.execute and search_receipts
                else not args.execute,
                "hard",
                {
                    "execute": bool(args.execute),
                    "receipt_count": len(search_receipts),
                    "invalid_count": sum(1 for row in search_receipts if row.get("replay", {}).get("passed") is not True),
                },
            ),
        ]
    )
    if args.execute:
        hard_gates.extend(
            [
                gate(
                    "type_handling_failures_reduced",
                    int(verifier_comparison.get("type_handling_failure_delta") or 0) < 0,
                    "hard",
                    verifier_comparison,
                ),
                gate(
                    "intended_behavior_not_regressed",
                    int(verifier_comparison.get("behavior_pass_delta") or 0) >= 0,
                    "hard",
                    verifier_comparison,
                ),
            ]
        )
    behavior_passes = int(
        dict_or_empty(dict_or_empty(verifier_report.get("correctness_labels")).get("stage_counts")).get("intended_behavior_passed")
        or 0
    )
    runtime_loaded = int(
        dict_or_empty(dict_or_empty(verifier_report.get("correctness_labels")).get("stage_counts")).get("runtime_loaded")
        or 0
    )
    hard_failed = [row for row in hard_gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "RED" if hard_failed else ("GREEN" if args.execute else "YELLOW")
    issue_counts = Counter(
        issue
        for row in repaired
        for issue in list(dict_or_empty(row.get("semantic_ir_repair_apply")).get("source_issue_labels") or [])
    )
    report = {
        "policy": "project_theseus_strict_generator_semantic_ir_repair_apply_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "execute": bool(args.execute),
        "inputs": {
            "config": rel(config_path),
            "config_sha256": file_hash(config_path),
            "repair_bridge": rel(bridge_path),
            "repair_bridge_sha256": file_hash(bridge_path),
            "source_candidates": rel(candidates_path),
            "source_candidates_sha256": file_hash(candidates_path),
        },
        "summary": {
            "source_candidate_rows": len(source_candidates),
            "selected_candidate_rows": len(selected),
            "repaired_candidate_rows": len(repaired),
            "skipped_candidate_rows": len(skipped),
            "repair_issue_counts": dict(sorted(issue_counts.items())),
            "runtime_loaded_repaired_attempts": runtime_loaded,
            "behavior_passed_repaired_attempts": behavior_passes,
            "behavior_pass_rate": ratio(behavior_passes, len(repaired)),
            "semantic_ir_ready_repaired_candidates": sum(
                1 for row in repaired if dict_or_empty(row.get("semantic_ir")).get("state") == "READY"
            ),
            "localized_patch_candidate_count": localized_patch_count,
            "source_type_handling_failures": verifier_comparison.get("source_type_handling_failures", 0),
            "repaired_type_handling_failures": verifier_comparison.get("repaired_type_handling_failures", 0),
            "type_handling_failure_delta": verifier_comparison.get("type_handling_failure_delta", 0),
            "behavior_pass_delta": verifier_comparison.get("behavior_pass_delta", 0),
            "verifier_guided_search_receipt_count": len(search_receipts),
            "verifier_guided_search_exact_count": sum(
                int(row.get("summary", {}).get("verified_exact_count") or 0) for row in search_receipts
            ),
            "verifier_guided_search_assisted_pass_count": sum(
                int(bool(row.get("summary", {}).get("assisted_repair_pass"))) for row in search_receipts
            ),
            "verifier_guided_search_replay_invalid_count": sum(
                int(row.get("replay", {}).get("passed") is not True) for row in search_receipts
            ),
            "hard_gap_count": len(hard_failed),
            "gvr_state_machine_state": gvr_contract.get("trigger_state"),
            "gvr_state_count": (gvr_contract.get("summary") or {}).get("state_count", 0),
            "gvr_mutation_passed_count": (gvr_contract.get("summary") or {}).get("mutation_passed_count", 0),
            "gvr_mutation_case_count": (gvr_contract.get("summary") or {}).get("mutation_case_count", 0),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            **NO_CHEAT,
        },
        "gates": hard_gates,
        "hard_gaps": hard_failed,
        "skipped": skipped[:32],
        "private_source_verifier": source_verifier_report,
        "private_verifier": verifier_report,
        "verifier_comparison": verifier_comparison,
        "verifier_guided_search": search_receipts,
        "gvr_state_machine": gvr_contract,
        "rules": {
            "input_boundary": "generated private candidates plus model-emitted prefix and semantic-IR repair obligations only",
            "verifier_boundary": "private tests are loaded only after repairs are produced and only for verification",
            "credit_boundary": "semantic-IR repaired rows are deterministic GVR evidence and never learned-generation evidence",
        },
        **NO_CHEAT,
        "non_claims": [
            "This report does not claim learned code generation.",
            "This report does not train a model.",
            "This report does not run public benchmarks or use public benchmark artifacts.",
            "Any repaired behavior pass is tool/GVR-style semantic repair evidence only.",
            "A reduction in type-handling faults is not a learned semantic-quality claim.",
        ],
    }
    return report, repaired


def verifier_guided_search_receipts(
    source_candidates: list[dict[str, Any]],
    repaired_candidates: list[dict[str, Any]],
    source_verifier: dict[str, Any],
    repaired_verifier: dict[str, Any],
    *,
    execute: bool,
) -> list[dict[str, Any]]:
    if not execute:
        return []
    source_traces = verifier_trace_by_candidate(source_verifier)
    repaired_traces = verifier_trace_by_candidate(repaired_verifier)
    repaired_by_source = {
        str(dict_or_empty(row.get("semantic_ir_repair_apply")).get("source_candidate_sha256") or ""): row
        for row in repaired_candidates
        if str(dict_or_empty(row.get("semantic_ir_repair_apply")).get("source_candidate_sha256") or "")
    }
    row_by_code_hash = {
        sha256_text(str(row.get("code") or "")): row
        for row in [*source_candidates, *repaired_candidates]
        if str(row.get("code") or "")
    }

    def integrity(proposal: verifier_guided_search.Proposal) -> dict[str, Any]:
        receipt = semantic_ir.candidate_receipt(proposal.code)
        row = row_by_code_hash.get(sha256_text(proposal.code), {})
        candidate_audit = recompute_candidate_integrity(row)
        if proposal.origin == "model_one_shot":
            origin_verified = bool(candidate_audit.get("integrity_verified"))
        else:
            repair_meta = dict_or_empty(row.get("semantic_ir_repair_apply"))
            origin_verified = bool(
                proposal.origin == "deterministic_repair"
                and repair_meta.get("policy") == "project_theseus_strict_generator_semantic_ir_repair_apply_v1"
                and int(repair_meta.get("candidate_generation_credit") or 0) == 0
                and repair_meta.get("learned_generation_claim_allowed") is False
            )
        return {
            "independently_recomputed": True,
            "valid": bool(
                origin_verified
                and receipt.get("state") == "READY"
                and receipt.get("roundtrip_ast_equal") is True
            ),
            "candidate_sha256": sha256_text(proposal.code),
            "family": candidate_audit.get("recomputed_candidate_family"),
            "fallback_or_template": candidate_audit.get("recomputed_candidate_family") == "fallback_or_template",
            "origin_independently_recomputed": True,
            "verified_origin": proposal.origin if origin_verified else "unverified",
        }

    def verify(proposal: verifier_guided_search.Proposal) -> dict[str, Any]:
        candidate_sha = sha256_text(proposal.code)
        trace = source_traces.get(candidate_sha) or repaired_traces.get(candidate_sha)
        if not trace:
            raise verifier_guided_search.SearchContractFault("VERIFIER_TRACE_MISSING", candidate_sha)
        failure = str(trace.get("failure_class") or "")
        row = row_by_code_hash.get(candidate_sha, {})
        repair_meta = dict_or_empty(row.get("semantic_ir_repair_apply"))
        return {
            "passed": bool(trace.get("intended_behavior_passed") or trace.get("passed")),
            "verification_stage": str(trace.get("verification_stage") or "unknown"),
            "verification_reward": float(trace.get("verification_reward") or 0.0),
            "fault_codes": [failure] if failure else [],
            "repair_scope": [str(item) for item in list(repair_meta.get("changed_atom_ids") or [])],
            "message_code": failure or str(trace.get("verification_stage") or "unknown"),
            "evidence_hash": verifier_guided_search.stable_hash(
                {
                    "candidate_sha256": candidate_sha,
                    "stage": trace.get("verification_stage"),
                    "reward": trace.get("verification_reward"),
                    "passed": bool(trace.get("intended_behavior_passed") or trace.get("passed")),
                }
            ),
            "verifier_id": "private_code_lm_candidate_verifier",
        }

    def repair(
        proposal: verifier_guided_search.Proposal,
        _feedback: dict[str, Any],
    ) -> list[verifier_guided_search.Proposal]:
        source_sha = sha256_text(proposal.code)
        row = repaired_by_source.get(source_sha)
        if not row:
            return []
        return [
            verifier_guided_search.Proposal(
                code=str(row.get("code") or ""),
                origin="deterministic_repair",
            )
        ]

    receipts: list[dict[str, Any]] = []
    for row in source_candidates:
        source_sha = sha256_text(str(row.get("code") or ""))
        receipt = verifier_guided_search.run_search(
            [
                verifier_guided_search.Proposal(
                    code=str(row.get("code") or ""),
                    origin="model_one_shot",
                    model_receipt_hash=str(row.get("candidate_sha256") or source_sha),
                )
            ],
            verify=verify,
            repair=repair,
            integrity=integrity,
            budget=verifier_guided_search.SearchBudget(
                max_proposals=2,
                max_verifier_calls=2,
                max_depth=1,
                max_repair_branches=1,
                max_wall_ms=5_000,
            ),
            task_ref_hash=verifier_guided_search.stable_hash(str(row.get("task_id") or source_sha)),
        )
        receipt["replay"] = verifier_guided_search.validate_replay(receipt)
        receipts.append(receipt)
    return receipts


def verifier_trace_by_candidate(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for row in list(report.get("verification_attempt_labels") or []):
        if not isinstance(row, dict):
            continue
        candidate_sha = str(row.get("code_sha256") or row.get("candidate_sha256") or "")
        if candidate_sha and candidate_sha not in traces:
            traces[candidate_sha] = row
    return traces


def candidate_rows_for_repair(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("phase") or "") != "private_eval":
            continue
        if str(row.get("candidate_generation_mode") or "") != "token_level_code_decoder":
            continue
        if row.get("token_level_code_generation_learned") is not True:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def apply_local_repair(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    loop = dict_or_empty(row.get("loop_plan_adequacy"))
    failures = {str(item) for item in list(loop.get("failures") or [])}
    expectation = dict_or_empty(loop.get("expectation"))
    plan = str(expectation.get("plan") or dict_or_empty(row.get("body_structure_decode")).get("predicted_plan") or "")
    if "missing_expected_update_call" not in failures:
        return None, "missing_expected_update_call_absent"
    if "MAX" not in plan and "MIN" not in plan:
        return None, "unsupported_plan_for_local_repair"
    parsed = parse_single_function(str(row.get("code") or ""))
    if parsed is None:
        return None, "source_code_not_single_function"
    function, lines = parsed
    learned_prefix_tokens = list(
        dict_or_empty(row.get("body_structure_decode")).get("learned_plan_prefix_tokens") or []
    )
    source_ir = semantic_ir.candidate_receipt(
        str(row.get("code") or ""),
        learned_prefix_tokens=learned_prefix_tokens,
        residual_lineage=[f"candidate:{str(row.get('candidate_sha256') or semantic_ir.stable_hash(row.get('code') or ''))}"],
        include_graph=True,
    )
    if source_ir.get("state") != "READY":
        return None, "source_semantic_ir_not_ready"
    repair = aggregate_call_repair(
        function,
        lines,
        reducer="max" if "MAX" in plan else "min",
        source_ir=source_ir,
        comparison_key="str" if "SLOT:EXPR_CALL_STR" in learned_prefix_tokens else "",
    )
    if repair is None:
        return None, "unsupported_body_shape_for_aggregate_call_repair"
    repaired = copy.deepcopy(row)
    repaired["code"] = repair["code"]
    repaired["candidate_sha256"] = sha256_text(repair["code"])
    repaired["candidate_generation_mode"] = "semantic_ir_localized_repair_noncredit"
    repaired["candidate_generation_mode_detail"] = "semantic_ir_aggregate_call_update_repair"
    repaired["candidate_generation_credit"] = 0
    repaired["semantic_ir_repair_credit"] = 1
    repaired["token_level_code_generation_learned"] = False
    repaired["benchmark_promotion_eligible"] = False
    repaired["phase"] = "private_eval"
    repaired["rank_score"] = float(row.get("rank_score") or 0.0)
    provenance = dict_or_empty(repaired.get("provenance"))
    provenance.update(
        {
            "candidate_family": "semantic_ir_repair_noncredit",
            "semantic_ir_repair_applied": True,
            "semantic_ir_repair_policy": "project_theseus_strict_generator_semantic_ir_repair_apply_v1",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "solutions_used_for_generation": False,
            "tests_used_for_generation": False,
            "public_data_used": False,
            "teacher_used": False,
        }
    )
    repaired["provenance"] = provenance
    repaired["semantic_ir_repair_apply"] = {
        "policy": "project_theseus_strict_generator_semantic_ir_repair_apply_v1",
        "source_candidate_sha256": str(row.get("candidate_sha256") or ""),
        "repair_id": repair["repair_id"],
        "repair_family": repair["repair_family"],
        "source_issue_labels": sorted(failures),
        "plan": plan,
        "reducer": repair["reducer"],
        "changed_lines": repair["changed_lines"],
        "changed_atom_ids": repair["changed_atom_ids"],
        "dependent_atom_ids": repair["dependent_atom_ids"],
        "semantic_ir_token_sha256": repair["semantic_ir_token_sha256"],
        "semantic_ir_program_sha256": repair["semantic_ir_program_sha256"],
        "comparison_key": repair["comparison_key"],
        "candidate_generation_credit": 0,
        "learned_generation_claim_allowed": False,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "external_inference_calls": 0,
        "non_claims": [
            "deterministic localized semantic-IR repair",
            "not learned generation",
            "not a template/fallback promotion artifact",
        ],
    }
    repaired["semantic_ir"] = repair["semantic_ir_receipt"]
    for key in ("private_verifier_label", "private_task_residual_label"):
        repaired.pop(key, None)
    return repaired, "repaired"


def parse_single_function(code: str) -> tuple[ast.FunctionDef, list[str]] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != 1:
        return None
    return funcs[0], code.splitlines()


def aggregate_call_repair(
    function: ast.FunctionDef,
    lines: list[str],
    *,
    reducer: str,
    source_ir: dict[str, Any],
    comparison_key: str,
) -> dict[str, Any] | None:
    if len(function.body) < 3:
        return None
    assign = function.body[0]
    loop = function.body[1]
    ret = function.body[2]
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    local_name = assign.targets[0].id
    if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name) or len(loop.body) != 1:
        return None
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Name) or ret.value.id != local_name:
        return None
    update = loop.body[0]
    if not isinstance(update, ast.Assign) or len(update.targets) != 1 or not isinstance(update.targets[0], ast.Name):
        return None
    if update.targets[0].id != local_name:
        return None
    if not isinstance(update.value, ast.BinOp) or not isinstance(update.value.op, ast.Add):
        return None
    loaded_names = {node.id for node in ast.walk(update.value) if isinstance(node, ast.Name)}
    loop_target = loop.target.id
    if local_name not in loaded_names or loop_target not in loaded_names:
        return None
    repaired_function = copy.deepcopy(function)
    repaired_assign = repaired_function.body[0]
    repaired_loop = repaired_function.body[1]
    if not isinstance(repaired_assign, ast.Assign) or not isinstance(repaired_loop, ast.For):
        return None
    repaired_assign.value = ast.Constant(value=None)
    repaired_loop.body[0] = ast.Assign(
        targets=[ast.Name(id=local_name, ctx=ast.Store())],
        value=ast.IfExp(
            test=ast.Compare(
                left=ast.Name(id=local_name, ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
            body=ast.Name(id=loop_target, ctx=ast.Load()),
            orelse=ast.Call(
                func=ast.Name(id=reducer, ctx=ast.Load()),
                args=[ast.Name(id=local_name, ctx=ast.Load()), ast.Name(id=loop_target, ctx=ast.Load())],
                keywords=(
                    [ast.keyword(arg="key", value=ast.Name(id=comparison_key, ctx=ast.Load()))]
                    if comparison_key
                    else []
                ),
            ),
        ),
    )
    ast.fix_missing_locations(repaired_function)
    tokens = semantic_ir.statements_to_tokens(repaired_function.body)
    compiled_body, compile_receipt = semantic_ir.compile_body_tokens(tokens)
    if compile_receipt.get("state") != "READY" or not compiled_body:
        return None
    header = lines[function.lineno - 1].rstrip()
    code = header + "\n" + "\n".join(f"    {line}" if line else "" for line in compiled_body.splitlines()) + "\n"
    repaired_ir = semantic_ir.candidate_receipt(code, include_graph=True)
    if repaired_ir.get("state") != "READY":
        return None
    graph = dict_or_empty(source_ir.get("program_graph"))
    changed_lines = {int(getattr(assign, "lineno", 0) or 0), int(getattr(update, "lineno", 0) or 0)}
    changed_atom_ids = sorted(
        str(atom.get("atom_id"))
        for atom in list(graph.get("atoms") or [])
        if isinstance(atom, dict)
        and int(dict_or_empty(atom.get("source_span")).get("line") or 0) in changed_lines
        and str(atom.get("intent") or "") in {"state_update", "value_expression", "literal_value"}
    )
    dependent_atom_ids = sorted(
        str(edge.get("to_atom"))
        for edge in list(graph.get("dependency_edges") or [])
        if isinstance(edge, dict) and str(edge.get("from_atom") or "") in set(changed_atom_ids)
    )
    if not changed_atom_ids:
        return None
    return {
        "code": code,
        "repair_id": f"semantic_ir_aggregate_{reducer}_update_v1",
        "repair_family": "aggregate_call_update",
        "reducer": reducer,
        "comparison_key": comparison_key,
        "changed_lines": ["initializer", "loop_update"],
        "changed_atom_ids": changed_atom_ids,
        "dependent_atom_ids": dependent_atom_ids,
        "semantic_ir_token_sha256": compile_receipt.get("token_sha256"),
        "semantic_ir_program_sha256": repaired_ir.get("program_sha256"),
        "semantic_ir_receipt": {key: value for key, value in repaired_ir.items() if key != "program_graph"},
    }


def compare_verifier_reports(source: dict[str, Any], repaired: dict[str, Any]) -> dict[str, Any]:
    source_attempts = list(source.get("verification_attempt_labels") or [])
    repaired_attempts = list(repaired.get("verification_attempt_labels") or [])
    source_type = sum(1 for row in source_attempts if dict_or_empty(row).get("failure_class") == "type_handling")
    repaired_type = sum(1 for row in repaired_attempts if dict_or_empty(row).get("failure_class") == "type_handling")
    source_behavior = sum(1 for row in source_attempts if dict_or_empty(row).get("intended_behavior_passed") is True)
    repaired_behavior = sum(1 for row in repaired_attempts if dict_or_empty(row).get("intended_behavior_passed") is True)
    return {
        "policy": "project_theseus_semantic_ir_source_repair_ablation_v1",
        "source_attempt_count": len(source_attempts),
        "repaired_attempt_count": len(repaired_attempts),
        "source_type_handling_failures": source_type,
        "repaired_type_handling_failures": repaired_type,
        "type_handling_failure_delta": repaired_type - source_type,
        "source_behavior_passes": source_behavior,
        "repaired_behavior_passes": repaired_behavior,
        "behavior_pass_delta": repaired_behavior - source_behavior,
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        "non_claims": [
            "deterministic semantic-IR repair ablation only",
            "not learned generation or model promotion evidence",
        ],
    }


def private_rows_for_candidates(config: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_cfg = dict_or_empty(config.get("data"))
    paths = eval_jsonl_paths(data_cfg)
    wanted = {str(row.get("task_id") or "") for row in candidates if str(row.get("task_id") or "")}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for raw in read_jsonl(resolve(path)):
            if not isinstance(raw, dict):
                continue
            enforce_private_flags(raw, data_cfg, path)
            task_id = str(raw.get("task_id") or "")
            if task_id in wanted and task_id not in rows_by_id:
                copy_row = dict(raw)
                copy_row["split"] = "eval"
                rows_by_id[task_id] = copy_row
    return [rows_by_id[task_id] for task_id in sorted(wanted) if task_id in rows_by_id]


def run_private_verifier(private_rows: list[dict[str, Any]], repaired: list[dict[str, Any]]) -> dict[str, Any]:
    if not private_rows or not repaired:
        return {
            "eval_task_count": len(private_rows),
            "candidate_count": len(repaired),
            "reason": "no_private_rows_or_repaired_candidates",
            "correctness_labels": {"stage_counts": {}},
            **NO_CHEAT,
        }
    from code_lm_private_verifier import evaluate_private_candidates  # noqa: PLC0415

    report = evaluate_private_candidates(private_rows, repaired)
    report["candidate_count"] = len(repaired)
    report["public_training_rows"] = 0
    report["external_inference_calls"] = 0
    report["fallback_template_router_tool_credit_count"] = 0
    report["candidate_generation_credit"] = 0
    report["score_semantics"] = (
        "Private verifier labels for deterministic semantic-IR repaired candidates. "
        "This is GVR/tool-style repair evidence only, not learned-generation evidence."
    )
    return report


def eval_jsonl_paths(data_cfg: dict[str, Any]) -> list[str]:
    broad = dict_or_empty(data_cfg.get("broad_private_heldout_eval")).get("eval_jsonl")
    if isinstance(broad, list):
        return [str(item) for item in broad if str(item)]
    if isinstance(broad, str) and broad:
        return [broad]
    fallback = data_cfg.get("eval_jsonl")
    if isinstance(fallback, list):
        return [str(item) for item in fallback if str(item)]
    return [str(fallback)] if str(fallback or "") else []


def enforce_private_flags(row: dict[str, Any], data_cfg: dict[str, Any], path: str) -> None:
    if not data_cfg.get("require_private_flags", True):
        return
    bad = [str(flag) for flag in data_cfg.get("forbidden_row_flags", []) if bool(row.get(str(flag)))]
    if bad:
        raise ValueError(f"forbidden public flags in {path}: {bad}")


def repaired_candidates_noncredit(rows: list[dict[str, Any]]) -> bool:
    return all(
        int(row.get("candidate_generation_credit") or 0) == 0
        and row.get("token_level_code_generation_learned") is False
        and dict_or_empty(row.get("provenance")).get("learned_generation_claim_allowed") is False
        and row.get("benchmark_promotion_eligible") is False
        for row in rows
    )


def no_cheat_clean(rows: list[dict[str, Any]]) -> bool:
    counts = no_cheat_counts(rows)
    return all(value == 0 for value in counts.values())


def no_cheat_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "public_training_rows": sum(int(row.get("public_training_rows") or 0) for row in rows),
        "external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in rows),
        "fallback_credit_rows": sum(int(row.get("fallback_template_router_tool_credit_count") or 0) for row in rows),
        "candidate_generation_credit": sum(int(row.get("candidate_generation_credit") or 0) for row in rows),
    }


def skip_record(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "task_ref_hash": stable_hash(str(row.get("task_id") or row.get("source_task_id") or ""))[:16],
        "candidate_sha256": str(row.get("candidate_sha256") or ""),
        "reason": reason,
    }


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "hard_gaps": report.get("hard_gaps", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Strict Generator Semantic IR Repair Apply",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Repaired candidates: `{summary.get('repaired_candidate_rows', 0)}`",
        f"- Runtime-loaded repaired attempts: `{summary.get('runtime_loaded_repaired_attempts', 0)}`",
        f"- Behavior-passed repaired attempts: `{summary.get('behavior_passed_repaired_attempts', 0)}`",
        f"- Behavior pass rate: `{summary.get('behavior_pass_rate', 0)}`",
        "",
        "## Non-Claims",
    ]
    for item in report.get("non_claims", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def stable_hash(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except Exception:
        return str(value)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
