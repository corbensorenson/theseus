#!/usr/bin/env python3
"""Recomputed candidate-family integrity for Project Theseus.

This module is intentionally independent of candidate self-declared eligibility
flags. It classifies code candidates from replayable metadata, generation mode,
origin/source strings, and static Python code shape so benchmark and promotion
reports can separate learned generation from structural/body-inventory help.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import viea_spine_records
import semantic_ir
from neural_seed_open_vocab import decode_target_tokens
from neural_seed_token_decoder_rendering import PLAN_BODY_START_TOKEN, split_learned_plan_prefix_tokens
from neural_seed_token_decoder_support import body_tokens


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "reports" / "student_code_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "candidate_integrity_audit.json"
DEFAULT_MD = ROOT / "reports" / "candidate_integrity_audit.md"
AUTO_CANDIDATE_SOURCES = (
    ROOT / "reports" / "neural_seed_token_decoder_candidates_strict_body_tokens.jsonl",
)
AUTO_CANDIDATE_GLOBS = (
    "reports/student_code_candidates_private_*.jsonl",
    "reports/student_code_candidates_*private*.jsonl",
)

PROMOTION_FAMILIES = {
    "learned_full_body_token",
    "transformer_hybrid",
    "symliquid",
}
KNOWN_FAMILIES = {
    "hand_authored_contract_body",
    "private_ngram_body",
    "structural_adapter",
    "neural_action_selector",
    "learned_full_body_token",
    "transformer_hybrid",
    "symliquid",
    "deterministic_tool",
    "fallback_or_template",
    "unknown",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--sample-limit", type=int, default=40)
    args = parser.parse_args()

    requested_source = resolve(args.candidates)
    explicit_source = args.candidates != rel(DEFAULT_CANDIDATES)
    candidate_source, source_resolution = resolve_candidate_source(requested_source, explicit=explicit_source)
    rows = read_jsonl(candidate_source)
    report = build_candidate_integrity_report(
        rows,
        source_path=candidate_source,
        sample_limit=max(0, int(args.sample_limit)),
    )
    report["source_resolution"] = source_resolution
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def resolve_candidate_source(requested: Path, *, explicit: bool) -> tuple[Path, dict[str, Any]]:
    """Resolve the candidate JSONL used by the canonical audit.

    Explicit --candidates input is always respected. The default legacy path is
    allowed to auto-resolve to a newer private/strict candidate source when the
    legacy path is absent or empty, so the canonical latest-view audit does not
    go RED only because a stale file name was retained.
    """
    requested_rows = jsonl_row_count(requested)
    if explicit or requested_rows > 0:
        return requested, {
            "policy": "candidate_integrity_source_resolution_v1",
            "requested": rel(requested),
            "selected": rel(requested),
            "explicit": explicit,
            "auto_resolved": False,
            "requested_row_count": requested_rows,
            "reason": "explicit_source" if explicit else "default_source_present",
        }

    candidates: list[Path] = []
    candidates.extend(path for path in AUTO_CANDIDATE_SOURCES if path.exists())
    for pattern in AUTO_CANDIDATE_GLOBS:
        candidates.extend(ROOT.glob(pattern))
    viable = [path for path in candidates if jsonl_row_count(path) > 0]
    if not viable:
        return requested, {
            "policy": "candidate_integrity_source_resolution_v1",
            "requested": rel(requested),
            "selected": rel(requested),
            "explicit": explicit,
            "auto_resolved": False,
            "requested_row_count": requested_rows,
            "reason": "no_non_empty_registered_candidate_source",
        }
    selected = max(viable, key=lambda path: path.stat().st_mtime)
    return selected, {
        "policy": "candidate_integrity_source_resolution_v1",
        "requested": rel(requested),
        "selected": rel(selected),
        "explicit": explicit,
        "auto_resolved": True,
        "requested_row_count": requested_rows,
        "selected_row_count": jsonl_row_count(selected),
        "candidate_source_count": len(viable),
        "reason": "default_source_missing_or_empty",
        "selection_rule": "newest non-empty private/strict registered candidate JSONL",
    }


def recompute_candidate_integrity(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    benchmark_integrity = (
        row.get("benchmark_integrity")
        if isinstance(row.get("benchmark_integrity"), dict)
        else get_path(provenance, ["benchmark_integrity"], {})
    )
    if not isinstance(benchmark_integrity, dict):
        benchmark_integrity = {}

    code = str(row.get("code") or row.get("candidate_code") or "")
    generation_mode = str(
        row.get("candidate_generation_mode")
        or get_path(provenance, ["candidate_generation_mode"], "")
        or ""
    )
    origin = str(row.get("origin") or row.get("candidate_source") or "")
    candidate_source = str(row.get("candidate_source") or "")
    body_kind = str(
        row.get("candidate_body_structure_kind")
        or get_path(provenance, ["candidate_body_structure_kind"], "")
        or ""
    )
    generation_contract = str(row.get("candidate_generation_contract") or "")
    source_module = str(row.get("source_module") or get_path(provenance, ["source_module"], ""))
    generation_inputs = [
        str(item)
        for item in (
            row.get("generation_inputs")
            if isinstance(row.get("generation_inputs"), list)
            else get_path(provenance, ["generation_inputs"], [])
        )
        if str(item)
    ]

    text = " ".join(
        [
            generation_mode,
            origin,
            candidate_source,
            body_kind,
            generation_contract,
            source_module,
            str(get_path(provenance, ["training_target"], "")),
            " ".join(generation_inputs),
        ]
    ).lower()
    shape = code_shape(code)
    direct_plan_body_trace = verify_direct_plan_body_trace(row, code)
    family, reasons = classify_family(
        text,
        shape,
        row,
        benchmark_integrity,
        direct_plan_body_trace=direct_plan_body_trace,
    )

    claimed = {
        "token_level_code_generation_learned": truthy(
            row.get("token_level_code_generation_learned")
            if "token_level_code_generation_learned" in row
            else get_path(provenance, ["token_level_code_generation_learned"], False)
        ),
        "template_like_candidate": truthy(
            row.get("template_like_candidate")
            if "template_like_candidate" in row
            else get_path(provenance, ["template_like_candidate"], False)
        ),
        "benchmark_promotion_eligible": truthy(
            row.get("benchmark_promotion_eligible")
            if "benchmark_promotion_eligible" in row
            else get_path(provenance, ["benchmark_promotion_eligible"], False)
        ),
        "grammar_masked_learned_token_candidate": truthy(
            row.get("grammar_masked_learned_token_candidate")
            if "grammar_masked_learned_token_candidate" in row
            else get_path(provenance, ["grammar_masked_learned_token_candidate"], False)
        ),
        "full_body_token_candidate": truthy(
            row.get("full_body_token_candidate")
            if "full_body_token_candidate" in row
            else get_path(provenance, ["full_body_token_candidate"], False)
        ),
        "expression_memory_fallback": truthy(
            row.get("expression_memory_fallback")
            if "expression_memory_fallback" in row
            else get_path(provenance, ["expression_memory_fallback"], False)
        ),
    }

    pure_learned_generation = family in PROMOTION_FAMILIES and shape["syntax_valid"] and shape["has_function"]
    integrity_verified = bool(
        pure_learned_generation
        and not shape["unconditional_trivial_return"]
        and not shape["inert_stub_like"]
        and not claimed["expression_memory_fallback"]
        and not bool(row.get("loop_closure_generated"))
        and not truthy(row.get("placeholder_scaffold_body"))
        and benchmark_integrity.get("public_tests_used") is not True
        and benchmark_integrity.get("public_solutions_used") is not True
        and benchmark_integrity.get("canonical_solution_used") is not True
    )

    mismatches = []
    if claimed["benchmark_promotion_eligible"] and not integrity_verified:
        mismatches.append("claimed_promotion_eligible_but_recomputed_not_integrity_verified")
    if claimed["token_level_code_generation_learned"] and family not in PROMOTION_FAMILIES:
        mismatches.append("claimed_token_level_learned_but_family_not_pure_learned")
    if claimed["template_like_candidate"] and family != "fallback_or_template":
        mismatches.append("claimed_template_but_family_not_template")
    claimed_learned_surface = (
        claimed["benchmark_promotion_eligible"]
        or claimed["token_level_code_generation_learned"]
        or claimed["grammar_masked_learned_token_candidate"]
        or claimed["full_body_token_candidate"]
    )
    if claimed_learned_surface and (not claimed["template_like_candidate"]) and family == "fallback_or_template":
        mismatches.append("claimed_non_template_but_recomputed_template_or_fallback")
    if claimed["grammar_masked_learned_token_candidate"] and family not in PROMOTION_FAMILIES:
        mismatches.append("claimed_grammar_masked_learned_but_family_not_pure_learned")
    if claimed["full_body_token_candidate"] and not shape["has_function"]:
        mismatches.append("claimed_full_body_but_no_function_def")
    if claimed_learned_surface and shape["inert_stub_like"]:
        mismatches.append("claimed_learned_candidate_but_inert_stub_like")

    return {
        "policy": "project_theseus_recomputed_candidate_integrity_v1",
        "candidate_sha256": sha256_text(code),
        "recomputed_candidate_family": family,
        "integrity_verified": integrity_verified,
        "pure_learned_generation": pure_learned_generation,
        "candidate_family_confidence": confidence_for(family, reasons, shape),
        "classification_reasons": reasons,
        "self_declared_flags": claimed,
        "integrity_mismatches": mismatches,
        "code_shape": shape,
        "direct_plan_body_trace": direct_plan_body_trace,
    }


def classify_family(
    text: str,
    shape: dict[str, Any],
    row: dict[str, Any],
    benchmark_integrity: dict[str, Any],
    *,
    direct_plan_body_trace: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if row.get("phase") == "private_baseline" or row.get("substrate_adapter") == "shared_null_baseline":
        reasons.append("baseline_or_null_candidate")
        return "fallback_or_template", reasons

    template_tokens = [
        "fallback",
        "no_admissible",
        "template",
        "placeholder",
        "prompt_program_induction",
        "prompt_program_decoder",
        "deterministic_program_prior",
        "repair_template",
        "baseline_prompt_stub",
    ]
    if any(token in text for token in template_tokens) or truthy(row.get("placeholder_scaffold_body")):
        reasons.append("template_or_fallback_token_in_provenance")
        return "fallback_or_template", reasons

    if shape["unconditional_trivial_return"]:
        reasons.append("trivial_fallback_shape")
        return "fallback_or_template", reasons

    if "private_contract_role_body" in text or "contract_role_body_synthesis" in text:
        reasons.append("private_contract_role_body_inventory")
        return "hand_authored_contract_body", reasons

    if (
        "private_multistatement_body_ngram" in text
        or "private_composition_body_ngram" in text
        or "private_body_ngram" in text
        or "seeded_body_ngram" in text
        or truthy(row.get("private_body_ngram_candidate"))
    ):
        reasons.append("private_train_body_ngram_inventory")
        return "private_ngram_body", reasons

    if direct_plan_body_trace and direct_plan_body_trace.get("valid") is True:
        if "transformer" in text or "hybrid" in text:
            reasons.append("independently_verified_direct_transformer_plan_body_trace")
            return "transformer_hybrid", reasons
        reasons.append("independently_verified_direct_learned_plan_body_trace")
        return "learned_full_body_token", reasons

    structural_tokens = [
        "structural_action",
        "strict_action",
        "structural_adapter",
        "skeleton",
        "semantic_adapter",
        "semantic_plan",
        "execution_shape",
        "local_adapter",
        "contract_guided",
        "contract_transduced",
        "causal_contract",
        "interface_floor",
        "body_template_selector",
        "receiver_inventory",
        "prototype",
        "broad_transfer_residual",
    ]
    if any(token in text for token in structural_tokens) or truthy(row.get("structural_action_candidate")):
        reasons.append("structural_or_adapter_candidate_family")
        return "structural_adapter", reasons

    if "deterministic_tool" in text or "tool_trace" in text or "sympy" in text or "lean" in text or "z3" in text:
        reasons.append("deterministic_tool_candidate_family")
        return "deterministic_tool", reasons

    action_selector_tokens = [
        "action_selector",
        "action_generator",
        "fixed_renderer",
        "action_renderer",
        "grammar_safe_action_renderer",
        "action=",
    ]
    if any(token in text for token in action_selector_tokens):
        reasons.append("neural_action_selector_with_fixed_renderer_not_generation")
        return "neural_action_selector", reasons

    if "transformer" in text or "hybrid" in text:
        reasons.append("transformer_or_hybrid_candidate_family")
        return "transformer_hybrid", reasons

    if "symliquid" in text or "recurrent_state_decoder" in text:
        reasons.append("symliquid_candidate_family")
        return "symliquid", reasons

    learned_mode = (
        "full_body_token_beam" in text
        or "greedy_body_token_decoder" in text
        or "token_decoder" in text
        or "private_train_body_tokens" in text
    )
    learned_flags = (
        (
            truthy(row.get("compositional_token_candidate"))
            and truthy(row.get("full_body_token_candidate"))
        )
        or (
            "token_level_code_decoder" in text
            and "private_train_body_tokens" in text
        )
        and not truthy(row.get("expression_memory_fallback"))
        and benchmark_integrity.get("public_tests_used") is not True
        and benchmark_integrity.get("public_solutions_used") is not True
    )
    if learned_mode and learned_flags and (shape["has_function"] or not shape["syntax_valid"]):
        reasons.append("strict_full_body_token_candidate_shape_and_mode")
        return "learned_full_body_token", reasons

    if learned_mode:
        reasons.append("learned_mode_without_required_shape_or_flags")
        return "unknown", reasons

    reasons.append("no_recomputed_family_rule_matched")
    return "unknown", reasons


def verify_direct_plan_body_trace(row: dict[str, Any], code: str) -> dict[str, Any]:
    mode = str(row.get("candidate_generation_mode") or "")
    if mode != "direct_decoder_only_causal_semantic_plan_body_tokens":
        return {"applicable": False, "valid": False, "faults": []}
    faults: list[str] = []
    raw_tokens = row.get("decoded_target_tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens or not all(isinstance(token, str) for token in raw_tokens):
        return {
            "applicable": True,
            "valid": False,
            "faults": ["decoded_target_token_trace_missing"],
        }
    tokens = [str(token) for token in raw_tokens]
    if sha256_text(" ".join(tokens)) != str(row.get("decoded_token_sha256") or ""):
        faults.append("decoded_target_token_hash_mismatch")
    body_stream, prefix_metadata = split_learned_plan_prefix_tokens(tokens)
    prefix = list(prefix_metadata.get("learned_plan_prefix_tokens") or [])
    transition_prefix: list[str] = []
    for token in prefix:
        if not semantic_ir.plan_prefix_token_allowed(
            transition_prefix,
            token,
            body_start_token=PLAN_BODY_START_TOKEN,
        ):
            faults.append("semantic_plan_transition_invalid")
            break
        transition_prefix.append(token)
    if prefix_metadata.get("semantic_ir_plan_complete") is not True:
        faults.append("semantic_plan_incomplete")
    decoded_body_tokens, decode_receipt = decode_target_tokens(body_stream)
    if decode_receipt.get("state") != "READY" or decode_receipt.get("faults"):
        faults.append("body_trace_decode_fault")
    expected_body_tokens = function_body_tokens(code)
    if not expected_body_tokens:
        faults.append("candidate_function_body_missing")
    elif strip_implicit_terminal_dedents(decoded_body_tokens) != strip_implicit_terminal_dedents(
        expected_body_tokens
    ):
        faults.append("decoded_body_trace_code_mismatch")
    return {
        "applicable": True,
        "valid": not faults,
        "faults": faults,
        "decoded_target_token_count": len(tokens),
        "learned_plan_token_count": len(prefix),
        "direct_body_token_count": len(body_stream),
        "body_token_sha256": sha256_text(" ".join(expected_body_tokens)),
        "failure_behavior": "reject_without_family_credit" if faults else "verified_direct_body_subsequence",
    }


def function_body_tokens(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or not functions[0].body:
        return []
    function = functions[0]
    lines = str(code).splitlines()
    start = int(function.body[0].lineno) - 1
    end = int(getattr(function, "end_lineno", len(lines)))
    body_source = textwrap.dedent("\n".join(lines[start:end]))
    return body_tokens(body_source)


def strip_implicit_terminal_dedents(tokens: list[str]) -> list[str]:
    values = list(tokens)
    while values and values[-1] == "DEDENT:":
        values.pop()
    return values


def code_shape(code: str) -> dict[str, Any]:
    shape = {
        "syntax_valid": False,
        "has_function": False,
        "function_count": 0,
        "statement_count": 0,
        "return_count": 0,
        "loop_count": 0,
        "if_count": 0,
        "call_count": 0,
        "import_count": 0,
        "parameter_use_count": 0,
        "parameter_names": [],
        "unconditional_trivial_return": False,
        "inert_stub_like": False,
        "parse_error": "",
    }
    if not code.strip():
        shape["parse_error"] = "empty_code"
        shape["unconditional_trivial_return"] = True
        return shape
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        shape["parse_error"] = f"{exc.__class__.__name__}:{exc.lineno}:{exc.offset}"
        return shape
    shape["syntax_valid"] = True
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    shape["has_function"] = bool(functions)
    shape["function_count"] = len(functions)
    body_nodes = functions[0].body if functions else tree.body
    parameter_names: set[str] = set()
    if functions:
        args = functions[0].args
        parameter_names.update(arg.arg for arg in args.posonlyargs)
        parameter_names.update(arg.arg for arg in args.args)
        parameter_names.update(arg.arg for arg in args.kwonlyargs)
        if args.vararg:
            parameter_names.add(args.vararg.arg)
        if args.kwarg:
            parameter_names.add(args.kwarg.arg)
    shape["statement_count"] = len(body_nodes)
    loaded_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            shape["return_count"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            shape["loop_count"] += 1
        elif isinstance(node, ast.If):
            shape["if_count"] += 1
        elif isinstance(node, ast.Call):
            shape["call_count"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            shape["import_count"] += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded_names.add(node.id)
    used_parameters = sorted(parameter_names & loaded_names)
    shape["parameter_names"] = sorted(parameter_names)
    shape["parameter_use_count"] = len(used_parameters)
    shape["unconditional_trivial_return"] = unconditional_trivial_return(body_nodes)
    shape["inert_stub_like"] = inert_stub_like(body_nodes, parameter_names=parameter_names, loaded_names=loaded_names)
    return shape


def unconditional_trivial_return(body_nodes: list[ast.stmt]) -> bool:
    meaningful = [node for node in body_nodes if not isinstance(node, (ast.Pass, ast.Expr))]
    if len(meaningful) != 1 or not isinstance(meaningful[0], ast.Return):
        return False
    return return_value_is_trivial(meaningful[0].value)


def return_value_is_trivial(value: ast.AST | None) -> bool:
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        return value.value in {None, False, True, 0, 1, "", b""}
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and not value.elts:
        return True
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    return False


def all_returns_trivial(body_nodes: list[ast.stmt]) -> bool:
    returns = [node for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])) if isinstance(node, ast.Return)]
    return bool(returns) and all(return_value_is_trivial(node.value) for node in returns)


def inert_stub_like(body_nodes: list[ast.stmt], *, parameter_names: set[str], loaded_names: set[str]) -> bool:
    """Reject syntax-shaped bodies that do not condition on callable inputs.

    Candidate integrity is not a semantic verifier, but promotion-grade learned
    generation should at least use the prompt-visible callable interface. This
    catches bodies such as ``if not isinstance: return True`` that are syntactic
    artifacts rather than task-conditioned code.
    """

    if not body_nodes:
        return True
    if unconditional_trivial_return(body_nodes):
        return True
    if all_returns_trivial(body_nodes):
        return True
    if parameter_names and not (parameter_names & loaded_names):
        return True
    meaningful = [node for node in body_nodes if not isinstance(node, (ast.Pass, ast.Expr))]
    if len(meaningful) == 1 and isinstance(meaningful[0], ast.If):
        names = {node.id for node in ast.walk(meaningful[0]) if isinstance(node, ast.Name)}
        if not (parameter_names & names):
            return True
    return False


def build_candidate_integrity_report(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    sample_limit: int = 40,
) -> dict[str, Any]:
    audits = []
    family_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    claimed_promotion_by_family: Counter[str] = Counter()
    integrity_verified_by_family: Counter[str] = Counter()
    syntax_invalid_by_family: Counter[str] = Counter()
    samples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        audit = recompute_candidate_integrity(row)
        family = str(audit["recomputed_candidate_family"])
        family_counts[family] += 1
        if not audit["code_shape"]["syntax_valid"]:
            syntax_invalid_by_family[family] += 1
        if audit["self_declared_flags"]["benchmark_promotion_eligible"]:
            claimed_promotion_by_family[family] += 1
        if audit["integrity_verified"]:
            integrity_verified_by_family[family] += 1
        for mismatch in audit["integrity_mismatches"]:
            mismatch_counts[mismatch] += 1
        if audit["integrity_mismatches"] and len(samples) < sample_limit:
            samples.append(
                {
                    "manifest_index": index,
                    "task_id": str(row.get("task_id") or ""),
                    "origin": str(row.get("origin") or "")[:240],
                    "candidate_generation_mode": str(row.get("candidate_generation_mode") or ""),
                    "family": family,
                    "mismatches": audit["integrity_mismatches"],
                    "candidate_sha256": audit["candidate_sha256"],
                }
            )
        audits.append(audit)

    hard_mismatch_count = sum(mismatch_counts.values())
    integrity_verified_count = sum(1 for audit in audits if audit["integrity_verified"])
    spine_receipt = viea_spine_records.materialized_view_consumer_receipt(
        "candidate_integrity_harness",
        required_groups=[
            "claim_ledger_entries",
            "artifact_records",
            "failure_boundaries",
            "generation_mode_records",
        ],
    )
    trigger_state = "GREEN" if rows and hard_mismatch_count == 0 and integrity_verified_count > 0 else "YELLOW" if rows else "RED"
    if trigger_state == "GREEN" and not spine_receipt["ready"]:
        trigger_state = "YELLOW"
    summary = {
        "candidate_count": len(rows),
        "audited_candidate_count": len(audits),
        "family_counts": dict(sorted(family_counts.items())),
        "claimed_promotion_by_family": dict(sorted(claimed_promotion_by_family.items())),
        "integrity_verified_by_family": dict(sorted(integrity_verified_by_family.items())),
        "integrity_verified_candidate_count": integrity_verified_count,
        "integrity_mismatch_count": hard_mismatch_count,
        "integrity_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "syntax_invalid_by_family": dict(sorted(syntax_invalid_by_family.items())),
        "promotion_families": sorted(PROMOTION_FAMILIES),
        "known_families": sorted(KNOWN_FAMILIES),
        "viea_spine_view_ready": spine_receipt["ready"],
        "viea_spine_view_record_count": spine_receipt["record_count"],
        "viea_spine_claim_ledger_entry_count": spine_receipt["claim_ledger_entry_count"],
        "viea_spine_generation_mode_record_count": spine_receipt["generation_mode_record_count"],
    }
    return {
        "policy": "project_theseus_candidate_integrity_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "source": rel(source_path),
        "summary": summary,
        "viea_integrity_records": build_viea_integrity_records(
            source_path=source_path,
            trigger_state=trigger_state,
            summary=summary,
        ),
        "viea_spine_consumer_receipt": spine_receipt,
        "private_repair_plan": private_repair_plan(family_counts, mismatch_counts, integrity_verified_count),
        "mismatch_samples": samples,
        "score_semantics": "audit only; recomputes candidate family and promotion eligibility without trusting candidate self-declared flags",
        "external_inference_calls": 0,
    }


def build_viea_integrity_records(
    *,
    source_path: Path,
    trigger_state: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    source_ref = rel(source_path)
    audit_id = viea_spine_records.stable_id("candidate_integrity_audit", source_ref, summary)
    family_counts = summary.get("family_counts") if isinstance(summary.get("family_counts"), dict) else {}
    verified_by_family = summary.get("integrity_verified_by_family") if isinstance(summary.get("integrity_verified_by_family"), dict) else {}
    syntax_invalid_by_family = summary.get("syntax_invalid_by_family") if isinstance(summary.get("syntax_invalid_by_family"), dict) else {}
    common = {
        "run_id": audit_id,
        "audit_scope": "candidate_family_integrity",
        "source_path": source_ref,
        "candidate_count": summary.get("candidate_count"),
        "integrity_verified_candidate_count": summary.get("integrity_verified_candidate_count"),
        "integrity_mismatch_count": summary.get("integrity_mismatch_count"),
        "support_state": "SUPPORTED" if trigger_state == "GREEN" else "RESIDUAL",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    generation_modes = []
    for family, count in sorted(family_counts.items()):
        verified = int(verified_by_family.get(family, 0) or 0)
        generation_modes.append(
            {
                **common,
                "record_type": "generation_mode",
                "record_id": viea_spine_records.stable_id("candidate_generation_mode", audit_id, family),
                "candidate_family": family,
                "candidate_count": int(count or 0),
                "integrity_verified_candidate_count": verified,
                "syntax_invalid_count": int(syntax_invalid_by_family.get(family, 0) or 0),
                "learned_generation_claim_allowed": bool(family in PROMOTION_FAMILIES and verified > 0),
                "state": "integrity_verified_family" if verified > 0 else "observed_non_promoting_family",
                "status": "AUDITED",
                "non_claim": "Family records classify candidates; only downstream verifier/promotion gates may make capability claims.",
            }
        )
    mismatch_count = int(summary.get("integrity_mismatch_count") or 0)
    return {
        "claim_record": {
            **common,
            "record_type": "claim_record",
            "record_id": viea_spine_records.stable_id("candidate_integrity_claim", audit_id),
            "claim_id": viea_spine_records.stable_id("claim_candidate_integrity", audit_id),
            "state": "independent_candidate_integrity_audited",
            "status": trigger_state,
            "verifier_state": "family_recomputed_from_code_and_provenance",
            "evidence_ref": "reports/candidate_integrity_audit.json",
        },
        "proof_carrying_claim": {
            **common,
            "record_type": "proof_carrying_claim",
            "record_id": viea_spine_records.stable_id("candidate_integrity_proof", audit_id),
            "proof_claim_id": viea_spine_records.stable_id("proof_candidate_integrity", audit_id),
            "state": "candidate_flags_not_trusted",
            "status": trigger_state,
            "verifier_state": "independent_recompute",
            "evidence_ref": "reports/candidate_integrity_audit.json",
        },
        "authority_use_receipt": {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": viea_spine_records.stable_id("candidate_integrity_authority", audit_id),
            "state": "audit_only_no_promotion",
            "status": "READY",
            "authority_scope": ["candidate_family_recompute", "promotion_claim_guard"],
        },
        "generation_modes": generation_modes,
        "failure_boundary": {
            **common,
            "record_type": "failure_boundary",
            "record_id": viea_spine_records.stable_id("candidate_integrity_failure_boundary", audit_id),
            "failure_id": viea_spine_records.stable_id("candidate_integrity_mismatch_boundary", audit_id),
            "state": "no_integrity_mismatches" if mismatch_count == 0 else "integrity_mismatches_present",
            "status": "READY" if mismatch_count == 0 else "RESIDUAL",
            "terminal": False,
            "structured_non_solved": mismatch_count > 0,
            "fallback_return_used": False,
        },
        "artifact_graph_record": {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": viea_spine_records.stable_id("candidate_integrity_artifact", audit_id),
            "artifact_kind": "candidate_integrity_report",
            "evidence_ref": "reports/candidate_integrity_audit.json",
            "content_hash": viea_spine_records.stable_hash(summary),
        },
        "evidence_transition_record": {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": viea_spine_records.stable_id("candidate_integrity_evidence_transition", audit_id),
            "state": "candidate_jsonl_to_integrity_audit",
            "status": trigger_state,
            "evidence_ref": "reports/candidate_integrity_audit.json",
        },
    }


def private_repair_plan(
    family_counts: Counter[str],
    mismatch_counts: Counter[str],
    integrity_verified_count: int,
) -> list[dict[str, Any]]:
    plan = []
    if mismatch_counts.get("claimed_promotion_eligible_but_recomputed_not_integrity_verified", 0):
        plan.append(
            {
                "target": "candidate_flag_contract",
                "priority": "high",
                "action": "Stop emitting benchmark_promotion_eligible=true for private_ngram_body, structural_adapter, neural_action_selector, fallback_or_template, and any candidate family not independently verified as learned_full_body_token, transformer_hybrid, or symliquid.",
                "evidence": {
                    "claimed_promotion_mismatches": mismatch_counts.get("claimed_promotion_eligible_but_recomputed_not_integrity_verified", 0),
                },
            }
        )
    if family_counts.get("private_ngram_body", 0):
        plan.append(
            {
                "target": "private_ngram_body_quarantine",
                "priority": "high",
                "action": "Keep private ngram bodies runnable as diagnostic/private pressure, but report their pass rate separately and exclude them from learned-generation promotion.",
                "evidence": {"private_ngram_body_count": family_counts.get("private_ngram_body", 0)},
            }
        )
    if family_counts.get("structural_adapter", 0):
        plan.append(
            {
                "target": "structural_adapter_ablation",
                "priority": "medium",
                "action": "Keep structural adapters as a separate candidate family and use family-filtered private replay to measure their utility without claiming pure token generation.",
                "evidence": {"structural_adapter_count": family_counts.get("structural_adapter", 0)},
            }
        )
    if family_counts.get("neural_action_selector", 0):
        plan.append(
            {
                "target": "neural_action_selector_quarantine",
                "priority": "high",
                "action": "Keep fixed-renderer action selectors as prompt/signature baselines or tool routers only. They cannot support learned-generation or promotion claims without a separate blind information-flow audit and a generator that synthesizes bodies rather than selecting catalog entries.",
                "evidence": {"neural_action_selector_count": family_counts.get("neural_action_selector", 0)},
            }
        )
    if integrity_verified_count == 0:
        plan.append(
            {
                "target": "learned_full_body_candidate_generation",
                "priority": "high",
                "action": "Repair the full-body token generator until at least one private heldout slice has verified learned_full_body_token candidates before any public calibration.",
                "evidence": {"integrity_verified_candidate_count": 0},
            }
        )
    else:
        plan.append(
            {
                "target": "learned_full_body_selection_quality",
                "priority": "high",
                "action": "Increase coverage and selected-pass quality of verified learned_full_body_token candidates; do not use private body inventory to mask low learned coverage.",
                "evidence": {"integrity_verified_candidate_count": integrity_verified_count},
            }
        )
    return plan


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Candidate Integrity Audit",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Candidates: `{summary.get('candidate_count')}`",
        f"- Integrity-verified candidates: `{summary.get('integrity_verified_candidate_count')}`",
        f"- Integrity mismatches: `{summary.get('integrity_mismatch_count')}`",
        f"- VIEA spine view ready: `{summary.get('viea_spine_view_ready')}`",
        f"- VIEA spine records: `{summary.get('viea_spine_view_record_count')}` claim/proof `{summary.get('viea_spine_claim_ledger_entry_count')}`",
        "",
        "## Family Counts",
        "",
    ]
    for family, count in (summary.get("family_counts") or {}).items():
        lines.append(f"- `{family}`: `{count}`")
    lines.extend(["", "## Mismatch Counts", ""])
    mismatch_counts = summary.get("integrity_mismatch_counts") or {}
    if mismatch_counts:
        for reason, count in mismatch_counts.items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    if report.get("mismatch_samples"):
        lines.extend(["", "## Samples", ""])
        for sample in report["mismatch_samples"][:20]:
            lines.append(
                f"- `{sample.get('family')}` `{sample.get('candidate_generation_mode')}` "
                f"{sample.get('mismatches')} sha=`{sample.get('candidate_sha256')}`"
            )
    if report.get("private_repair_plan"):
        lines.extend(["", "## Private Repair Plan", ""])
        for item in report["private_repair_plan"]:
            lines.append(f"- `{item.get('priority')}` `{item.get('target')}`: {item.get('action')}")
    lines.append("")
    return "\n".join(lines)


def confidence_for(family: str, reasons: list[str], shape: dict[str, Any]) -> float:
    if family == "unknown":
        return 0.2
    if not shape.get("syntax_valid"):
        return 0.95
    if reasons:
        return 0.9
    return 0.5


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
