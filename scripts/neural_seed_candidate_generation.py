#!/usr/bin/env python3
"""Candidate beam generation helpers for neural seed token decoders.

This module owns task-blind token beam expansion, final beam sorting, and decoded
candidate merging. It uses only model logits and generated token prefixes; it
never reads tests, solutions, public benchmark payloads, answer templates, or
verifier outcomes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from neural_seed_code_proposer_comparator import stable_hash  # noqa: E402
from neural_seed_static_coherence import (  # noqa: E402
    candidate_static_coherence,
    expression_hygiene_counts,
    local_static_type_bindings,
)
from neural_seed_token_decoder_support import (  # noqa: E402
    body_like_target_mode,
    body_tokens_for_target_mode,
    decode_body_tokens,
    grammar_constrained_token_choices,
    normalize_body_text,
    syntax_complete_body_prefix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCM_CONTEXT_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"


def vcm_context_governor_receipt(path: Path = DEFAULT_VCM_CONTEXT_GOVERNOR) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except json.JSONDecodeError:
            report = {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    trigger_state = str(report.get("trigger_state") or "")
    hard_gap_count = int_number(summary.get("hard_gap_count"))
    ready = bool(
        path.exists()
        and trigger_state == "GREEN"
        and hard_gap_count == 0
        and str(summary.get("mission_brief_status") or "") == "ready"
        and str(summary.get("deletion_closure_status") or "") == "closed"
        and str(summary.get("scif_status") or "") == "ready"
    )
    receipt_basis = {
        "path": rel_path(path),
        "created_utc": report.get("created_utc"),
        "trigger_state": trigger_state,
        "summary": summary,
    }
    return {
        "policy": str(report.get("policy") or ""),
        "report": rel_path(path),
        "receipt_id": f"direct_generator_vcm_governor-{stable_payload_hash(receipt_basis)[:16]}",
        "trigger_state": trigger_state,
        "ready": ready,
        "hard_gap_count": hard_gap_count,
        "warning_count": int_number(summary.get("warning_count")),
        "mission_brief_status": str(summary.get("mission_brief_status") or ""),
        "deletion_closure_status": str(summary.get("deletion_closure_status") or ""),
        "scif_status": str(summary.get("scif_status") or ""),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def vcm_candidate_metadata(vcm_receipt: dict[str, Any]) -> dict[str, Any]:
    ready = bool(vcm_receipt.get("ready"))
    return {
        "vcm_context_governor_ready": ready,
        "vcm_context_governor_receipt_id": str(vcm_receipt.get("receipt_id") or ""),
        "vcm_context_adequacy_state": (
            "governed_sufficient_for_direct_generation"
            if ready
            else "missing_or_insufficient_governed_generation_context"
        ),
        "vcm_context_fail_closed": not ready,
        "vcm_context_policy": "direct_candidate_generation_requires_governed_vcm_context_v1",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def vcm_context_blocked_candidate_batches(task_count: int, vcm_receipt: dict[str, Any]) -> list[list[dict[str, Any]]]:
    metadata = vcm_candidate_metadata(vcm_receipt)
    return [
        [
            {
                "body": "",
                "decoded_tokens": [],
                "rank_score": -999.0,
                "decoded_token_count": 0,
                "decoded_token_sha256": stable_hash(f"vcm_context_blocked:{index}:{metadata['vcm_context_governor_receipt_id']}"),
                "beam_source": "governed_vcm_context_missing_fail_closed",
                "structured_failure": "missing_or_insufficient_governed_generation_context",
                "candidate_generation_credit": 0,
                **metadata,
            }
        ]
        for index in range(max(0, int(task_count)))
    ]


def direct_generator_vcm_smoke() -> dict[str, Any]:
    receipt = vcm_context_governor_receipt()
    metadata = vcm_candidate_metadata(receipt)
    return {
        "policy": "project_theseus_direct_generator_vcm_context_smoke_v1",
        "ready": bool(receipt.get("ready")),
        "receipt": receipt,
        "metadata": metadata,
        "generation_boundary": {
            "checked_before_model_decode": True,
            "fail_closed_on_missing_context": True,
            "blocked_candidate_body": "",
            "blocked_candidate_generation_credit": 0,
        },
        "score_semantics": "Direct generator context smoke only; no training, decoding, public calibration, or learned-generation promotion claim.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def direct_generator_viea_records(
    smoke: dict[str, Any],
    *,
    report_ref: str = "reports/neural_seed_token_decoder_comparator.json",
) -> list[dict[str, Any]]:
    receipt = smoke.get("receipt") if isinstance(smoke.get("receipt"), dict) else {}
    metadata = smoke.get("metadata") if isinstance(smoke.get("metadata"), dict) else {}
    ready = bool(smoke.get("ready"))
    suffix = stable_payload_hash({"report_ref": report_ref, "receipt_id": receipt.get("receipt_id")})[:16]
    run_id = f"direct_generator_vcm-{suffix}"
    claim_id = f"claim_direct_generator_vcm-{suffix}"
    common = {
        "run_id": run_id,
        "producer_surface": "neural_seed_candidate_generation",
        "support_state": "SUPPORTED" if ready else "BLOCKED",
        "vcm_context_governor_ready": ready,
        "vcm_context_governor_receipt_id": str(receipt.get("receipt_id") or ""),
        "vcm_context_adequacy_state": str(metadata.get("vcm_context_adequacy_state") or ""),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return [
        {
            **common,
            "record_type": "context_transaction_record",
            "record_id": f"direct_generator_context_transaction-{suffix}",
            "transaction_id": f"direct_generator_context_txn-{suffix}",
            "operation": "read",
            "mounts": ["vcm_context_governor", "direct_generator_boundary"],
            "read_set": [rel_path(DEFAULT_VCM_CONTEXT_GOVERNOR)],
            "write_set": [report_ref],
            "taint_labels": ["governed_context_receipt", "candidate_generation_boundary"],
            "closure_state": "closed_before_model_decode" if ready else "blocked_before_model_decode",
            "faults": [] if ready else ["vcm_context_governor_not_ready"],
        },
        {
            **common,
            "record_type": "context_adequacy_record",
            "record_id": f"direct_generator_context_adequacy-{suffix}",
            "adequacy_id": f"direct_generator_context_adequacy-{suffix}",
            "target_claim_id": claim_id,
            "adequacy_state": str(metadata.get("vcm_context_adequacy_state") or ""),
            "governor_ready": ready,
            "governor_receipt_id": str(receipt.get("receipt_id") or ""),
            "fail_closed": not ready,
            "compression_path": "vcm_governor_receipt_to_direct_generator_boundary",
            "semantic_units": [
                {
                    "title": "VCM context governor receipt",
                    "source_path": rel_path(DEFAULT_VCM_CONTEXT_GOVERNOR),
                    "address": rel_path(DEFAULT_VCM_CONTEXT_GOVERNOR),
                    "taints": ["governed_context_receipt"],
                }
            ],
        },
        {
            **common,
            "record_type": "runtime_adapter_invocation",
            "record_id": f"direct_generator_runtime_adapter-{suffix}",
            "adapter_id": "neural_seed_candidate_generation.generate_candidates",
            "status": "READY" if ready else "BLOCKED",
            "checked_before_model_decode": True,
        },
        {
            **common,
            "record_type": "resource_budget_record",
            "record_id": f"direct_generator_resource_budget-{suffix}",
            "budget_id": f"direct_generator_budget-{suffix}",
            "heavy_training_started_by_record": False,
            "score_semantics": "planned boundary smoke only; no training or decode executed by this record.",
        },
        {
            **common,
            "record_type": "generation_mode_record",
            "record_id": f"direct_generator_generation_mode-{suffix}",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "state": "direct_generation_boundary_context_gate_only",
            "non_claim": "This proves the direct generator has a governed-context boundary, not that generated code is correct.",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": f"direct_generator_failure_boundary-{suffix}",
            "failure_id": f"direct_generator_boundary-{suffix}",
            "fallback_return_used": False,
            "structured_non_solved": not ready,
            "terminal": False,
            "status": "READY" if ready else "BLOCKED",
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": f"direct_generator_authority_use-{suffix}",
            "authority_scope": ["local_model_decode_boundary", "governed_vcm_context_read"],
            "state": "local_decode_boundary_no_public_scoring_no_external_inference",
            "status": "READY" if ready else "BLOCKED",
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": f"direct_generator_artifact-{suffix}",
            "artifact_kind": "direct_generator_vcm_boundary_smoke",
            "content_hash": stable_payload_hash(smoke),
            "evidence_ref": report_ref,
            "context_refs": [rel_path(DEFAULT_VCM_CONTEXT_GOVERNOR), str(receipt.get("receipt_id") or "")],
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": f"direct_generator_claim-{suffix}",
            "claim_id": claim_id,
            "state": "direct_generator_context_boundary_present",
            "status": "GREEN" if ready else "RED",
            "evidence_ref": report_ref,
            "claim_boundary": "context_boundary_only_not_capability_promotion",
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": f"direct_generator_evidence_transition-{suffix}",
            "state": "vcm_governor_receipt_to_direct_generator_boundary_smoke",
            "status": "SUPPORTED" if ready else "BLOCKED",
            "evidence_ref": report_ref,
        },
    ]


def int_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def stable_payload_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def generate_candidates(
    model: Any,
    source_rows: list[list[int]],
    target_vocab: dict[str, int],
    *,
    max_target_tokens: int,
    fanout_top_k: int,
    grammar_top_k: int,
    decode_beam_width: int = 0,
    decode_branching_factor: int = 0,
    target_mode: str = "body_tokens",
    body_token_decode_policy: str = "lightweight_python_v1",
    torch: Any,
    device: Any,
    allowed_name_sets: list[set[str]] | None = None,
    vcm_context_receipt: dict[str, Any] | None = None,
    require_vcm_context: bool = True,
) -> list[list[dict[str, Any]]]:
    vcm_receipt = vcm_context_receipt or vcm_context_governor_receipt()
    if require_vcm_context and not bool(vcm_receipt.get("ready")):
        return vcm_context_blocked_candidate_batches(len(source_rows), vcm_receipt)
    vcm_metadata = vcm_candidate_metadata(vcm_receipt)
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    bos = target_vocab["<bos>"]
    eos = target_vocab["<eos>"]
    src = torch.tensor(source_rows, dtype=torch.long, device=device)
    model.eval()
    all_rows = []
    with torch.no_grad():
        for i in range(src.shape[0]):
            allowed_names = allowed_name_sets[i] if allowed_name_sets and i < len(allowed_name_sets) else None
            incremental = hasattr(model, "init_decode_state") and hasattr(model, "decode_next_logits")
            initial_state = model.init_decode_state(src[i : i + 1]) if incremental else None
            beam_width = int(decode_beam_width or 0)
            if beam_width <= 0:
                beam_width = max(8, int(fanout_top_k or 1) * 4)
            beam_width = max(int(fanout_top_k or 1), beam_width)
            branch_factor = int(decode_branching_factor or 0)
            if branch_factor <= 0:
                branch_factor = max(2, min(6, int(fanout_top_k or 1) + 1))
            beams: list[dict[str, Any]] = [
                {"generated": [bos], "logprob": 0.0, "done": False, "state": initial_state}
            ]
            completed: list[dict[str, Any]] = []
            for _step in range(max_target_tokens - 1):
                expanded: list[dict[str, Any]] = []
                for beam in beams:
                    generated = list(beam["generated"])
                    if bool(beam.get("done")):
                        expanded.append(beam)
                        continue
                    next_state = None
                    if incremental:
                        logits_row, next_state = model.decode_next_logits(generated[-1], beam["state"])
                        logits = logits_row[0]
                    else:
                        tgt = torch.tensor([generated], dtype=torch.long, device=device)
                        logits = model(src[i : i + 1], tgt)[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    if target_mode == "body_tokens":
                        next_choices = grammar_constrained_token_choices(
                            probs,
                            inverse,
                            generated,
                            eos_id=eos,
                            grammar_top_k=grammar_top_k,
                            max_choices=branch_factor,
                            token_policy=body_token_decode_policy,
                            allowed_names=allowed_names,
                            torch=torch,
                        )
                    else:
                        next_choices = unconstrained_token_choices(
                            probs,
                            inverse,
                            generated,
                            eos_id=eos,
                            max_choices=branch_factor,
                            torch=torch,
                        )
                    for next_id, prob in next_choices:
                        next_generated = generated + [int(next_id)]
                        next_logprob = float(beam["logprob"]) + math.log(max(float(prob), 1e-9))
                        expanded.append(
                            {
                                "generated": next_generated,
                                "logprob": next_logprob,
                                "done": int(next_id) == eos,
                                "state": next_state,
                            }
                        )
                        prefix_tokens = [inverse.get(idx, "<unk>") for idx in next_generated[1:]]
                        if target_mode == "body_tokens" and int(next_id) != eos and syntax_complete_body_prefix(prefix_tokens):
                            completed.append(
                                {
                                    "generated": next_generated + [eos],
                                    "logprob": next_logprob,
                                    "done": True,
                                    "state": next_state,
                                    "completion_source": "syntax_complete_prefix",
                                }
                            )
                if not expanded:
                    break
                expanded.sort(
                    key=decode_beam_sort_key,
                    reverse=True,
                )
                beams = expanded[:beam_width]
                if all(bool(row.get("done")) for row in beams):
                    break
            seen: set[str] = set()
            task_rows = []
            ranked_beams = list(beams) + completed
            ranked_beams.sort(key=lambda row: final_decode_beam_sort_key(row, inverse, target_mode=target_mode), reverse=True)
            for beam in ranked_beams:
                generated = list(beam["generated"])
                decoded_tokens = [inverse.get(idx, "<unk>") for idx in generated[1:]]
                body = decode_body_tokens(decoded_tokens) if target_mode == "body_tokens" else ""
                token_sha = stable_hash(" ".join(str(idx) for idx in generated))
                body_sha = (
                    stable_hash(normalize_body_text(body))
                    if target_mode == "body_tokens"
                    else stable_hash(" ".join(decoded_tokens))
                )
                if body_sha in seen:
                    continue
                seen.add(body_sha)
                task_rows.append(
                    {
                        "body": body,
                        "decoded_tokens": decoded_tokens,
                        "rank_score": round(float(beam["logprob"]) / max(1, len(generated) - 1), 8),
                        "decoded_token_count": len(generated) - 1,
                        "decoded_token_sha256": token_sha,
                        "beam_source": str(
                            beam.get("completion_source")
                            or (
                                "direct_grammar_constrained_token_beam"
                                if target_mode == "body_tokens"
                                else f"direct_{target_mode}_token_beam"
                            )
                        ),
                        **vcm_metadata,
                    }
                )
                if len(task_rows) >= int(fanout_top_k or 1):
                    break
            if not task_rows:
                generated = [bos, eos]
                decoded_tokens = ["<eos>"]
                task_rows.append(
                    {
                        "body": "",
                        "decoded_tokens": decoded_tokens,
                        "rank_score": -999.0,
                        "decoded_token_count": 1,
                        "decoded_token_sha256": stable_hash(" ".join(str(idx) for idx in generated)),
                        "beam_source": "direct_grammar_constrained_token_beam_empty",
                        **vcm_metadata,
                    }
                )
            all_rows.append(task_rows)
    return all_rows


def unconstrained_token_choices(
    probs: Any,
    inverse: dict[int, str],
    generated: list[int],
    *,
    eos_id: int,
    max_choices: int,
    torch: Any,
) -> list[tuple[int, float]]:
    top_values, top_indices = torch.topk(probs, k=min(max(max_choices * 3, max_choices, 4), probs.numel()))
    choices: list[tuple[int, float]] = []
    seen: set[int] = set()
    for value, idx in zip(top_values, top_indices):
        next_id = int(idx)
        if next_id in seen:
            continue
        tok = inverse.get(next_id, "<unk>")
        if tok in {"<pad>", "<bos>", "<unk>"}:
            continue
        if next_id == eos_id and len(generated) <= 1:
            continue
        seen.add(next_id)
        choices.append((next_id, float(value)))
        if len(choices) >= max(1, max_choices):
            break
    if choices:
        return choices
    return [(eos_id, float(max(float(probs[eos_id]), 1e-9)))]


def decode_beam_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    generated = list(row.get("generated") or [])
    norm = float(row.get("logprob") or 0.0) / max(1, len(generated) - 1)
    done_bonus = 0.02 if bool(row.get("done")) else 0.0
    return norm + done_bonus, -0.0001 * len(generated)


def final_decode_beam_sort_key(
    row: dict[str, Any],
    inverse: dict[int, str],
    *,
    target_mode: str,
) -> tuple[Any, ...]:
    generated = list(row.get("generated") or [])
    if not body_like_target_mode(target_mode):
        norm = float(row.get("logprob") or 0.0) / max(1, len(generated) - 1)
        done = int(bool(row.get("done")))
        return done, 0, 0, 0, 0, 0, 0, 0, 0.0, norm, -0.0001 * len(generated)
    decoded_tokens = [inverse.get(idx, "<unk>") for idx in generated[1:]]
    body = decode_body_tokens(body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode))
    syntax_ok = 0
    literal_call_free = 0
    invalid_receiver_free = 0
    builtin_type_descriptor_receiver_free = 0
    bare_builtin_condition_free = 0
    invalid_known_builtin_arity_free = 0
    invalid_known_local_receiver_free = 0
    invalid_known_local_call_free = 0
    invalid_known_local_iter_free = 0
    invalid_multi_assign_free = 0
    mutating_method_return_value_free = 0
    ignored_pure_call_expression_free = 0
    parameter_free_literal_return_free = 0
    no_effect_expression_free = 0
    undefined_free = 0
    valued_return = 0
    nontrivial_return = 0
    top_level_valued_return = 0
    nested_return_free = 0
    bare_return_free = 0
    parameter_use = 0
    coherence_score = -999.0
    try:
        parsed = ast.parse("def _theseus_candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n")
        syntax_ok = 1
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
        if function is not None:
            hygiene = expression_hygiene_counts(function, local_static_types=local_static_type_bindings(function))
            literal_call_free = int(int(hygiene.get("literal_call_count", 0)) == 0)
            invalid_receiver_free = int(int(hygiene.get("invalid_receiver_count", 0)) == 0)
            builtin_type_descriptor_receiver_free = int(
                int(hygiene.get("builtin_type_descriptor_receiver_count", 0)) == 0
            )
            bare_builtin_condition_free = int(int(hygiene.get("bare_builtin_condition_count", 0)) == 0)
            invalid_known_builtin_arity_free = int(int(hygiene.get("invalid_known_builtin_arity_count", 0)) == 0)
            invalid_known_local_receiver_free = int(
                int(hygiene.get("invalid_known_local_receiver_count", 0)) == 0
            )
            invalid_known_local_call_free = int(int(hygiene.get("invalid_known_local_call_count", 0)) == 0)
            invalid_known_local_iter_free = int(int(hygiene.get("invalid_known_local_iter_count", 0)) == 0)
            invalid_multi_assign_free = int(int(hygiene.get("invalid_multi_assign_from_scalar_count", 0)) == 0)
            mutating_method_return_value_free = int(
                int(hygiene.get("mutating_method_return_value_count", 0)) == 0
            )
            ignored_pure_call_expression_free = int(
                int(hygiene.get("ignored_pure_call_expression_count", 0)) == 0
            )
            no_effect_expression_free = int(int(hygiene.get("no_effect_expression_count", 0)) == 0)
            coherence = candidate_static_coherence(
                "def _theseus_candidate(data, other=None):\n"
                + "\n".join(f"    {line}" for line in body.splitlines())
                + "\n"
            )
            parameter_free_literal_return_free = int(
                int(coherence.get("parameter_free_literal_expression_return_count", 999)) == 0
            )
            undefined_free = int(int(coherence.get("undefined_name_count", 999)) == 0)
            valued_return = int(int(coherence.get("valued_return_count", 0)) > 0)
            nontrivial_return = int(int(coherence.get("nontrivial_return_count", 0)) > 0)
            top_level_valued_return = int(int(coherence.get("top_level_valued_return_count", 0)) > 0)
            nested_return_free = int(
                int(coherence.get("nested_return_count", 999)) == 0
                or int(coherence.get("top_level_valued_return_count", 0)) > 0
            )
            bare_return_free = int(int(coherence.get("bare_return_count", 999)) == 0)
            parameter_use = int(int(coherence.get("used_parameter_count", 0)) > 0)
            coherence_score = float(coherence.get("score") or -999.0)
    except SyntaxError:
        syntax_ok = 0
    norm = float(row.get("logprob") or 0.0) / max(1, len(generated) - 1)
    return (
        syntax_ok,
        literal_call_free,
        invalid_receiver_free,
        builtin_type_descriptor_receiver_free,
        bare_builtin_condition_free,
        invalid_known_builtin_arity_free,
        invalid_known_local_receiver_free,
        invalid_known_local_call_free,
        invalid_known_local_iter_free,
        invalid_multi_assign_free,
        mutating_method_return_value_free,
        ignored_pure_call_expression_free,
        parameter_free_literal_return_free,
        no_effect_expression_free,
        undefined_free,
        nontrivial_return,
        valued_return,
        top_level_valued_return,
        nested_return_free,
        bare_return_free,
        parameter_use,
        coherence_score,
        norm,
        -0.0001 * len(generated),
    )



def merge_decoded_candidates(
    decoded: list[list[dict[str, Any]]],
    extra: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    return [list(base) + list(additional) for base, additional in zip(decoded, extra)]
