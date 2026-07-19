#!/usr/bin/env python3
"""Materialize licensed auxiliary objectives for canonical MoECOT arms.

This owner materializes code denoising and the KERC English objective views. It
does not train another model or grant capability credit to deterministic record
validation and compilation.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from moecot_language_supervision import (
    BoundedRows,
    now,
    read_json,
    relative,
    resolve,
    sha256_file,
    write_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import (
    bound_logical_tokens,
    decode_target_tokens,
    encode_tokens,
    populate_open_vocab,
)
from kernel_english_protocol import (
    KernelProtocolFault,
    KERC_VERIFIER_DIMENSIONS,
    SEMANTIC_EVIDENCE_TIERS,
    TRAINING_OBJECTIVES,
    TRAINING_VERIFICATION_POLICY,
    canonical_json,
    compile_training_views,
    kernel_training_contract,
    learned_residual_view,
    stable_hash,
    validate_training_disposition,
    validate_training_record,
)
from kerc_content_cache import (
    CACHE_POLICY,
    cache_key as kerc_cache_key,
    dependency_bindings,
    load_receipt as load_kerc_cache_receipt,
    publish_receipt as publish_kerc_cache_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")
KERC_KERNEL_OBJECTIVES = {
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
}
KERC_STRUCTURED_SOURCE_OBJECTIVES = {
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
    "answer_packet_to_surface_v1",
}
KERC_SEQUENCE_BUCKET_POLICY = "project_theseus_kerc_exact_sequence_buckets_v1"
KERC_POINTER_TOKEN_RE = re.compile(
    r"(?:@[A-Z][A-Za-z0-9_]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\Z"
)
KERC_POINTER_CONTROL_TOKENS = {
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    ":",
    ",",
    '"',
    "\\",
    " ",
    "\n",
    "\r",
    "\t",
}
KERC_COMPACT_TOKEN_PREFIXES = (
    "VERSION:",
    "SERIALIZATION:",
    "NODE_",
    "OP:",
    "MOD:",
    "POL:",
    "QUANT:",
    "CONF:",
    "DERIV:",
    "SPANS:",
    "ROLE:",
    "HANDLE:",
    "CONCEPT:",
    "NUMBER:",
    "QUANTITY:",
    "TEMPORAL:",
    "TEXT:",
    "SYMBOL:",
    "NODE_REF:",
    "LIST_",
    "AMBIG_",
    "PROB:",
    "EVIDENCE:",
    "BYTE:",
    "BOOL:",
    "NULL",
    "ROOT:",
    "PROGRAM_END",
    "ANSWER_VERSION:",
    "CLAIM_",
    "PRED:",
    "DECISION_",
    "DISPOSITION:",
    "UNCERTAINTY:",
    "CONTROLLING:",
    "AMBIGUITY_ID:",
    "REQUIRED_TERM:",
    "REQUIRED_CAVEAT:",
    "STYLE:",
    "ANSWER_END",
    "MACRO:",
)
KERC_SOURCE_CATALOG_POLICY = "project_theseus_kerc_semantic_source_catalog_v1"
KERC_SEMANTIC_PROGRAM_POLICY = "project_theseus_kerc_semantic_supervision_program_v1"
KERC_SEMANTIC_CORPUS_POLICY = "project_theseus_kerc_semantic_corpus_materialization_v1"
KERC_SELECTION_POLICY = "project_theseus_kerc_constraint_aware_selection_v1"
KERC_PARALLEL_MATERIALIZATION_POLICY = (
    "project_theseus_kerc_bounded_parallel_materialization_v1"
)
KERC_CONTEXT_COUNTERFACTUAL_POLICY = (
    "project_theseus_kerc_grounded_context_counterfactual_v1"
)


class KercCodeToken(str):
    """Lossless token text carrying its typed code-space through byte bounding."""

    code_space: str

    def __new__(cls, value: str, code_space: str) -> "KercCodeToken":
        instance = str.__new__(cls, value)
        instance.code_space = code_space
        return instance


KERC_CONTEXT_COUNTERFACTUAL_OBJECTIVES = {
    "surface_direct_control_v1",
    "kernel_program_to_answer_packet_v1",
}
KERC_CONTEXT_COUNTERFACTUAL_FAILED_DIMENSIONS = (
    "semantic_consistency",
    "answer_decision_consistency",
)


def _replace_exact_strings(value: Any, replacements: dict[str, str]) -> Any:
    """Replace bound identities without performing unsafe substring rewrites."""

    if isinstance(value, dict):
        return {
            key: _replace_exact_strings(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_strings(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _replace_interaction_context(
    prompt: dict[str, Any], *, replacement: str | None
) -> bool:
    """Replace or remove the governed document-context entry in a prompt."""

    interaction = prompt.get("interaction")
    if not isinstance(interaction, list):
        residual = prompt.get("residual")
        interaction = residual.get("interaction") if isinstance(residual, dict) else None
    if not isinstance(interaction, list):
        return False
    matches = [
        index
        for index, entry in enumerate(interaction)
        if isinstance(entry, list)
        and len(entry) == 3
        and entry[0] == "document_context"
        and entry[1] == "content"
    ]
    if len(matches) != 1:
        return False
    index = matches[0]
    if replacement is None:
        del interaction[index]
    else:
        interaction[index][2] = replacement
    return True


def _rehash_bound_program(prompt: dict[str, Any]) -> None:
    program = prompt.get("program")
    if not isinstance(program, dict):
        return
    unsigned = {key: value for key, value in program.items() if key != "program_sha256"}
    program["program_sha256"] = stable_hash(unsigned)


def _rehash_answer_packet(packet: dict[str, Any]) -> None:
    unsigned = {
        key: value for key, value in packet.items() if key != "answer_packet_sha256"
    }
    packet["answer_packet_sha256"] = stable_hash(unsigned)


def _grounded_context_record(record: dict[str, Any]) -> dict[str, str] | None:
    annotation = record.get("interaction_annotation")
    source = record.get("source_annotation")
    provenance = record.get("provenance")
    if (
        not isinstance(annotation, dict)
        or annotation.get("kind") != "licensed_grounded_question_context"
        or not isinstance(source, dict)
        or not isinstance(provenance, dict)
    ):
        return None
    context = str(source.get("context") or "")
    answer = str(source.get("response") or record.get("surface_target") or "")
    context_sha256 = str(annotation.get("context_sha256") or "")
    if not context or not answer or not re.fullmatch(r"sha256:[0-9a-f]{64}", context_sha256):
        return None
    return {
        "record_sha256": str(record.get("record_sha256") or ""),
        "source_group": str(provenance.get("source_group") or ""),
        "context": context,
        "answer": answer,
        "context_sha256": context_sha256,
    }


def attach_grounded_context_counterfactuals(
    compiled_views: dict[str, list[dict[str, Any]]],
    selected_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Attach source-disjoint verifier-only context interventions.

    Donors stay within the same split, never contain the original answer, and
    receive fresh program/answer checksums. The negative signal therefore tests
    answer support rather than a stale-hash shortcut.
    """

    grounded_by_split: dict[str, list[dict[str, str]]] = {}
    grounded_by_record: dict[str, dict[str, str]] = {}
    for split, records in selected_records.items():
        grounded = [
            item
            for record in records
            if (item := _grounded_context_record(record)) is not None
        ]
        grounded_by_split[split] = sorted(
            grounded, key=lambda item: item["record_sha256"]
        )
        grounded_by_record.update(
            {item["record_sha256"]: item for item in grounded}
        )

    counts: Counter[str] = Counter()
    missing_donors: list[str] = []
    for split, views in compiled_views.items():
        split_grounded = grounded_by_split.get(split) or []
        for view in views:
            objective = str(view.get("objective") or "")
            source_record = grounded_by_record.get(
                str(view.get("source_record_sha256") or "")
            )
            if source_record is None or objective not in KERC_CONTEXT_COUNTERFACTUAL_OBJECTIVES:
                continue
            donor_candidates = [
                donor
                for donor in split_grounded
                if donor["source_group"] != source_record["source_group"]
                and source_record["answer"].casefold().strip()
                not in donor["context"].casefold()
            ]
            donor_candidates.sort(
                key=lambda donor: stable_hash(
                    {
                        "source_record_sha256": source_record["record_sha256"],
                        "donor_record_sha256": donor["record_sha256"],
                        "objective": objective,
                    }
                )
            )
            if not donor_candidates:
                missing_donors.append(source_record["record_sha256"])
                continue
            donor = donor_candidates[0]
            original_prompt = json.loads(str(view["prompt"]))
            old_hash = source_record["context_sha256"]
            new_hash = donor["context_sha256"]
            old_hash_b64 = base64.b64encode(old_hash.encode()).decode()
            new_hash_b64 = base64.b64encode(new_hash.encode()).decode()
            labels = [1] * len(KERC_VERIFIER_DIMENSIONS)
            labels[0] = 0
            labels[4] = 0
            counterfactuals: list[dict[str, Any]] = []

            withheld_prompt = copy.deepcopy(original_prompt)
            if not _replace_interaction_context(withheld_prompt, replacement=None):
                raise ValueError(
                    "grounded KERC prompt is missing its document context: "
                    + source_record["record_sha256"]
                )
            withheld_prompt_text = canonical_json(withheld_prompt)
            counterfactuals.append(
                {
                    "policy": KERC_CONTEXT_COUNTERFACTUAL_POLICY,
                    "strategy": "context_withheld",
                    "prompt": withheld_prompt_text,
                    "prompt_sha256": stable_hash(withheld_prompt_text.encode()),
                    "target": str(view["target"]),
                    "target_sha256": str(view["target_sha256"]),
                    "labels": labels,
                    "failed_dimensions": list(
                        KERC_CONTEXT_COUNTERFACTUAL_FAILED_DIMENSIONS
                    ),
                    "donor_record_sha256": "",
                    "donor_source_group": "",
                    "generator_loss_enabled": False,
                    "unique_source_credit": 0,
                    "candidate_generation_credit": 0,
                }
            )

            shuffled_prompt = _replace_exact_strings(
                copy.deepcopy(original_prompt),
                {old_hash: new_hash, old_hash_b64: new_hash_b64},
            )
            if not _replace_interaction_context(
                shuffled_prompt, replacement=donor["context"]
            ):
                raise ValueError(
                    "grounded KERC prompt is missing its shuffled context: "
                    + source_record["record_sha256"]
                )
            _rehash_bound_program(shuffled_prompt)
            shuffled_target = str(view["target"])
            if objective == "kernel_program_to_answer_packet_v1":
                packet = _replace_exact_strings(
                    json.loads(shuffled_target),
                    {old_hash: new_hash, old_hash_b64: new_hash_b64},
                )
                _rehash_answer_packet(packet)
                shuffled_target = canonical_json(packet)
            shuffled_prompt_text = canonical_json(shuffled_prompt)
            counterfactuals.append(
                {
                    "policy": KERC_CONTEXT_COUNTERFACTUAL_POLICY,
                    "strategy": "context_shuffled",
                    "prompt": shuffled_prompt_text,
                    "prompt_sha256": stable_hash(shuffled_prompt_text.encode()),
                    "target": shuffled_target,
                    "target_sha256": stable_hash(shuffled_target.encode()),
                    "labels": labels,
                    "failed_dimensions": list(
                        KERC_CONTEXT_COUNTERFACTUAL_FAILED_DIMENSIONS
                    ),
                    "donor_record_sha256": donor["record_sha256"],
                    "donor_source_group": donor["source_group"],
                    "donor_context_sha256": new_hash,
                    "answer_absent_from_donor_context": True,
                    "generator_loss_enabled": False,
                    "unique_source_credit": 0,
                    "candidate_generation_credit": 0,
                }
            )
            view["kerc_context_counterfactuals"] = counterfactuals
            evaluator_only = list(view.get("evaluator_only_fields") or [])
            if "kerc_context_counterfactuals" not in evaluator_only:
                evaluator_only.append("kerc_context_counterfactuals")
            view["evaluator_only_fields"] = evaluator_only
            for counterfactual in counterfactuals:
                counts[
                    f"{split}:{objective}:{counterfactual['strategy']}"
                ] += 1

    return {
        "policy": KERC_CONTEXT_COUNTERFACTUAL_POLICY,
        "counts": dict(sorted(counts.items())),
        "total_count": sum(counts.values()),
        "missing_donor_record_sha256": sorted(set(missing_donors)),
        "generator_loss_enabled": False,
        "unique_source_credit": 0,
        "candidate_generation_credit": 0,
        "claim_scope": "source-grounded_counterfactual_support_sensitivity_only",
    }


def kerc_code_tokens(text: str) -> list[str]:
    """Losslessly tokenize typed Kernel/answer JSON as structural atoms.

    Generic surface tokenization fragmented every quoted Kernel token into JSON
    punctuation and word pieces. That made a 2,970-token Kernel program consume
    tens of thousands of model positions. Quoted JSON atoms are already bounded
    by the reversible byte codec, so keep each as one logical atom (or bounded
    adjacent atoms) while preserving exact concatenation.
    """

    raw: list[KercCodeToken] = []
    for atom in _exact_json_string_atoms(text):
        space = _kerc_code_space_text(atom)
        raw.extend(
            KercCodeToken(piece, space)
            for piece in bound_logical_tokens([atom])
        )
    tokens: list[str] = []
    index = 0
    while index < len(raw):
        if (
            raw[index] == "@"
            and index + 1 < len(raw)
            and str(raw[index + 1]).replace("_", "").isalnum()
        ):
            tokens.append(
                KercCodeToken("@" + str(raw[index + 1]), "V_P")
            )
            index += 2
            continue
        tokens.append(raw[index])
        index += 1
    if "".join(tokens) != str(text):
        raise ValueError("KERC code tokenizer failed exact reconstruction")
    return tokens


def _exact_json_string_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    outside_start = 0
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        if outside_start < index:
            atoms.extend(exact_text_tokens(text[outside_start:index]))
        string_start = index
        index += 1
        escaped = False
        while index < len(text):
            character = text[index]
            index += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                break
        else:
            raise ValueError("KERC code tokenizer encountered unterminated JSON string")
        atoms.append(text[string_start:index])
        outside_start = index
    if outside_start < len(text):
        atoms.extend(exact_text_tokens(text[outside_start:]))
    if "".join(atoms) != text:
        raise ValueError("KERC JSON atom tokenizer failed exact reconstruction")
    return atoms


def kerc_surface_tokens(text: str) -> list[str]:
    """Tokenize arbitrary KERC surface text without oversized unknown atoms."""

    tokens = bound_logical_tokens(exact_text_tokens(text))
    if "".join(tokens) != str(text):
        raise ValueError("KERC surface tokenizer failed exact reconstruction")
    return tokens


def kerc_code_space(token: str) -> str:
    declared = getattr(token, "code_space", None)
    if declared in {"V_K", "V_P", "V_S"}:
        return str(declared)
    return _kerc_code_space_text(str(token))


def _kerc_code_space_text(token: str) -> str:
    value = str(token)
    if len(value) >= 3 and value[0] == '"' and value[-1] == '"':
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if (
            isinstance(decoded, str)
            and len(decoded) >= 2
            and decoded[1:].startswith(KERC_COMPACT_TOKEN_PREFIXES)
        ):
            transport_space = {"K": "V_K", "P": "V_P", "S": "V_S"}.get(
                decoded[0]
            )
            if transport_space:
                return transport_space
    if (
        value in KERC_POINTER_CONTROL_TOKENS
        or value.isspace()
        or KERC_POINTER_TOKEN_RE.fullmatch(value)
    ):
        return "V_P"
    return "V_K"


def build_kerc_code_vocabulary(
    private_train_views: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Fit V_K/V_P on private-train positive targets only."""

    kernel_counts: Counter[str] = Counter()
    pointer_counts: Counter[str] = Counter()
    source_view_count = 0
    for view in private_train_views:
        if str(view.get("objective") or "") not in KERC_KERNEL_OBJECTIVES:
            continue
        source_view_count += 1
        for token in kerc_code_tokens(str(view.get("target") or "")):
            (pointer_counts if kerc_code_space(token) == "V_P" else kernel_counts)[
                token
            ] += 1
    if not source_view_count or not kernel_counts or not pointer_counts:
        raise ValueError("KERC code vocabulary requires compiler/core private-train views")
    kernel_vocab = {"<pad>": 0, "<unk>": 1}
    pointer_vocab = {"<pad>": 0, "<unk>": 1}
    populate_open_vocab(
        kernel_vocab,
        kernel_counts,
        max_vocab=int(contract["kernel_max_vocab"]),
        stream="target",
    )
    populate_open_vocab(
        pointer_vocab,
        pointer_counts,
        max_vocab=int(contract["pointer_max_vocab"]),
        stream="target",
    )
    payload = {
        "policy": "project_theseus_kerc_dual_code_vocabulary_v1",
        "fit_split": "private_train",
        "fit_positive_targets_only": True,
        "dev_eval_vocabulary_fit_count": 0,
        "verifier_corruption_vocabulary_fit_count": 0,
        "surface_vocabulary_owner": "canonical_moecot_target_vocab",
        "kernel_max_vocab": int(contract["kernel_max_vocab"]),
        "pointer_max_vocab": int(contract["pointer_max_vocab"]),
        "kernel_vocab": kernel_vocab,
        "pointer_vocab": pointer_vocab,
        "kernel_observed_token_count": int(sum(kernel_counts.values())),
        "pointer_observed_token_count": int(sum(pointer_counts.values())),
        "source_view_count": source_view_count,
        "tokenizer": "lossless_exact_json_with_typed_handle_coalescing_v1",
        "byte_fallback_required": True,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
    }
    payload["contract_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def encode_kerc_view_target(
    view: dict[str, Any],
    *,
    target_vocab: dict[str, int],
    code_vocabulary: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    objective = str(view.get("objective") or "")
    target = str(view.get("target") or "")
    if objective not in KERC_KERNEL_OBJECTIVES:
        return encode_tokens(kerc_surface_tokens(target), target_vocab, stream="target")
    ids: list[int] = []
    unknown = 0
    fallback_tokens = 0
    by_space = {"V_K": 0, "V_P": 0}
    kernel_vocab = code_vocabulary.get("kernel_vocab") or {}
    pointer_vocab = code_vocabulary.get("pointer_vocab") or {}
    for token in kerc_code_tokens(target):
        space = kerc_code_space(token)
        vocab = pointer_vocab if space == "V_P" else kernel_vocab
        encoded, receipt = encode_tokens([token], vocab, stream="target")
        ids.extend(encoded)
        unknown += int(receipt.get("unknown_token_count") or 0)
        fallback_tokens += int(receipt.get("fallback_token_count") or 0)
        by_space[space] += len(encoded)
    return ids, {
        "policy": "project_theseus_kerc_dual_code_encoding_v1",
        "unknown_token_count": unknown,
        "fallback_token_count": fallback_tokens,
        "encoded_token_count": len(ids),
        "encoded_tokens_by_space": by_space,
        "code_vocabulary_sha256": code_vocabulary.get("contract_sha256"),
        "failure_behavior": "reject_without_surface_or_template_fallback",
    }


def encode_kerc_view_source(
    view: dict[str, Any],
    *,
    source_vocab: dict[str, int],
    code_vocabulary: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    objective = str(view.get("objective") or "")
    prompt = str(view.get("prompt") or "")
    if objective not in KERC_STRUCTURED_SOURCE_OBJECTIVES:
        return encode_tokens(kerc_surface_tokens(prompt), source_vocab, stream="source")
    ids, receipt = encode_kerc_view_target(
        {
            "objective": "kernel_program_to_answer_packet_v1",
            "target": prompt,
        },
        target_vocab={},
        code_vocabulary=code_vocabulary,
    )
    return ids, {
        **receipt,
        "policy": "project_theseus_kerc_structured_source_encoding_v1",
        "objective": objective,
        "source_uses_dual_code_vocabulary": True,
    }


def encode_kerc_global_target(
    text: str,
    *,
    code_vocabulary: dict[str, Any],
    kernel_offset: int,
    pointer_offset: int,
) -> tuple[list[int], dict[str, Any]]:
    """Encode a Kernel/answer target into disjoint global V_K/V_P ranges."""

    ids: list[int] = []
    unknown = 0
    fallback_tokens = 0
    by_space = {"V_K": 0, "V_P": 0}
    kernel_vocab = code_vocabulary.get("kernel_vocab") or {}
    pointer_vocab = code_vocabulary.get("pointer_vocab") or {}
    for token in kerc_code_tokens(text):
        space = kerc_code_space(token)
        vocab = pointer_vocab if space == "V_P" else kernel_vocab
        offset = pointer_offset if space == "V_P" else kernel_offset
        encoded, receipt = encode_tokens([token], vocab, stream="target")
        ids.extend(offset + int(value) for value in encoded)
        unknown += int(receipt.get("unknown_token_count") or 0)
        fallback_tokens += int(receipt.get("fallback_token_count") or 0)
        by_space[space] += len(encoded)
    return ids, {
        "policy": "project_theseus_kerc_global_dual_code_encoding_v1",
        "unknown_token_count": unknown,
        "fallback_token_count": fallback_tokens,
        "encoded_token_count": len(ids),
        "encoded_tokens_by_space": by_space,
        "kernel_offset": int(kernel_offset),
        "pointer_offset": int(pointer_offset),
        "code_vocabulary_sha256": code_vocabulary.get("contract_sha256"),
        "failure_behavior": "reject_without_surface_or_template_fallback",
    }


def decode_kerc_global_target(
    ids: list[int],
    *,
    code_vocabulary: dict[str, Any],
    kernel_offset: int,
    pointer_offset: int,
) -> tuple[str, dict[str, Any]]:
    kernel_inverse = {
        int(value): str(token)
        for token, value in (code_vocabulary.get("kernel_vocab") or {}).items()
    }
    pointer_inverse = {
        int(value): str(token)
        for token, value in (code_vocabulary.get("pointer_vocab") or {}).items()
    }
    logical: list[str] = []
    by_space = {"V_K": 0, "V_P": 0}
    for global_id in ids:
        value = int(global_id)
        if kernel_offset <= value < pointer_offset:
            token = kernel_inverse.get(value - kernel_offset)
            space = "V_K"
        elif value >= pointer_offset:
            token = pointer_inverse.get(value - pointer_offset)
            space = "V_P"
        else:
            token = None
            space = ""
        if token is None:
            return "", {
                "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
                "state": "FAULT",
                "reason": "unassigned_or_cross_space_token",
                "token_id": value,
                "failure_behavior": "reject_without_surface_or_template_fallback",
            }
        logical.append(token)
        by_space[space] += 1
    decoded, receipt = decode_target_tokens(logical)
    if receipt.get("state") != "READY":
        return "", {
            "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
            "state": "FAULT",
            "reason": "byte_fallback_decode_fault",
            "open_vocab": receipt,
            "failure_behavior": "reject_without_surface_or_template_fallback",
        }
    return "".join(decoded), {
        "policy": "project_theseus_kerc_global_dual_code_decoding_v1",
        "state": "READY",
        "decoded_tokens_by_space": by_space,
        "exact_reconstruction": True,
        "fallback_return_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel-english", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    cfg = (
        validate_kernel_english_config(config)
        if args.kernel_english
        else validate_config(config)
    )
    if args.kernel_english:
        report = (
            materialize_kernel_english(config, config_path)
            if args.execute
            else inspect_kernel_english(config, config_path)
        )
    else:
        report = materialize(config, config_path) if args.execute else inspect(config, config_path)
    write_json(resolve(args.out or cfg["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PLANNED"} else 2


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("source_conditioned_pretraining")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_source_conditioned_pretraining_v1":
        raise ValueError("unexpected source-conditioned pretraining policy")
    if tuple((cfg.get("rows_by_arm") or {}).keys()) != ARM_IDS:
        raise ValueError("source-conditioned row arm set/order mismatch")
    if int((cfg.get("rows_by_arm") or {}).get("english") or 0) != 0:
        raise ValueError("code-denoising source cannot be assigned to the English arm")
    if not 0.0 < float(cfg.get("deletion_fraction") or 0.0) < 0.5:
        raise ValueError("deletion fraction must be bounded between zero and one half")
    if int(cfg.get("maximum_windows_per_document") or 0) <= 0:
        raise ValueError("maximum windows per document must be positive")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(cfg.get(key) or 0):
            raise ValueError(f"source-conditioned no-cheat counter must remain zero: {key}")
    return cfg


def validate_kernel_english_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("kernel_english_training")
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("policy") != "project_theseus_moecot_kernel_english_stage_v1":
        raise ValueError("unexpected KERC training-stage policy")
    disposition = validate_training_disposition(cfg)
    full_kerc_enabled = disposition.get("full_kerc_training_enabled") is True
    if tuple(cfg.get("objective_order") or ()) != TRAINING_OBJECTIVES:
        raise ValueError("KERC objective order/identity mismatch")
    rows = cfg.get("records_by_split") or {}
    if tuple(rows) != ("private_train", "private_dev", "private_eval"):
        raise ValueError("KERC record split set/order mismatch")
    selection = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
    expected_selection_fields = [
        "record_sha256",
        "raw_source_sha256",
        "split",
        "semantic_supervision.claim_authority",
        "semantic_supervision.objective_authority",
        "interaction_annotation.kind",
    ]
    if (
        selection.get("policy") != KERC_SELECTION_POLICY
        or not str(selection.get("ranking_seed") or "")
        or selection.get("selection_visible_fields") != expected_selection_fields
        or selection.get("answer_text_visible") is not False
        or selection.get("model_outcomes_visible") is not False
        or selection.get("exclude_raw_sources_present_in_multiple_splits") is not True
        or selection.get("preserve_grounded_question_quota") is not True
        or selection.get("preserve_decision_grade_objective_floors") is not True
        or selection.get("fill_policy") != "content_hash_rank_after_constraints"
    ):
        raise ValueError("KERC constraint-aware selection contract is incomplete")
    if full_kerc_enabled:
        if any(int(value or 0) <= 0 for value in rows.values()):
            raise ValueError("KERC record floors must be positive for every split")
        if not cfg.get("allowed_licenses"):
            raise ValueError("KERC stage requires an explicit license allowlist")
        if not str(cfg.get("verification_ledger_jsonl") or "").strip():
            raise ValueError("KERC stage requires a separate verification ledger")
        if not str(cfg.get("semantic_source_catalog_json") or "").strip():
            raise ValueError("KERC stage requires a semantic source catalog")
        validate_kerc_semantic_program(cfg)
        validate_kerc_semantic_corpus_config(cfg)
    elif any(int(value or 0) != 0 for value in rows.values()):
        raise ValueError("retired KERC stage must request zero records")
    if int(cfg.get("maximum_sequence_tokens") or 0) <= 0:
        raise ValueError("KERC maximum sequence tokens must be positive")
    sequence_buckets = cfg.get("sequence_buckets") or {}
    bucket_rows = sequence_buckets.get("buckets") or []
    if (
        sequence_buckets.get("policy") != KERC_SEQUENCE_BUCKET_POLICY
        or sequence_buckets.get("routing")
        != "encoded_length_only_without_target_semantic_metadata"
        or not isinstance(bucket_rows, list)
        or [row.get("bucket_id") for row in bucket_rows]
        != ["standard_8k", "exact_high_fan_in_16k"]
        or [int(row.get("maximum_sequence_tokens") or 0) for row in bucket_rows]
        != [8192, int(cfg["maximum_sequence_tokens"])]
        or [int(row.get("maximum_batch_size") or 0) for row in bucket_rows]
        != [2, 1]
        or sequence_buckets.get("truncation_allowed") is not False
        or sequence_buckets.get("row_drop_allowed") is not False
        or sequence_buckets.get("long_bucket_capability_credit") is not False
    ):
        raise ValueError("KERC exact sequence-bucket contract is incomplete")
    if not 1 <= int(cfg.get("batch_size") or 0) <= 16:
        raise ValueError("KERC batch size must be bounded")
    execution = (
        cfg.get("materialization_execution")
        if isinstance(cfg.get("materialization_execution"), dict)
        else {}
    )
    if (
        execution.get("policy") != KERC_PARALLEL_MATERIALIZATION_POLICY
        or not 1 <= int(execution.get("validation_workers") or 0) <= 32
        or not 1 <= int(execution.get("compilation_workers") or 0) <= 32
        or not 1 <= int(execution.get("batch_rows") or 0) <= 256
        or execution.get("deterministic_input_order") is not True
        or execution.get("raw_line_content_binding_required") is not True
        or execution.get("worker_failure_behavior")
        != "reject_record_or_abort_stage_without_partial_trust"
    ):
        raise ValueError("KERC bounded parallel materialization contract is incomplete")
    cache = (
        cfg.get("materialization_cache")
        if isinstance(cfg.get("materialization_cache"), dict)
        else {}
    )
    if (
        cache.get("policy") != CACHE_POLICY
        or not isinstance(cache.get("enabled"), bool)
        or not str(cache.get("cache_root") or "")
        or cache.get("role") != "canonical_kerc_stage_materializer"
        or cache.get("exact_dependency_binding_required") is not True
        or cache.get("output_identity_revalidation_required") is not True
        or cache.get("cache_miss_behavior")
        != "recompute_complete_stage_and_publish_atomic_receipt"
    ):
        raise ValueError("KERC content-addressed stage cache contract is incomplete")
    vocabulary = cfg.get("code_vocabulary") or {}
    if (
        vocabulary.get("policy") != "project_theseus_kerc_dual_code_vocabulary_v1"
        or vocabulary.get("fit_split") != "private_train"
        or vocabulary.get("surface_vocabulary_owner")
        != "canonical_moecot_target_vocab"
        or vocabulary.get("byte_fallback_required") is not True
        or vocabulary.get("dev_eval_vocabulary_fit_forbidden") is not True
        or int(vocabulary.get("kernel_max_vocab") or 0) < 512
        or int(vocabulary.get("pointer_max_vocab") or 0) < 512
    ):
        raise ValueError("KERC dual-code vocabulary contract is incomplete")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(cfg.get(key) or 0):
            raise ValueError(f"KERC no-cheat counter must remain zero: {key}")
    return cfg


def validate_kerc_semantic_program(cfg: dict[str, Any]) -> dict[str, Any]:
    program = cfg.get("semantic_supervision")
    program = program if isinstance(program, dict) else {}
    if program.get("policy") != KERC_SEMANTIC_PROGRAM_POLICY:
        raise ValueError("KERC semantic-supervision program policy mismatch")
    tiers = program.get("tiers") if isinstance(program.get("tiers"), dict) else {}
    if tuple(tiers) != tuple(SEMANTIC_EVIDENCE_TIERS):
        raise ValueError("KERC semantic evidence tier set/order mismatch")
    for tier, contract in SEMANTIC_EVIDENCE_TIERS.items():
        configured = tiers.get(tier) if isinstance(tiers.get(tier), dict) else {}
        expected = {
            "claim_authority": contract["claim_authority"],
            "maximum_optimizer_sampling_weight": float(
                contract["maximum_optimizer_sampling_weight"]
            ),
            "training_only": contract["allowed_splits"] == {"private_train"},
        }
        if configured != expected:
            raise ValueError(f"KERC semantic evidence tier contract mismatch: {tier}")
    floors = program.get("minimum_decision_grade_records_by_split_and_objective") or {}
    requested = cfg.get("records_by_split") or {}
    if tuple(floors) != tuple(requested):
        raise ValueError("KERC decision-grade split floor set/order mismatch")
    for split, objective_floors in floors.items():
        if (
            not isinstance(objective_floors, dict)
            or tuple(objective_floors) != TRAINING_OBJECTIVES
        ):
            raise ValueError(f"KERC decision-grade objective floor set/order invalid: {split}")
        for objective, floor in objective_floors.items():
            if not 0 <= int(floor) <= int(requested[split]):
                raise ValueError(
                    f"KERC decision-grade objective floor invalid: {split}:{objective}"
                )
    record_caps = program.get("maximum_train_record_share_by_tier") or {}
    probability_caps = program.get("maximum_train_optimizer_probability_by_tier") or {}
    if set(record_caps) != {"local_parser_silver", "governed_openai_residual"}:
        raise ValueError("KERC train record-share caps are incomplete")
    if set(probability_caps) != {"governed_openai_residual"}:
        raise ValueError("KERC optimizer-probability cap is incomplete")
    if not 0.0 <= float(record_caps["local_parser_silver"]) <= 0.9:
        raise ValueError("KERC parser-silver record share may not exceed 0.9")
    if not 0.0 <= float(record_caps["governed_openai_residual"]) <= 0.1:
        raise ValueError("KERC teacher residual record share may not exceed 0.1")
    if not 0.0 <= float(probability_caps["governed_openai_residual"]) <= 0.02:
        raise ValueError("KERC teacher residual optimizer probability may not exceed 0.02")
    if program.get("public_semantic_benchmarks_training_forbidden") is not True:
        raise ValueError("KERC public semantic benchmarks must remain calibration-only")
    if program.get("silver_can_satisfy_decision_grade_floor") is not False:
        raise ValueError("KERC silver rows may not satisfy decision-grade floors")
    qualifications = program.get("source_qualification")
    if not isinstance(qualifications, list) or not qualifications:
        raise ValueError("KERC semantic source qualification ledger is required")
    identities: set[str] = set()
    for row in qualifications:
        if not isinstance(row, dict):
            raise ValueError("KERC semantic source qualification row is invalid")
        source_id = str(row.get("source_id") or "")
        disposition = str(row.get("disposition") or "")
        if (
            not source_id
            or source_id in identities
            or str(row.get("intended_tier") or "") not in SEMANTIC_EVIDENCE_TIERS
            or not disposition
            or not str(row.get("license_spdx") or "")
            or not str(row.get("source_url") or "")
        ):
            raise ValueError(f"KERC semantic source qualification invalid: {source_id}")
        if row.get("public_benchmark_surface") is True and disposition.startswith(
            "eligible"
        ):
            raise ValueError(f"public semantic benchmark cannot be training-eligible: {source_id}")
        identities.add(source_id)
    return program


def validate_kerc_semantic_corpus_config(cfg: dict[str, Any]) -> dict[str, Any]:
    corpus = cfg.get("semantic_corpus_materialization")
    corpus = corpus if isinstance(corpus, dict) else {}
    if corpus.get("policy") != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("KERC semantic corpus materialization policy mismatch")
    content_cache = corpus.get("content_cache")
    if (
        not isinstance(content_cache, dict)
        or content_cache.get("policy")
        != "project_theseus_kerc_content_addressed_run_cache_v1"
        or not isinstance(content_cache.get("enabled"), bool)
        or not str(content_cache.get("root") or "")
        or not str(content_cache.get("producer_role") or "")
        or not str(content_cache.get("verifier_role") or "")
        or content_cache.get("producer_role") == content_cache.get("verifier_role")
        or content_cache.get("family_identity_policy")
        != "project_theseus_kerc_source_family_identity_v1"
        or content_cache.get("producer_candidate_layer") != "candidate_record_v1"
        or content_cache.get("producer_finalization_layer")
        != "candidate_finalization_v1"
        or content_cache.get("verifier_semantic_layer") != "semantic_admission_v1"
        or content_cache.get("common_change_invalidates_all_families") is not True
        or content_cache.get("family_local_change_invalidates_only_that_family")
        is not True
    ):
        raise ValueError("KERC content-addressed cache contract invalid")
    source_names = ["dolly", "masc", "oasst2"]
    if isinstance(corpus.get("gum"), dict):
        source_names.insert(2, "gum")
    sources = {name: corpus.get(name) or {} for name in source_names}
    for name, source in sources.items():
        path_key = (
            "archive_path"
            if name == "masc"
            else "source_root"
            if name == "gum"
            else "path"
        )
        source_path_ready = (
            isinstance(source.get("files"), dict)
            and tuple(source["files"]) == ("train", "validation")
            and all(
                str(row.get("path") or "")
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}", str(row.get("content_sha256") or "")
                )
                for row in source["files"].values()
            )
            if name == "oasst2"
            else bool(str(source.get(path_key) or ""))
        )
        if (
            not source_path_ready
            or not str(source.get("dataset_id") or "")
            or not str(source.get("dataset_revision") or "")
            or not str(source.get("source_url") or "").startswith("https://")
            or not str(source.get("license_evidence_url") or "").startswith("https://")
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(source.get("content_sha256") or ""))
            or not str(source.get("license_spdx") or "")
            or tuple(source.get("records_by_split") or {})
            != ("private_train", "private_dev", "private_eval")
            or any(int(value or 0) < 0 for value in (source.get("records_by_split") or {}).values())
            or not set(source.get("allowed_objectives") or {}) <= set(TRAINING_OBJECTIVES)
            or not source.get("allowed_objectives")
        ):
            raise ValueError(f"KERC semantic corpus source contract invalid: {name}")
    requested = cfg.get("records_by_split") or {}
    grounded_counts = sources["dolly"].get("grounded_question_records_by_split")
    grounded_objectives = sources["dolly"].get("grounded_question_allowed_objectives")
    grounded_forms = sources["dolly"].get("grounded_question_required_forms")
    if (
        not isinstance(grounded_counts, dict)
        or tuple(grounded_counts) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in grounded_counts.values())
        or not isinstance(grounded_objectives, list)
        or set(grounded_objectives) != set(TRAINING_OBJECTIVES)
        or not isinstance(grounded_forms, list)
        or len(grounded_forms) < 4
        or len(set(grounded_forms)) != len(grounded_forms)
        or any(not str(value) for value in grounded_forms)
        or not str(sources["dolly"].get("grounded_question_claim_scope") or "")
    ):
        raise ValueError("KERC Dolly grounded-question contract invalid")
    frame_ambiguity = sources["masc"].get("contextual_frame_ambiguity")
    if (
        not isinstance(frame_ambiguity, dict)
        or frame_ambiguity.get("policy")
        != "project_theseus_kerc_masc_train_only_contextual_frame_ambiguity_v1"
        or frame_ambiguity.get("fit_split") != "private_train"
        or int(frame_ambiguity.get("minimum_distinct_frames") or 0) < 2
        or int(frame_ambiguity.get("minimum_total_occurrences") or 0)
        < int(frame_ambiguity.get("minimum_distinct_frames") or 0)
        or not str(frame_ambiguity.get("claim_scope") or "")
    ):
        raise ValueError("KERC MASC contextual-frame ambiguity contract invalid")
    composite_counts = sources["masc"].get("composite_semantic_records_by_split")
    decision_counts = sources["masc"].get("decision_semantic_records_by_split")
    if (
        not isinstance(composite_counts, dict)
        or tuple(composite_counts) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in composite_counts.values())
        or int(sources["masc"].get("composite_semantic_minimum_frames") or 0) < 2
        or int(sources["masc"].get("composite_semantic_maximum_frames") or 0)
        < int(sources["masc"].get("composite_semantic_minimum_frames") or 0)
        or int(sources["masc"].get("composite_semantic_unique_source_credit") or 0) != 0
        or not str(sources["masc"].get("composite_semantic_claim_scope") or "")
    ):
        raise ValueError("KERC MASC composite-semantic contract invalid")
    if (
        not isinstance(decision_counts, dict)
        or tuple(decision_counts) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in decision_counts.values())
        or int(sources["masc"].get("decision_semantic_minimum_annotations") or 0) < 2
        or int(sources["masc"].get("decision_semantic_maximum_annotations") or 0)
        < int(sources["masc"].get("decision_semantic_minimum_annotations") or 0)
        or int(sources["masc"].get("decision_semantic_unique_source_credit") or 0) != 0
        or not str(sources["masc"].get("decision_semantic_claim_scope") or "")
    ):
        raise ValueError("KERC MASC decision-semantic contract invalid")
    event_coreference = sources["masc"].get("event_coreference")
    event_counts = (
        event_coreference.get("records_by_split")
        if isinstance(event_coreference, dict)
        else None
    )
    event_mentions = (
        event_coreference.get("mentions_by_split")
        if isinstance(event_coreference, dict)
        else None
    )
    event_document_map = (
        event_coreference.get("document_map")
        if isinstance(event_coreference, dict)
        else None
    )
    rejected_event_groups = (
        event_coreference.get("expected_rejected_groups")
        if isinstance(event_coreference, dict)
        else None
    )
    if (
        not isinstance(event_coreference, dict)
        or event_coreference.get("policy")
        != "project_theseus_kerc_masc_manual_event_coreference_v1"
        or event_coreference.get("alignment_contract")
        != "complete_named_gate_group_dual_independent_token_alignment_v1"
        or event_coreference.get("source_compaction_contract")
        != "uniform_radius_mention_centered_source_windows_v1"
        or not str(event_coreference.get("original_event_root") or "")
        or not isinstance(event_document_map, dict)
        or len(event_document_map) < 2
        or len(set(event_document_map.values())) != len(event_document_map)
        or any(
            not str(filename).endswith(".xml") or not str(document_id)
            for filename, document_id in event_document_map.items()
        )
        or not isinstance(event_counts, dict)
        or tuple(event_counts) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in event_counts.values())
        or sum(int(value) for value in event_counts.values())
        != int(event_coreference.get("expected_admitted_group_count", -1))
        or not isinstance(event_mentions, dict)
        or tuple(event_mentions) != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in event_mentions.values())
        or sum(int(value) for value in event_mentions.values())
        != int(event_coreference.get("expected_admitted_mention_count", -1))
        or int(event_coreference.get("expected_observed_group_count") or 0)
        < int(event_coreference.get("expected_admitted_group_count") or 0)
        or int(event_coreference.get("expected_observed_mention_count") or 0)
        < int(event_coreference.get("expected_admitted_mention_count") or 0)
        or int(event_coreference.get("expected_rejected_group_count", -1))
        != len(rejected_event_groups or [])
        or not isinstance(rejected_event_groups, list)
        or any(
            not isinstance(row, dict)
            or set(row) != {"document_id", "annotation_set_name"}
            or not str(row["document_id"])
            or not str(row["annotation_set_name"])
            for row in rejected_event_groups
        )
        or int(event_coreference.get("unique_source_credit") or 0) != 0
        or not str(event_coreference.get("claim_scope") or "")
    ):
        raise ValueError("KERC MASC event-coreference contract invalid")
    mpqa_relations = sources["masc"].get("mpqa_relations")
    mpqa_relation_counts = (
        mpqa_relations.get("records_by_split")
        if isinstance(mpqa_relations, dict)
        else None
    )
    mpqa_rejection_counts = (
        mpqa_relations.get("expected_rejection_reason_counts")
        if isinstance(mpqa_relations, dict)
        else None
    )
    mpqa_dev_documents = (
        mpqa_relations.get("private_dev_documents")
        if isinstance(mpqa_relations, dict)
        else None
    )
    mpqa_eval_documents = (
        mpqa_relations.get("private_eval_documents")
        if isinstance(mpqa_relations, dict)
        else None
    )
    if (
        not isinstance(mpqa_relations, dict)
        or mpqa_relations.get("policy")
        != "project_theseus_kerc_masc_manual_mpqa_relation_v1"
        or mpqa_relations.get("relation_contract")
        != "complete_manual_mpqa_expression_attitude_target_source_chain_v1"
        or mpqa_relations.get("source_compaction_contract")
        != "uniform_radius_relation_member_source_windows_v1"
        or not str(mpqa_relations.get("original_mpqa_root") or "")
        or not isinstance(mpqa_dev_documents, list)
        or not mpqa_dev_documents
        or len(set(mpqa_dev_documents)) != len(mpqa_dev_documents)
        or any(not str(value) for value in mpqa_dev_documents)
        or not isinstance(mpqa_eval_documents, list)
        or not mpqa_eval_documents
        or len(set(mpqa_eval_documents)) != len(mpqa_eval_documents)
        or any(not str(value) for value in mpqa_eval_documents)
        or set(mpqa_dev_documents) & set(mpqa_eval_documents)
        or not isinstance(mpqa_relation_counts, dict)
        or tuple(mpqa_relation_counts)
        != ("private_train", "private_dev", "private_eval")
        or any(int(value) < 0 for value in mpqa_relation_counts.values())
        or sum(int(value) for value in mpqa_relation_counts.values())
        != int(mpqa_relations.get("expected_admitted_relation_count", -1))
        or int(mpqa_relations.get("expected_observed_linked_expression_count") or 0)
        < int(mpqa_relations.get("expected_admitted_relation_count") or 0)
        or int(mpqa_relations.get("expected_admitted_source_member_count") or 0) <= 0
        or int(mpqa_relations.get("expected_admitted_attitude_count") or 0) <= 0
        or int(mpqa_relations.get("expected_admitted_target_count") or 0) <= 0
        or not isinstance(mpqa_rejection_counts, dict)
        or not mpqa_rejection_counts
        or any(not str(key) or int(value) < 0 for key, value in mpqa_rejection_counts.items())
        or sum(int(value) for value in mpqa_rejection_counts.values())
        != int(mpqa_relations.get("expected_observed_linked_expression_count", -1))
        - int(mpqa_relations.get("expected_admitted_relation_count", -1))
        or int(mpqa_relations.get("unique_source_credit") or 0) != 0
        or not str(mpqa_relations.get("claim_scope") or "")
    ):
        raise ValueError("KERC MASC MPQA-relation contract invalid")
    def validate_gum(gum: dict[str, Any]) -> None:
        gum_genres = gum.get("allowed_genre_licenses")
        gum_dev = gum.get("private_dev_documents")
        gum_eval = gum.get("private_eval_documents")
        gum_documents = gum.get("documents_by_split")
        gum_records = gum.get("records_by_split")
        gum_secondary = gum.get("secondary_edges_by_split")
        entity_coreference = gum.get("entity_coreference") or {}
        entity_count_maps = [
            entity_coreference.get(name)
            for name in (
                "records_by_split",
                "identity_records_by_split",
                "bridge_records_by_split",
                "mentions_by_split",
                "components_by_split",
            )
        ]
        if (
            not isinstance(gum_genres, dict)
            or set(gum_genres)
            != {"academic", "bio", "court", "interview", "news", "voyage"}
            or any(not str(value) for value in gum_genres.values())
            or not isinstance(gum_dev, list)
            or len(gum_dev) != 12
            or len(set(gum_dev)) != len(gum_dev)
            or not isinstance(gum_eval, list)
            or len(gum_eval) != 12
            or len(set(gum_eval)) != len(gum_eval)
            or set(gum_dev) & set(gum_eval)
            or any(
                not re.fullmatch(r"GUM_[a-z0-9_]+", str(value))
                for value in [*gum_dev, *gum_eval]
            )
            or not isinstance(gum_documents, dict)
            or tuple(gum_documents)
            != ("private_train", "private_dev", "private_eval")
            or sum(int(value) for value in gum_documents.values())
            != int(gum.get("expected_selected_document_count", -1))
            or int(gum_documents["private_dev"]) != len(gum_dev)
            or int(gum_documents["private_eval"]) != len(gum_eval)
            or not isinstance(gum_records, dict)
            or tuple(gum_records)
            != ("private_train", "private_dev", "private_eval")
            or any(int(value) <= 0 for value in gum_records.values())
            or not isinstance(gum_secondary, dict)
            or tuple(gum_secondary)
            != ("private_train", "private_dev", "private_eval")
            or any(int(value) < 0 for value in gum_secondary.values())
            or int(gum.get("minimum_relation_types_per_split") or 0) < 2
            or int(gum.get("minimum_weak_tail_count_per_split") or 0) < 1
            or gum.get("official_partitions_admitted") != ["train"]
            or gum.get("official_partitions_quarantined")
            != ["dev", "test", "test2"]
            or set(gum.get("allowed_objectives") or [])
            != {
                "surface_to_kernel_program_v1",
                "kernel_program_to_answer_packet_v1",
            }
            or not str(gum.get("claim_scope") or "")
            or entity_coreference.get("policy")
            != "project_theseus_kerc_gum_human_entity_coreference_v1"
            or entity_coreference.get("relation_contract")
            != "complete_source_declared_identity_component_or_bridge_endpoint_graph_v1"
            or entity_coreference.get("source_compaction_contract")
            != "uniform_sentence_bounded_mention_window_v1"
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(entity_coreference.get("content_sha256") or ""),
            )
            or int(entity_coreference.get("expected_selected_document_count") or 0)
            != int(gum.get("expected_selected_document_count") or -1)
            or any(
                not isinstance(values, dict)
                or tuple(values) != ("private_train", "private_dev", "private_eval")
                or any(int(value) <= 0 for value in values.values())
                for values in entity_count_maps
            )
            or any(
                int(entity_coreference["records_by_split"][split])
                != int(entity_coreference["identity_records_by_split"][split])
                + int(entity_coreference["bridge_records_by_split"][split])
                for split in ("private_train", "private_dev", "private_eval")
            )
            or not str(entity_coreference.get("claim_scope") or "")
        ):
            raise ValueError("KERC GUM discourse/coreference contract invalid")

    gum = sources.get("gum")
    if gum is not None:
        validate_gum(gum)
    behavior_counts = sources["oasst2"].get("explicit_behavior_records_by_split")
    if (
        not isinstance(behavior_counts, dict)
        or tuple(behavior_counts) != ("private_train", "private_dev", "private_eval")
        or any(
            set((behavior_counts.get(split) or {})) != {"CLARIFY", "ABSTAIN"}
            or any(int(value) < 0 for value in behavior_counts[split].values())
            for split in behavior_counts
        )
        or not str(sources["oasst2"].get("explicit_behavior_claim_scope") or "")
    ):
        raise ValueError("KERC OASST2 explicit behavior contract invalid")
    for split in requested:
        total = sum(int(source["records_by_split"][split]) for source in sources.values())
        total += int(grounded_counts[split])
        total += sum(int(value) for value in behavior_counts[split].values())
        if total != int(requested[split]):
            raise ValueError(f"KERC semantic corpus split total mismatch: {split}")
    floors = cfg["semantic_supervision"][
        "minimum_decision_grade_records_by_split_and_objective"
    ]
    for split, objective_floors in floors.items():
        for objective, floor in objective_floors.items():
            available = sum(
                int(source["records_by_split"][split])
                for source in sources.values()
                if objective in source["allowed_objectives"]
            )
            if objective in grounded_objectives:
                available += int(grounded_counts[split])
            if objective in sources["oasst2"]["allowed_objectives"]:
                available += sum(int(value) for value in behavior_counts[split].values())
            if available < int(floor):
                raise ValueError(
                    f"KERC semantic corpus cannot satisfy objective floor: {split}:{objective}"
                )
    groups = sources["masc"].get("document_groups") or {}
    if tuple(groups) != ("private_dev", "private_eval"):
        raise ValueError("KERC MASC heldout document groups are incomplete")
    dev = {str(value) for value in groups["private_dev"]}
    evaluation = {str(value) for value in groups["private_eval"]}
    if not dev or not evaluation or dev & evaluation:
        raise ValueError("KERC MASC heldout document groups overlap or are empty")
    oasst = sources["oasst2"]
    if (
        oasst.get("required_valid_realization_ranks") != [0, 1]
        or not 0.0 <= float(oasst.get("minimum_quality", -1.0)) <= 1.0
        or set(oasst.get("maximum_label_values") or {})
        != {"spam", "lang_mismatch", "pii", "not_appropriate"}
        or any(
            not 0.0 <= float(value) <= 1.0
            for value in (oasst.get("maximum_label_values") or {}).values()
        )
        or any(
            int(oasst.get(key) or 0) <= 0
            for key in (
                "maximum_current_characters",
                "maximum_response_characters",
                "maximum_context_characters",
                "maximum_compiled_context_bytes",
                "minimum_prior_turns",
                "maximum_prior_turns",
            )
        )
        or int(oasst.get("minimum_prior_turns") or 0)
        > int(oasst.get("maximum_prior_turns") or 0)
    ):
        raise ValueError("KERC OASST2 conversation-tree contract is incomplete")
    for key in ("minimum_source_groups_by_split", "minimum_source_sentences_by_split"):
        values = corpus.get(key) or {}
        if tuple(values) != ("private_train", "private_dev", "private_eval") or any(
            int(value or 0) <= 0 for value in values.values()
        ):
            raise ValueError(f"KERC semantic corpus diversity floor invalid: {key}")
    if int(corpus.get("maximum_source_characters") or 0) < 256:
        raise ValueError("KERC semantic corpus source-character cap is too small")
    economics = corpus.get("residual_economics")
    if (
        not isinstance(economics, dict)
        or economics.get("policy")
        != "project_theseus_kerc_residual_economics_v1"
        or not isinstance(economics.get("allocation_lambda_grid_bits"), list)
        or not economics.get("allocation_lambda_grid_bits")
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in economics.get("allocation_lambda_grid_bits") or []
        )
        or not math.isfinite(
            float(
                economics.get(
                    "maximum_dev_importance_weighted_structural_distortion", -1.0
                )
            )
        )
        or not 0.0
        <= float(
            economics.get(
                "maximum_dev_importance_weighted_structural_distortion", -1.0
            )
        )
        <= 1.0
        or economics.get("importance_fit_split") != "private_train"
        or economics.get("importance_calibration_split") != "private_dev"
        or economics.get("importance_final_evaluation_split") != "private_eval"
        or economics.get("utility_claim") is not False
    ):
        raise ValueError("KERC residual economics contract is invalid")
    for key in (
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
    ):
        if int(corpus.get(key) or 0):
            raise ValueError(f"KERC semantic corpus no-cheat counter must remain zero: {key}")
    return corpus


def load_kerc_semantic_source_catalog(
    path: Path, cfg: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = read_json(path) if path.is_file() else {}
    gaps: list[str] = []
    if payload.get("policy") != KERC_SOURCE_CATALOG_POLICY:
        return {}, ["kernel_semantic_source_catalog_policy_invalid"]
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return {}, ["kernel_semantic_source_catalog_empty"]
    sources: dict[str, dict[str, Any]] = {}
    allowed_licenses = {str(value).lower() for value in cfg.get("allowed_licenses") or []}
    for index, row in enumerate(raw_sources):
        if not isinstance(row, dict):
            gaps.append(f"kernel_semantic_source_catalog_row_invalid:{index}")
            continue
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id or dataset_id in sources:
            gaps.append(f"kernel_semantic_source_catalog_identity_invalid:{index}")
            continue
        required = (
            str(row.get("dataset_revision") or "").strip()
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("content_sha256") or ""))
            and str(row.get("license_spdx") or "").lower() in allowed_licenses
            and row.get("permitted_use") == "model_training"
            and row.get("training_allowed") is True
            and row.get("public_benchmark_surface") is False
            and row.get("public_benchmark_payload") is False
        )
        tiers = row.get("allowed_evidence_tiers")
        objectives = row.get("allowed_objectives")
        license_matrix = row.get("per_record_license_matrix")
        license_matrix_valid = (
            license_matrix is None
            or isinstance(license_matrix, dict)
            and bool(license_matrix)
            and all(
                str(value).lower() in allowed_licenses
                for value in license_matrix.values()
            )
        )
        if (
            not required
            or not isinstance(tiers, list)
            or not tiers
            or any(str(tier) not in SEMANTIC_EVIDENCE_TIERS for tier in tiers)
            or not isinstance(objectives, list)
            or not objectives
            or any(str(objective) not in TRAINING_OBJECTIVES for objective in objectives)
            or not license_matrix_valid
        ):
            gaps.append(f"kernel_semantic_source_catalog_contract_invalid:{dataset_id}")
            continue
        sources[dataset_id] = row
    return sources, sorted(set(gaps))


def validate_kerc_record_source(
    record: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> str:
    provenance = record.get("provenance") or {}
    dataset_id = str(provenance.get("dataset_id") or "")
    source = sources.get(dataset_id)
    if source is None:
        return "semantic_source_absent_from_catalog"
    semantic = record.get("semantic_supervision") or {}
    authorized_objectives = {
        objective
        for objective, authorized in (semantic.get("objective_authority") or {}).items()
        if authorized is True
    }
    license_matrix = source.get("per_record_license_matrix")
    admitted_licenses = (
        {str(value).lower() for value in license_matrix.values()}
        if isinstance(license_matrix, dict) and license_matrix
        else {str(source.get("license_spdx") or "").lower()}
    )
    checks = (
        str(provenance.get("dataset_revision") or "") == str(source.get("dataset_revision") or ""),
        str(provenance.get("license_spdx") or "").lower() in admitted_licenses,
        str(semantic.get("evidence_tier") or "")
        in {str(value) for value in source.get("allowed_evidence_tiers") or []},
        str(semantic.get("annotation_source_sha256") or "")
        == str(source.get("content_sha256") or ""),
        authorized_objectives
        <= {str(value) for value in source.get("allowed_objectives") or []},
    )
    return "" if all(checks) else "semantic_source_catalog_binding_mismatch"


def inspect_kernel_english(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    disposition = validate_training_disposition(cfg)
    if disposition.get("full_kerc_training_enabled") is not True:
        return kernel_english_base_report(config_path, cfg, "GREEN", [])
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return kernel_english_base_report(
            config_path, cfg, "PLANNED", ["kernel_english_stage_not_materialized"]
        )
    payload = read_json(manifest_path)
    gaps = validate_kernel_english_manifest(payload, cfg)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def _preflight_kernel_record_line(payload: tuple[int, str]) -> dict[str, Any]:
    line_number, raw = payload
    binding = stable_hash(raw.encode("utf-8"))
    try:
        record = validate_training_record(json.loads(raw))
    except Exception as exc:
        return {
            "line_number": line_number,
            "raw_line_sha256": binding,
            "state": "REJECTED",
            "fault_code": str(getattr(exc, "code", "KERC_RECORD_INVALID")),
        }
    try:
        validate_kernel_record_learned_abi(record, line_number=line_number)
    except Exception as exc:
        return {
            "line_number": line_number,
            "raw_line_sha256": binding,
            "state": "FATAL",
            "fault_code": str(getattr(exc, "code", "KERC_LEARNED_ABI_INVALID")),
            "detail": str(exc),
        }
    return {
        "line_number": line_number,
        "raw_line_sha256": binding,
        "record_sha256": str(record["record_sha256"]),
        "record_binding_sha256": stable_hash(record),
        "record": record,
        "state": "ACCEPTED",
    }


def _bounded_batches(rows: Any, batch_rows: int) -> Any:
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_rows:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_preflighted_kernel_records(
    records_path: Path, execution: dict[str, Any]
) -> Any:
    workers = min(
        int(execution["validation_workers"]), max(1, int(os.cpu_count() or 1))
    )
    batch_rows = int(execution["batch_rows"])

    def source_rows() -> Any:
        with records_path.open(encoding="utf-8") as records_handle:
            for line_number, raw in enumerate(records_handle, 1):
                if raw.strip():
                    yield line_number, raw

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in _bounded_batches(source_rows(), batch_rows):
            results = (
                list(executor.map(_preflight_kernel_record_line, batch))
                if executor is not None
                else [_preflight_kernel_record_line(row) for row in batch]
            )
            if len(results) != len(batch):
                raise ValueError("KERC validation worker batch cardinality mismatch")
            for (line_number, raw), result in zip(batch, results):
                observed_binding = stable_hash(raw.encode("utf-8"))
                if (
                    int(result.get("line_number") or -1) != line_number
                    or result.get("raw_line_sha256") != observed_binding
                ):
                    raise ValueError(
                        f"KERC validation worker content binding mismatch at line {line_number}"
                    )
                state = str(result.get("state") or "")
                if state == "FATAL":
                    raise ValueError(
                        "KERC learned ABI worker rejected governed record at line "
                        f"{line_number}: {result.get('fault_code')}:{result.get('detail')}"
                    )
                if state == "REJECTED":
                    yield None, str(result.get("fault_code") or "KERC_RECORD_INVALID")
                    continue
                if state != "ACCEPTED":
                    raise ValueError(
                        f"KERC validation worker returned invalid state at line {line_number}"
                    )
                record = result.get("record")
                if (
                    not isinstance(record, dict)
                    or str(record.get("record_sha256") or "")
                    != result.get("record_sha256")
                    or stable_hash(record) != result.get("record_binding_sha256")
                ):
                    raise ValueError(
                        f"KERC validation worker record identity mismatch at line {line_number}"
                    )
                yield record, None
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _compile_kernel_record_worker(
    payload: tuple[str, dict[str, Any]]
) -> dict[str, Any]:
    split, record = payload
    record_binding = stable_hash(record)
    return {
        "split": split,
        "record_sha256": str(record["record_sha256"]),
        "record_binding_sha256": record_binding,
        "views": compile_training_views(record),
    }


def compile_selected_kernel_views(
    selected: dict[str, list[dict[str, Any]]], execution: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    started = time.perf_counter()
    workers = min(
        int(execution["compilation_workers"]), max(1, int(os.cpu_count() or 1))
    )
    batch_rows = int(execution["batch_rows"])
    ordered = [
        (split, record)
        for split, records in selected.items()
        for record in records
    ]
    compiled = {split: [] for split in selected}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in _bounded_batches(ordered, batch_rows):
            results = (
                list(executor.map(_compile_kernel_record_worker, batch))
                if executor is not None
                else [_compile_kernel_record_worker(row) for row in batch]
            )
            if len(results) != len(batch):
                raise ValueError("KERC compilation worker batch cardinality mismatch")
            for (split, record), result in zip(batch, results):
                if (
                    result.get("split") != split
                    or result.get("record_sha256") != record["record_sha256"]
                    or result.get("record_binding_sha256") != stable_hash(record)
                ):
                    raise ValueError(
                        "KERC compilation worker content binding mismatch: "
                        + str(record["record_sha256"])
                    )
                compiled[split].extend(result["views"])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return compiled, {
        "policy": KERC_PARALLEL_MATERIALIZATION_POLICY,
        "validation_workers": min(
            int(execution["validation_workers"]), max(1, int(os.cpu_count() or 1))
        ),
        "compilation_workers": workers,
        "batch_rows": batch_rows,
        "compiled_record_count": len(ordered),
        "compiled_view_count": sum(len(rows) for rows in compiled.values()),
        "compilation_seconds": round(time.perf_counter() - started, 6),
        "deterministic_input_order": True,
        "raw_line_content_binding_required": True,
        "fallback_return_count": 0,
    }


def kernel_stage_cache_contract(
    *,
    cfg: dict[str, Any],
    records_path: Path,
    ledger_path: Path,
    source_catalog_path: Path,
    stage_metadata_path: Path,
    stage_root: Path,
) -> tuple[Path, str, list[dict[str, Any]], dict[str, Path]]:
    cache = cfg["materialization_cache"]
    dependencies = dependency_bindings(
        {
            "governed_records": records_path,
            "verification_ledger": ledger_path,
            "semantic_source_catalog": source_catalog_path,
            "language_stage_metadata": stage_metadata_path,
            "kernel_protocol_owner": ROOT / "scripts" / "kernel_english_protocol.py",
            "stage_materializer_owner": Path(__file__).resolve(),
            "language_tokenizer_owner": ROOT / "scripts" / "moecot_language_tokenizer.py",
            "open_vocabulary_owner": ROOT / "scripts" / "neural_seed_open_vocab.py",
            "cache_integrity_owner": ROOT / "scripts" / "kerc_content_cache.py",
        }
    )
    encoded_config = canonical_json(cfg).encode("utf-8")
    dependencies.append(
        {
            "id": "kernel_stage_config_contract",
            "kind": "canonical_payload",
            "path": "configs/moecot_language_arm_training.json#kernel_english_training",
            "sha256": hashlib.sha256(encoded_config).hexdigest(),
            "size_bytes": len(encoded_config),
        }
    )
    dependencies.sort(key=lambda row: str(row["id"]))
    outputs = {
        "code_vocabulary": stage_root / "code_vocabulary_v1.json",
        "manifest": stage_root / "manifest.json",
        "private_train": stage_root / "private_train.jsonl",
        "private_dev": stage_root / "private_dev.jsonl",
        "private_eval": stage_root / "private_eval.jsonl",
    }
    return (
        resolve(str(cache["cache_root"])),
        str(cache["role"]),
        dependencies,
        outputs,
    )


def materialize_kernel_english(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    cfg = validate_kernel_english_config(config)
    disposition = validate_training_disposition(cfg)
    if disposition.get("full_kerc_training_enabled") is not True:
        report = kernel_english_base_report(config_path, cfg, "GREEN", [])
        stage_root = resolve(cfg["stage_root"])
        stage_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(stage_root / "manifest.json", report)
        return report
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    records_path = resolve(cfg["records_jsonl"])
    ledger_path = resolve(cfg["verification_ledger_jsonl"])
    source_catalog_path = resolve(cfg["semantic_source_catalog_json"])
    stage_metadata_path = resolve(config["stage_dir"]) / "stage_metadata_v1.json"
    missing = []
    if not records_path.is_file():
        missing.append("kernel_english_records_missing")
    if not ledger_path.is_file():
        missing.append("kernel_english_verification_ledger_missing")
    if not source_catalog_path.is_file():
        missing.append("kernel_english_semantic_source_catalog_missing")
    if not stage_metadata_path.is_file():
        missing.append("kernel_english_language_stage_metadata_missing")
    if missing:
        report = kernel_english_base_report(
            config_path,
            cfg,
            "RED",
            missing,
        )
        write_json_atomic(stage_root / "manifest.json", report)
        return report

    cache_root, cache_role, cache_dependencies, cache_outputs = (
        kernel_stage_cache_contract(
            cfg=cfg,
            records_path=records_path,
            ledger_path=ledger_path,
            source_catalog_path=source_catalog_path,
            stage_metadata_path=stage_metadata_path,
            stage_root=stage_root,
        )
    )
    stage_cache_key = kerc_cache_key(
        role=cache_role,
        dependencies=cache_dependencies,
    )
    if cfg["materialization_cache"]["enabled"]:
        cached_report = load_kerc_cache_receipt(
            cache_root,
            role=cache_role,
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="manifest",
        )
        if cached_report is not None:
            cache_gaps = validate_kernel_english_manifest(cached_report, cfg)
            if not cache_gaps:
                return cached_report

    ledger, ledger_gaps = load_kernel_verification_ledger(ledger_path)
    source_catalog, source_catalog_gaps = load_kerc_semantic_source_catalog(
        source_catalog_path, cfg
    )
    metadata = read_json(stage_metadata_path)
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    candidates: dict[str, list[dict[str, Any]]] = {
        split: [] for split in cfg["records_by_split"]
    }
    execution = cfg["materialization_execution"]
    validation_started = time.perf_counter()
    rejection_counts: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    for record, validation_fault in iter_preflighted_kernel_records(
        records_path, execution
    ):
        if validation_fault:
            rejection_counts[validation_fault] += 1
            continue
        if record is None:
            raise ValueError("KERC validator accepted an empty record")
        split = str(record["split"])
        candidate_count[split] += 1
        receipt = record["verification_receipt"]
        ledger_receipt = ledger.get(str(receipt["receipt_id"]))
        if ledger_receipt is None:
            rejection_counts["verification_receipt_absent_from_ledger"] += 1
            continue
        if ledger_receipt != receipt:
            rejection_counts["verification_receipt_ledger_mismatch"] += 1
            continue
        if str(record["provenance"]["license_spdx"]).lower() not in {
            str(value).lower() for value in cfg["allowed_licenses"]
        }:
            rejection_counts["license_not_allowed"] += 1
            continue
        source_gap = validate_kerc_record_source(record, source_catalog)
        if source_gap:
            rejection_counts[source_gap] += 1
            continue
        candidates[split].append(record)
    validation_seconds = round(time.perf_counter() - validation_started, 6)

    selected, selection_receipt = select_kernel_records(candidates, cfg)
    overlaps = kernel_english_split_overlap(selected)
    gaps = [*ledger_gaps, *source_catalog_gaps, *overlaps["hard_gaps"]]
    semantic_program = validate_kerc_semantic_program(cfg)
    evidence_counts_by_split: dict[str, Counter[str]] = {
        split: Counter(
            str((record.get("semantic_supervision") or {}).get("evidence_tier") or "")
            for record in records
        )
        for split, records in selected.items()
    }
    decision_grade_tiers = {
        tier
        for tier, contract in SEMANTIC_EVIDENCE_TIERS.items()
        if contract["claim_authority"] == "decision_grade_reference"
    }
    decision_grade_counts_by_split_and_objective = {
        split: {
            objective: sum(
                1
                for record in records
                if str((record.get("semantic_supervision") or {}).get("evidence_tier") or "")
                in decision_grade_tiers
                and (record.get("semantic_supervision") or {})
                .get("objective_authority", {})
                .get(objective)
                is True
            )
            for objective in TRAINING_OBJECTIVES
        }
        for split, records in selected.items()
    }
    for split, objective_floors in semantic_program[
        "minimum_decision_grade_records_by_split_and_objective"
    ].items():
        for objective, floor in objective_floors.items():
            observed = decision_grade_counts_by_split_and_objective.get(split, {}).get(
                objective, 0
            )
            if observed < int(floor):
                gaps.append(
                    f"insufficient_decision_grade_kernel_records:{split}:{objective}:"
                    f"{observed}:{int(floor)}"
                )
    train_records = selected.get("private_train") or []
    train_count = len(train_records)
    train_counts = evidence_counts_by_split.get("private_train") or Counter()
    for tier, cap in semantic_program["maximum_train_record_share_by_tier"].items():
        share = train_counts[tier] / max(1, train_count)
        if share > float(cap) + 1e-12:
            gaps.append(f"kernel_semantic_record_share_exceeded:{tier}:{share:.8f}:{cap}")
    train_weight_by_tier: Counter[str] = Counter()
    for record in train_records:
        semantic = record["semantic_supervision"]
        train_weight_by_tier[str(semantic["evidence_tier"])] += float(
            semantic["optimizer_sampling_weight"]
        )
    train_weight_total = sum(train_weight_by_tier.values())
    for tier, cap in semantic_program["maximum_train_optimizer_probability_by_tier"].items():
        probability = train_weight_by_tier[tier] / max(1e-12, train_weight_total)
        if probability > float(cap) + 1e-12:
            gaps.append(
                f"kernel_semantic_optimizer_probability_exceeded:{tier}:"
                f"{probability:.8f}:{cap}"
            )
    artifacts: dict[str, Any] = {}
    objective_counts: Counter[str] = Counter()
    objective_counts_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in selected
    }
    encoded_length_stats: dict[str, Any] = {}
    sequence_bucket_counts_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in selected
    }
    all_source_hashes: set[str] = set()
    raw_source_bytes = 0
    verifier_corruption_count = 0
    compiled_views, materialization_execution = compile_selected_kernel_views(
        selected, execution
    )
    materialization_execution["validation_seconds"] = validation_seconds
    materialization_execution["validated_candidate_count"] = sum(
        candidate_count.values()
    )
    context_counterfactuals = attach_grounded_context_counterfactuals(
        compiled_views, selected
    )
    expected_context_counterfactual_count = 4 * sum(
        int(value)
        for value in (
            (((cfg.get("semantic_corpus_materialization") or {}).get("dolly") or {}))
            .get("grounded_question_records_by_split", {})
        ).values()
    )
    context_counterfactuals["expected_total_count"] = (
        expected_context_counterfactual_count
    )
    if int(context_counterfactuals["total_count"]) != expected_context_counterfactual_count:
        gaps.append(
            "kernel_context_counterfactual_count_mismatch:"
            f"{context_counterfactuals['total_count']}:"
            f"{expected_context_counterfactual_count}"
        )
    if context_counterfactuals["missing_donor_record_sha256"]:
        gaps.append(
            "kernel_context_counterfactual_donor_missing:"
            + str(len(context_counterfactuals["missing_donor_record_sha256"]))
        )
    if not compiled_views.get("private_train"):
        raise ValueError(
            "KERC stage has no admitted private-train views: "
            + json.dumps(dict(rejection_counts), sort_keys=True)
        )
    code_vocabulary = build_kerc_code_vocabulary(
        compiled_views["private_train"], cfg["code_vocabulary"]
    )
    code_vocabulary_path = stage_root / "code_vocabulary_v1.json"
    write_json_atomic(code_vocabulary_path, code_vocabulary)
    for split, records in selected.items():
        wanted = int(cfg["records_by_split"][split])
        if len(records) != wanted:
            gaps.append(f"insufficient_kernel_records:{split}:{len(records)}:{wanted}")
        views: list[dict[str, Any]] = []
        source_lengths: list[int] = []
        target_lengths: list[int] = []
        sequence_lengths: list[int] = []
        for record in records:
            all_source_hashes.add(str(record["raw_source_sha256"]))
            raw_source_bytes += len(str(record["source_text"]).encode("utf-8"))
        for view in compiled_views[split]:
                source_body_ids, source_receipt = encode_kerc_view_source(
                    view,
                    source_vocab=source_vocab,
                    code_vocabulary=code_vocabulary,
                )
                trusted_prefix = list(view.get("trusted_source_prefix_tokens") or [])
                if len(trusted_prefix) != 1 or trusted_prefix[0] not in source_vocab:
                    gaps.append(f"kernel_view_trusted_prefix_invalid:{view['row_id']}")
                    continue
                source_ids = [int(source_vocab[trusted_prefix[0]]), *source_body_ids]
                target_ids, target_receipt = encode_kerc_view_target(
                    view, target_vocab=target_vocab, code_vocabulary=code_vocabulary
                )
                if int(source_receipt.get("unknown_token_count") or 0) or int(
                    target_receipt.get("unknown_token_count") or 0
                ):
                    gaps.append(f"kernel_view_unrepresentable:{view['row_id']}")
                    continue
                verifier_negative = view.get("kerc_verifier_negative") or {}
                negative_target = str(verifier_negative.get("target") or "")
                negative_ids, negative_receipt = encode_kerc_view_target(
                    {**view, "target": negative_target},
                    target_vocab=target_vocab,
                    code_vocabulary=code_vocabulary,
                )
                if (
                    not negative_target
                    or verifier_negative.get("generator_loss_enabled") is not False
                    or int(negative_receipt.get("unknown_token_count") or 0)
                ):
                    gaps.append(
                        f"kernel_view_verifier_corruption_invalid:{view['row_id']}"
                    )
                    continue
                sequence_tokens = len(source_ids) + len(target_ids) + 4
                negative_sequence_tokens = len(source_ids) + len(negative_ids) + 4
                if sequence_tokens > int(cfg["maximum_sequence_tokens"]):
                    gaps.append(
                        f"kernel_view_requires_truncation:{view['row_id']}:{sequence_tokens}"
                    )
                    continue
                if negative_sequence_tokens > int(cfg["maximum_sequence_tokens"]):
                    gaps.append(
                        "kernel_view_verifier_corruption_requires_truncation:"
                        f"{view['row_id']}:{negative_sequence_tokens}"
                    )
                    continue
                counterfactual_lengths: list[tuple[int, int]] = []
                counterfactual_invalid = False
                for counterfactual in view.get("kerc_context_counterfactuals") or []:
                    counterfactual_prompt = str(counterfactual.get("prompt") or "")
                    counterfactual_target = str(counterfactual.get("target") or "")
                    counterfactual_source_ids, counterfactual_source_receipt = (
                        encode_kerc_view_source(
                            {**view, "prompt": counterfactual_prompt},
                            source_vocab=source_vocab,
                            code_vocabulary=code_vocabulary,
                        )
                    )
                    counterfactual_target_ids, counterfactual_target_receipt = (
                        encode_kerc_view_target(
                            {**view, "target": counterfactual_target},
                            target_vocab=target_vocab,
                            code_vocabulary=code_vocabulary,
                        )
                    )
                    counterfactual_sequence_tokens = (
                        1
                        + len(trusted_prefix)
                        + len(counterfactual_source_ids)
                        + len(counterfactual_target_ids)
                        + 3
                    )
                    if (
                        not counterfactual_prompt
                        or not counterfactual_target
                        or counterfactual.get("generator_loss_enabled") is not False
                        or int(counterfactual_source_receipt.get("unknown_token_count") or 0)
                        or int(counterfactual_target_receipt.get("unknown_token_count") or 0)
                        or counterfactual_sequence_tokens
                        > int(cfg["maximum_sequence_tokens"])
                    ):
                        gaps.append(
                            "kernel_view_context_counterfactual_invalid:"
                            f"{view['row_id']}:{counterfactual.get('strategy')}"
                        )
                        counterfactual_invalid = True
                        break
                    counterfactual_lengths.append(
                        (len(counterfactual_target_ids), counterfactual_sequence_tokens)
                    )
                if counterfactual_invalid:
                    continue
                source_lengths.append(len(source_ids))
                target_lengths.append(len(target_ids))
                target_lengths.append(len(negative_ids))
                sequence_lengths.extend((sequence_tokens, negative_sequence_tokens))
                for observed_length in (sequence_tokens, negative_sequence_tokens):
                    sequence_bucket_counts_by_split[split][
                        "standard_8k"
                        if observed_length <= 8192
                        else "exact_high_fan_in_16k"
                    ] += 1
                target_lengths.extend(length[0] for length in counterfactual_lengths)
                sequence_lengths.extend(length[1] for length in counterfactual_lengths)
                for _, observed_length in counterfactual_lengths:
                    sequence_bucket_counts_by_split[split][
                        "standard_8k"
                        if observed_length <= 8192
                        else "exact_high_fan_in_16k"
                    ] += 1
                objective_counts[str(view["objective"])] += 1
                objective_counts_by_split[split][str(view["objective"])] += 1
                verifier_corruption_count += 1
                views.append(view)
        path = stage_root / f"{split}.jsonl"
        write_jsonl_atomic(path, views)
        artifacts[f"english:{split}"] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(views),
            "unique_record_count": len(records),
            "bytes": path.stat().st_size,
        }
        encoded_length_stats[split] = {
            "maximum_source_tokens": max(source_lengths or [0]),
            "maximum_target_tokens": max(target_lengths or [0]),
            "maximum_sequence_tokens": max(sequence_lengths or [0]),
        }

    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract(),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": True,
        "source": {
            "path": relative(records_path),
            "sha256": sha256_file(records_path),
            "license_policy": "row_level_explicit_allowlist",
        },
        "verification_ledger": {
            "path": relative(ledger_path),
            "sha256": sha256_file(ledger_path),
            "receipt_count": len(ledger),
            "producer_separate_from_training_rows": True,
        },
        "semantic_source_catalog": {
            "path": relative(source_catalog_path),
            "sha256": sha256_file(source_catalog_path),
            "policy": KERC_SOURCE_CATALOG_POLICY,
            "source_count": len(source_catalog),
        },
        "semantic_supervision": {
            "policy": KERC_SEMANTIC_PROGRAM_POLICY,
            "evidence_record_counts_by_split": {
                split: dict(counts) for split, counts in evidence_counts_by_split.items()
            },
            "decision_grade_record_counts_by_split_and_objective": (
                decision_grade_counts_by_split_and_objective
            ),
            "minimum_decision_grade_records_by_split_and_objective": dict(
                semantic_program["minimum_decision_grade_records_by_split_and_objective"]
            ),
            "train_weight_by_tier": dict(train_weight_by_tier),
            "train_optimizer_probability_by_tier": {
                tier: round(weight / max(1e-12, train_weight_total), 10)
                for tier, weight in train_weight_by_tier.items()
            },
            "silver_supports_decision_grade_claims": False,
            "teacher_residual_supports_decision_grade_claims": False,
        },
        "code_vocabulary": {
            "path": relative(code_vocabulary_path),
            "sha256": sha256_file(code_vocabulary_path),
            "policy": code_vocabulary["policy"],
            "contract_sha256": code_vocabulary["contract_sha256"],
            "fit_split": code_vocabulary["fit_split"],
            "kernel_vocab_count": len(code_vocabulary["kernel_vocab"]),
            "pointer_vocab_count": len(code_vocabulary["pointer_vocab"]),
            "kernel_max_vocab": code_vocabulary["kernel_max_vocab"],
            "pointer_max_vocab": code_vocabulary["pointer_max_vocab"],
        },
        "artifacts": artifacts,
        "candidate_record_count_by_split": dict(candidate_count),
        "selection": selection_receipt,
        "materialization_execution": materialization_execution,
        "materialization_cache": {
            "policy": CACHE_POLICY,
            "enabled": bool(cfg["materialization_cache"]["enabled"]),
            "role": cache_role,
            "cache_key_sha256": stage_cache_key,
            "dependency_count": len(cache_dependencies),
            "exact_dependency_binding_required": True,
            "output_identity_revalidation_required": True,
            "cache_hit_authorizes_capability_claim": False,
        },
        "selected_record_count_by_split": {
            split: len(records) for split, records in selected.items()
        },
        "compiled_view_count_by_objective": dict(objective_counts),
        "compiled_view_count_by_split_and_objective": {
            split: dict(counts) for split, counts in objective_counts_by_split.items()
        },
        "unique_raw_source_count": len(all_source_hashes),
        "unique_raw_source_bytes": raw_source_bytes,
        "derived_view_unique_data_credit": 0,
        "derived_view_optimizer_exposure_count": sum(objective_counts.values()),
        "verifier_corruption_count": verifier_corruption_count,
        "verifier_corruptions_receive_generator_loss": False,
        "context_counterfactuals": context_counterfactuals,
        "split_overlap_audit": overlaps,
        "encoded_length_stats": encoded_length_stats,
        "sequence_buckets": {
            "policy": KERC_SEQUENCE_BUCKET_POLICY,
            "routing": "encoded_length_only_without_target_semantic_metadata",
            "counts_by_split": {
                split: dict(counts)
                for split, counts in sequence_bucket_counts_by_split.items()
            },
            "truncation_count": 0,
            "row_drop_count": 0,
            "long_bucket_capability_credit": False,
        },
        "rejection_counts": dict(rejection_counts),
        "failure_behavior": "reject_without_template_literal_tool_or_router_fallback",
        "score_semantics": "KERC learned-objective data readiness; not learned capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": sorted(set(gaps)),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    if cfg["materialization_cache"]["enabled"] and not gaps:
        publish_kerc_cache_receipt(
            cache_root,
            role=cache_role,
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="manifest",
        )
    return report


def validate_kernel_record_learned_abi(
    record: dict[str, Any], *, line_number: int
) -> None:
    """Fail before selection when a governed record cannot reach learned stages.

    Corpus admission and packet replay are insufficient if the compact learned ABI
    cannot represent the same record.  Running this check while streaming the source
    makes an owner defect fail at its first source line instead of after the complete
    bounded-selection pass.
    """

    try:
        learned_residual_view(
            record["kernel_packet"]["residual"], hrl_state=record["hrl_state"]
        )
    except KernelProtocolFault as exc:
        raise ValueError(
            "KERC learned ABI rejected governed record "
            f"at line {line_number} ({record.get('record_sha256')}): {exc}"
        ) from exc


def select_kernel_records(
    candidates: dict[str, list[dict[str, Any]]], cfg: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select a deterministic source-disjoint set while preserving declared strata.

    Selection can inspect only immutable identities, split, provenance authority, the
    per-objective authority mask, and the grounded-context kind. It cannot inspect
    answer text, model output, verifier score, or public benchmark metadata.
    """

    selection = cfg["selection"]
    seed = str(selection["ranking_seed"])
    split_order = tuple(cfg["records_by_split"])
    raw_source_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in candidates.items():
        for record in records:
            raw_source_splits[str(record["raw_source_sha256"])].add(split)
    cross_split_raw_sources = {
        identity for identity, splits in raw_source_splits.items() if len(splits) > 1
    }
    eligible = {
        split: [
            record
            for record in candidates.get(split, [])
            if str(record["raw_source_sha256"]) not in cross_split_raw_sources
        ]
        for split in split_order
    }

    grounded_quotas = (
        (((cfg.get("semantic_corpus_materialization") or {}).get("dolly") or {}))
        .get("grounded_question_records_by_split", {})
    )
    objective_floors = cfg["semantic_supervision"][
        "minimum_decision_grade_records_by_split_and_objective"
    ]

    rank_cache: dict[str, str] = {}

    def rank(record: dict[str, Any]) -> str:
        identity = str(record["record_sha256"])
        if identity not in rank_cache:
            rank_cache[identity] = stable_hash(
                {
                    "policy": KERC_SELECTION_POLICY,
                    "seed": seed,
                    "record_sha256": identity,
                }
            )
        return rank_cache[identity]

    def grounded(record: dict[str, Any]) -> bool:
        return (
            (record.get("interaction_annotation") or {}).get("kind")
            == "licensed_grounded_question_context"
        )

    def decision_grade_authority(record: dict[str, Any], objective: str) -> bool:
        semantic = record.get("semantic_supervision") or {}
        return (
            semantic.get("claim_authority") == "decision_grade_reference"
            and (semantic.get("objective_authority") or {}).get(objective) is True
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    objective_counts_by_split: dict[str, dict[str, int]] = {}
    grounded_counts: dict[str, int] = {}
    objective_priority_by_split: dict[str, list[str]] = {}
    for split in split_order:
        pool = sorted(eligible[split], key=rank)
        requested = int(cfg["records_by_split"][split])
        if len(pool) < requested:
            raise ValueError(
                f"KERC selection capacity infeasible: {split}:{len(pool)}:{requested}"
            )
        chosen: dict[str, dict[str, Any]] = {}

        def admit(record: dict[str, Any]) -> None:
            chosen.setdefault(str(record["record_sha256"]), record)

        grounded_quota = int(grounded_quotas.get(split) or 0)
        grounded_pool = [record for record in pool if grounded(record)]
        if len(grounded_pool) < grounded_quota:
            raise ValueError(
                "KERC grounded selection quota infeasible: "
                f"{split}:{len(grounded_pool)}:{grounded_quota}"
            )
        for record in grounded_pool[:grounded_quota]:
            admit(record)

        floors = {key: int(value) for key, value in objective_floors[split].items()}
        margins = {
            objective: sum(
                decision_grade_authority(record, objective) for record in pool
            )
            - floor
            for objective, floor in floors.items()
        }
        infeasible = {
            objective: margin for objective, margin in margins.items() if margin < 0
        }
        if infeasible:
            raise ValueError(
                f"KERC objective selection floor infeasible: {split}:{infeasible}"
            )
        objective_priority = sorted(
            floors,
            key=lambda objective: (margins[objective], TRAINING_OBJECTIVES.index(objective)),
        )
        objective_priority_by_split[split] = objective_priority
        for objective in objective_priority:
            observed = sum(
                decision_grade_authority(record, objective)
                for record in chosen.values()
            )
            needed = floors[objective] - observed
            if needed <= 0:
                continue
            options = [
                record
                for record in pool
                if str(record["record_sha256"]) not in chosen
                and decision_grade_authority(record, objective)
            ]
            unmet_objectives = {
                other
                for other in floors
                if sum(
                    decision_grade_authority(current, other)
                    for current in chosen.values()
                )
                < floors[other]
            }
            options.sort(
                key=lambda record: (
                    -sum(
                        decision_grade_authority(record, other)
                        for other in unmet_objectives
                    ),
                    rank(record),
                )
            )
            if len(options) < needed:
                raise ValueError(
                    "KERC objective selection floor became infeasible: "
                    f"{split}:{objective}:{len(options)}:{needed}"
                )
            for record in options[:needed]:
                admit(record)

        if len(chosen) > requested:
            raise ValueError(
                f"KERC selection constraints exceed capacity: {split}:{len(chosen)}:{requested}"
            )
        for record in pool:
            if len(chosen) >= requested:
                break
            admit(record)
        if len(chosen) != requested:
            raise ValueError(
                f"KERC selection did not fill capacity: {split}:{len(chosen)}:{requested}"
            )
        rows = sorted(chosen.values(), key=rank)
        selected[split] = rows
        grounded_counts[split] = sum(grounded(record) for record in rows)
        objective_counts_by_split[split] = {
            objective: sum(
                decision_grade_authority(record, objective) for record in rows
            )
            for objective in TRAINING_OBJECTIVES
        }
        for objective, floor in floors.items():
            if objective_counts_by_split[split][objective] < floor:
                raise ValueError(
                    "KERC objective floor not preserved: "
                    f"{split}:{objective}:{objective_counts_by_split[split][objective]}:{floor}"
                )
        if grounded_counts[split] < grounded_quota:
            raise ValueError(
                "KERC grounded quota not preserved: "
                f"{split}:{grounded_counts[split]}:{grounded_quota}"
            )

    receipt = {
        "policy": KERC_SELECTION_POLICY,
        "ranking_seed": seed,
        "selection_visible_fields": list(selection["selection_visible_fields"]),
        "answer_text_visible": False,
        "model_outcomes_visible": False,
        "candidate_count_by_split": {
            split: len(candidates.get(split, [])) for split in split_order
        },
        "eligible_count_by_split": {
            split: len(eligible[split]) for split in split_order
        },
        "selected_count_by_split": {
            split: len(selected[split]) for split in split_order
        },
        "cross_split_raw_source_exclusion_count": len(cross_split_raw_sources),
        "cross_split_raw_source_excluded_record_count": sum(
            str(record["raw_source_sha256"]) in cross_split_raw_sources
            for records in candidates.values()
            for record in records
        ),
        "cross_split_raw_source_sha256": sorted(cross_split_raw_sources),
        "grounded_question_count_by_split": grounded_counts,
        "decision_grade_objective_count_by_split": objective_counts_by_split,
        "objective_priority_by_split": objective_priority_by_split,
        "selection_sha256": stable_hash(
            {
                split: [record["record_sha256"] for record in selected[split]]
                for split in split_order
            }
        ),
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return selected, receipt


def kernel_english_split_overlap(
    selected: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for split, records in selected.items():
        groups[split] = {str(row["provenance"]["source_group"]) for row in records}
        sources[split] = {str(row["raw_source_sha256"]) for row in records}
    group_overlap = 0
    source_overlap = 0
    for left_index, left in enumerate(selected):
        for right in tuple(selected)[left_index + 1 :]:
            group_overlap += len(groups[left] & groups[right])
            source_overlap += len(sources[left] & sources[right])
    gaps = []
    if group_overlap:
        gaps.append(f"kernel_source_group_cross_split_overlap:{group_overlap}")
    if source_overlap:
        gaps.append(f"kernel_raw_source_cross_split_overlap:{source_overlap}")
    return {
        "source_group_overlap_count": group_overlap,
        "raw_source_overlap_count": source_overlap,
        "content_bound_disjoint": not gaps,
        "hard_gaps": gaps,
    }


def load_kernel_verification_ledger(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    receipts: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            gaps.append(f"kernel_verification_ledger_json_invalid:{line_number}")
            continue
        receipt_id = str(row.get("receipt_id") or "") if isinstance(row, dict) else ""
        if not receipt_id:
            gaps.append(f"kernel_verification_ledger_receipt_id_missing:{line_number}")
            continue
        if receipt_id in receipts:
            gaps.append(f"kernel_verification_ledger_receipt_duplicate:{receipt_id}")
            continue
        if row.get("policy") != TRAINING_VERIFICATION_POLICY:
            gaps.append(f"kernel_verification_ledger_policy_invalid:{receipt_id}")
            continue
        if row.get("accepted") is not True:
            gaps.append(f"kernel_verification_ledger_unaccepted:{receipt_id}")
            continue
        receipts[receipt_id] = row
    return receipts, sorted(set(gaps))


def validate_kernel_english_manifest(
    payload: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    gaps: list[str] = []
    if payload.get("trigger_state") != "GREEN" or payload.get("hard_gaps") != []:
        gaps.append("kernel_stage_not_green")
    if payload.get("policy") != cfg["policy"]:
        gaps.append("kernel_stage_policy_mismatch")
    if payload.get("contract_sha256") != kernel_english_stage_contract_sha256(cfg):
        gaps.append("kernel_stage_contract_identity_mismatch")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    expected_view_count = 0
    for split, record_count in cfg["records_by_split"].items():
        key = f"english:{split}"
        artifact = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"kernel_stage_artifact_identity_mismatch:{key}")
        row_count = int(artifact.get("row_count") or 0)
        if not int(record_count) <= row_count <= int(record_count) * len(TRAINING_OBJECTIVES):
            gaps.append(f"kernel_stage_view_count_mismatch:{key}")
        expected_view_count += row_count
        if int(artifact.get("unique_record_count") or 0) != int(record_count):
            gaps.append(f"kernel_stage_record_count_mismatch:{key}")
    overlap = payload.get("split_overlap_audit") or {}
    if not bool(overlap.get("content_bound_disjoint")):
        gaps.append("kernel_stage_split_overlap")
    if int(payload.get("derived_view_unique_data_credit") or 0):
        gaps.append("kernel_stage_derived_view_unique_credit_nonzero")
    if int(payload.get("verifier_corruption_count") or 0) != expected_view_count:
        gaps.append("kernel_stage_verifier_corruption_count_mismatch")
    if payload.get("verifier_corruptions_receive_generator_loss") is not False:
        gaps.append("kernel_stage_verifier_corruption_generator_credit")
    counterfactuals = payload.get("context_counterfactuals") or {}
    counterfactual_counts = counterfactuals.get("counts") or {}
    expected_counterfactual_count = 4 * sum(
        int(value)
        for value in (
            (((cfg.get("semantic_corpus_materialization") or {}).get("dolly") or {}))
            .get("grounded_question_records_by_split", {})
        ).values()
    )
    if (
        counterfactuals.get("policy") != KERC_CONTEXT_COUNTERFACTUAL_POLICY
        or int(counterfactuals.get("total_count") or 0)
        != sum(int(value) for value in counterfactual_counts.values())
        or int(counterfactuals.get("total_count") or 0)
        != expected_counterfactual_count
        or int(counterfactuals.get("expected_total_count") or 0)
        != expected_counterfactual_count
        or counterfactuals.get("missing_donor_record_sha256") != []
        or counterfactuals.get("generator_loss_enabled") is not False
        or int(counterfactuals.get("unique_source_credit") or 0)
        or int(counterfactuals.get("candidate_generation_credit") or 0)
    ):
        gaps.append("kernel_stage_context_counterfactual_contract_invalid")
    sequence_buckets = payload.get("sequence_buckets") or {}
    bucket_counts = sequence_buckets.get("counts_by_split") or {}
    observed_bucket_rows = sum(
        int(value)
        for counts in bucket_counts.values()
        for value in (counts or {}).values()
    )
    if (
        sequence_buckets.get("policy") != KERC_SEQUENCE_BUCKET_POLICY
        or sequence_buckets.get("routing")
        != "encoded_length_only_without_target_semantic_metadata"
        or observed_bucket_rows
        != 2 * expected_view_count + expected_counterfactual_count
        or int(sequence_buckets.get("truncation_count") or 0)
        or int(sequence_buckets.get("row_drop_count") or 0)
        or sequence_buckets.get("long_bucket_capability_credit") is not False
    ):
        gaps.append("kernel_stage_sequence_bucket_contract_invalid")
    ledger = payload.get("verification_ledger") or {}
    ledger_path = resolve(str(ledger.get("path") or ""))
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != str(ledger.get("sha256") or "")
    ):
        gaps.append("kernel_stage_verification_ledger_identity_mismatch")
    if ledger.get("producer_separate_from_training_rows") is not True:
        gaps.append("kernel_stage_verification_ledger_not_independent")
    source_catalog = payload.get("semantic_source_catalog") or {}
    source_catalog_path = resolve(str(source_catalog.get("path") or ""))
    if (
        not source_catalog_path.is_file()
        or sha256_file(source_catalog_path) != str(source_catalog.get("sha256") or "")
        or source_catalog.get("policy") != KERC_SOURCE_CATALOG_POLICY
    ):
        gaps.append("kernel_stage_semantic_source_catalog_identity_mismatch")
    semantic = payload.get("semantic_supervision") or {}
    program = validate_kerc_semantic_program(cfg)
    if semantic.get("policy") != KERC_SEMANTIC_PROGRAM_POLICY:
        gaps.append("kernel_stage_semantic_supervision_policy_mismatch")
    if semantic.get("minimum_decision_grade_records_by_split_and_objective") != program.get(
        "minimum_decision_grade_records_by_split_and_objective"
    ):
        gaps.append("kernel_stage_decision_grade_floor_mismatch")
    decision_counts = (
        semantic.get("decision_grade_record_counts_by_split_and_objective") or {}
    )
    for split, objective_floors in program[
        "minimum_decision_grade_records_by_split_and_objective"
    ].items():
        for objective, floor in objective_floors.items():
            if int((decision_counts.get(split) or {}).get(objective) or 0) < int(floor):
                gaps.append(
                    f"kernel_stage_decision_grade_floor_not_met:{split}:{objective}"
                )
    if semantic.get("silver_supports_decision_grade_claims") is not False:
        gaps.append("kernel_stage_silver_claim_authority_invalid")
    if semantic.get("teacher_residual_supports_decision_grade_claims") is not False:
        gaps.append("kernel_stage_teacher_claim_authority_invalid")
    code = payload.get("code_vocabulary") or {}
    code_path = resolve(str(code.get("path") or ""))
    code_payload = read_json(code_path) if code_path.is_file() else {}
    if (
        not code_path.is_file()
        or sha256_file(code_path) != str(code.get("sha256") or "")
        or code_payload.get("policy")
        != "project_theseus_kerc_dual_code_vocabulary_v1"
        or code_payload.get("fit_split") != "private_train"
        or int(code_payload.get("dev_eval_vocabulary_fit_count") or 0) != 0
        or int(code_payload.get("verifier_corruption_vocabulary_fit_count") or 0) != 0
        or code_payload.get("contract_sha256") != code.get("contract_sha256")
    ):
        gaps.append("kernel_stage_code_vocabulary_identity_mismatch")
    else:
        unsigned = {
            key: value for key, value in code_payload.items() if key != "contract_sha256"
        }
        observed_contract = "sha256:" + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if observed_contract != code_payload["contract_sha256"]:
            gaps.append("kernel_stage_code_vocabulary_contract_mismatch")
        if (
            len(code_payload.get("kernel_vocab") or {})
            != int(code.get("kernel_vocab_count") or 0)
            or len(code_payload.get("pointer_vocab") or {})
            != int(code.get("pointer_vocab_count") or 0)
        ):
            gaps.append("kernel_stage_code_vocabulary_count_mismatch")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(payload.get(key) or 0):
            gaps.append(f"kernel_stage_nonzero_boundary:{key}")
    return sorted(set(gaps))


def kernel_english_stage_contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def kernel_english_base_report(
    config_path: Path,
    cfg: dict[str, Any],
    state: str,
    gaps: list[str],
) -> dict[str, Any]:
    disposition = validate_training_disposition(cfg)
    enabled = disposition.get("full_kerc_training_enabled") is True
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection" if enabled else "retired_from_first_long_run",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": kernel_english_stage_contract_sha256(cfg),
        "learned_pipeline_contract": kernel_training_contract() if enabled else {},
        "architecture_disposition": disposition,
        "full_kerc_training_enabled": enabled,
        "retained_mechanisms": list(disposition.get("retained_mechanisms") or []),
        "required_records_by_split": dict(cfg["records_by_split"]),
        "verification_ledger_required": enabled,
        "artifacts": {},
        "selected_record_count_by_split": {
            split: 0 for split in cfg["records_by_split"]
        },
        "compiled_view_count_by_objective": {},
        "unique_raw_source_count": 0,
        "derived_view_unique_data_credit": 0,
        "split_overlap_audit": {
            "source_group_overlap_count": 0,
            "raw_source_overlap_count": 0,
            "content_bound_disjoint": True,
            "hard_gaps": [],
        },
        "hard_gaps": gaps,
        "score_semantics": (
            "KERC learned-objective data readiness; not learned capability"
            if enabled
            else "bounded pre-training architecture disposition; full KERC receives zero optimizer exposure"
        ),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    manifest_path = resolve(cfg["stage_root"]) / "manifest.json"
    if not manifest_path.is_file():
        return base_report(config_path, cfg, "PLANNED", ["stage_not_materialized"])
    payload = read_json(manifest_path)
    gaps = validate_manifest(payload, cfg, config)
    return {
        **payload,
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": "RED" if gaps else "GREEN",
        "hard_gaps": gaps,
    }


def materialize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = validate_config(config)
    started = time.perf_counter()
    stage_root = resolve(cfg["stage_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    dependencies = source_conditioning_dependencies(config)
    metadata = read_json(resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    supervision_targets = supervision_target_hashes(config)
    selectors = {
        arm: BoundedRows(int(count))
        for arm, count in cfg["rows_by_arm"].items()
        if int(count) > 0
    }
    language_to_arm = {
        language: arm
        for arm, languages in (cfg.get("arm_languages") or {}).items()
        for language in languages
    }
    source_path = resolve(cfg["source_jsonl"])
    rejections: Counter[str] = Counter()
    candidate_count: Counter[str] = Counter()
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            arm = language_to_arm.get(str(source.get("language") or "").lower())
            if arm not in selectors:
                continue
            reason = source_rejection(source, cfg)
            if reason:
                rejections[reason] += 1
                continue
            for row in denoising_rows(source, arm, cfg, source_vocab, target_vocab):
                candidate_count[arm] += 1
                if row["target_sha256"] in supervision_targets:
                    rejections["supervision_target_overlap"] += 1
                    continue
                selectors[arm].add(row["selection_sha256"], row)

    artifacts: dict[str, Any] = {}
    copy_coverage: dict[str, Any] = {}
    gaps: list[str] = []
    for arm, selector in selectors.items():
        rows = selector.rows()
        wanted = int(cfg["rows_by_arm"][arm])
        if len(rows) != wanted:
            gaps.append(f"insufficient_rows:{arm}:{len(rows)}:{wanted}")
        path = stage_root / f"{arm}.jsonl"
        write_jsonl_atomic(path, rows)
        artifacts[arm] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "bytes": path.stat().st_size,
        }
        fractions = [float(row["target_token_copy_fraction"]) for row in rows]
        copy_coverage[arm] = {
            "mean_target_token_copy_fraction": round(
                sum(fractions) / max(1, len(fractions)), 8
            ),
            "minimum_target_token_copy_fraction": round(min(fractions or [0.0]), 8),
        }
    report = {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "materialized",
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "dependencies": dependencies,
        "source": {
            "path": relative(source_path),
            "sha256": sha256_file(source_path),
            "license_policy": "row_level_permissive_allowlist",
        },
        "artifacts": artifacts,
        "candidate_count_by_arm": dict(candidate_count),
        "copy_coverage_by_arm": copy_coverage,
        "rejection_counts": dict(rejections),
        "supervision_target_overlap_count": int(rejections["supervision_target_overlap"]),
        "corruption": {
            "mode": "deterministic_span_deletion_reconstruction",
            "deletion_fraction": float(cfg["deletion_fraction"]),
            "maximum_spans": int(cfg["maximum_deletion_spans"]),
            "seed": int(cfg["seed"]),
        },
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "score_semantics": "licensed source-conditioned objective readiness; not edit capability",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "hard_gaps": gaps,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json_atomic(stage_root / "manifest.json", report)
    return report


def denoising_rows(
    source: dict[str, Any],
    arm: str,
    cfg: dict[str, Any],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    text = str(source.get("text") or "")
    logical = exact_text_tokens(text)
    minimum = int(cfg["minimum_target_logical_tokens"])
    maximum = int(cfg["maximum_target_logical_tokens"])
    if len(logical) < minimum:
        return []
    source_identity = str(source.get("text_sha256") or hashlib.sha256(text.encode()).hexdigest())
    starts = list(range(0, len(logical) - minimum + 1, maximum))
    ranked_starts = sorted(
        starts,
        key=lambda start: hashlib.sha256(f"{source_identity}:{start}:{cfg['seed']}".encode()).hexdigest(),
    )[: int(cfg["maximum_windows_per_document"])]
    rows = []
    for start in ranked_starts:
        target_tokens = logical[start : start + maximum]
        if len(target_tokens) < minimum:
            continue
        corruption_identity = hashlib.sha256(
            f"{source_identity}:{start}:{cfg['seed']}".encode()
        ).hexdigest()
        damaged_tokens = delete_spans(target_tokens, cfg, corruption_identity)
        target = "".join(target_tokens)
        damaged = "".join(damaged_tokens)
        if not target.strip() or damaged == target:
            continue
        language = str(source.get("language") or arm)
        prompt = (
            f"Reconstruct the complete original {language} excerpt from this damaged excerpt. "
            "Return only the original excerpt.\n\n"
            f"Damaged excerpt:\n{damaged}"
        )
        source_ids, source_receipt = encode_tokens(
            exact_text_tokens(prompt), source_vocab, stream="source"
        )
        target_ids, target_receipt = encode_tokens(
            exact_text_tokens(target), target_vocab, stream="target"
        )
        if int(source_receipt.get("unknown_token_count") or 0) or int(
            target_receipt.get("unknown_token_count") or 0
        ):
            continue
        if len(source_ids) > int(cfg["maximum_source_encoded_tokens"]) or len(
            target_ids
        ) > int(cfg["maximum_target_encoded_tokens"]):
            continue
        source_token_set = set(exact_text_tokens(prompt))
        copy_fraction = sum(token in source_token_set for token in exact_text_tokens(target)) / max(
            1, len(exact_text_tokens(target))
        )
        digest = hashlib.sha256(
            f"{arm}:{source_identity}:{start}:{corruption_identity}".encode()
        ).hexdigest()
        rows.append(
            {
                "row_id": f"moecot-denoise-{digest[:20]}",
                "split": "private_train",
                "arm_id": arm,
                "objective": "source_conditioned_span_deletion_reconstruction_v1",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "target_token_copy_fraction": round(copy_fraction, 8),
                "selection_sha256": digest,
                "source_identity": {
                    "repo": source.get("repo"),
                    "path": source.get("path"),
                    "text_sha256": source_identity,
                    "window_start": start,
                    "license_spdx": source.get("license_spdx"),
                },
                "public_benchmark": False,
                "public_tests_included": False,
                "public_benchmark_solutions_included": False,
                "external_inference": False,
            }
        )
    return rows


def delete_spans(tokens: list[str], cfg: dict[str, Any], identity: str) -> list[str]:
    rng = random.Random(int(identity[:16], 16))
    delete_count = max(1, round(len(tokens) * float(cfg["deletion_fraction"])))
    spans = min(int(cfg["maximum_deletion_spans"]), delete_count)
    removed: set[int] = set()
    remaining = delete_count
    for span_index in range(spans):
        width = max(1, remaining // (spans - span_index))
        start = rng.randrange(max(1, len(tokens) - width + 1))
        removed.update(range(start, min(len(tokens), start + width)))
        remaining = max(0, delete_count - len(removed))
    while len(removed) < delete_count:
        removed.add(rng.randrange(len(tokens)))
    return [token for index, token in enumerate(tokens) if index not in removed]


def source_rejection(source: dict[str, Any], cfg: dict[str, Any]) -> str:
    if source.get("public_benchmark") is not False:
        return "public_benchmark_state_not_false"
    if source.get("public_tests_included") is not False:
        return "public_tests_present"
    if source.get("public_benchmark_solutions_included") is not False:
        return "public_solutions_present"
    if str(source.get("license_spdx") or "").lower() not in {
        str(value).lower() for value in cfg["allowed_licenses"]
    }:
        return "license_not_allowed"
    if not str(source.get("text") or "").strip():
        return "empty_text"
    return ""


def supervision_target_hashes(config: dict[str, Any]) -> set[str]:
    root = resolve(config["supervision"]["stage_root"])
    hashes: set[str] = set()
    for path in sorted(root.glob("private_*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                hashes.add(str(row.get("target_sha256") or ""))
    return hashes


def source_conditioning_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    """Bind every mutable input that changes source-conditioned row selection."""

    cfg = validate_config(config)
    source_path = resolve(cfg["source_jsonl"])
    metadata_path = resolve(config["stage_dir"]) / "stage_metadata_v1.json"
    supervision_root = resolve(config["supervision"]["stage_root"])
    supervision_paths = [
        path
        for path in [
            supervision_root / "manifest.json",
            *sorted(supervision_root.glob("private_*/*.jsonl")),
        ]
        if path.is_file()
    ]
    supervision_files = [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in supervision_paths
    ]
    supervision_digest = hashlib.sha256(
        json.dumps(supervision_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_jsonl": {
            "path": relative(source_path),
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "stage_metadata": {
            "path": relative(metadata_path),
            "sha256": sha256_file(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
        "supervision_stage": {
            "root": relative(supervision_root),
            "file_count": len(supervision_files),
            "files": supervision_files,
            "sha256": supervision_digest,
        },
    }


def contract_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(
    payload: dict[str, Any], cfg: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    gaps = []
    if payload.get("policy") != cfg["policy"]:
        gaps.append("policy_mismatch")
    if payload.get("contract_sha256") != contract_sha256(cfg):
        gaps.append("contract_identity_mismatch")
    recorded_dependencies = payload.get("dependencies") or {}
    try:
        current_dependencies = source_conditioning_dependencies(config)
    except (FileNotFoundError, KeyError, OSError) as exc:
        gaps.append(f"dependency_identity_unavailable:{type(exc).__name__}")
        current_dependencies = {}
    for dependency in ("source_jsonl", "stage_metadata", "supervision_stage"):
        if recorded_dependencies.get(dependency) != current_dependencies.get(dependency):
            gaps.append(f"dependency_identity_mismatch:{dependency}")
    for arm, wanted in cfg["rows_by_arm"].items():
        if int(wanted) <= 0:
            continue
        artifact = (payload.get("artifacts") or {}).get(arm) or {}
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            gaps.append(f"artifact_identity_mismatch:{arm}")
        if int(artifact.get("row_count") or 0) != int(wanted):
            gaps.append(f"row_count_mismatch:{arm}")
    for key in ("public_training_rows_written", "public_benchmark_payload_count", "external_inference_calls", "fallback_return_count"):
        if int(payload.get(key) or 0):
            gaps.append(f"nonzero_boundary:{key}")
    return gaps


def base_report(
    config_path: Path, cfg: dict[str, Any], state: str, gaps: list[str]
) -> dict[str, Any]:
    return {
        "policy": cfg["policy"],
        "created_utc": now(),
        "mode": "inspection",
        "trigger_state": state,
        "config": relative(config_path),
        "contract_sha256": contract_sha256(cfg),
        "hard_gaps": gaps,
        "score_semantics": "licensed source-conditioned objective readiness; not capability",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
