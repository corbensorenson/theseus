#!/usr/bin/env python3
"""Prompt/signature-only source text helpers for strict-generator MLX decode.

The strict generator may see natural-language prompt text, entry point,
callable signature, and prompt-derived tags. It must not see tests, solutions,
categories, return-shape labels, required constructs, or target-derived decoder
fields during generation.
"""

from __future__ import annotations

import keyword
from collections import Counter
from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty, get_path
from neural_seed_token_decoder_comparator import (
    visible_identifier_parts,
    visible_prompt_intent_tags,
    visible_prompt_operation_tags,
    visible_prompt_type_shape_tags,
    visible_subword_parts,
)


STRICT_GENERATOR_SOURCE_TEXT_POLICY = "prompt_signature_only_v1"
STRICT_GENERATOR_FORBIDDEN_SOURCE_MARKERS = (
    "solution_body ",
    "solution_expr ",
    "tests ",
    "hidden_tests ",
    "expected ",
    "answer ",
    "category ",
    "concept_residual_label ",
    "residual_concept ",
    "targeted_private_residual_family_v3 ",
    "broad_private_family_v1 ",
    "source_task_id ",
    "card_id ",
    "return_shape ",
    "type_family ",
    "required_constructs ",
    "visible_operation_tags ",
)


def checkpoint_source_text_style(config: dict[str, Any], vocab_payload: dict[str, Any]) -> str:
    recorded = str(vocab_payload.get("source_text_style") or vocab_payload.get("source_text_policy") or "").strip()
    if recorded:
        return recorded
    budget_id = str(vocab_payload.get("budget_id") or "").strip()
    for row in dict_or_empty(config.get("strict_generator_pretraining")).get("budgets", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or row.get("budget_id") or "") == budget_id:
            style = str(row.get("source_text_style") or "").strip()
            if style:
                return style
    return str(get_path(config, ["pretraining_initialization", "full_state_warmup", "source_text_style"], "prompt_signature_metadata_v2"))


def strict_generator_visible_argument_names(row: dict[str, Any]) -> list[str]:
    """Use only the callable signature surface admitted for generation."""

    try:
        argc = int(get_path(row, ["decoder_contract", "visible_arg_count_hint"], 1) or 1)
    except (TypeError, ValueError):
        argc = 1
    names = ["data"]
    if argc >= 2:
        names.append("other")
    if argc >= 3:
        names.append("extra")
    return names[: max(1, argc)]


def strict_generator_visible_signature(row: dict[str, Any], args: list[str]) -> str:
    entry = str(row.get("entry_point") or "solve").strip() or "solve"
    safe_entry = entry if entry.isidentifier() and not keyword.iskeyword(entry) else "solve"
    return f"def {safe_entry}({', '.join(args)}):"


def strict_generator_decode_source_text(
    row: dict[str, Any],
    fields: list[Any],
    *,
    source_text_style: str,
    source_vocab: dict[str, int],
) -> str:
    _ = fields
    prompt = " ".join(str(row.get("prompt") or "").split())
    entry = str(row.get("entry_point") or "solve").strip() or "solve"
    args = strict_generator_visible_argument_names(row)
    signature = strict_generator_visible_signature(row, args)
    entry_parts = visible_identifier_parts(entry)
    arg_parts = [part for arg in args for part in visible_identifier_parts(arg)]
    subword_basis = "\n".join(part for part in [prompt, " ".join(entry_parts), " ".join(arg_parts)] if part)
    intent_tags = visible_prompt_intent_tags(subword_basis)
    operation_tags = visible_prompt_operation_tags(subword_basis)
    type_shape_tags = visible_prompt_type_shape_tags(subword_basis)
    subwords = visible_subword_parts(subword_basis)
    chunks = [
        prompt,
        entry,
        "visible_intent_tags " + " ".join(intent_tags) if intent_tags else "",
        "prompt_operation_hints " + " ".join(operation_tags) if operation_tags else "",
        "visible_type_shape_tags " + " ".join(type_shape_tags) if type_shape_tags else "",
        f"signature {signature}",
        "arguments " + " ".join(args) if args else "",
        "entry_point_parts " + " ".join(entry_parts) if entry_parts else "",
        "argument_parts " + " ".join(arg_parts) if arg_parts else "",
        "visible_subwords " + " ".join(subwords) if subwords else "",
    ]
    style = str(source_text_style or "").strip()
    if style and "source_style" in source_vocab:
        chunks.append(f"source_style {style}")
    return "\n".join(chunk for chunk in chunks if chunk)


def _long_fragments(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    fragments: list[str] = []
    for line in text.splitlines():
        clean = " ".join(line.split())
        if len(clean) >= 24:
            fragments.append(clean)
    collapsed = " ".join(text.split())
    if len(collapsed) >= 32:
        fragments.append(collapsed[:160])
    return list(dict.fromkeys(fragments))[:12]


def strict_generator_source_text_audit(rows: list[dict[str, Any]], source_texts: list[str]) -> dict[str, Any]:
    marker_hits: Counter[str] = Counter()
    solution_fragment_hits = 0
    test_fragment_hits = 0
    intent_tag_rows = 0
    operation_hint_rows = 0
    type_shape_tag_rows = 0
    checked_rows = min(len(rows), len(source_texts))
    for row, text in zip(rows, source_texts):
        lowered = "\n" + str(text or "").lower()
        for marker in STRICT_GENERATOR_FORBIDDEN_SOURCE_MARKERS:
            if f"\n{marker}" in lowered:
                marker_hits[marker.strip()] += 1
        normalized_text = " ".join(str(text or "").split())
        intent_tag_rows += 1 if "\nvisible_intent_tags " in lowered else 0
        operation_hint_rows += 1 if "\nprompt_operation_hints " in lowered else 0
        type_shape_tag_rows += 1 if "\nvisible_type_shape_tags " in lowered else 0
        if any(fragment and fragment in normalized_text for fragment in _long_fragments(row.get("solution_body"))):
            solution_fragment_hits += 1
        if any(fragment and fragment in normalized_text for fragment in _long_fragments(row.get("tests"))):
            test_fragment_hits += 1
    clean = not marker_hits and solution_fragment_hits == 0 and test_fragment_hits == 0
    return {
        "policy": "project_theseus_strict_generator_source_text_audit_v1",
        "source_text_policy": STRICT_GENERATOR_SOURCE_TEXT_POLICY,
        "checked_rows": checked_rows,
        "source_text_rows": len(source_texts),
        "forbidden_field_reads_by_builder": 0,
        "forbidden_marker_hit_count": sum(marker_hits.values()),
        "forbidden_marker_hits": dict(sorted(marker_hits.items())),
        "solution_body_fragment_leak_count": solution_fragment_hits,
        "test_fragment_leak_count": test_fragment_hits,
        "visible_intent_tag_rows": intent_tag_rows,
        "prompt_operation_hint_rows": operation_hint_rows,
        "visible_type_shape_tag_rows": type_shape_tag_rows,
        "visible_operation_tags_emitted": False,
        "prompt_operation_hints_emitted": operation_hint_rows > 0,
        "category_or_family_fields_emitted": False,
        "clean": clean,
        "score_semantics": (
            "Strict generator source text is constructed from prompt, entry point, visible argument count, "
            "prompt-derived intent/type-shape tags, prompt-only operation hints, signature text, identifier "
            "pieces, and tokenizer subword repair only. Prompt operation hints are derived from visible words "
            "and are not hidden operation-family labels. It does not read tests, solutions, categories, "
            "return-shape labels, required constructs, or target-derived decoder fields."
        ),
    }


def visible_parameter_type_hints_from_source_text(text: str) -> dict[str, str]:
    """Infer coarse input type hints from visible source text only."""

    argument_names: list[str] = []
    tags: set[str] = set()
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            for raw in stripped.removeprefix("arguments ").replace(",", " ").split():
                name = raw.strip("* ,:()")
                if name.isidentifier() and name not in {"self", "cls"} and name not in argument_names:
                    argument_names.append(name)
        elif stripped.startswith("visible_type_shape_tags "):
            tags.update(stripped.removeprefix("visible_type_shape_tags ").split())
    if not tags:
        tags.update(visible_prompt_type_shape_tags(text))
    if not argument_names:
        argument_names = ["data"]
    primary = argument_names[0]
    inferred = ""
    if tags.intersection(
        {
            "shape_graph",
            "shape_intervals",
            "shape_nested",
            "shape_pairs",
            "shape_ranked_counts",
            "shape_records",
            "shape_sequence",
        }
    ):
        inferred = "list"
    elif tags.intersection({"shape_mapping"}):
        inferred = "dict"
    elif tags.intersection({"shape_text", "shape_path_text"}):
        inferred = "str"
    elif tags == {"shape_numeric"} or ("shape_numeric" in tags and len(tags) == 1):
        inferred = "int"
    return {primary: inferred} if inferred else {}
