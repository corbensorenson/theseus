#!/usr/bin/env python3
"""Visible prompt/signature helpers for the neural seed token decoder.

This module is a mechanical extraction from
``neural_seed_token_decoder_comparator.py``. It preserves the existing helper
contracts while keeping source conditioning and split-audit logic in a bounded
module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from neural_seed_code_proposer_comparator import (
    deterministic_sample,
    dict_or_empty,
    get_path,
    row_text,
    stable_hash,
)
from neural_seed_token_decoder_support import target_tokens


def broad_private_heldout_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = json.loads(json.dumps(row))
        copied["evaluation_split"] = "broad_private_heldout"
        copied["candidate_source"] = "neural_seed_token_decoder_comparator.broad_private_heldout"
        copied.setdefault("provenance", {})["evaluation_split"] = "broad_private_heldout"
        copied.setdefault("provenance", {})["broad_private_heldout_eval"] = True
        out.append(copied)
    return out


def deterministic_family_balanced_sample(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    *,
    family_key_name: str,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return deterministic_sample(rows, limit, seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[strict_disjoint_family_key(row, family_key_name)].append(row)
    selected: list[dict[str, Any]] = []
    families = sorted(by_family, key=lambda family: stable_hash(f"{seed}:family:{family}"))
    for family in families:
        if len(selected) >= limit:
            break
        selected.extend(deterministic_sample(by_family[family], 1, seed + len(selected)))
    if len(selected) < limit:
        selected_keys = {str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True))) for row in selected}
        remainder = [
            row
            for row in rows
            if str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True))) not in selected_keys
        ]
        selected.extend(deterministic_sample(remainder, limit - len(selected), seed + 991))
    return selected[:limit]


def decoder_source_text(row: dict[str, Any], fields: list[Any]) -> str:
    """Visible prompt/signature text with identifier pieces made learnable.

    The shared classifier tokenizer treats underscores as part of one token,
    which turns heldout function names such as ``bpg_top_k_frequent_1000917``
    into mostly unseen IDs. This expands only the already-visible fields listed
    in ``text_views``; it does not add category, contract, tests, solutions, or
    residual labels.
    """

    base = row_text(row, fields)
    additions: list[str] = []
    field_names = {str(field) for field in fields}
    intent_tags = visible_prompt_intent_tags(base)
    if intent_tags:
        additions.append("visible_intent_tags " + " ".join(intent_tags))
    if "visible_operation_tags" in field_names:
        operation_tags = visible_prompt_operation_tags(base)
        if operation_tags:
            additions.append("visible_operation_tags " + " ".join(operation_tags))
    type_shape_tags = visible_prompt_type_shape_tags(base)
    if type_shape_tags:
        additions.append("visible_type_shape_tags " + " ".join(type_shape_tags))
    if "entry_point" in field_names:
        entry = str(row.get("entry_point") or "")
        parts = visible_identifier_parts(entry)
        if parts:
            additions.append("entry_point_parts " + " ".join(parts))
    args = visible_callable_argument_names(row)
    if args:
        additions.append("signature " + visible_callable_signature(row, args))
        additions.append("arguments " + " ".join(args))
        arg_parts = [part for arg in args for part in visible_identifier_parts(arg)]
        if arg_parts:
            additions.append("argument_parts " + " ".join(arg_parts))
    subwords = visible_subword_parts(base)
    if subwords:
        additions.append("visible_subwords " + " ".join(subwords))
    return "\n".join(chunk for chunk in [base, *additions] if chunk)


def visible_callable_argument_names(row: dict[str, Any]) -> list[str]:
    contract = dict_or_empty(row.get("decoder_contract"))
    argc = int(contract.get("visible_arg_count_hint") or 1)
    names = ["data"]
    if argc >= 2:
        names.append("other")
    if argc >= 3:
        names.append("extra")
    return names[: max(1, argc)]


def visible_callable_signature(row: dict[str, Any], args: list[str]) -> str:
    entry = str(row.get("entry_point") or "solve")
    return f"def {entry}({', '.join(args)}):"


VISIBLE_PROMPT_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intent_parse", ("parse", "extract", "read", "split", "tokenize")),
    ("intent_string", ("string", "text", "substring", "char", "casefold", "lowercase", "uppercase")),
    ("intent_numeric", ("integer", "number", "numeric", "sum", "signed", "gcd", "average", "median")),
    ("intent_collection", ("list", "array", "values", "items", "records", "sequence", "iterable")),
    ("intent_mapping", ("dict", "map", "key", "value", "lookup", "group")),
    ("intent_count", ("count", "frequency", "frequent", "histogram", "tally")),
    ("intent_sort", ("sort", "sorted", "order", "rank", "top")),
    ("intent_unique", ("unique", "dedup", "duplicate", "distinct")),
    ("intent_filter", ("filter", "threshold", "keep", "remove", "select", "matching")),
    ("intent_window", ("window", "consecutive", "adjacent", "run", "delta")),
    ("intent_stack", ("balanced", "bracket", "parentheses", "stack")),
    ("intent_graph", ("graph", "node", "edge", "component", "path", "hop")),
    ("intent_dynamic_programming", ("subsequence", "non-adjacent", "knapsack", "dynamic")),
    ("intent_interval", ("interval", "range", "overlap", "merge")),
    ("intent_plan", ("plan", "task", "dependency", "blocked", "progress")),
    ("intent_storage", ("file", "storage", "quota", "sync", "path")),
    ("intent_device", ("device", "node", "room", "speaker", "worker")),
)


def visible_prompt_intent_tags(text: str) -> list[str]:
    """Generic prompt/entry-point intent tags from visible text only.

    This is source conditioning, not a semantic renderer. It never reads tests,
    solutions, decoder contracts, public benchmark metadata, or hidden family
    labels. Tags are intentionally broad capabilities, not answer identifiers.
    """

    normalized = " ".join(visible_identifier_parts(text))
    tokens = set(visible_identifier_parts(text))
    tags: list[str] = []
    for tag, needles in VISIBLE_PROMPT_INTENT_RULES:
        if any(visible_keyword_match(needle, normalized=normalized, tokens=tokens) for needle in needles):
            tags.append(tag)
    # Common visible multiword idioms that are useful but still not hidden
    # answer metadata.
    if "run length" in normalized or "rle" in normalized:
        tags.append("intent_run_length")
    if "query string" in normalized:
        tags.append("intent_query_string")
    if "top k" in normalized:
        tags.append("intent_top_k")
    return list(dict.fromkeys(tags))[:24]


def visible_prompt_operation_tags(text: str) -> list[str]:
    """Prompt/signature-only operation hints for source conditioning.

    These tags are deliberately derived only from the visible prompt and
    callable name text. They do not inspect task category, decoder contracts,
    tests, solutions, return shape, required constructs, or benchmark labels.
    They are used as learnable source tokens, not as renderers or templates.
    """

    normalized = " ".join(visible_identifier_parts(text))
    tokens = set(visible_identifier_parts(text))
    tags: list[str] = []

    def has(*needles: str) -> bool:
        return all(visible_keyword_match(needle, normalized=normalized, tokens=tokens) for needle in needles)

    def any_has(*needles: str) -> bool:
        return any(visible_keyword_match(needle, normalized=normalized, tokens=tokens) for needle in needles)

    if (has("signed") and any_has("integer", "integers", "number", "numbers")) or has("extract", "integers"):
        tags.append("op_signed_number_scan")
    if "run length" in normalized or "runlength" in normalized or has("consecutive", "equal"):
        tags.append("op_run_length_encode")
    if any_has("gcd") and any_has("absolute", "positive", "integer", "integers", "value", "values"):
        tags.append("op_gcd_reduce")
    if any_has("absolute", "abs") and any_has("positive", "integer", "integers"):
        tags.append("op_abs_positive_filter")
    if any_has("longest") and any_has("run") and any_has("even", "odd", "parity"):
        tags.append("op_parity_run_length")
    if any_has("adjacent", "delta", "deltas", "windowed") and any_has("clip", "clipping", "range"):
        tags.append("op_windowed_delta")
    if any_has("clip", "clipping", "clamp", "clamped", "clamping") and any_has("range", "lo", "hi"):
        tags.append("op_clip_to_range")
    if has("query", "string") or has("key", "values"):
        tags.append("op_query_key_value_parse")
    if (
        any_has("min", "max", "median", "average", "gcd", "delta", "deltas", "adjacent", "absolute", "round", "rounded")
        and any_has("count", "numeric", "number", "numbers", "integer", "integers", "values", "sum")
    ):
        tags.append("op_numeric_summary")
    if any_has("threshold", "meets") and any_has("filter", "whose", "records", "labels"):
        tags.append("op_threshold_filter")
    if any_has("top", "frequent", "frequency", "histogram", "count") and any_has("k", "most"):
        tags.append("op_frequency_top_k")
    if any_has("pair", "pairs") and any_has("sum", "sums", "add"):
        tags.append("op_pairwise_arithmetic")
    if any_has("group", "grouped", "records") and any_has("key", "field", "by"):
        tags.append("op_group_by_key")
    if any_has("normalize", "normalise", "lower", "strip") and any_has("filter", "sort", "sorted"):
        tags.append("op_normalize_filter_sort")
    if any_has("dedup", "deduplicate", "duplicate", "duplicates", "unique", "distinct") and any_has("stable", "order"):
        tags.append("op_stable_dedup")
    if any_has("balanced", "bracket", "parentheses"):
        tags.append("op_stack_balance")
    if any_has("merge", "interval", "overlap", "ranges"):
        tags.append("op_interval_merge")
    if any_has("path", "graph", "node", "edge", "component"):
        tags.append("op_graph_walk")
    if any_has("dependency", "dependencies", "blocked", "plan", "tasks"):
        tags.append("op_dependency_plan")

    return list(dict.fromkeys(tags))[:24]


def visible_prompt_type_shape_tags(text: str) -> list[str]:
    """Prompt/signature-only coarse type and container-shape hints.

    These tags intentionally stay broad. They are derived only from visible
    prompt, function-name, and signature words so they can condition the
    learned generator without importing hidden return shapes, task categories,
    tests, solutions, benchmark labels, or decoder-contract fields.
    """

    normalized = " ".join(visible_identifier_parts(text))
    tokens = set(visible_identifier_parts(text))
    tags: list[str] = []

    def has(*needles: str) -> bool:
        return all(visible_keyword_match(needle, normalized=normalized, tokens=tokens) for needle in needles)

    def any_has(*needles: str) -> bool:
        return any(visible_keyword_match(needle, normalized=normalized, tokens=tokens) for needle in needles)

    if any_has("list", "lists", "array", "arrays", "sequence", "sequences", "iterable", "items", "values"):
        tags.append("shape_sequence")
    if any_has("dict", "dictionary", "map", "mapping", "lookup", "key", "keys"):
        tags.append("shape_mapping")
    if any_has("string", "strings", "text", "char", "chars", "word", "words", "line", "lines", "substring"):
        tags.append("shape_text")
    if any_has("integer", "integers", "number", "numbers", "numeric", "float", "score", "scores", "sum", "average"):
        tags.append("shape_numeric")
    if any_has("record", "records", "row", "rows", "table", "columns", "field", "fields"):
        tags.append("shape_records")
    if any_has("pair", "pairs", "tuple", "tuples"):
        tags.append("shape_pairs")
    if any_has("nested", "flatten", "matrix", "grid"):
        tags.append("shape_nested")
    if any_has("graph", "node", "nodes", "edge", "edges", "path", "component", "components"):
        tags.append("shape_graph")
    if any_has("interval", "intervals", "range", "ranges", "overlap", "merge"):
        tags.append("shape_intervals")
    if any_has("file", "path", "paths", "storage", "directory"):
        tags.append("shape_path_text")
    if has("top", "k") or any_has("frequency", "frequent", "count", "counts", "histogram"):
        tags.append("shape_ranked_counts")
    if any_has("true", "false", "bool", "boolean", "predicate", "valid", "balanced"):
        tags.append("shape_boolean")
    return list(dict.fromkeys(tags))[:24]


def visible_keyword_match(needle: str, *, normalized: str, tokens: set[str]) -> bool:
    """Match visible prompt words without substring leakage.

    This keeps visible-operation tags honest: a single-word hint such as
    ``top`` or ``k`` must be present as its own visible token, so words like
    ``stopwords`` and ``tokens`` cannot accidentally emit ``op_frequency_top_k``.
    Multiword/hyphenated hints still use phrase matching over normalized
    visible text.
    """

    phrase = str(needle or "").lower().replace("-", " ").strip()
    if not phrase:
        return False
    if " " in phrase:
        return f" {phrase} " in f" {normalized} "
    return phrase in tokens


def visible_identifier_parts(value: str) -> list[str]:
    pieces: list[str] = []
    for raw in str(value or "").replace("_", " ").replace("-", " ").split():
        token = "".join(ch.lower() if ch.isalnum() else " " for ch in raw).strip()
        if not token or token.isdigit():
            continue
        pieces.append(token)
    return pieces


def visible_subword_parts(value: str) -> list[str]:
    """Prompt/signature-only subwords for disjoint visible vocabulary.

    This is a tokenizer repair, not a semantic route. It lets held-out visible
    words share character fragments with training words without exposing hidden
    family labels, tests, solutions, return shape, or required constructs.
    """

    out: list[str] = []
    seen: set[str] = set()
    for token in visible_identifier_parts(value):
        if len(token) < 4:
            continue
        fragments = [f"ch_{ch}" for ch in sorted(set(token))]
        fragments.extend(f"bg_{token[idx:idx + 2]}" for idx in range(max(0, len(token) - 1)))
        fragments.extend(
            [
                f"swp_{token[:3]}",
                f"sws_{token[-3:]}",
            ]
        )
        fragments.extend(f"swg_{token[idx:idx + 3]}" for idx in range(max(0, len(token) - 2)))
        for fragment in fragments:
            if fragment not in seen:
                seen.add(fragment)
                out.append(fragment)
    return out[:160]


def decoder_source_text_policy(fields: list[Any]) -> dict[str, Any]:
    field_names = [str(field) for field in fields]
    return {
        "base_visible_fields": field_names,
        "entry_point_identifier_split": "entry_point" in set(field_names),
        "visible_prompt_signature_subwords": True,
        "visible_prompt_intent_tags": True,
        "visible_prompt_intent_policy": "generic_prompt_entrypoint_keyword_tags_v1",
        "visible_prompt_operation_tags": "visible_operation_tags" in set(field_names),
        "visible_prompt_operation_policy": "prompt_signature_visible_operation_tags_v1",
        "visible_prompt_type_shape_tags": True,
        "visible_prompt_type_shape_policy": "prompt_signature_visible_type_shape_tags_v1",
        "uses_only_allowed_prompt_signature_fields": True,
        "forbidden_fields_added": [],
        "eval_tests_or_solutions_used": False,
        "public_data_used": False,
    }


def strict_disjoint_family_key(row: dict[str, Any], key_name: str = "concept_residual_label") -> str:
    if key_name and key_name not in {"auto", "strict"}:
        value = get_path(row, key_name.split("."), None)
        if value not in (None, "", []):
            return str(value)
    return str(
        row.get("concept_residual_label")
        or row.get("category")
        or row.get("residual_concept")
        or get_path(row, ["decoder_contract", "semantic_family"], "")
        or row.get("targeted_private_residual_family_v3")
        or get_path(row, ["decoder_contract", "type_family"], "")
        or "unknown"
    )


def family_disjoint_split_audit(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    holdout_families: list[str],
    family_key_name: str,
    target_mode: str,
) -> dict[str, Any]:
    train_families = {strict_disjoint_family_key(row, family_key_name) for row in train_rows}
    eval_families = {strict_disjoint_family_key(row, family_key_name) for row in eval_rows}

    def values(rows: list[dict[str, Any]], key: str) -> set[str]:
        return {str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()}

    train_solution_hashes = {stable_hash(str(row.get("solution_body") or "").strip()) for row in train_rows if row.get("solution_body")}
    eval_solution_hashes = {stable_hash(str(row.get("solution_body") or "").strip()) for row in eval_rows if row.get("solution_body")}
    train_token_hashes = {
        stable_hash(" ".join(target_tokens(str(row.get("solution_body") or ""), target_mode=target_mode)))
        for row in train_rows
        if row.get("solution_body")
    }
    eval_token_hashes = {
        stable_hash(" ".join(target_tokens(str(row.get("solution_body") or ""), target_mode=target_mode)))
        for row in eval_rows
        if row.get("solution_body")
    }
    return {
        "policy": "project_theseus_family_disjoint_split_audit_v0",
        "family_key": family_key_name,
        "holdout_families": holdout_families,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_family_count": len(train_families),
        "eval_family_count": len(eval_families),
        "train_families": sorted(train_families),
        "eval_families": sorted(eval_families),
        "overlap": {
            "family_overlap_count": len(train_families & eval_families),
            "family_overlap": sorted(train_families & eval_families),
            "prompt_overlap_count": len(values(train_rows, "prompt") & values(eval_rows, "prompt")),
            "entry_point_overlap_count": len(values(train_rows, "entry_point") & values(eval_rows, "entry_point")),
            "source_task_id_overlap_count": len(values(train_rows, "source_task_id") & values(eval_rows, "source_task_id")),
            "task_id_overlap_count": len(values(train_rows, "task_id") & values(eval_rows, "task_id")),
            "solution_body_sha256_overlap_count": len(train_solution_hashes & eval_solution_hashes),
            "target_token_template_sha256_overlap_count": len(train_token_hashes & eval_token_hashes),
        },
        "uses_family_labels_only_for_split": True,
        "family_labels_visible_to_generator": False,
        "eval_tests_visible_to_generator": False,
        "eval_solutions_visible_to_generator": False,
        "public_data_used": False,
        "teacher_used": False,
    }


def family_disjoint_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = json.loads(json.dumps(row))
        phase = str(copied.get("phase") or "")
        copied["phase"] = "private_baseline" if phase == "private_baseline" else f"family_disjoint_{phase}" if phase else "family_disjoint"
        copied["evaluation_split"] = "family_disjoint"
        copied["candidate_source"] = "neural_seed_token_decoder_comparator.family_disjoint"
        copied.setdefault("provenance", {})["evaluation_split"] = "family_disjoint"
        copied.setdefault("provenance", {})["family_disjoint_eval"] = True
        out.append(copied)
    return out
