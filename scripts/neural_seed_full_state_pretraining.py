#!/usr/bin/env python3
"""Full-state corpus pretraining row builders for the neural seed comparator.

This module owns admitted local Python corpus collection, prompt/signature-like
source text construction, no-cheat quality filters, and vocab-extension
summaries. It does not train models, run public calibration, inspect eval tests
or solutions, or call a teacher.
"""

from __future__ import annotations

import ast
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import (  # noqa: E402
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    ratio,
    rel,
    stable_hash,
)
from neural_seed_static_coherence import (  # noqa: E402
    expression_complexity_score,
    expression_is_parameter_copy_call,
    expression_is_parameter_identity_copy,
    expression_is_static_literal_only,
    expression_uses_any_name,
)
from neural_seed_token_decoder_support import encoded_target_payload_ids, encode_target_rows, target_tokens  # noqa: E402
from neural_seed_visible_source import (  # noqa: E402
    visible_identifier_parts,
    visible_prompt_intent_tags,
    visible_prompt_operation_tags,
    visible_prompt_type_shape_tags,
    visible_subword_parts,
)
from narrow_corpus_pretraining_spine import basic_tokens as bpe_basic_tokens  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def full_state_pretraining_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict_or_empty(get_path(config, ["pretraining_initialization", "full_state_warmup"], {}))
    if not cfg:
        cfg = dict_or_empty(config.get("full_state_pretraining"))
    return cfg


def full_state_pretraining_vocab_bodies(config: dict[str, Any], *, seed: int) -> list[str]:
    cfg = full_state_pretraining_config(config)
    if not bool(cfg.get("enabled", False)):
        return []
    if not bool(cfg.get("extend_target_vocab", True)):
        return []
    max_examples = int(cfg.get("target_vocab_max_examples") or cfg.get("max_examples") or 512)
    max_files = int(cfg.get("target_vocab_max_files") or cfg.get("max_files") or 128)
    max_function_body_chars = int(cfg.get("max_function_body_chars") or 5000)
    examples, _summary = collect_full_state_python_examples(
        cfg,
        max_files=max_files,
        max_examples=max_examples,
        min_target_tokens=1,
        max_function_body_chars=max_function_body_chars,
        seed=seed,
    )
    return [str(row["body"]) for row in examples]


def full_state_pretraining_vocab_source_texts(config: dict[str, Any], *, seed: int) -> list[str]:
    cfg = full_state_pretraining_config(config)
    if not bool(cfg.get("enabled", False)):
        return []
    if not bool(cfg.get("extend_source_vocab", True)):
        return []
    max_examples = int(cfg.get("source_vocab_max_examples") or cfg.get("target_vocab_max_examples") or cfg.get("max_examples") or 512)
    max_files = int(cfg.get("source_vocab_max_files") or cfg.get("target_vocab_max_files") or cfg.get("max_files") or 128)
    max_function_body_chars = int(cfg.get("max_function_body_chars") or 5000)
    examples, _summary = collect_full_state_python_examples(
        cfg,
        max_files=max_files,
        max_examples=max_examples,
        min_target_tokens=1,
        max_function_body_chars=max_function_body_chars,
        seed=seed,
    )
    return [str(row["source_text"]) for row in examples]


def full_state_target_vocab_extension_summary(bodies: list[str]) -> dict[str, Any]:
    token_counter: Counter[str] = Counter()
    for body in bodies:
        token_counter.update(target_tokens(str(body), target_mode="body_tokens"))
    return {
        "enabled": bool(bodies),
        "body_count": len(bodies),
        "unique_body_token_count": len(token_counter),
        "body_token_count": sum(token_counter.values()),
        "score_semantics": "Target vocab extension is built only from admitted self-supervised corpus function bodies; it changes token support, not eval tasks or generator-visible answer metadata.",
    }


def full_state_source_vocab_extension_summary(source_texts: list[str]) -> dict[str, Any]:
    token_counter: Counter[str] = Counter()
    for text in source_texts:
        token_counter.update(source_summary_tokens(str(text)))
    return {
        "enabled": bool(source_texts),
        "source_text_count": len(source_texts),
        "unique_source_token_count": len(token_counter),
        "source_token_count": sum(token_counter.values()),
        "score_semantics": "Source vocab extension is built only from admitted self-supervised corpus function prompt/signature metadata and docstring source text. It changes token support for full-state pretraining, not eval tasks, tests, solutions, or answer metadata.",
    }


def source_summary_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in bpe_basic_tokens(str(text)):
        if isinstance(token, (list, tuple)):
            tokens.append("".join(str(part) for part in token))
        else:
            tokens.append(str(token))
    return tokens


def filter_complete_target_examples(
    examples: list[dict[str, Any]],
    *,
    max_target: int,
    target_mode: str,
    target_vocab: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject targets that cannot fit with BOS/EOS instead of truncating them.

    Prefix-only target truncation is not a harmless batching detail: it teaches
    programs without their closing control flow or return. This filter is
    target-local and runs before train/eval splitting. It never reads verifier
    outcomes, heldout rows, public benchmark payloads, or answer metadata.
    """

    payload_limit = max(0, int(max_target) - 2)
    kept: list[dict[str, Any]] = []
    lengths: list[int] = []
    kept_lengths: list[int] = []
    rejected_lengths: list[int] = []
    fault_count = 0
    for row in examples:
        try:
            body = str(row.get("body") or "")
            if target_vocab:
                encoded, encode_receipt = encoded_target_payload_ids(
                    body,
                    vocab=target_vocab,
                    target_mode=target_mode,
                )
                if int(encode_receipt.get("unknown_token_count") or 0) > 0:
                    fault_count += 1
                    continue
                token_count = len(encoded)
            else:
                token_count = len(target_tokens(body, target_mode=target_mode))
        except Exception:  # Typed target encoders fail closed at admission.
            fault_count += 1
            continue
        lengths.append(token_count)
        if token_count <= payload_limit:
            kept.append(row)
            kept_lengths.append(token_count)
        else:
            rejected_lengths.append(token_count)

    ordered = sorted(lengths)

    def quantile(fraction: float) -> int:
        if not ordered:
            return 0
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return int(ordered[index])

    return kept, {
        "policy": "complete_target_sequence_admission_v1",
        "target_mode": str(target_mode or ""),
        "max_target_tokens": int(max_target),
        "max_target_payload_tokens": payload_limit,
        "candidate_example_count": len(examples),
        "complete_example_count": len(kept),
        "oversized_example_count": len(rejected_lengths),
        "target_encoding_fault_count": fault_count,
        "complete_example_rate": ratio(len(kept), len(examples)),
        "candidate_target_token_positions": sum(lengths),
        "complete_target_token_positions": sum(kept_lengths),
        "excluded_target_token_positions": sum(rejected_lengths),
        "target_token_length_p50": quantile(0.50),
        "target_token_length_p90": quantile(0.90),
        "target_token_length_p95": quantile(0.95),
        "target_token_length_p99": quantile(0.99),
        "target_sequence_truncation_count": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "score_semantics": (
            "AST/body target length admission only. Oversized or unencodable target rows are excluded "
            "before splitting so no prefix-only target is trained or evaluated."
        ),
    }


def build_full_state_pretraining_rows(
    config: dict[str, Any],
    *,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    max_source: int,
    max_target: int,
    target_mode: str,
    seed: int,
) -> dict[str, Any]:
    """Build same-architecture warmup rows from admitted local corpus code.

    This is self-supervised function-body reconstruction over local project
    source. It deliberately does not read eval rows, public benchmark payloads,
    teacher rows, tests, or solution metadata. The output is encoded into the
    strict comparator's current source/target vocab so the actual model instance
    can be warmed up before private task-pair adaptation.
    """

    cfg = full_state_pretraining_config(config)
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "active": False,
            "reason": "full_state_pretraining_disabled",
            "source_rows": [],
            "target_rows": [],
            "summary": {"enabled": False, "active": False},
        }
    max_files = int(cfg.get("max_files") or 64)
    max_examples = int(cfg.get("max_examples") or 256)
    min_target_tokens = int(cfg.get("min_target_tokens") or 8)
    max_function_body_chars = int(cfg.get("max_function_body_chars") or 5000)
    examples, collection_summary = collect_full_state_python_examples(
        cfg,
        max_files=max_files,
        max_examples=max_examples,
        min_target_tokens=min_target_tokens,
        max_function_body_chars=max_function_body_chars,
        seed=seed,
    )
    examples, target_completeness = filter_complete_target_examples(
        examples,
        max_target=max_target,
        target_mode=target_mode,
        target_vocab=target_vocab,
    )
    manifest_path = resolve(str(cfg.get("corpus_manifest") or "data/training_sources/narrow_corpus_manifest.json"))
    eval_fraction = max(0.0, min(0.5, float(cfg.get("eval_fraction") or 0.08)))
    default_eval_examples = max(32, int(len(examples) * eval_fraction)) if examples else 0
    max_eval_examples = max(0, int(cfg.get("max_eval_examples") or default_eval_examples))
    eval_count = min(len(examples) // 3, max_eval_examples, default_eval_examples) if examples else 0
    eval_examples = examples[:eval_count]
    train_examples = examples[eval_count:] or examples
    if train_examples is examples:
        eval_examples = []
    source_texts = [str(row["source_text"]) for row in train_examples]
    body_texts = [str(row["body"]) for row in train_examples]
    eval_source_texts = [str(row["source_text"]) for row in eval_examples]
    eval_body_texts = [str(row["body"]) for row in eval_examples]
    source_rows = encode_many(source_texts, source_vocab, max_source) if source_texts else []
    target_rows = (
        encode_target_rows(body_texts, target_vocab, max_target, target_mode=target_mode)
        if body_texts
        else []
    )
    eval_source_rows = encode_many(eval_source_texts, source_vocab, max_source) if eval_source_texts else []
    eval_target_rows = (
        encode_target_rows(eval_body_texts, target_vocab, max_target, target_mode=target_mode)
        if eval_body_texts
        else []
    )
    unknown_target_token_count = 0
    total_target_token_count = 0
    unknown_source_token_count = 0
    total_source_token_count = 0
    unk_id = int(target_vocab.get("<unk>", 1))
    target_pad_id = int(target_vocab.get("<pad>", 0))
    for row in target_rows:
        for token_id in row:
            if int(token_id) == target_pad_id:
                continue
            total_target_token_count += 1
            if int(token_id) == unk_id:
                unknown_target_token_count += 1
    source_unk_id = int(source_vocab.get("<unk>", 1))
    source_pad_id = int(source_vocab.get("<pad>", 0))
    for row in source_rows:
        for token_id in row:
            if int(token_id) == source_pad_id:
                continue
            total_source_token_count += 1
            if int(token_id) == source_unk_id:
                unknown_source_token_count += 1
    public_payload_admitted = int(collection_summary.get("public_benchmark_payload_admitted_count") or 0)
    active = bool(source_rows and target_rows and public_payload_admitted == 0)
    summary = {
        "enabled": True,
        "active": active,
        "policy": str(cfg.get("policy") or "comparator_compatible_python_function_full_state_warmup_v1"),
        "corpus_manifest": rel(manifest_path),
        "manifest_source_count": int(collection_summary.get("manifest_source_count") or 0),
        "admitted_python_files": int(collection_summary.get("admitted_python_files") or 0),
        "example_count": len(examples),
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "encoded_source_rows": len(source_rows),
        "encoded_target_rows": len(target_rows),
        "encoded_eval_source_rows": len(eval_source_rows),
        "encoded_eval_target_rows": len(eval_target_rows),
        "source_unknown_token_count": unknown_source_token_count,
        "source_total_token_count": total_source_token_count,
        "source_unknown_token_rate": ratio(unknown_source_token_count, total_source_token_count),
        "target_unknown_token_count": unknown_target_token_count,
        "target_total_token_count": total_target_token_count,
        "target_unknown_token_rate": ratio(unknown_target_token_count, total_target_token_count),
        "stale_hash_count": int(collection_summary.get("stale_hash_count") or 0),
        "public_benchmark_payload_admitted_count": public_payload_admitted,
        "skip_reasons": dict(collection_summary.get("skip_reasons") or {}),
        "target_sequence_admission": target_completeness,
        "target_vocab_extended_from_corpus": bool(cfg.get("extend_target_vocab", True)),
        "source_text_style": str(cfg.get("source_text_style") or "prompt_signature_metadata_v2"),
        "source_alignment_semantics": (
            "Corpus pretraining sources are derived from visible function names, argument names, "
            "docstrings, and generic visible-intent tags only, formatted to resemble the "
            "prompt/signature source seen by the strict comparator. Function bodies are "
            "target-only."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "public_training_rows": 0,
    }
    return {
        "enabled": True,
        "active": active,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "eval_source_rows": eval_source_rows,
        "eval_target_rows": eval_target_rows,
        "examples": [
            {
                "path": row["path"],
                "function": row["function"],
                "source_sha256": stable_hash(str(row["source_text"])),
                "body_sha256": stable_hash(str(row["body"])),
                "body_token_count": len(target_tokens(str(row["body"]), target_mode=target_mode)),
                "quality": dict_or_empty(row.get("quality")),
            }
            for row in train_examples[:16]
        ],
        "eval_examples": [
            {
                "path": row["path"],
                "function": row["function"],
                "source_sha256": stable_hash(str(row["source_text"])),
                "body_sha256": stable_hash(str(row["body"])),
                "body_token_count": len(target_tokens(str(row["body"]), target_mode=target_mode)),
                "quality": dict_or_empty(row.get("quality")),
            }
            for row in eval_examples[:8]
        ],
        "summary": summary,
    }


def collect_full_state_python_examples(
    cfg: dict[str, Any],
    *,
    max_files: int,
    max_examples: int,
    min_target_tokens: int,
    max_function_body_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = resolve(str(cfg.get("corpus_manifest") or "data/training_sources/narrow_corpus_manifest.json"))
    manifest = read_json(manifest_path)
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    # Keep a deterministic bottom-k sample over the entire admitted corpus.
    # Stopping once max_examples is reached biases the corpus toward whichever
    # packages happen to sort first in the manifest.
    sampled: list[tuple[int, int, dict[str, Any]]] = []
    seen_source_body: set[str] = set()
    skip_reasons: Counter[str] = Counter()
    stale_hash_count = 0
    admitted_python_files = 0
    public_payload_admitted = 0
    eligible_example_count = 0
    exact_source_body_duplicate_count = 0
    exact_body_hashes: set[str] = set()
    quality_cfg = dict_or_empty(cfg.get("quality_filter"))
    for source in sources:
        row = dict_or_empty(source)
        if not bool(row.get("admitted")):
            skip_reasons[str(row.get("reason") or "not_admitted")] += 1
            continue
        if bool(row.get("public_benchmark_payload_detected")):
            public_payload_admitted += 1
            skip_reasons["public_benchmark_payload_detected"] += 1
            continue
        rel_path = str(row.get("path") or "")
        if not rel_path.endswith(".py"):
            skip_reasons["non_python_source"] += 1
            continue
        path = resolve(rel_path)
        outside_root = not is_relative_to(path, ROOT)
        if not path.exists():
            skip_reasons["missing_source_path"] += 1
            continue
        if outside_root and not bool(row.get("license_allowed")):
            skip_reasons["outside_root_without_license_admission"] += 1
            continue
        if admitted_python_files >= max_files:
            skip_reasons["max_files_reached"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            skip_reasons[f"read_error:{exc.__class__.__name__}"] += 1
            continue
        char_count = int(row.get("char_count") or len(text))
        text = text[:max(0, char_count)]
        expected_hash = str(row.get("sha256") or "")
        if expected_hash and stable_hash(text) != expected_hash:
            stale_hash_count += 1
            skip_reasons["manifest_hash_mismatch"] += 1
            continue
        admitted_python_files += 1
        for example in python_function_body_pretraining_examples(
            rel_path,
            text,
            max_function_body_chars=max_function_body_chars,
            source_text_style=str(cfg.get("source_text_style") or "prompt_signature_metadata_v2"),
        ):
            quality_reject_reason = corpus_pretraining_quality_reject_reason(example, quality_cfg)
            if quality_reject_reason:
                skip_reasons[f"quality_filter:{quality_reject_reason}"] += 1
                continue
            target_token_count = len(target_tokens(str(example["body"]), target_mode="body_tokens"))
            if target_token_count < min_target_tokens:
                skip_reasons["too_few_target_tokens"] += 1
                continue
            source_body_hash = stable_hash(
                json.dumps(
                    [str(example.get("source_text") or ""), str(example.get("body") or "")],
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            if source_body_hash in seen_source_body:
                exact_source_body_duplicate_count += 1
                skip_reasons["exact_source_body_duplicate"] += 1
                continue
            seen_source_body.add(source_body_hash)
            exact_body_hashes.add(stable_hash(str(example.get("body") or "")))
            eligible_example_count += 1
            rank = int(
                stable_hash(
                    f"{seed}:{source_body_hash}:{example.get('path')}:{example.get('function')}"
                ),
                16,
            )
            heap_row = (-rank, eligible_example_count, example)
            if len(sampled) < max_examples:
                heapq.heappush(sampled, heap_row)
            elif rank < -sampled[0][0]:
                heapq.heapreplace(sampled, heap_row)
    examples = [row for _rank, _ordinal, row in sorted(sampled, key=lambda item: (-item[0], item[1]))]
    source_token_counter: Counter[str] = Counter()
    target_token_counter: Counter[str] = Counter()
    selected_body_hashes: set[str] = set()
    for example in examples:
        source_token_counter.update(source_summary_tokens(str(example.get("source_text") or "")))
        target_token_counter.update(target_tokens(str(example.get("body") or ""), target_mode="body_tokens"))
        selected_body_hashes.add(stable_hash(str(example.get("body") or "")))
    return examples, {
        "manifest_source_count": len(sources),
        "admitted_python_files": admitted_python_files,
        "example_count": len(examples),
        "eligible_example_count_before_sampling": eligible_example_count,
        "selection_policy": "deterministic_full_stream_bottom_k_v1",
        "exact_source_body_duplicate_count": exact_source_body_duplicate_count,
        "eligible_unique_body_count": len(exact_body_hashes),
        "selected_unique_body_count": len(selected_body_hashes),
        "selected_single_pass_source_token_positions": sum(source_token_counter.values()),
        "selected_single_pass_target_token_positions": sum(target_token_counter.values()),
        "selected_source_vocabulary_size": len(source_token_counter),
        "selected_target_vocabulary_size": len(target_token_counter),
        "stale_hash_count": stale_hash_count,
        "public_benchmark_payload_admitted_count": public_payload_admitted,
        "skip_reasons": dict(skip_reasons.most_common()),
        "quality_filter": corpus_pretraining_quality_filter_summary(quality_cfg),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "public_training_rows": 0,
    }


def python_function_body_pretraining_examples(
    rel_path: str,
    text: str,
    *,
    max_function_body_chars: int,
    source_text_style: str = "prompt_signature_metadata_v2",
) -> list[dict[str, Any]]:
    try:
        parsed = ast.parse(text)
    except SyntaxError:
        return []
    examples: list[dict[str, Any]] = []
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        body_nodes = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr) or not isinstance(getattr(stmt, "value", None), ast.Constant)]
        body_nodes = body_nodes or list(node.body)
        try:
            body = "\n".join(ast.unparse(statement) for statement in body_nodes).strip()
        except (AttributeError, TypeError, ValueError):
            continue
        if not body or len(body) > max_function_body_chars:
            continue
        args = [arg.arg for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)]
        if node.args.vararg is not None:
            args.append(node.args.vararg.arg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg.arg)
        doc = ast.get_docstring(node) or ""
        clean_doc = " ".join(str(doc).split())
        identifier_parts = visible_identifier_parts(node.name)
        arg_parts = [part for arg in args for part in visible_identifier_parts(arg)]
        source_text = python_function_body_pretraining_source_text(
            rel_path,
            node.name,
            args,
            doc,
            source_text_style=source_text_style,
        )
        quality = corpus_function_body_quality(node)
        examples.append(
            {
                "path": rel_path,
                "function": node.name,
                "source_text_style": source_text_style,
                "source_text": source_text,
                "body": body,
                "prompt_source": "docstring" if clean_doc else "identifier_fallback",
                "prompt_character_count": len(clean_doc),
                "quality": quality,
            }
        )
    return examples


def python_function_body_pretraining_source_text(
    rel_path: str,
    function_name: str,
    args: list[str],
    doc: str,
    *,
    source_text_style: str,
) -> str:
    """Build no-cheat corpus source text in the same shape as eval input.

    The strict private decoder sees natural-language prompt text plus an
    entry-point name. Corpus pretraining cannot use heldout/eval payloads, but
    local Python functions do expose analogous visible source metadata:
    docstring, function name, and argument names. The body remains target-only.
    """

    identifier_parts = visible_identifier_parts(function_name)
    arg_parts = [part for arg in args for part in visible_identifier_parts(arg)]
    clean_doc = " ".join(str(doc or "").split())[:512]
    prompt_text = clean_doc or f"Implement Python function {function_name}."
    signature_text = f"def {function_name}({', '.join(args)}):"
    prompt_like = "\n".join(part for part in [prompt_text, function_name] if part)
    intent_tags = visible_prompt_intent_tags(prompt_like)
    operation_tags = visible_prompt_operation_tags(prompt_like)
    type_shape_tags = visible_prompt_type_shape_tags(prompt_like)
    visible_subwords = visible_subword_parts(prompt_like)
    aligned_chunks = [
        prompt_text,
        function_name,
        "visible_intent_tags " + " ".join(intent_tags) if intent_tags else "",
        "prompt_operation_hints " + " ".join(operation_tags) if operation_tags else "",
        "visible_type_shape_tags " + " ".join(type_shape_tags) if type_shape_tags else "",
        f"signature {signature_text}",
        "arguments " + " ".join(args) if args else "",
        "entry_point_parts " + " ".join(identifier_parts) if identifier_parts else "",
        "argument_parts " + " ".join(arg_parts) if arg_parts else "",
        "visible_subwords " + " ".join(visible_subwords) if visible_subwords else "",
    ]
    if source_text_style == "legacy_metadata_v1":
        chunks = [
            f"path {rel_path}",
            f"python function {function_name}",
            "function_name_parts " + " ".join(identifier_parts) if identifier_parts else "",
            "arguments " + " ".join(args) if args else "",
            "argument_parts " + " ".join(arg_parts) if arg_parts else "",
            "docstring " + clean_doc if clean_doc else "",
        ]
    else:
        chunks = [
            *aligned_chunks,
            f"source_style {source_text_style}",
        ]
    return "\n".join(part for part in chunks if part)


def corpus_pretraining_quality_filter_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    return {
        "enabled": True,
        "policy": str(cfg.get("policy") or "nontrivial_algorithmic_function_body_filter_v1"),
        "exclude_dunder_functions": bool(cfg.get("exclude_dunder_functions", True)),
        "reject_inert_bodies": bool(cfg.get("reject_inert_bodies", True)),
        "reject_identity_copy_shortcuts": bool(cfg.get("reject_identity_copy_shortcuts", True)),
        "min_body_statement_count": int(cfg.get("min_body_statement_count") or 1),
        "min_nontrivial_signal_count": int(cfg.get("min_nontrivial_signal_count") or 2),
        "require_parameter_or_call": bool(cfg.get("require_parameter_or_call", True)),
        "score_semantics": (
            "Filters admitted clean corpus functions before full-state warmup using AST-local body quality only. "
            "It never reads eval tests, eval solutions, public benchmark payloads, or answer metadata."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def corpus_pretraining_quality_reject_reason(example: dict[str, Any], cfg: dict[str, Any]) -> str:
    if not bool(cfg.get("enabled", False)):
        return ""
    quality = dict_or_empty(example.get("quality"))
    if bool(cfg.get("exclude_dunder_functions", True)) and bool(quality.get("dunder_function")):
        return "dunder_function"
    min_body_statement_count = int(cfg.get("min_body_statement_count") or 1)
    if int(quality.get("body_statement_count") or 0) < min_body_statement_count:
        return "too_few_body_statements"
    if bool(cfg.get("reject_inert_bodies", True)) and bool(quality.get("inert_body")):
        return "inert_body"
    if bool(cfg.get("reject_identity_copy_shortcuts", True)) and bool(quality.get("identity_copy_shortcut_body")):
        return "identity_copy_shortcut_body"
    min_nontrivial_signal_count = int(cfg.get("min_nontrivial_signal_count") or 0)
    if int(quality.get("nontrivial_signal_count") or 0) < min_nontrivial_signal_count:
        return "too_few_nontrivial_signals"
    if bool(cfg.get("require_parameter_or_call", True)):
        if int(quality.get("parameter_load_count") or 0) <= 0 and int(quality.get("call_count") or 0) <= 0:
            return "no_parameter_or_call_use"
    return ""


def corpus_function_body_quality(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    params = [arg.arg for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)]
    if node.args.vararg is not None:
        params.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        params.append(node.args.kwarg.arg)
    param_set = set(params)
    body_nodes = [
        stmt
        for index, stmt in enumerate(node.body)
        if not (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(getattr(stmt.value, "value", None), str)
        )
    ]
    loads: Counter[str] = Counter()
    stores: Counter[str] = Counter()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                loads[child.id] += 1
            elif isinstance(child.ctx, (ast.Store, ast.Del)):
                stores[child.id] += 1
    returns = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
    valued_return_count = sum(1 for child in returns if child.value is not None)
    bare_return_count = sum(1 for child in returns if child.value is None)
    literal_only_return_count = sum(
        1 for child in returns if child.value is not None and expression_is_static_literal_only(child.value)
    )
    identity_copy_return_count = sum(
        1 for child in returns if expression_is_parameter_identity_copy(child.value, param_set)
    )
    nontrivial_return_count = sum(
        1
        for child in returns
        if child.value is not None
        and (
            expression_uses_any_name(child.value, param_set)
            or expression_complexity_score(child.value) >= 2
        )
        and not expression_is_static_literal_only(child.value)
    )
    parameter_load_count = sum(int(loads.get(param, 0)) for param in params)
    control_flow_count = sum(1 for child in ast.walk(node) if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)))
    loop_count = sum(1 for child in ast.walk(node) if isinstance(child, (ast.For, ast.While)))
    assignment_count = sum(1 for child in ast.walk(node) if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)))
    call_count = sum(1 for child in ast.walk(node) if isinstance(child, ast.Call))
    identity_copy_call_count = sum(1 for child in ast.walk(node) if expression_is_parameter_copy_call(child, param_set))
    isinstance_guard_call_count = sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "isinstance"
        and bool(child.args)
        and expression_uses_any_name(child.args[0], param_set)
    )
    comprehension_count = sum(
        1
        for child in ast.walk(node)
        if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )
    local_store_name_count = len([name for name in stores if name not in param_set])
    nontrivial_signal_count = sum(
        int(value > 0)
        for value in [
            parameter_load_count,
            control_flow_count,
            assignment_count,
            call_count,
            comprehension_count,
            nontrivial_return_count,
            local_store_name_count,
        ]
    )
    inert_body = (
        not body_nodes
        or (
            valued_return_count == 0
            and assignment_count == 0
            and call_count == 0
            and control_flow_count == 0
            and comprehension_count == 0
        )
        or (
            len(body_nodes) <= 1
            and nontrivial_return_count == 0
            and parameter_load_count == 0
            and assignment_count == 0
            and call_count == 0
            and comprehension_count == 0
        )
    )
    identity_copy_shortcut_body = (
        valued_return_count > 0
        and identity_copy_return_count == valued_return_count
        and assignment_count == 0
        and loop_count == 0
        and comprehension_count == 0
        and local_store_name_count == 0
        and call_count <= identity_copy_call_count + isinstance_guard_call_count
    )
    return {
        "policy": "corpus_function_body_quality_v1",
        "function": node.name,
        "dunder_function": bool(node.name.startswith("__") and node.name.endswith("__")),
        "parameter_count": len(params),
        "parameter_load_count": parameter_load_count,
        "body_statement_count": len(body_nodes),
        "valued_return_count": valued_return_count,
        "bare_return_count": bare_return_count,
        "literal_only_return_count": literal_only_return_count,
        "identity_copy_return_count": identity_copy_return_count,
        "identity_copy_shortcut_body": identity_copy_shortcut_body,
        "nontrivial_return_count": nontrivial_return_count,
        "control_flow_count": control_flow_count,
        "loop_count": loop_count,
        "assignment_count": assignment_count,
        "call_count": call_count,
        "comprehension_count": comprehension_count,
        "local_store_name_count": local_store_name_count,
        "nontrivial_signal_count": nontrivial_signal_count,
        "inert_body": inert_body,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }
