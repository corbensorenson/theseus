#!/usr/bin/env python3
"""Language-profiled reversible tokenization for MoECOT training and replay."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Iterable

from neural_seed_open_vocab import decode_target_tokens, encode_tokens


PROFILE_BY_CATEGORY = {
    "english_conversation_instruction": "exact_text_v1",
    "english_broad": "exact_text_v1",
    "python": "exact_text_v1",
    "javascript_typescript": "exact_text_v1",
    "html_css": "exact_text_v1",
    "rust": "exact_text_v1",
}
_INVERSE_VOCAB_CACHE: OrderedDict[
    tuple[int, int], tuple[dict[str, int], dict[int, str]]
] = OrderedDict()
_INVERSE_VOCAB_CACHE_LIMIT = 8

EXACT_TOKEN_RE = re.compile(
    r"\r\n|\r|\n|[^\S\r\n]+|[A-Za-z_$][A-Za-z_$0-9]*|"
    r"0[xX][0-9A-Fa-f_]+|0[bB][01_]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|.",
    re.DOTALL,
)


def profile_for_category(category: str) -> str:
    try:
        return PROFILE_BY_CATEGORY[str(category)]
    except KeyError as exc:
        raise ValueError(f"unknown MoECOT tokenizer category: {category}") from exc


def logical_tokens(text: str, *, category: str) -> list[str]:
    profile = profile_for_category(category)
    # The final alternative intentionally consumes exactly one character. Keeping
    # whitespace as explicit tokens makes concatenation a lossless decoder.
    return exact_text_tokens(text)


def exact_text_tokens(text: str) -> list[str]:
    value = str(text)
    if not value:
        return []
    tokens = [match.group(0) for match in EXACT_TOKEN_RE.finditer(value)]
    if "".join(tokens) != value:
        raise ValueError("exact text tokenizer failed to cover input")
    return tokens


def decode_logical_tokens(tokens: Iterable[str], *, category: str) -> tuple[str, dict[str, Any]]:
    profile = profile_for_category(category)
    decoded, open_vocab = decode_target_tokens(tokens)
    if open_vocab.get("state") != "READY":
        return "", {
            "policy": "project_theseus_moecot_language_tokenizer_v1",
            "profile": profile,
            "state": "FAULT",
            "open_vocab": open_vocab,
            "failure_behavior": "reject_without_fallback",
        }
    text = "".join(decoded)
    return text, {
        "policy": "project_theseus_moecot_language_tokenizer_v1",
        "profile": profile,
        "state": "READY",
        "open_vocab": open_vocab,
        "fallback_return_count": 0,
    }


def encode_document(
    text: str,
    target_vocab: dict[str, int],
    *,
    category: str,
) -> tuple[list[str], list[int], dict[str, Any]]:
    value = str(text)
    profile = profile_for_category(category)
    tokens = logical_tokens(value, category=category)
    encoded, open_vocab = encode_tokens(tokens, target_vocab, stream="target")
    inverse = inverse_vocab(target_vocab)
    decoded_tokens = [inverse_token(inverse, token_id) for token_id in encoded]
    reconstructed, decode_receipt = decode_logical_tokens(decoded_tokens, category=category)
    roundtrip = roundtrip_receipt(value, reconstructed, profile=profile)
    return tokens, encoded, {
        **open_vocab,
        "policy": "project_theseus_moecot_language_tokenizer_v1",
        "category": category,
        "profile": profile,
        "roundtrip": roundtrip,
        "decode_state": decode_receipt["state"],
        "failure_behavior": "reject_without_fallback",
        "fallback_return_count": 0,
    }


def roundtrip_receipt(source: str, reconstructed: str, *, profile: str) -> dict[str, Any]:
    source_semantics = hashlib.sha256(source.encode()).hexdigest()
    reconstructed_semantics = hashlib.sha256(reconstructed.encode()).hexdigest()
    valid = source == reconstructed
    mode = "exact_utf8"
    return {
        "state": "GREEN" if valid else "RED",
        "mode": mode,
        "source_digest": source_semantics,
        "reconstructed_digest": reconstructed_semantics,
        "exact_text_equal": source == reconstructed,
    }
def inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    key = (id(vocab), len(vocab))
    cached = _INVERSE_VOCAB_CACHE.get(key)
    if cached is not None and cached[0] is vocab:
        _INVERSE_VOCAB_CACHE.move_to_end(key)
        return cached[1]
    inverse = {int(value): str(token) for token, value in vocab.items()}
    _INVERSE_VOCAB_CACHE[key] = (vocab, inverse)
    _INVERSE_VOCAB_CACHE.move_to_end(key)
    while len(_INVERSE_VOCAB_CACHE) > _INVERSE_VOCAB_CACHE_LIMIT:
        _INVERSE_VOCAB_CACHE.popitem(last=False)
    return inverse


def inverse_token(inverse: dict[int, str], token_id: int) -> str:
    try:
        return inverse[int(token_id)]
    except KeyError as exc:
        raise ValueError(f"target token id is absent from vocabulary: {token_id}") from exc
