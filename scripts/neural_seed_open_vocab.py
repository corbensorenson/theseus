#!/usr/bin/env python3
"""Reversible byte fallback for Theseus neural-seed token streams.

The code generator keeps frequent source/Python tokens atomic. Tokens outside
the bounded vocabulary are encoded as UTF-8 bytes between typed boundaries.
This removes destructive ``<unk>`` substitution without loading an external
tokenizer, changing candidate semantics, or synthesizing code.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Iterable


SOURCE_BYTE_BEGIN = "<source_token_bytes>"
SOURCE_BYTE_END = "</source_token_bytes>"
TARGET_BYTE_BEGIN = "<target_token_bytes>"
TARGET_BYTE_END = "</target_token_bytes>"
BYTE_PREFIX = "<byte:"
BYTE_PIECE_PREFIX = "<bytes:"
BYTE_TOKENS = tuple(f"<byte:{value:02x}>" for value in range(256))
MAX_TOKEN_BYTES = 512
DEFAULT_PIECE_BUDGET = 512
_PIECE_INVENTORY_CACHE: OrderedDict[
    int,
    tuple[dict[str, int], dict[int, list[tuple[bytes, str]]]],
] = OrderedDict()
_PIECE_INVENTORY_CACHE_LIMIT = 32
_ENCODED_PIECE_CACHE: OrderedDict[
    tuple[int, int, bytes], tuple[dict[str, int], tuple[str, ...]],
] = OrderedDict()
_ENCODED_PIECE_CACHE_LIMIT = 65_536


def reserve_byte_fallback_tokens(
    vocab: dict[str, int],
    *,
    max_vocab: int,
    stream: str,
) -> dict[str, int]:
    begin, end = stream_boundaries(stream)
    required = [begin, end, *BYTE_TOKENS]
    if int(max_vocab) < len(vocab) + len([token for token in required if token not in vocab]):
        raise ValueError(f"max_vocab_too_small_for_{stream}_byte_fallback")
    for token in required:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def populate_open_vocab(
    vocab: dict[str, int],
    counts: Counter[str],
    *,
    max_vocab: int,
    stream: str,
    piece_budget: int = DEFAULT_PIECE_BUDGET,
) -> dict[str, int]:
    """Add frequent whole tokens plus corpus-derived reversible byte pieces."""

    reserve_byte_fallback_tokens(vocab, max_vocab=max_vocab, stream=stream)
    remaining = max(0, int(max_vocab) - len(vocab))
    piece_slots = min(max(0, int(piece_budget)), remaining // 3)
    whole_slots = max(0, remaining - piece_slots)
    for token, _count in counts.most_common():
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= int(max_vocab) - piece_slots:
            break

    residual_counts = Counter({token: count for token, count in counts.items() if token not in vocab})
    ngrams: Counter[bytes] = Counter()
    for token, count in residual_counts.items():
        payload = str(token).encode("utf-8")
        for width in (4, 3, 2):
            for index in range(0, max(0, len(payload) - width + 1)):
                ngrams[payload[index : index + width]] += int(count)
    for piece, _count in ngrams.most_common(piece_slots):
        marker = byte_piece_token(piece)
        if marker not in vocab:
            vocab[marker] = len(vocab)
        if len(vocab) >= int(max_vocab):
            break

    if len(vocab) < int(max_vocab):
        for token, _count in counts.most_common():
            if token not in vocab:
                vocab[token] = len(vocab)
            if len(vocab) >= int(max_vocab):
                break
    return vocab


def encode_tokens(
    tokens: Iterable[str],
    vocab: dict[str, int],
    *,
    stream: str,
) -> tuple[list[int], dict[str, Any]]:
    logical_tokens = [str(token) for token in tokens]
    begin, end = stream_boundaries(stream)
    fallback_active = begin in vocab and end in vocab and all(token in vocab for token in BYTE_TOKENS)
    ids: list[int] = []
    fallback_token_count = 0
    fallback_byte_count = 0
    unknown_token_count = 0
    unknown_id = int(vocab.get("<unk>", 1))
    for token in logical_tokens:
        if token in vocab:
            ids.append(int(vocab[token]))
            continue
        payload = token.encode("utf-8")
        if fallback_active and 0 < len(payload) <= MAX_TOKEN_BYTES:
            pieces = encode_byte_pieces(payload, vocab)
            ids.append(int(vocab[begin]))
            ids.extend(int(vocab[piece]) for piece in pieces)
            ids.append(int(vocab[end]))
            fallback_token_count += 1
            fallback_byte_count += len(payload)
            continue
        ids.append(unknown_id)
        unknown_token_count += 1
    return ids, {
        "policy": "neural_seed_reversible_byte_fallback_v1",
        "stream": stream,
        "fallback_active": fallback_active,
        "logical_token_count": len(logical_tokens),
        "encoded_token_count": len(ids),
        "fallback_token_count": fallback_token_count,
        "fallback_byte_count": fallback_byte_count,
        "unknown_token_count": unknown_token_count,
        "failure_behavior": "explicit_unknown_only_when_byte_bound_exceeded",
    }


def bound_logical_tokens(
    tokens: Iterable[str], *, maximum_token_bytes: int = MAX_TOKEN_BYTES
) -> list[str]:
    """Losslessly split oversized logical tokens before byte fallback encoding.

    The open-vocabulary codec deliberately rejects a single fallback span above
    ``MAX_TOKEN_BYTES``. Structured formats can nevertheless contain legitimate
    long atoms (for example, base64-encoded exact byte fields). This helper keeps
    the codec's per-span bound while producing adjacent UTF-8-safe spans whose
    concatenation is byte-identical to the original token stream.
    """

    bound = int(maximum_token_bytes)
    if bound <= 0:
        raise ValueError("maximum_token_bytes_must_be_positive")
    logical_tokens = [str(token) for token in tokens]
    result: list[str] = []
    for raw in logical_tokens:
        token = str(raw)
        if len(token.encode("utf-8")) <= bound:
            result.append(token)
            continue
        chunk: list[str] = []
        chunk_bytes = 0
        for character in token:
            character_bytes = len(character.encode("utf-8"))
            if character_bytes > bound:
                raise ValueError("unicode_scalar_exceeds_open_vocab_span_bound")
            if chunk and chunk_bytes + character_bytes > bound:
                result.append("".join(chunk))
                chunk = []
                chunk_bytes = 0
            chunk.append(character)
            chunk_bytes += character_bytes
        if chunk:
            result.append("".join(chunk))
    if "".join(result) != "".join(logical_tokens):
        raise ValueError("bounded_logical_tokens_failed_exact_reconstruction")
    return result


def decode_target_tokens(tokens: Iterable[str]) -> tuple[list[str], dict[str, Any]]:
    out: list[str] = []
    payload = bytearray()
    active = False
    fallback_token_count = 0
    faults: list[dict[str, Any]] = []
    for index, raw in enumerate(tokens):
        token = str(raw)
        if not active:
            if token == TARGET_BYTE_BEGIN:
                active = True
                payload.clear()
            elif token == TARGET_BYTE_END or is_byte_token(token):
                faults.append(fault("BYTE_FALLBACK_BOUNDARY", index, token))
            else:
                out.append(token)
            continue
        if is_byte_token(token):
            if len(payload) >= MAX_TOKEN_BYTES:
                faults.append(fault("BYTE_FALLBACK_TOO_LONG", index, token))
                active = False
                payload.clear()
                continue
            payload.extend(byte_token_bytes(token))
            if len(payload) > MAX_TOKEN_BYTES:
                faults.append(fault("BYTE_FALLBACK_TOO_LONG", index, token))
                active = False
                payload.clear()
            continue
        if token == TARGET_BYTE_END:
            if not payload:
                faults.append(fault("BYTE_FALLBACK_EMPTY", index, token))
            else:
                try:
                    out.append(bytes(payload).decode("utf-8"))
                    fallback_token_count += 1
                except UnicodeDecodeError:
                    faults.append(fault("BYTE_FALLBACK_UTF8", index, token))
            active = False
            payload.clear()
            continue
        faults.append(fault("BYTE_FALLBACK_TOKEN_EXPECTED", index, token))
        active = False
        payload.clear()
    if active:
        faults.append(fault("BYTE_FALLBACK_TRUNCATED", len(out), "end_of_stream"))
    return out, {
        "policy": "neural_seed_reversible_byte_fallback_v1",
        "state": "READY" if not faults else "FAULT",
        "fallback_token_count": fallback_token_count,
        "faults": faults,
        "failure_behavior": "reject_without_fallback",
        "candidate_generation_credit": 0,
        "public_training_rows": 0,
        "external_inference_calls": 0,
    }


def active_target_span(prefix: Iterable[str]) -> dict[str, Any]:
    before: list[str] = []
    payload = bytearray()
    active = False
    for raw in prefix:
        token = str(raw)
        if not active:
            if token == TARGET_BYTE_BEGIN:
                active = True
                payload.clear()
            else:
                before.append(token)
            continue
        if is_byte_token(token):
            if len(payload) < MAX_TOKEN_BYTES:
                payload.extend(byte_token_bytes(token))
            continue
        if token == TARGET_BYTE_END:
            try:
                before.append(bytes(payload).decode("utf-8"))
            except UnicodeDecodeError:
                before.append("<byte_fallback_fault>")
            active = False
            payload.clear()
            continue
        before.append("<byte_fallback_fault>")
        active = False
        payload.clear()
    return {
        "active": active,
        "prefix_before_span": before,
        "payload": bytes(payload),
        "payload_length": len(payload),
    }


def stream_boundaries(stream: str) -> tuple[str, str]:
    if str(stream) == "source":
        return SOURCE_BYTE_BEGIN, SOURCE_BYTE_END
    if str(stream) == "target":
        return TARGET_BYTE_BEGIN, TARGET_BYTE_END
    raise ValueError(f"unknown_open_vocab_stream:{stream}")


def is_byte_token(token: str) -> bool:
    value = str(token)
    if len(value) == len("<byte:00>") and value.startswith(BYTE_PREFIX) and value.endswith(">"):
        payload = value[len(BYTE_PREFIX) : -1]
        return len(payload) == 2 and all(char in "0123456789abcdef" for char in payload)
    if value.startswith(BYTE_PIECE_PREFIX) and value.endswith(">"):
        payload = value[len(BYTE_PIECE_PREFIX) : -1]
        return (
            bool(payload)
            and len(payload) % 2 == 0
            and all(char in "0123456789abcdef" for char in payload)
        )
    return False


def byte_token_value(token: str) -> int:
    return int(str(token)[len(BYTE_PREFIX) : -1], 16)


def byte_token_bytes(token: str) -> bytes:
    value = str(token)
    if value.startswith(BYTE_PREFIX):
        return bytes([byte_token_value(value)])
    if value.startswith(BYTE_PIECE_PREFIX) and value.endswith(">"):
        return bytes.fromhex(value[len(BYTE_PIECE_PREFIX) : -1])
    raise ValueError(f"not_byte_piece:{value}")


def byte_piece_token(payload: bytes) -> str:
    return f"{BYTE_PIECE_PREFIX}{bytes(payload).hex()}>"


def encode_byte_pieces(payload: bytes, vocab: dict[str, int]) -> list[str]:
    cache_key = (id(vocab), len(vocab), bytes(payload))
    cached = _ENCODED_PIECE_CACHE.get(cache_key)
    if cached is not None and cached[0] is vocab:
        _ENCODED_PIECE_CACHE.move_to_end(cache_key)
        return list(cached[1])
    by_first_byte = byte_piece_inventory(vocab)
    length = len(payload)
    best: list[list[str] | None] = [None] * (length + 1)
    best[length] = []
    for index in range(length - 1, -1, -1):
        candidates: list[list[str]] = []
        for raw, token in by_first_byte.get(payload[index], []):
            if payload.startswith(raw, index) and best[index + len(raw)] is not None:
                candidates.append([token, *list(best[index + len(raw)] or [])])
        if candidates:
            best[index] = min(candidates, key=lambda row: (len(row), row))
    encoded = (
        [BYTE_TOKENS[value] for value in payload]
        if best[0] is None
        else list(best[0])
    )
    _ENCODED_PIECE_CACHE[cache_key] = (vocab, tuple(encoded))
    _ENCODED_PIECE_CACHE.move_to_end(cache_key)
    while len(_ENCODED_PIECE_CACHE) > _ENCODED_PIECE_CACHE_LIMIT:
        _ENCODED_PIECE_CACHE.popitem(last=False)
    return encoded


def byte_piece_inventory(vocab: dict[str, int]) -> dict[int, list[tuple[bytes, str]]]:
    cache_key = id(vocab)
    cached = _PIECE_INVENTORY_CACHE.get(cache_key)
    if cached is not None and cached[0] is vocab:
        _PIECE_INVENTORY_CACHE.move_to_end(cache_key)
        return cached[1]
    pieces: dict[bytes, str] = {
        bytes([value]): BYTE_TOKENS[value]
        for value in range(256)
        if BYTE_TOKENS[value] in vocab
    }
    for token in vocab:
        if str(token).startswith(BYTE_PIECE_PREFIX) and str(token).endswith(">"):
            try:
                raw = byte_token_bytes(str(token))
            except ValueError:
                continue
            if raw:
                pieces[raw] = str(token)
    by_first_byte: dict[int, list[tuple[bytes, str]]] = {}
    for raw, token in pieces.items():
        by_first_byte.setdefault(raw[0], []).append((raw, token))
    for rows in by_first_byte.values():
        rows.sort(key=lambda item: (-len(item[0]), item[0]))
    _PIECE_INVENTORY_CACHE[cache_key] = (vocab, by_first_byte)
    _PIECE_INVENTORY_CACHE.move_to_end(cache_key)
    while len(_PIECE_INVENTORY_CACHE) > _PIECE_INVENTORY_CACHE_LIMIT:
        _PIECE_INVENTORY_CACHE.popitem(last=False)
    return by_first_byte


def fault(kind: str, index: int, token: str) -> dict[str, Any]:
    return {
        "fault_type": kind,
        "token_index": int(index),
        "token": str(token)[:80],
        "failure_behavior": "reject_without_fallback",
    }
