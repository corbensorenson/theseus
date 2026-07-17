from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import build_vocab, encode_many
from neural_seed_open_vocab import (
    BYTE_TOKENS,
    MAX_TOKEN_BYTES,
    TARGET_BYTE_BEGIN,
    TARGET_BYTE_END,
    bound_logical_tokens,
    decode_target_tokens,
    encode_tokens,
    reserve_byte_fallback_tokens,
)
from neural_seed_token_decoder_rendering import decode_body_tokens
from neural_seed_token_decoder_support import body_tokens, token_allowed_by_policy


def target_only_byte_vocab() -> dict[str, int]:
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    return reserve_byte_fallback_tokens(vocab, max_vocab=300, stream="target")


def test_source_byte_fallback_removes_unknown_substitution() -> None:
    vocab = build_vocab(["common token"], max_vocab=300, byte_fallback=True)
    encoded = encode_many(["rare_identifier_987654"], vocab, max_len=64)[0]

    assert vocab["<unk>"] not in [token for token in encoded if token != vocab["<pad>"]]


def test_target_byte_fallback_roundtrips_python_tokens() -> None:
    logical = body_tokens("result_with_rare_name = input_with_rare_name + 17\nreturn result_with_rare_name")
    vocab = target_only_byte_vocab()
    ids, encode_receipt = encode_tokens(logical, vocab, stream="target")
    inverse = {value: token for token, value in vocab.items()}
    encoded_tokens = [inverse[token_id] for token_id in ids]
    decoded, decode_receipt = decode_target_tokens(encoded_tokens)

    assert encode_receipt["unknown_token_count"] == 0
    assert encode_receipt["fallback_token_count"] == len(logical)
    assert decode_receipt["state"] == "READY"
    assert decoded == logical
    assert decode_body_tokens(encoded_tokens) == (
        "result_with_rare_name = input_with_rare_name + 17\nreturn result_with_rare_name"
    )


def test_target_byte_span_is_grammar_checked_after_reconstruction() -> None:
    vocab = target_only_byte_vocab()
    ids, _receipt = encode_tokens(["NAME:rare_identifier"], vocab, stream="target")
    inverse = {value: token for token, value in vocab.items()}
    span = [inverse[token_id] for token_id in ids]
    prefix = ["NAME:return"]

    for token in span[:-1]:
        assert token_allowed_by_policy(
            prefix,
            token,
            policy="strict_body_token_legality_v1",
            allowed_names={"rare_identifier"},
        )
        prefix.append(token)
    assert span[-1] == TARGET_BYTE_END
    assert token_allowed_by_policy(
        prefix,
        span[-1],
        policy="strict_body_token_legality_v1",
        allowed_names={"rare_identifier"},
    )
    assert TARGET_BYTE_BEGIN in span


def test_piece_codec_reuses_one_vocab_inventory_for_many_tokens() -> None:
    texts = [f"shared_identifier_prefix_{index}" for index in range(200)]
    vocab = build_vocab(texts, max_vocab=512, byte_fallback=True)
    logical = [f"shared_identifier_prefix_unseen_{index}" for index in range(1000)]

    first, first_receipt = encode_tokens(logical, vocab, stream="source")
    second, second_receipt = encode_tokens(logical, vocab, stream="source")

    assert first == second
    assert first_receipt["unknown_token_count"] == 0
    assert second_receipt["unknown_token_count"] == 0


def test_truncated_and_invalid_utf8_spans_fail_closed() -> None:
    truncated, truncated_receipt = decode_target_tokens([TARGET_BYTE_BEGIN, BYTE_TOKENS[65]])
    invalid, invalid_receipt = decode_target_tokens(
        [TARGET_BYTE_BEGIN, BYTE_TOKENS[0xFF], TARGET_BYTE_END]
    )

    assert truncated == []
    assert truncated_receipt["state"] == "FAULT"
    assert truncated_receipt["faults"][0]["fault_type"] == "BYTE_FALLBACK_TRUNCATED"
    assert invalid == []
    assert invalid_receipt["state"] == "FAULT"
    assert invalid_receipt["faults"][0]["fault_type"] == "BYTE_FALLBACK_UTF8"


def test_over_bound_token_reports_unknown_instead_of_partial_encoding() -> None:
    vocab = target_only_byte_vocab()
    ids, receipt = encode_tokens(["x" * (MAX_TOKEN_BYTES + 1)], vocab, stream="target")

    assert ids == [vocab["<unk>"]]
    assert receipt["unknown_token_count"] == 1
    assert receipt["fallback_token_count"] == 0


def test_explicit_bounding_preserves_long_utf8_atoms_without_weakening_codec_limit() -> None:
    logical = ["prefix", "\u03bb" * (MAX_TOKEN_BYTES + 3), "suffix"]
    bounded = bound_logical_tokens(logical)
    vocab = target_only_byte_vocab()
    ids, receipt = encode_tokens(bounded, vocab, stream="target")
    inverse = {value: token for token, value in vocab.items()}
    decoded, decode_receipt = decode_target_tokens([inverse[token_id] for token_id in ids])

    assert all(len(token.encode("utf-8")) <= MAX_TOKEN_BYTES for token in bounded)
    assert "".join(bounded) == "".join(logical)
    assert receipt["unknown_token_count"] == 0
    assert decode_receipt["state"] == "READY"
    assert "".join(decoded) == "".join(logical)


def test_reconstructed_disallowed_name_is_rejected_by_grammar() -> None:
    vocab = target_only_byte_vocab()
    ids, _receipt = encode_tokens(["NAME:hidden_identifier"], vocab, stream="target")
    inverse = {value: token for token, value in vocab.items()}
    span = [inverse[token_id] for token_id in ids]
    prefix = ["NAME:return"]

    for token in span[:-1]:
        assert token_allowed_by_policy(
            prefix,
            token,
            policy="strict_body_token_legality_v1",
            allowed_names={"visible_identifier"},
        )
        prefix.append(token)
    assert not token_allowed_by_policy(
        prefix,
        span[-1],
        policy="strict_body_token_legality_v1",
        allowed_names={"visible_identifier"},
    )
