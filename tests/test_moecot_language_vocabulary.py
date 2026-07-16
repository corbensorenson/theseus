from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_language_tokenizer import exact_text_tokens  # noqa: E402
from moecot_language_vocabulary import (  # noqa: E402
    KERC_SOURCE_CONTROL_TOKENS,
    SPECIAL,
    contract_sha256,
    coverage,
    reserve_required_tokens,
    validate_payload,
)
from neural_seed_open_vocab import populate_open_vocab  # noqa: E402


def test_exact_vocabulary_makes_common_cross_language_tokens_atomic() -> None:
    text = (
        "function add(value) { return value + 1; }\n"
        "fn add(value: i32) -> i32 { value + 1 }\n"
        "<button class=\"primary\">Save</button>\n"
    )
    counts = Counter(exact_text_tokens(text * 20))
    vocab = dict(SPECIAL)
    populate_open_vocab(vocab, counts, max_vocab=1024, stream="target", piece_budget=128)
    for token in ("function", "fn", "{", "}", "\n", " "):
        assert token in vocab
    assert coverage(counts, vocab)["atomic_token_ratio"] > 0.95


def test_vocabulary_contract_requires_full_byte_fallback_inventory() -> None:
    counts = Counter(exact_text_tokens("const value = 1;\n" * 20))
    source = dict(SPECIAL)
    target = dict(SPECIAL)
    populate_open_vocab(source, counts, max_vocab=512, stream="source", piece_budget=64)
    populate_open_vocab(target, counts, max_vocab=512, stream="target", piece_budget=64)
    cfg = {
        "policy": "project_theseus_moecot_exact_language_vocabulary_v1",
        "source_max_vocab": len(source),
        "target_max_vocab": len(target),
        "required_source_control_tokens": [],
        "required_target_control_tokens": [],
    }
    payload = {
        "policy": cfg["policy"],
        "contract_sha256": contract_sha256(cfg),
        "source_vocab": source,
        "target_vocab": target,
    }
    assert validate_payload(payload, cfg) == []
    del payload["target_vocab"]["<byte:ff>"]
    assert "target_vocabulary_size_mismatch" in validate_payload(payload, cfg)
    assert "target_byte_inventory_incomplete" in validate_payload(payload, cfg)


def test_kerc_control_tokens_are_reserved_but_cannot_be_injected_as_raw_text() -> None:
    counts = Counter(exact_text_tokens("Explain the result clearly.\n" * 20))
    source = dict(SPECIAL)
    target = dict(SPECIAL)
    reserve_required_tokens(source, KERC_SOURCE_CONTROL_TOKENS)
    populate_open_vocab(source, counts, max_vocab=512, stream="source", piece_budget=64)
    populate_open_vocab(target, counts, max_vocab=512, stream="target", piece_budget=64)
    cfg = {
        "policy": "project_theseus_moecot_exact_language_vocabulary_v1",
        "source_max_vocab": len(source),
        "target_max_vocab": len(target),
        "required_source_control_tokens": list(KERC_SOURCE_CONTROL_TOKENS),
        "required_target_control_tokens": [],
    }
    payload = {
        "policy": cfg["policy"],
        "contract_sha256": contract_sha256(cfg),
        "source_vocab": source,
        "target_vocab": target,
    }
    assert validate_payload(payload, cfg) == []
    assert all(token in source for token in KERC_SOURCE_CONTROL_TOKENS)
    assert exact_text_tokens(KERC_SOURCE_CONTROL_TOKENS[0]) != [
        KERC_SOURCE_CONTROL_TOKENS[0]
    ]
