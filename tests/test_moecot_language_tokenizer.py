from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_language_tokenizer import (  # noqa: E402
    encode_document,
    exact_text_tokens,
    logical_tokens,
    profile_for_category,
)
from neural_seed_open_vocab import populate_open_vocab  # noqa: E402
from collections import Counter  # noqa: E402


def test_exact_profiles_preserve_language_significant_text() -> None:
    fixtures = {
        "english_broad": "First line.\n\nSecond line with  two spaces.\n",
        "javascript_typescript": "const greet = (name: string) => `Hi, ${name}!`;\n",
        "html_css": "<main aria-label=\"x\">Hi</main>\n.x > p { color: #0af; }\n",
        "rust": "fn borrow<'a>(x: &'a str) -> &'a str { x }\n",
    }
    for category, source in fixtures.items():
        tokens = logical_tokens(source, category=category)
        assert "".join(tokens) == source
        assert profile_for_category(category) == "exact_text_v1"


def test_profiled_encoding_roundtrips_through_bounded_open_vocab() -> None:
    source = "const answer = (value: number) => value + 1;\n"
    counts = Counter(exact_text_tokens("const value = 1;\n"))
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    populate_open_vocab(vocab, counts, max_vocab=1024, stream="target", piece_budget=128)

    _tokens, _encoded, receipt = encode_document(
        source, vocab, category="javascript_typescript"
    )

    assert receipt["unknown_token_count"] == 0
    assert receipt["roundtrip"]["state"] == "GREEN"
    assert receipt["roundtrip"]["exact_text_equal"] is True
    assert receipt["fallback_return_count"] == 0


def test_python_pretraining_profile_preserves_exact_source_text() -> None:
    source = "def add(value):\n    return value + 1\n"
    tokens = logical_tokens(source, category="python")
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    populate_open_vocab(vocab, Counter(tokens), max_vocab=1024, stream="target", piece_budget=128)

    _tokens, _encoded, receipt = encode_document(source, vocab, category="python")

    assert receipt["profile"] == "exact_text_v1"
    assert receipt["roundtrip"]["state"] == "GREEN"
    assert receipt["roundtrip"]["mode"] == "exact_utf8"
    assert receipt["roundtrip"]["exact_text_equal"] is True


def test_unknown_category_fails_closed() -> None:
    try:
        profile_for_category("klingon")
    except ValueError as exc:
        assert "unknown MoECOT tokenizer category" in str(exc)
    else:
        raise AssertionError("unknown tokenizer category was accepted")
