from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ast
import semantic_ir

from neural_seed_full_state_pretraining import (
    python_function_body_pretraining_examples,
    python_function_body_pretraining_source_text,
)
from strict_generator_pretraining_spine import select_staged_row_examples
from strict_generator_mlx_source_text import strict_generator_decode_source_text
from strict_generator_mlx_decode_eval import token_is_target_control_metadata


def labels(text: str) -> list[str]:
    return [line.split(" ", 1)[0] for line in text.splitlines()]


def test_corpus_and_decode_source_contracts_keep_signature_before_subwords() -> None:
    prompt = "Normalize tokens, filter stop words, preserve stable encounter order, and return unique values."
    corpus = python_function_body_pretraining_source_text(
        "licensed/example.py",
        "stable_normalized_tokens",
        ["data", "stopwords"],
        prompt,
        source_text_style="prompt_signature_metadata_v2",
    )
    runtime = strict_generator_decode_source_text(
        {
            "prompt": prompt,
            "entry_point": "stable_normalized_tokens",
            "decoder_contract": {"argument_roles": {"data": "tokens", "stopwords": "filter"}},
        },
        [],
        source_text_style="prompt_signature_metadata_v2",
        source_vocab={"source_style": 1},
    )
    for text in (corpus, runtime):
        order = labels(text)
        assert order.index("signature") < order.index("visible_subwords")
        assert order.index("arguments") < order.index("visible_subwords")
        assert "prompt_operation_hints" in order
        assert "signature" in " ".join(text.split()[:96])
    assert "path licensed/example.py" not in corpus


def test_legacy_source_contract_remains_explicitly_separate() -> None:
    text = python_function_body_pretraining_source_text(
        "licensed/example.py",
        "f",
        ["data"],
        "Return the number of values.",
        source_text_style="legacy_metadata_v1",
    )
    assert text.startswith("path licensed/example.py")


def test_corpus_body_extraction_is_ast_canonical_with_multiline_string() -> None:
    source = '''
def report(einfo):
    """Build an error report."""
    etype, value = einfo
    if isinstance(etype, type):
        etype = etype.__name__
    head = """first
column-zero payload
last"""
    return head + str(etype) + str(value)
'''
    rows = python_function_body_pretraining_examples(
        "licensed/report.py",
        source,
        max_function_body_chars=5000,
    )
    assert len(rows) == 1
    wrapper = "def report(einfo):\n" + "\n".join(f"    {line}" if line else "" for line in rows[0]["body"].splitlines())
    parsed = ast.parse(wrapper)
    assert isinstance(parsed.body[0], ast.FunctionDef)
    assert "column-zero payload" in rows[0]["body"]
    assert rows[0]["prompt_source"] == "docstring"
    assert rows[0]["prompt_character_count"] > 0


def test_semantic_plan_tokens_cannot_reenter_python_body() -> None:
    target_mode = semantic_ir.PLAN_BODY_TARGET_MODE

    assert token_is_target_control_metadata("IRP:DEPTH:1", target_mode=target_mode)
    assert token_is_target_control_metadata("IRP:CALL:range", target_mode=target_mode)
    assert token_is_target_control_metadata("SLOT:BODY_START", target_mode=target_mode)
    assert not token_is_target_control_metadata("NAME:data", target_mode=target_mode)
    assert not token_is_target_control_metadata("OP:+", target_mode=target_mode)


def test_quality_selection_prioritizes_descriptive_prompt_body_pairs() -> None:
    examples = [
        {
            "path": f"licensed/{index}.py",
            "function": f"function_{index}",
            "prompt_source": "docstring" if index < 4 else "identifier_fallback",
            "prompt_character_count": 80 if index < 4 else 0,
            "quality": {"nontrivial_return_count": 1, "parameter_load_count": 1},
        }
        for index in range(10)
    ]
    selected, summary = select_staged_row_examples(
        examples,
        max_examples=4,
        seed=23,
        selection_cfg={
            "policy": "quality_balanced_visible_prompt_v1",
            "min_quality_score": 1.0,
            "high_quality_fraction": 1.0,
            "descriptive_prompt_fraction": 0.75,
            "min_prompt_characters": 24,
        },
    )

    assert sum(row["prompt_source"] == "docstring" for row in selected) >= 3
    assert summary["prompt_alignment_after"]["descriptive_docstring_rate"] >= 0.75
