from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from blind_runtime_guard import (  # noqa: E402
    BlindRuntimeFault,
    blind_view,
    guard_ranking_inputs,
    issue_blind_capability,
    validate_cached_artifact,
)


def capability():
    return issue_blind_capability(
        allowed_fields={"prompt", "signature", "runtime_context"},
        purpose="candidate_generation",
    )


def private_row() -> dict:
    return {
        "prompt": "Add two integers.",
        "signature": "def add(a: int, b: int) -> int",
        "runtime_context": {"language": "python"},
        "solution_body": "return a + b",
        "tests": ["assert add(1, 2) == 3"],
        "category": "arithmetic",
    }


def test_forbidden_fields_fail_through_index_get_attribute_helper_and_alias() -> None:
    view = blind_view(private_row(), capability())
    alias = view

    assert view["prompt"] == "Add two integers."
    for read in (
        lambda: view["solution_body"],
        lambda: view.get("tests"),
        lambda: view.category,
        lambda: alias["category"],
    ):
        with pytest.raises(BlindRuntimeFault, match="forbidden_field_access"):
            read()


def test_nested_derived_labels_and_cache_payloads_fail_closed() -> None:
    nested = {
        "prompt": "Visible",
        "signature": "f()",
        "runtime_context": {
            "derived": {"required_constructs": ["loop"]},
        },
    }
    with pytest.raises(BlindRuntimeFault, match="nested_forbidden_fields"):
        blind_view(nested, capability())["runtime_context"]

    cache = {
        "visible_fields": ["prompt", "signature", "runtime_context"],
        "payload": {
            "prompt": "Visible",
            "signature": "f()",
            "runtime_context": {},
            "answer": "hidden",
        },
    }
    with pytest.raises(
        BlindRuntimeFault,
        match="cached_artifact_forbidden_fields",
    ):
        validate_cached_artifact(cache, capability())


def test_ranking_receives_only_runtime_guarded_views() -> None:
    ranked = guard_ranking_inputs([private_row(), private_row()], capability())

    assert len(ranked) == 2
    assert ranked[0]["signature"] == "def add(a: int, b: int) -> int"
    with pytest.raises(BlindRuntimeFault, match="forbidden_field_access"):
        _ = ranked[0]["category"]


def test_capability_cannot_include_forbidden_fields() -> None:
    with pytest.raises(
        BlindRuntimeFault,
        match="capability_contains_forbidden_fields",
    ):
        issue_blind_capability(
            allowed_fields={"prompt", "solution"},
            purpose="invalid",
        )
