#!/usr/bin/env python3
"""Independent hidden checks for the P4 recovery replacement task."""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError


SOURCE = Path("pydantic_ai_slim/pydantic_ai/models/openrouter.py")
FAIL = "P4V2R2R1_FAIL_PYDANTIC_AI_OPENROUTER"


class ModelAPIError(Exception):
    def __init__(self, *, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name


def top_level(tree: ast.Module, kind: type[ast.AST], name: str) -> ast.AST:
    matches = [
        node
        for node in tree.body
        if isinstance(node, kind) and getattr(node, "name", None) == name
    ]
    assert len(matches) == 1, f"missing or duplicate {name}"
    return matches[0]


def owner_method(tree: ast.Module, owner: str, name: str) -> ast.AST:
    cls = top_level(tree, ast.ClassDef, owner)
    assert isinstance(cls, ast.ClassDef)
    matches = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"missing or duplicate {owner}.{name}"
    return matches[0]


def execute_shape_and_helper(text: str, tree: ast.Module) -> tuple[type, Any]:
    shape = top_level(tree, ast.ClassDef, "_OpenRouterNoCompletionResponse")
    helper = top_level(tree, ast.FunctionDef, "_raise_for_no_completion")
    module = ast.Module(body=[copy.deepcopy(shape), copy.deepcopy(helper)], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Literal": Literal,
        "BaseModel": BaseModel,
        "ValidationError": ValidationError,
        "ModelAPIError": ModelAPIError,
    }
    exec(compile(module, str(SOURCE), "exec", dont_inherit=True), namespace)
    return namespace["_OpenRouterNoCompletionResponse"], namespace["_raise_for_no_completion"]


def call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.append(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.append(item.func.attr)
    return names


def main() -> int:
    try:
        text = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "exceptions"
            for alias in node.names
        }
        assert "ModelAPIError" in imported

        shape, helper = execute_shape_and_helper(text, tree)
        fields = shape.model_fields
        assert fields["choices"].is_required()
        assert not fields["error"].is_required()
        assert not fields["provider"].is_required()

        for body, expected_model in (
            ({"choices": None}, "configured/model"),
            ({"choices": None, "provider": "Google", "model": "body/model"}, "body/model"),
        ):
            try:
                helper(body, "configured/model", ValidationError.from_exception_data("source", []))
            except ModelAPIError as exc:
                assert exc.model_name == expected_model
                assert "null `choices`" in str(exc)
            else:
                raise AssertionError("valid no-completion shape was not classified")
        for body in (
            {},
            {"choices": None, "provider": {"name": "Google"}},
            {"choices": None, "error": {"message": "bad"}},
            {"choices": []},
        ):
            helper(body, "configured/model", ValidationError.from_exception_data("source", []))

        completion = owner_method(tree, "OpenRouterModel", "_validate_completion")
        completion_calls = call_names(completion)
        assert completion_calls.count("_raise_for_no_completion") == 1
        helper_call = next(
            node
            for node in ast.walk(completion)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_raise_for_no_completion"
        )
        nested_call = next(
            node
            for node in ast.walk(completion)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_OpenRouterNestedProviderResponse"
        )
        assert helper_call.lineno < nested_call.lineno

        streamed = owner_method(tree, "OpenRouterStreamedResponse", "_validate_response")
        assert isinstance(streamed, ast.AsyncFunctionDef)
        streamed_calls = call_names(streamed)
        assert streamed_calls.count("_raise_for_no_completion") == 1
        assert any(isinstance(node, ast.Raise) and node.exc is None for node in ast.walk(streamed))
        assert any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(streamed))
        print("P4V2R2R1_HIDDEN_PASS")
        return 0
    except Exception as exc:  # noqa: BLE001 - hidden verifier must fail closed.
        print(f"{FAIL}:{type(exc).__name__}:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
