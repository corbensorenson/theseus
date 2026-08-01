#!/usr/bin/env python3
"""Candidate-hidden visible-feedback check for the P4 recovery task."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


MARKER = "P4V2R2R1_VISIBLE_OPENROUTER_NULL_CHOICES"
SOURCE = Path("pydantic_ai_slim/pydantic_ai/models/openrouter.py")


def main() -> int:
    try:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {
            getattr(node, "name", "")
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        assert "_OpenRouterNoCompletionResponse" in names
        assert "_raise_for_no_completion" in names
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_raise_for_no_completion"
        ]
        assert len(calls) == 2
        print("P4V2R2R1_VISIBLE_PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{MARKER}:{type(exc).__name__}:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
