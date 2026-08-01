#!/usr/bin/env python3
"""Semantic-IR v2 transport repair with explicit header-inference custody."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import theseus_semantic_ir_v2 as v2


HEADER = v2.HEADER


def grammar() -> str:
    return v2.grammar()


def complete(text: str) -> bool:
    normalized, _ = normalize(text)
    return v2.complete(normalized)


def parse(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized, inferred = normalize(text)
    result = v2.parse(normalized, task, root)
    receipt = dict(result.get("semantic_receipt") or {})
    receipt.update({
        "declared_transport": "theseus_semantic_ir_v2_labeled",
        "version_header_inferred_from_bound_parser": inferred,
        "header_inference_answer_bearing": False,
    })
    result["semantic_receipt"] = receipt
    return result


def normalize(text: str) -> tuple[str, bool]:
    raw = v2.unwrap(text)
    if raw.startswith(f"{HEADER}\n"):
        return raw, False
    if (
        raw.startswith("SOURCE ")
        and "\nALL_OBLIGATIONS " in raw
        and "\nUNIT " in raw
        and raw.endswith("\nEND")
    ):
        return f"{HEADER}\n{raw}", True
    return raw, False

