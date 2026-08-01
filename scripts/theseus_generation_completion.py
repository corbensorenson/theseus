#!/usr/bin/env python3
"""Completion predicates shared by successor local-model instruments."""

from __future__ import annotations

import re


FENCE = re.compile(r"^```(?:text)?\s*(.*?)\s*```$", flags=re.DOTALL)
EDIT_HEADER = "THESEUS_EDIT_V1"
PLAN_HEADER = "THESEUS_PLAN_V1"
SEMANTIC_HEADER = "THESEUS_SEMANTIC_IR_V1"


def candidate_envelope_complete(text: str) -> bool:
    """Recognize a terminal protocol envelope without judging its correctness."""
    raw = str(text or "").strip()
    fenced = FENCE.match(raw)
    if raw.startswith("```"):
        if not fenced:
            return False
        raw = fenced.group(1).strip()
    if not raw.endswith("\nEND"):
        return False
    if raw.startswith(SEMANTIC_HEADER):
        return bool(
            re.search(r"^SOURCE [^\n]+$", raw, flags=re.MULTILINE)
            and re.search(r"^OBLIGATIONS [^\n]+$", raw, flags=re.MULTILINE)
            and re.search(r"^UNIT [^\n]+\n<<<\n.*?\n>>>", raw, flags=re.MULTILINE | re.DOTALL)
            and re.search(r"^LOSS [^\n]+\nEND$", raw, flags=re.MULTILINE)
        )
    if raw.startswith(PLAN_HEADER):
        return bool(
            "\nPLAN\n" in raw
            and "\nTARGET\n" in raw
            and f"\n{EDIT_HEADER}\n" in raw
            and re.search(r"^REPLACE [^\n]+\n<<<\n.*?\n>>>\nEND$", raw, flags=re.MULTILINE | re.DOTALL)
        )
    if raw.startswith(EDIT_HEADER):
        return bool(
            re.search(r"^REPLACE [^\n]+\n<<<\n.*?\n>>>\nEND$", raw, flags=re.MULTILINE | re.DOTALL)
        )
    return False
