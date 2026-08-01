#!/usr/bin/env python3
"""Prospective Semantic-IR transport with delimiter-only list normalization."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

import theseus_semantic_ir_v2 as v2


HEADER = v2.HEADER
OBLIGATION_ID_RE = re.compile(r"[A-Z][A-Z0-9_]*")
LIST_FIELD_RE = re.compile(r"^(ALL_OBLIGATIONS|OBLIGATIONS|LOSS) (.+)$")


def grammar() -> str:
    return (
        v2.grammar()
        + "\nObligation-list delimiters may be canonical commas, whitespace, "
        "unquoted brackets, or single/double-quoted brackets. Identifier values "
        "and order are never inferred or changed."
    )


def complete(text: str) -> bool:
    normalized, _ = normalize_with_receipt(text)
    return v2.complete(normalized)


def parse(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized, normalization = normalize_with_receipt(text)
    result = v2.parse(normalized, task, root)
    receipt = dict(result.get("semantic_receipt") or {})
    receipt.update({
        "declared_transport": "theseus_semantic_ir_v2r2_labeled",
        "version_header_inferred_from_bound_parser": normalization[
            "version_header_inferred_from_bound_parser"
        ],
        "header_inference_answer_bearing": False,
        "obligation_list_normalization": normalization,
    })
    result["semantic_receipt"] = receipt
    return result


def normalize(text: str) -> tuple[str, bool]:
    normalized, receipt = normalize_with_receipt(text)
    return normalized, bool(receipt["version_header_inferred_from_bound_parser"])


def normalize_with_receipt(text: str) -> tuple[str, dict[str, Any]]:
    raw = v2.unwrap(text)
    header_inferred = False
    if not raw.startswith(f"{HEADER}\n") and (
        raw.startswith("SOURCE ")
        and "\nALL_OBLIGATIONS " in raw
        and "\nUNIT " in raw
        and raw.endswith("\nEND")
    ):
        raw = f"{HEADER}\n{raw}"
        header_inferred = True

    normalized_lines: list[str] = []
    field_receipts: list[dict[str, Any]] = []
    inside_replacement = False
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if line == "<<<" and not inside_replacement:
            inside_replacement = True
            normalized_lines.append(line)
            continue
        if line == ">>>" and inside_replacement:
            inside_replacement = False
            normalized_lines.append(line)
            continue
        match = None if inside_replacement else LIST_FIELD_RE.fullmatch(line)
        if match is None:
            normalized_lines.append(line)
            continue
        field, surface = match.groups()
        if field == "LOSS" and surface == "NONE":
            normalized_lines.append(line)
            continue
        parsed = canonical_obligation_list(surface)
        if parsed is None:
            normalized_lines.append(line)
            field_receipts.append({
                "line": line_number,
                "field": field,
                "state": "REJECTED_UNRECOGNIZED_SURFACE",
                "source_surface_sha256": hashlib.sha256(surface.encode()).hexdigest(),
                "identifier_values_invented": 0,
                "order_changed": False,
            })
            continue
        canonical, identifiers, surface_class = parsed
        normalized_lines.append(f"{field} {canonical}")
        field_receipts.append({
            "line": line_number,
            "field": field,
            "state": "NORMALIZED" if canonical != surface else "ALREADY_CANONICAL",
            "surface_class": surface_class,
            "source_surface_sha256": hashlib.sha256(surface.encode()).hexdigest(),
            "canonical_identifier_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "identifier_count": len(identifiers),
            "identifier_values_invented": 0,
            "order_changed": False,
        })
    receipt = {
        "policy": "theseus_semantic_ir_v2r2_obligation_list_normalization_v1",
        "version_header_inferred_from_bound_parser": header_inferred,
        "list_field_count": len(field_receipts),
        "normalized_field_count": sum(
            row["state"] == "NORMALIZED" for row in field_receipts
        ),
        "rejected_field_count": sum(
            row["state"] == "REJECTED_UNRECOGNIZED_SURFACE"
            for row in field_receipts
        ),
        "fields": field_receipts,
        "answer_bearing_transformation": False,
        "identifier_values_invented": 0,
        "identifier_order_preserved": True,
        "replacement_source_touched": False,
        "path_node_operation_source_digest_touched": False,
    }
    return "\n".join(normalized_lines), receipt


def canonical_obligation_list(
    surface: str,
) -> tuple[str, list[str], str] | None:
    """Canonicalize delimiters only; never infer, sort, or repair identifiers."""

    value = surface.strip()
    if not value:
        return None
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            return None
        inner = value[1:-1].strip()
        if not inner:
            return None
        if "'" in inner or '"' in inner:
            try:
                literal = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
            if not isinstance(literal, list) or not literal:
                return None
            if not all(isinstance(item, str) for item in literal):
                return None
            identifiers = list(literal)
            surface_class = "quoted_bracket_list"
        else:
            identifiers = split_unquoted_identifiers(inner)
            surface_class = "unquoted_bracket_list"
    else:
        if any(character in value for character in "[]'\""):
            return None
        identifiers = split_unquoted_identifiers(value)
        surface_class = "comma_list" if "," in value else "whitespace_list"
    if not identifiers or not all(OBLIGATION_ID_RE.fullmatch(item) for item in identifiers):
        return None
    return ",".join(identifiers), identifiers, surface_class


def split_unquoted_identifiers(value: str) -> list[str]:
    if "," in value:
        if not re.fullmatch(
            rf"{OBLIGATION_ID_RE.pattern}(?:\s*,\s*{OBLIGATION_ID_RE.pattern})*",
            value,
        ):
            return []
        return [item.strip() for item in value.split(",")]
    if not re.fullmatch(
        rf"{OBLIGATION_ID_RE.pattern}(?:\s+{OBLIGATION_ID_RE.pattern})*",
        value,
    ):
        return []
    return value.split()
