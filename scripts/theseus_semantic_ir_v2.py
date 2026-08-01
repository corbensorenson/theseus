#!/usr/bin/env python3
"""Labeled Semantic-IR transport that canonicalizes into the v1 lowerer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import theseus_p4_cognitive_compilation as p4


HEADER = "THESEUS_SEMANTIC_IR_V2"
UNIT_RE = re.compile(
    r"UNIT ([A-Z][A-Z0-9_-]*)\n"
    r"OBLIGATIONS ([A-Z0-9_,]+)\n"
    r"OP (REPLACE|INSERT_BEFORE|INSERT_AFTER)\n"
    r"PATH ([^\n ]+)\n"
    r"NODE ([A-Z0-9-]+)\n"
    r"NODE_SHA ([a-f0-9]{64})\n"
    r"<<<\n(.*?)\n>>>\nEND_UNIT",
    flags=re.DOTALL,
)


def grammar() -> str:
    return (
        f"{HEADER}\n"
        "SOURCE <semantic-source-digest>\n"
        "ALL_OBLIGATIONS <comma-separated exact obligation ids>\n"
        "UNIT <unit-id>\n"
        "OBLIGATIONS <comma-separated obligation ids covered by this unit>\n"
        "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>\n"
        "PATH <repository-relative path>\n"
        "NODE <exact node id from the symbol table>\n"
        "NODE_SHA <exact node sha256 from the symbol table>\n"
        "<<<\n<replacement source>\n>>>\n"
        "END_UNIT\n"
        "LOSS <NONE|comma-separated unresolved obligation ids>\n"
        "END"
    )


def complete(text: str) -> bool:
    raw = unwrap(text)
    return (
        raw.startswith(f"{HEADER}\n")
        and raw.endswith("\nEND")
        and bool(re.search(r"^SOURCE [a-f0-9]{64}$", raw, flags=re.MULTILINE))
        and bool(re.search(r"^ALL_OBLIGATIONS [A-Z0-9_,]+$", raw, flags=re.MULTILINE))
        and bool(UNIT_RE.search(raw))
        and bool(re.search(r"^LOSS (?:NONE|[A-Z0-9_,]+)\nEND$", raw, flags=re.MULTILINE))
    )


def parse(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    raw = unwrap(text)
    if not raw.startswith(f"{HEADER}\n") or not raw.endswith("\nEND"):
        return empty(["semantic_ir_v2_envelope_invalid"])
    source = re.search(r"^SOURCE ([a-f0-9]{64})$", raw, flags=re.MULTILINE)
    obligations = re.search(
        r"^ALL_OBLIGATIONS ([A-Z0-9_,]+)$", raw, flags=re.MULTILINE
    )
    loss = re.search(r"^LOSS (NONE|[A-Z0-9_,]+)$", raw, flags=re.MULTILINE)
    faults: list[str] = []
    if source is None:
        faults.append("semantic_ir_v2_source_missing")
    if obligations is None:
        faults.append("semantic_ir_v2_all_obligations_missing")
    if loss is None:
        faults.append("semantic_ir_v2_loss_missing")
    units = list(UNIT_RE.finditer(raw))
    if not units:
        faults.append("semantic_ir_v2_units_missing")
    scrubbed = raw
    for match in reversed(units):
        start, end = match.span()
        scrubbed = scrubbed[:start] + "" + scrubbed[end:]
    scrubbed = re.sub(
        r"^(THESEUS_SEMANTIC_IR_V2|SOURCE .+|ALL_OBLIGATIONS .+|LOSS .+|END)\s*$",
        "",
        scrubbed,
        flags=re.MULTILINE,
    ).strip()
    if scrubbed:
        faults.append("semantic_ir_v2_unparsed_text")
    if faults:
        return empty(sorted(set(faults)))
    canonical = [
        p4.IR_HEADER,
        f"SOURCE {source.group(1)}",
        f"OBLIGATIONS {obligations.group(1)}",
    ]
    unit_receipts: list[dict[str, str | list[str]]] = []
    for match in units:
        unit_id, refs, operation, path, node_id, node_sha, replacement = match.groups()
        canonical.extend([
            f"UNIT {unit_id} {refs} {operation} {path} {node_id} {node_sha}",
            "<<<",
            replacement,
            ">>>",
        ])
        unit_receipts.append({
            "unit_id": unit_id,
            "obligation_ids": refs.split(","),
            "operation": operation,
            "path": path,
            "node_id": node_id,
            "node_sha256": node_sha,
        })
    canonical.extend([f"LOSS {loss.group(1)}", "END"])
    lowered = p4.parse_semantic_ir("\n".join(canonical), task, root)
    receipt = dict(lowered.get("semantic_receipt") or {})
    receipt.update({
        "transport": "theseus_semantic_ir_v2_labeled",
        "canonical_lowerer": "theseus_semantic_ir_v1",
        "transport_units": unit_receipts,
    })
    lowered["semantic_receipt"] = receipt
    lowered["canonical_v1"] = "\n".join(canonical)
    return lowered


def unwrap(text: str) -> str:
    raw = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    return (fenced.group(1) if fenced else raw).strip()


def empty(faults: list[str]) -> dict[str, Any]:
    return {
        "actions": [],
        "faults": faults,
        "units": [],
        "semantic_receipt": {},
        "canonical_v1": "",
    }

