#!/usr/bin/env python3
"""Compact integrity-bound statement ABI for Semantic-IR adequacy.

The v4 statement inventory fixed mutation granularity but repeated a path,
full node digest, and descriptive label for every statement in addition to the
complete parent source.  On large files that representation consumed the host
watchdog during prompt ingestion.  This successor keeps every statement
address, strengthens the visible handle to 128 bits, groups addresses by path,
and resolves the full node digest independently before invoking the unchanged
production lowerer.  It never truncates the inventory or the parent source.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_runtime as base
import theseus_semantic_ir_production_adequacy_runtime_v4 as statement_runtime


HEADER = "THESEUS_SEMANTIC_IR_V4_COMPACT"
INTERNAL_HEADER = base.HEADER
CREATE_FILE_OPERATION = base.CREATE_FILE_OPERATION
CREATE_FILE_SUPPORTED = base.CREATE_FILE_SUPPORTED
STATEMENT_SCOPE_POLICY = "all_candidate_visible_python_statements_compact_128bit_v1"
NODE_HANDLE_HEX_BITS = 128

COMPACT_UNIT_RE = re.compile(
    r"UNIT ([A-Z][A-Z0-9_-]*)\n"
    r"OBLIGATIONS ([A-Z0-9_,]+)\n"
    r"OP (REPLACE|INSERT_BEFORE|INSERT_AFTER)\n"
    r"PATH ([^\n ]+)\n"
    r"NODE (N-[A-F0-9]{32})\n"
    r"<<<\n(.*?)\n>>>\nEND_UNIT",
    flags=re.DOTALL,
)
COMPACT_CREATE_FILE_UNIT_RE = re.compile(
    r"UNIT ([A-Z][A-Z0-9_-]*)\n"
    r"OBLIGATIONS ([A-Z0-9_,]+)\n"
    r"OP CREATE_FILE\n"
    r"PATH ([^\n ]+)\n"
    r"<<<\n(.*?)\n>>>\nEND_UNIT",
    flags=re.DOTALL,
)


def semantic_scope_symbol_table(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Return the complete v4 statement table with collision-checked handles."""

    table = statement_runtime.semantic_scope_symbol_table(root, task)
    nodes: list[dict[str, Any]] = []
    handles: set[str] = set()
    for source in p2a.dicts(table.get("nodes")):
        row = dict(source)
        handle_payload = "|".join(
            [
                str(row.get("path") or ""),
                str(row.get("node_type") or ""),
                str(int(row.get("start_line") or 0)),
                str(int(row.get("start_col") or 0)),
                str(int(row.get("end_line") or 0)),
                str(int(row.get("end_col") or 0)),
                str(row.get("sha256") or ""),
            ]
        )
        handle = "N-" + p2a.sha256_text(handle_payload)[:32].upper()
        if handle in handles:
            raise p2a.InstrumentFault("semantic_compact_node_handle_collision")
        handles.add(handle)
        row["id"] = handle
        nodes.append(row)
    return {
        **table,
        "nodes": nodes,
        "semantic_unit_policy": STATEMENT_SCOPE_POLICY,
        "compact_integrity_abi": {
            "policy": "semantic_node_handle_128bit_collision_checked_v1",
            "handle_bits": NODE_HANDLE_HEX_BITS,
            "full_node_sha256_candidate_visible": False,
            "full_node_sha256_resolved_independently": True,
            "descriptive_label_candidate_visible": False,
            "path_grouped_once": True,
            "inventory_complete": True,
            "inventory_truncated": False,
            "project_selected_node_cap_applied": False,
        },
    }


def render_common_context(
    root: Path, task: dict[str, Any], symbols: dict[str, Any] | None = None
) -> str:
    """Render a complete compact address inventory plus complete visible source."""

    table = symbols or semantic_scope_symbol_table(root, task)
    obligation_lines = [
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    ]
    dependency_lines = [
        f"{row['before']} -> {row['after']}"
        for row in p2a.dicts(task.get("obligation_dependencies"))
    ] or ["none"]
    symbol_lines: list[str] = []
    current_path = ""
    for row in p2a.dicts(table.get("nodes")):
        path = str(row.get("path") or "")
        if path != current_path:
            symbol_lines.append(f"PATH {path}")
            current_path = path
        symbol_lines.append(
            f"{row['id']} {row['node_type']} "
            f"{row['start_line']}:{row['start_col']}-{row['end_line']}:{row['end_col']}"
        )
    missing = p2a.strings(table.get("missing_allowed_effect_paths"))
    missing_lines = missing or ["none"]
    return (
        "[INFORMATION_MATCHED_OBLIGATIONS]\n" + "\n".join(obligation_lines)
        + "\n[OBLIGATION_DEPENDENCIES]\n" + "\n".join(dependency_lines)
        + f"\n[SEMANTIC_SOURCE_DIGEST]\n{table['source_digest']}"
        + "\n[COMPACT_SEMANTIC_NODE_ABI]\n"
        + "Each 128-bit NODE handle independently binds path, type, exact span, and full node digest.\n"
        + "\n".join(symbol_lines)
        + "\n[MISSING_ALLOWED_EFFECT_PATHS]\n" + "\n".join(missing_lines)
        + "\n[CANDIDATE_VISIBLE_REPOSITORY_CONTEXT]\n"
        + p2a.render_visible_context(root, task)
    )


def grammar() -> str:
    return base.grammar().replace(INTERNAL_HEADER, HEADER).replace(
        "NODE_SHA <exact candidate-visible semantic node sha256; omit for CREATE_FILE>\n",
        "",
    )


def complete(text: str) -> bool:
    raw, _ = base.canonicalize_with_receipt(text)
    return (
        raw.startswith(f"{HEADER}\n")
        and raw.endswith("\nEND")
        and all(
            re.search(
                rf"^{field} (?:NONE|[A-Z0-9_,]+)$", raw, flags=re.MULTILINE
            )
            for field in base.ROLE_FIELDS
        )
        and bool(re.search(r"^SOURCE [a-f0-9]{64}$", raw, flags=re.MULTILINE))
        and bool(COMPACT_UNIT_RE.search(raw) or COMPACT_CREATE_FILE_UNIT_RE.search(raw))
        and bool(
            re.search(
                r"^LOSS (?:NONE|[A-Z0-9_,]+)\nEND$",
                raw,
                flags=re.MULTILINE,
            )
        )
    )


def parse(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve compact handles internally, then use the unchanged base lowerer."""

    raw, canonicalization = base.canonicalize_with_receipt(text)
    if not raw.startswith(f"{HEADER}\n") or not raw.endswith("\nEND"):
        return base.empty(["semantic_ir_compact_envelope_invalid"], canonicalization)
    try:
        symbols = semantic_scope_symbol_table(root, task)
    except (OSError, p2a.InstrumentFault, ValueError):
        return base.empty(["semantic_symbol_table_invalid"], canonicalization)
    symbol_map = {
        str(row.get("id") or ""): row for row in p2a.dicts(symbols.get("nodes"))
    }
    resolution: list[dict[str, Any]] = []

    def expand(match: re.Match[str]) -> str:
        unit_id, refs, operation, path, node_id, replacement = match.groups()
        symbol = p2a.mapping(symbol_map.get(node_id))
        full_digest = str(symbol.get("sha256") or "0" * 64)
        resolution.append(
            {
                "unit_id": unit_id,
                "node_id": node_id,
                "resolved": bool(symbol),
                "candidate_supplied_full_digest": False,
                "full_digest_resolved_independently": bool(symbol),
            }
        )
        return (
            f"UNIT {unit_id}\nOBLIGATIONS {refs}\nOP {operation}\nPATH {path}\n"
            f"NODE {node_id}\nNODE_SHA {full_digest}\n<<<\n{replacement}\n>>>\nEND_UNIT"
        )

    expanded = COMPACT_UNIT_RE.sub(expand, raw)
    expanded = expanded.replace(f"{HEADER}\n", f"{INTERNAL_HEADER}\n", 1)
    result = base.parse(
        expanded,
        task,
        root,
        symbol_table_factory=semantic_scope_symbol_table,
    )
    receipt = dict(p2a.mapping(result.get("semantic_receipt")))
    receipt.update(
        {
            "schema": HEADER,
            "compact_integrity_abi": symbols.get("compact_integrity_abi"),
            "node_identity_resolution": resolution,
            "candidate_supplied_full_node_sha256": False,
            "full_node_sha256_resolved_independently": True,
        }
    )
    return {
        **result,
        "semantic_receipt": receipt,
        "canonical_ir": raw,
    }


def render_prompt(task: dict[str, Any], common_context: str) -> str:
    roles = base.obligation_roles(task)
    return (
        "[THESEUS_PRODUCTION_SEMANTIC_IR_COMPACT]\n"
        f"Implement this repository task: {task.get('natural_request')}\n\n"
        "Compile change obligations into the least-sufficient candidate-visible "
        "statement or declaration edit units. Prefer the smallest exact node that "
        "can express the change; do not reproduce a containing function when a "
        "nested statement target is sufficient. Carry dependency-required preserve "
        "obligations on those units. Non-goals are global invariants and must never "
        "be attached to a mutation. Copy source, path, NODE handle, and obligation "
        "identities exactly from candidate-visible context. Return only one complete artifact.\n\n"
        + common_context
        + "\n\nOUTPUT ONLY THIS SHAPE:\n"
        + f"{HEADER}\n"
        + "SOURCE <copy exact semantic source digest>\n"
        + f"ALL_OBLIGATIONS {base.encode_ids(roles['all'])}\n"
        + f"CHANGE_OBLIGATIONS {base.encode_ids(roles['change'])}\n"
        + f"PRESERVE_OBLIGATIONS {base.encode_ids(roles['preserve'])}\n"
        + f"NON_GOAL_OBLIGATIONS {base.encode_ids(roles['non_goal'])}\n"
        + "UNIT U1\n"
        + "OBLIGATIONS <change ids plus dependency-required preserve ids>\n"
        + "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>\n"
        + "PATH <copy exact path from the nearest PATH group>\n"
        + "NODE <copy exact 128-bit semantic node handle>\n"
        + "<<<\n<replacement source for only that node>\n>>>\n"
        + "END_UNIT\n"
        + "For a path listed under MISSING_ALLOWED_EFFECT_PATHS, instead use:\n"
        + "UNIT U2\n"
        + "OBLIGATIONS <change ids plus dependency-required preserve ids>\n"
        + "OP CREATE_FILE\n"
        + "PATH <copy exact declared missing repository-relative path>\n"
        + "<<<\n<complete new file source>\n>>>\n"
        + "END_UNIT\n"
        + "LOSS NONE\n"
        + "END"
    )


render_repair_prompt = base.render_repair_prompt
apply_actions = base.apply_actions
canonicalize_with_receipt = base.canonicalize_with_receipt
canonical_path_for_symbol = base.canonical_path_for_symbol
lower_replacement = base.lower_replacement
