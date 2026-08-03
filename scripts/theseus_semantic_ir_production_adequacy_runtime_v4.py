#!/usr/bin/env python3
"""Statement-granular successor runtime for production Semantic-IR adequacy.

The historical adequacy runtime intentionally remains immutable evidence.  This
successor repairs the output-amplification wall exposed by the v3 campaign: a
local edit inside a large function must have an exact statement target and must
not require the model to reproduce the whole function.  The inventory is
candidate-visible, complete for Python statements in the declared read ranges,
and never silently truncated to a project-selected node count.
"""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_runtime as base


HEADER = base.HEADER
CREATE_FILE_OPERATION = base.CREATE_FILE_OPERATION
CREATE_FILE_SUPPORTED = base.CREATE_FILE_SUPPORTED
STATEMENT_SCOPE_POLICY = "all_candidate_visible_python_statements_v1"


def semantic_scope_symbol_table(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Return every safe statement target in the declared visible source ranges."""

    missing = sorted(
        set(
            p2a.strings(
                p2a.mapping(task.get("candidate_visible_context")).get(
                    "missing_allowed_effect_paths"
                )
            )
        )
    )
    allowed = p2a.strings(task.get("allowed_effect_paths"))
    if (
        any(path not in allowed or p2a.unsafe_relative_path(path) for path in missing)
        or any((root / path).exists() for path in missing)
    ):
        raise p2a.InstrumentFault("declared_missing_effect_path_invalid")

    reads: dict[str, list[tuple[int, int]]] = {}
    for row in p2a.dicts(
        p2a.mapping(task.get("candidate_visible_context")).get("reads")
    ):
        reads.setdefault(str(row.get("path") or ""), []).append(
            (
                int(row.get("start_line") or 1),
                int(row.get("end_line") or 10**9),
            )
        )

    nodes: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for path in allowed:
        if path in missing:
            continue
        source_path = p2a.checked_source_path(root, path)
        text = source_path.read_text(encoding="utf-8")
        source_hashes[path] = p2a.sha256_text(text)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt) or not hasattr(node, "lineno"):
                continue
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", start) or start)
            if reads.get(path) and not any(
                start <= visible_end and end >= visible_start
                for visible_start, visible_end in reads[path]
            ):
                continue
            start_col = int(getattr(node, "col_offset", 0) or 0)
            end_col = int(
                getattr(node, "end_col_offset", len(lines[end - 1]))
                or len(lines[end - 1])
            )
            if start == end:
                segment = lines[start - 1][start_col:end_col]
            else:
                segment = "\n".join(
                    [
                        lines[start - 1][start_col:],
                        *lines[start : end - 1],
                        lines[end - 1][:end_col],
                    ]
                )
            node_hash = p2a.sha256_text(segment)
            node_type = type(node).__name__
            node_id = "N-" + p2a.sha256_text(
                f"{path}|{node_type}|{start}|{end}|{node_hash}"
            )[:16].upper()
            label = str(getattr(node, "name", "")) or ast.dump(
                node, annotate_fields=False, include_attributes=False
            )
            nodes.append(
                {
                    "id": node_id,
                    "sha256": node_hash,
                    "node_type": node_type,
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "start_col": start_col,
                    "end_col": end_col,
                    "label": re.sub(r"\s+", " ", label)[:100],
                }
            )

    nodes.sort(
        key=lambda row: (
            row["path"],
            row["start_line"],
            row["end_line"],
            row["node_type"],
        )
    )
    contract = p2a.mapping(task.get("semantic_ir_contract"))
    declared_capacity = int(
        contract.get("maximum_semantic_scope_nodes")
        or contract.get("maximum_symbol_nodes")
        or 0
    )
    if declared_capacity and declared_capacity < len(nodes):
        raise p2a.InstrumentFault("semantic_statement_inventory_would_be_truncated")

    if missing:
        identity_hashes = {**source_hashes, **{path: "MISSING_ALLOWED_EFFECT_PATH" for path in missing}}
        source_digest = p2a.stable_hash(
            {
                "allowed_file_sha256": identity_hashes,
                "missing_allowed_effect_paths": missing,
            }
        )
    else:
        identity_hashes = source_hashes
        source_digest = p2a.stable_hash(source_hashes)

    spans = [int(row["end_line"]) - int(row["start_line"]) + 1 for row in nodes]
    return {
        "source_digest": source_digest,
        "allowed_file_sha256": identity_hashes,
        "missing_allowed_effect_paths": missing,
        "nodes": nodes,
        "semantic_unit_policy": STATEMENT_SCOPE_POLICY,
        "inventory_complete": True,
        "inventory_truncated": False,
        "statement_scope_metrics": {
            "node_count": len(nodes),
            "single_line_node_count": sum(span == 1 for span in spans),
            "maximum_node_line_span": max(spans, default=0),
            "project_selected_node_cap_applied": False,
        },
    }


def parse(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    return base.parse(
        text,
        task,
        root,
        symbol_table_factory=semantic_scope_symbol_table,
    )


def render_prompt(task: dict[str, Any], common_context: str) -> str:
    prompt = base.render_prompt(task, common_context)
    return prompt.replace(
        "Compile change obligations into the least-sufficient semantic edit units.",
        "Compile change obligations into the least-sufficient candidate-visible "
        "statement or declaration edit units. Prefer the smallest exact node that "
        "can express the change; do not reproduce a containing function when a "
        "nested statement target is sufficient.",
    )


complete = base.complete
render_repair_prompt = base.render_repair_prompt
apply_actions = base.apply_actions
canonicalize_with_receipt = base.canonicalize_with_receipt
canonical_path_for_symbol = base.canonical_path_for_symbol
lower_replacement = base.lower_replacement
