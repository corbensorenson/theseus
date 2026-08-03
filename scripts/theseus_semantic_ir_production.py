#!/usr/bin/env python3
"""Canonical production Semantic-IR schema and deterministic lowerer.

Historical Semantic-IR transports remain immutable evidence.  This owner fixes
the production mechanics they exposed: change, preservation, and non-goal
obligations are distinct semantic roles, and redundant source coordinates may
be normalized only when they exactly agree with the candidate-visible symbol
identity.  No hidden evaluator or answer data is consumed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4
import theseus_p4s_cognitive_compilation as p4s
import theseus_semantic_ir_v2r2 as list_transport


HEADER = "THESEUS_SEMANTIC_IR_V3"
ROLE_FIELDS = (
    "ALL_OBLIGATIONS",
    "CHANGE_OBLIGATIONS",
    "PRESERVE_OBLIGATIONS",
    "NON_GOAL_OBLIGATIONS",
)
LIST_FIELDS = set(ROLE_FIELDS) | {"OBLIGATIONS", "LOSS"}
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
COORDINATE_PATH_RE = re.compile(
    r"^(.*):(\d+):(\d+)-(\d+):(\d+)$"
)
ROLE_KIND = {
    "CHANGE_OBLIGATIONS": "require",
    "PRESERVE_OBLIGATIONS": "preserve",
    "NON_GOAL_OBLIGATIONS": "non_goal",
}
SymbolTableFactory = Callable[[Path, dict[str, Any]], dict[str, Any]]


def grammar() -> str:
    return (
        f"{HEADER}\n"
        "SOURCE <semantic-source-digest>\n"
        "ALL_OBLIGATIONS <all exact ids in task order>\n"
        "CHANGE_OBLIGATIONS <exact require ids in task order|NONE>\n"
        "PRESERVE_OBLIGATIONS <exact preserve ids in task order|NONE>\n"
        "NON_GOAL_OBLIGATIONS <exact non-goal ids in task order|NONE>\n"
        "UNIT <stable-unit-id>\n"
        "OBLIGATIONS <change ids and dependency-required preserve ids>\n"
        "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>\n"
        "PATH <exact repository-relative path>\n"
        "NODE <exact candidate-visible semantic node id>\n"
        "NODE_SHA <exact candidate-visible semantic node sha256>\n"
        "<<<\n<replacement source>\n>>>\n"
        "END_UNIT\n"
        "LOSS <NONE|comma-separated unresolved obligation ids>\n"
        "END"
    )


def complete(text: str) -> bool:
    raw, _ = canonicalize_with_receipt(text)
    return (
        raw.startswith(f"{HEADER}\n")
        and raw.endswith("\nEND")
        and all(
            re.search(
                rf"^{field} (?:NONE|[A-Z0-9_,]+)$", raw, flags=re.MULTILINE
            )
            for field in ROLE_FIELDS
        )
        and bool(re.search(r"^SOURCE [a-f0-9]{64}$", raw, flags=re.MULTILINE))
        and bool(UNIT_RE.search(raw))
        and bool(
            re.search(
                r"^LOSS (?:NONE|[A-Z0-9_,]+)\nEND$",
                raw,
                flags=re.MULTILINE,
            )
        )
    )


def canonicalize_with_receipt(text: str) -> tuple[str, dict[str, Any]]:
    """Normalize list delimiters only, never values, roles, paths, or source."""

    raw = unwrap(text)
    normalized_lines: list[str] = []
    fields: list[dict[str, Any]] = []
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
        match = None if inside_replacement else re.fullmatch(
            r"([A-Z_]+) (.+)", line
        )
        if match is None or match.group(1) not in LIST_FIELDS:
            normalized_lines.append(line)
            continue
        field, surface = match.groups()
        if surface == "NONE" and field in LIST_FIELDS:
            normalized_lines.append(line)
            continue
        parsed = list_transport.canonical_obligation_list(surface)
        if parsed is None:
            normalized_lines.append(line)
            fields.append({
                "field": field,
                "line": line_number,
                "state": "REJECTED_UNRECOGNIZED_SURFACE",
            })
            continue
        canonical, identifiers, surface_class = parsed
        normalized_lines.append(f"{field} {canonical}")
        fields.append({
            "field": field,
            "line": line_number,
            "state": "NORMALIZED" if canonical != surface else "ALREADY_CANONICAL",
            "surface_class": surface_class,
            "identifier_count": len(identifiers),
            "identifier_values_invented": 0,
            "identifier_order_changed": False,
        })
    return "\n".join(normalized_lines), {
        "policy": "project_theseus_semantic_ir_production_canonicalization_v1",
        "fields": fields,
        "normalized_field_count": sum(row["state"] == "NORMALIZED" for row in fields),
        "rejected_field_count": sum(
            row["state"] == "REJECTED_UNRECOGNIZED_SURFACE" for row in fields
        ),
        "identifier_values_invented": 0,
        "identifier_order_changed": False,
        "replacement_source_touched": False,
        "target_identity_touched": False,
        "answer_bearing_transformation": False,
    }


def parse(
    text: str,
    task: dict[str, Any],
    root: Path,
    *,
    symbol_table_factory: SymbolTableFactory = p4s.semantic_scope_symbol_table,
) -> dict[str, Any]:
    raw, canonicalization = canonicalize_with_receipt(text)
    faults: list[str] = []
    if not raw.startswith(f"{HEADER}\n") or not raw.endswith("\nEND"):
        return empty(["semantic_ir_production_envelope_invalid"], canonicalization)

    symbols = symbol_table_factory(root, task)
    source = field_value(raw, "SOURCE", r"[a-f0-9]{64}")
    if source != symbols.get("source_digest"):
        faults.append("semantic_source_identity_invalid")

    obligations = p2a.dicts(task.get("obligations"))
    expected_all = [str(row.get("id") or "") for row in obligations]
    expected_by_kind = {
        kind: [
            str(row.get("id") or "")
            for row in obligations
            if row.get("kind") == kind
        ]
        for kind in ("require", "preserve", "non_goal")
    }
    declared_by_field: dict[str, list[str]] = {}
    for field in ROLE_FIELDS:
        value = field_value(raw, field, r"NONE|[A-Z0-9_,]+")
        declared_by_field[field] = (
            [] if value == "NONE" else value.split(",") if value else []
        )
    if declared_by_field["ALL_OBLIGATIONS"] != expected_all:
        faults.append("semantic_obligation_identity_or_order_invalid")
    for field, kind in ROLE_KIND.items():
        if declared_by_field[field] != expected_by_kind[kind]:
            faults.append(f"semantic_obligation_role_identity_invalid:{kind}")
    if set().union(*(set(values) for values in declared_by_field.values())) != set(expected_all):
        faults.append("semantic_obligation_role_partition_invalid")

    loss_value = field_value(raw, "LOSS", r"NONE|[A-Z0-9_,]+")
    loss = [] if loss_value == "NONE" else loss_value.split(",") if loss_value else []
    if not loss_value:
        faults.append("semantic_loss_record_missing")
    elif loss:
        if not set(loss).issubset(set(expected_all)):
            faults.append("semantic_loss_identity_invalid")
        faults.append("semantic_loss_unresolved")

    units = list(UNIT_RE.finditer(raw))
    if not units:
        faults.append("semantic_units_missing")
    scrubbed = raw
    for match in reversed(units):
        start, end = match.span()
        scrubbed = scrubbed[:start] + scrubbed[end:]
    scrubbed = re.sub(
        r"^(THESEUS_SEMANTIC_IR_V3|SOURCE .+|ALL_OBLIGATIONS .+|"
        r"CHANGE_OBLIGATIONS .+|PRESERVE_OBLIGATIONS .+|"
        r"NON_GOAL_OBLIGATIONS .+|LOSS .+|END)\s*$",
        "",
        scrubbed,
        flags=re.MULTILINE,
    ).strip()
    if scrubbed:
        faults.append("semantic_ir_production_unparsed_text")

    symbol_map = {str(row.get("id") or ""): row for row in p2a.dicts(symbols.get("nodes"))}
    change_ids = set(expected_by_kind["require"])
    preserve_ids = set(expected_by_kind["preserve"])
    non_goal_ids = set(expected_by_kind["non_goal"])
    covered_changes: set[str] = set()
    actions: list[dict[str, Any]] = []
    unit_receipts: list[dict[str, Any]] = []
    path_receipts: list[dict[str, Any]] = []
    ranges: dict[str, list[tuple[int, int]]] = {}
    dependencies = p2a.dicts(task.get("obligation_dependencies"))

    for match in units:
        unit_id, refs_raw, operation, path_surface, node_id, node_hash, replacement = match.groups()
        refs = refs_raw.split(",")
        refs_set = set(refs)
        if len(refs) != len(refs_set) or not refs_set.issubset(set(expected_all)):
            faults.append("semantic_unit_obligation_reference_invalid")
        if refs_set & non_goal_ids:
            faults.append("semantic_non_goal_attached_to_mutation")
        unit_changes = refs_set & change_ids
        if not unit_changes:
            faults.append("semantic_unit_change_obligation_missing")
        covered_changes.update(unit_changes)
        required = p4.dependency_ancestors(list(unit_changes), dependencies)
        if not required.issubset(refs_set):
            faults.append("semantic_unit_dependency_not_closed")
        if refs_set - change_ids - preserve_ids:
            faults.append("semantic_unit_role_invalid")

        symbol = p2a.mapping(symbol_map.get(node_id))
        canonical_path, path_receipt = canonical_path_for_symbol(path_surface, symbol)
        path_receipts.append(path_receipt)
        if (
            not symbol
            or canonical_path is None
            or symbol.get("sha256") != node_hash
        ):
            faults.append("semantic_target_identity_invalid")
            continue
        start, end = int(symbol["start_line"]), int(symbol["end_line"])
        if any(
            not (end < prior_start or start > prior_end)
            for prior_start, prior_end in ranges.setdefault(canonical_path, [])
        ):
            faults.append("semantic_units_overlap")
        ranges[canonical_path].append((start, end))
        original_lines = p2a.checked_source_path(root, canonical_path).read_text(
            encoding="utf-8"
        ).splitlines()
        original = original_lines[start - 1 : end]
        lowered = lower_replacement(operation, replacement, original, symbol)
        action = {
            "op": "REPLACE",
            "path": canonical_path,
            "start_line": start,
            "end_line": end,
            "replacement": lowered,
        }
        actions.append(action)
        unit_receipts.append({
            "id": unit_id,
            "obligation_ids": refs,
            "change_obligation_ids": [value for value in refs if value in change_ids],
            "preserve_obligation_ids": [value for value in refs if value in preserve_ids],
            "operation": operation,
            "path": canonical_path,
            "node_id": node_id,
            "node_sha256": node_hash,
            "replacement_sha256": p2a.sha256_text(replacement),
        })

    if covered_changes != change_ids:
        faults.append("semantic_change_obligation_coverage_incomplete")
    if len({row["id"] for row in unit_receipts}) != len(unit_receipts):
        faults.append("semantic_unit_identity_duplicate")
    allowed = set(p2a.strings(task.get("allowed_effect_paths")))
    if any(row["path"] not in allowed for row in actions):
        faults.append("semantic_effect_path_unauthorized")

    receipt = {
        "policy": "project_theseus_semantic_ir_production_receipt_v1",
        "schema": HEADER,
        "semantic_source_digest": symbols.get("source_digest"),
        "obligation_roles": {
            "change": expected_by_kind["require"],
            "preserve": expected_by_kind["preserve"],
            "non_goal": expected_by_kind["non_goal"],
        },
        "loss_obligation_ids": loss,
        "units": unit_receipts,
        "path_normalization": path_receipts,
        "canonicalization": canonicalization,
        "lowered_action_sha256": p2a.stable_hash(actions),
        "model_generated_ir": True,
        "deterministic_lowerer": True,
        "hidden_evaluator_fields_consumed": [],
        "answer_bearing_repair": False,
    }
    unique_faults = sorted(set(faults))
    return {
        "actions": actions if not unique_faults else [],
        "faults": unique_faults,
        "units": unit_receipts,
        "semantic_receipt": receipt,
        "canonical_ir": raw,
    }


def canonical_path_for_symbol(
    surface: str, symbol: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    exact = str(symbol.get("path") or "")
    receipt = {
        "surface": surface,
        "canonical": exact,
        "state": "REJECTED",
        "answer_bearing_transformation": False,
    }
    if exact and surface == exact:
        receipt["state"] = "EXACT"
        return exact, receipt
    match = COORDINATE_PATH_RE.fullmatch(surface)
    if not match or not exact:
        return None, receipt
    path, start_line, start_col, end_line, end_col = match.groups()
    observed = tuple(map(int, (start_line, start_col, end_line, end_col)))
    expected = (
        int(symbol.get("start_line") or 0),
        int(symbol.get("start_col") or 0),
        int(symbol.get("end_line") or 0),
        int(symbol.get("end_col") or 0),
    )
    if path != exact or observed != expected:
        return None, receipt
    receipt.update({
        "state": "REDUNDANT_EXACT_COORDINATES_REMOVED",
        "observed_coordinates": list(observed),
        "candidate_visible_symbol_coordinates": list(expected),
    })
    return exact, receipt


def lower_replacement(
    operation: str,
    replacement: str,
    original: list[str],
    symbol: dict[str, Any],
) -> str:
    if operation == "INSERT_BEFORE":
        return replacement + "\n" + "\n".join(original)
    if operation == "INSERT_AFTER":
        return "\n".join(original) + "\n" + replacement
    replacement_lines = replacement.splitlines() or [""]
    prefix = original[0][: int(symbol.get("start_col") or 0)]
    suffix = original[-1][int(symbol.get("end_col") or len(original[-1])) :]
    if len(replacement_lines) == 1:
        return prefix + replacement_lines[0] + suffix
    return "\n".join(
        [
            prefix + replacement_lines[0],
            *replacement_lines[1:-1],
            replacement_lines[-1] + suffix,
        ]
    )


def render_prompt(task: dict[str, Any], common_context: str) -> str:
    roles = obligation_roles(task)
    return (
        "[THESEUS_PRODUCTION_SEMANTIC_IR]\n"
        f"Implement this repository task: {task.get('natural_request')}\n\n"
        "Compile change obligations into the least-sufficient semantic edit units. "
        "Carry dependency-required preserve obligations on those units. Non-goals are "
        "global invariants and must never be attached to a mutation. Copy all source, "
        "path, node, hash, and obligation identities exactly from candidate-visible "
        "context. Return only one complete artifact.\n\n"
        + common_context
        + "\n\nOUTPUT ONLY THIS SHAPE:\n"
        + f"{HEADER}\n"
        + "SOURCE <copy exact semantic source digest>\n"
        + f"ALL_OBLIGATIONS {encode_ids(roles['all'])}\n"
        + f"CHANGE_OBLIGATIONS {encode_ids(roles['change'])}\n"
        + f"PRESERVE_OBLIGATIONS {encode_ids(roles['preserve'])}\n"
        + f"NON_GOAL_OBLIGATIONS {encode_ids(roles['non_goal'])}\n"
        + "UNIT U1\n"
        + "OBLIGATIONS <change ids plus dependency-required preserve ids>\n"
        + "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>\n"
        + "PATH <copy exact repository-relative path without coordinates>\n"
        + "NODE <copy exact semantic-scope node id>\n"
        + "NODE_SHA <copy exact semantic-scope node sha256>\n"
        + "<<<\n<replacement source for only that node>\n>>>\n"
        + "END_UNIT\n"
        + "LOSS NONE\n"
        + "END"
    )


def render_repair_prompt(
    original_prompt: str,
    first_artifact: str,
    parse_faults: list[str],
    verification: dict[str, Any],
) -> str:
    visible = p2a.mapping(verification.get("visible_verifier"))
    feedback = {
        "parse_or_lower_faults": list(parse_faults),
        "apply_faults": p2a.strings(verification.get("apply_faults")),
        "visible_verifier_returncode": visible.get("returncode"),
        "visible_verifier_stdout_complete": str(visible.get("stdout_tail") or ""),
        "visible_verifier_stderr_complete": str(visible.get("stderr_tail") or ""),
    }
    return (
        original_prompt
        + "\n\n[COMPLETE_PROVISIONAL_ARTIFACT]\n"
        + str(first_artifact or "")
        + "\n\n[COMPLETE_VISIBLE_FEEDBACK]\n"
        + json.dumps(feedback, sort_keys=True)
        + "\n\nReturn only one complete corrected artifact against the ORIGINAL snapshot. "
        "Do not emit a delta, plan, JSON, Markdown, or commentary."
    )


def obligation_roles(task: dict[str, Any]) -> dict[str, list[str]]:
    rows = p2a.dicts(task.get("obligations"))
    return {
        "all": [str(row.get("id") or "") for row in rows],
        "change": [str(row.get("id") or "") for row in rows if row.get("kind") == "require"],
        "preserve": [str(row.get("id") or "") for row in rows if row.get("kind") == "preserve"],
        "non_goal": [str(row.get("id") or "") for row in rows if row.get("kind") == "non_goal"],
    }


def encode_ids(values: list[str]) -> str:
    return ",".join(values) if values else "NONE"


def field_value(raw: str, field: str, pattern: str) -> str:
    match = re.search(rf"^{field} ({pattern})$", raw, flags=re.MULTILINE)
    return match.group(1) if match else ""


def unwrap(text: str) -> str:
    raw = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    return (fenced.group(1) if fenced else raw).strip()


def empty(faults: list[str], canonicalization: dict[str, Any]) -> dict[str, Any]:
    return {
        "actions": [],
        "faults": faults,
        "units": [],
        "semantic_receipt": {"canonicalization": canonicalization},
        "canonical_ir": "",
    }
