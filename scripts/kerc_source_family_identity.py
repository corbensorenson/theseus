#!/usr/bin/env python3
"""Content identities for independently replaceable KERC source families.

The semantic producer and verifier are large orchestration modules. Binding a
record to either whole file makes an unrelated family addition invalidate the
entire corpus. This module computes a conservative transitive closure over the
top-level functions and constants used by one family. Shared helpers remain in
every affected closure, so changing common behavior still invalidates all
dependent families.

The implementation parses source text only. The independent verifier can
therefore recompute producer identities without importing or executing producer
code.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


POLICY = "project_theseus_kerc_source_family_identity_v1"
ALGORITHM = "top_level_function_and_constant_transitive_source_closure_v1"

PRODUCER_FAMILY_ROOTS = {
    "dolly_direct": "dolly_record",
    "dolly_grounded": "dolly_grounded_question_record",
    "masc_frame": "masc_record",
    "masc_composite": "masc_composite_record",
    "masc_decision": "masc_decision_semantic_record",
    "masc_event_coreference": "masc_event_coreference_record",
    "masc_mpqa_relation": "masc_mpqa_relation_record",
    "gum_discourse": "gum_discourse_record",
    "oasst_dialogue": "oasst_record",
    "oasst_behavior": "oasst_behavior_record",
}

VERIFIER_FAMILY_ROOTS = {
    "dolly_direct": "verify_dolly_record",
    "dolly_grounded": "verify_dolly_grounded_record",
    "masc_frame": "verify_masc_record",
    "masc_composite": "verify_masc_composite_record",
    "masc_decision": "verify_masc_decision_record",
    "masc_event_coreference": "verify_masc_event_coreference_record",
    "masc_mpqa_relation": "verify_masc_mpqa_relation_record",
    "gum_discourse": "verify_gum_discourse_record",
    "oasst_dialogue": "verify_oasst_record",
    "oasst_behavior": "verify_oasst_behavior_record",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _top_level_sources(
    source: str,
) -> tuple[dict[str, tuple[ast.AST, str]], dict[str, tuple[ast.AST, str]]]:
    tree = ast.parse(source)
    functions: dict[str, tuple[ast.AST, str]] = {}
    constants: dict[str, tuple[ast.AST, str]] = {}
    for node in tree.body:
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise ValueError("unable to recover source-family identity segment")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = (node, segment)
            continue
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            constants[name] = (node, segment)
    return functions, constants


def _source_closure_receipt_from_index(
    *,
    functions: dict[str, tuple[ast.AST, str]],
    constants: dict[str, tuple[ast.AST, str]],
    source_label: str,
    role: str,
    family: str,
    root_function: str,
    external_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if root_function not in functions:
        raise ValueError(f"missing source-family root function: {root_function}")

    pending = [root_function]
    included_functions: set[str] = set()
    included_constants: set[str] = set()
    while pending:
        name = pending.pop()
        if name in included_functions:
            continue
        node, _segment = functions[name]
        included_functions.add(name)
        names = _loaded_names(node)
        pending.extend(
            sorted(
                referenced
                for referenced in names
                if referenced in functions and referenced not in included_functions
            )
        )
        included_constants.update(name for name in names if name in constants)

    constant_pending = list(included_constants)
    while constant_pending:
        name = constant_pending.pop()
        node, _segment = constants[name]
        for referenced in _loaded_names(node):
            if referenced in constants and referenced not in included_constants:
                included_constants.add(referenced)
                constant_pending.append(referenced)

    function_sources = {
        name: _sha256_bytes(functions[name][1].encode("utf-8"))
        for name in sorted(included_functions)
    }
    constant_sources = {
        name: _sha256_bytes(constants[name][1].encode("utf-8"))
        for name in sorted(included_constants)
    }
    external_sources = {
        name: _sha256_file(path.resolve())
        for name, path in sorted(external_paths.items())
    }
    identity_payload = {
        "policy": POLICY,
        "algorithm": ALGORITHM,
        "source_label": source_label,
        "role": role,
        "family": family,
        "root_function": root_function,
        "function_sources": function_sources,
        "constant_sources": constant_sources,
        "external_sources": external_sources,
    }
    return {
        **identity_payload,
        "identity_sha256": _sha256_bytes(_canonical(identity_payload).encode("utf-8")),
    }


def source_closure_receipt(
    *,
    source_path: Path,
    source_label: str,
    role: str,
    family: str,
    root_function: str,
    external_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Return a source-only, transitive identity for one family implementation."""

    functions, constants = _top_level_sources(
        source_path.read_text(encoding="utf-8")
    )
    return _source_closure_receipt_from_index(
        functions=functions,
        constants=constants,
        source_label=source_label,
        role=role,
        family=family,
        root_function=root_function,
        external_paths=external_paths,
    )


def family_identity_receipts(
    *,
    source_path: Path,
    source_label: str,
    role: str,
    family_roots: Mapping[str, str],
    external_paths: Mapping[str, Path],
    family_external_paths: Mapping[str, Mapping[str, Path]] | None = None,
) -> dict[str, dict[str, Any]]:
    functions, constants = _top_level_sources(
        source_path.read_text(encoding="utf-8")
    )
    return {
        family: _source_closure_receipt_from_index(
            functions=functions,
            constants=constants,
            source_label=source_label,
            role=role,
            family=family,
            root_function=root,
            external_paths={
                **external_paths,
                **(family_external_paths or {}).get(family, {}),
            },
        )
        for family, root in sorted(family_roots.items())
    }


def source_family(*, dataset_key: str, source_id: str) -> str:
    """Classify a canonical source identity without answer-derived metadata."""

    if source_id.startswith("dolly-grounded:"):
        return "dolly_grounded"
    if dataset_key == "dolly":
        return "dolly_direct"
    if source_id.startswith("masc-composite:"):
        return "masc_composite"
    if source_id.startswith("masc-decision:"):
        return "masc_decision"
    if source_id.startswith("masc-event-coref:"):
        return "masc_event_coreference"
    if source_id.startswith("masc-mpqa-relation:"):
        return "masc_mpqa_relation"
    if dataset_key == "gum" and source_id.startswith("gum-erst:"):
        return "gum_discourse"
    if dataset_key == "masc":
        return "masc_frame"
    if source_id.startswith("oasst2-behavior:"):
        return "oasst_behavior"
    if dataset_key == "oasst2":
        return "oasst_dialogue"
    raise ValueError(f"unknown KERC source family: {dataset_key}:{source_id}")
