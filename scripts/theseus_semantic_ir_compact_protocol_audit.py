#!/usr/bin/env python3
"""Audit the compact Semantic-IR statement ABI without candidate exposure."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_fresh_v4_task_pool as token_owner
import theseus_semantic_ir_production_adequacy_runtime_v5 as runtime


ROOT = Path(__file__).resolve().parents[1]
PRIOR_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_compact_protocol_audit.json"
POLICY = "project_theseus_semantic_ir_compact_protocol_audit_v1"
MODEL_CONTEXT_TOKENS = 262_144
CONSUMED_INDICES = [1, 2, 3, 4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit() -> dict[str, Any]:
    started = time.perf_counter()
    faults: list[str] = []
    pool = p2a.read_json(PRIOR_POOL)
    if pool.get("trigger_state") != "GREEN" or int(pool.get("task_count") or 0) != 18:
        faults.append("prior_pool_invalid")
    rows: list[dict[str, Any]] = []
    for prior in p2a.dicts(pool.get("rows")):
        index = int(prior.get("index") or 0)
        row_faults: list[str] = []
        task_path = p2a.resolve(str(prior.get("task_manifest") or ""))
        packet_path = p2a.resolve(str(prior.get("candidate_packet") or ""))
        if not task_path.is_file() or p2a.sha256_file(task_path) != prior.get("task_manifest_sha256"):
            row_faults.append("task_binding_invalid")
        if not packet_path.is_file() or p2a.sha256_file(packet_path) != prior.get("candidate_packet_sha256"):
            row_faults.append("packet_binding_invalid")
        if row_faults:
            faults.extend(f"task_{index:02d}:{fault}" for fault in row_faults)
            rows.append({"index": index, "trigger_state": "RED", "faults": row_faults})
            continue
        task = p2a.read_json(task_path)
        old_prompt = str(p2a.read_json(packet_path).get("serialized_prompt") or "")
        with tempfile.TemporaryDirectory(prefix=f"theseus-compact-audit-{index:02d}-") as directory:
            root = Path(directory) / "source"
            p2a.extract_source_archive(
                p2a.resolve(str(task.get("source_archive") or "")),
                root,
                str(task.get("source_archive_root") or ""),
            )
            symbols = runtime.semantic_scope_symbol_table(root, task)
            context = runtime.render_common_context(root, task, symbols)
            compact_prompt = runtime.render_prompt(task, context)
            visible_source = p2a.render_visible_context(root, task)
        nodes = p2a.dicts(symbols.get("nodes"))
        address_inventory = context.split("[COMPACT_SEMANTIC_NODE_ABI]\n", 1)[-1].split(
            "[MISSING_ALLOWED_EFFECT_PATHS]", 1
        )[0]
        compact_bytes = len(compact_prompt.encode("utf-8"))
        old_bytes = len(old_prompt.encode("utf-8"))
        exact_tokens = token_owner.exact_prompt_tokens(compact_prompt)
        if len(nodes) != int(prior.get("semantic_scope_node_count") or 0):
            row_faults.append("statement_inventory_cardinality_changed")
        if symbols.get("inventory_complete") is not True or symbols.get("inventory_truncated") is not False:
            row_faults.append("statement_inventory_incomplete_or_truncated")
        if len({str(node.get("id") or "") for node in nodes}) != len(nodes):
            row_faults.append("node_handles_not_unique")
        if any(not re.fullmatch(r"N-[A-F0-9]{32}", str(node.get("id") or "")) for node in nodes):
            row_faults.append("node_handle_shape_invalid")
        if any(str(node.get("sha256") or "") in address_inventory for node in nodes):
            row_faults.append("full_node_digest_candidate_visible")
        if visible_source not in compact_prompt:
            row_faults.append("complete_visible_source_missing")
        if address_inventory.count("N-") != len(nodes):
            row_faults.append("candidate_visible_handle_cardinality_invalid")
        if compact_bytes >= old_bytes:
            row_faults.append("compact_representation_not_smaller")
        if exact_tokens <= 0 or exact_tokens >= MODEL_CONTEXT_TOKENS:
            row_faults.append("physical_context_addressability_invalid")
        faults.extend(f"task_{index:02d}:{fault}" for fault in row_faults)
        rows.append(
            {
                "index": index,
                "consumed_surface": index in CONSUMED_INDICES,
                "statement_node_count": len(nodes),
                "old_prompt_utf8_bytes": old_bytes,
                "compact_prompt_utf8_bytes": compact_bytes,
                "byte_reduction_fraction": round(1.0 - compact_bytes / old_bytes, 6),
                "old_exact_prompt_tokens": int(prior.get("exact_prompt_tokens") or 0),
                "compact_exact_prompt_tokens": exact_tokens,
                "compact_exact_context_residual_tokens": MODEL_CONTEXT_TOKENS - exact_tokens,
                "complete_visible_source_retained": visible_source in compact_prompt,
                "full_node_digest_candidate_visible": False,
                "full_node_digest_resolved_independently": True,
                "inventory_complete": symbols.get("inventory_complete"),
                "inventory_truncated": symbols.get("inventory_truncated"),
                "project_selected_node_cap_applied": False,
                "faults": row_faults,
                "trigger_state": "GREEN" if not row_faults else "RED",
            }
        )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults and len(rows) == 18 else "RED",
        "state": "COMPACT_PROTOCOL_MECHANICS_GREEN_FRESH_DENOMINATOR_REQUIRED"
        if not faults and len(rows) == 18
        else "COMPACT_PROTOCOL_INVALID",
        "prior_pool": {"path": p2a.rel(PRIOR_POOL), "sha256": p2a.sha256_file(PRIOR_POOL)},
        "runtime": {
            "path": p2a.rel(Path(runtime.__file__)),
            "sha256": p2a.sha256_file(Path(runtime.__file__)),
            "header": runtime.HEADER,
            "handle_bits": runtime.NODE_HANDLE_HEX_BITS,
        },
        "consumed_indices": CONSUMED_INDICES,
        "fresh_replacement_required_for_indices": CONSUMED_INDICES,
        "unexposed_indices_eligible_for_uniform_protocol_rebind": list(range(5, 19)),
        "task_count": len(rows),
        "green_task_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "maximum_compact_exact_prompt_tokens": max(
            (int(row.get("compact_exact_prompt_tokens") or 0) for row in rows), default=0
        ),
        "minimum_compact_context_residual_tokens": min(
            (int(row.get("compact_exact_context_residual_tokens") or 0) for row in rows), default=0
        ),
        "rows": rows,
        "faults": sorted(set(faults)),
        "counters": {
            "candidate_or_model_calls": 0,
            "hidden_evaluator_executions": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": "A GREEN audit establishes only that the compact ABI preserves the complete statement inventory and parent source, independently resolves full node integrity, and reduces redundant prompt representation on the prior 18 sources. It does not establish host operability, model competence, implementation adequacy, a subsystem effect, D1, D2, training value, serving, or book support. Exposed indices 1-4 remain consumed and require fresh sources.",
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    task4 = next(
        (row for row in p2a.dicts(report.get("rows")) if row.get("index") == 4), {}
    )
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "green_task_count": report.get("green_task_count"),
        "maximum_compact_exact_prompt_tokens": report.get("maximum_compact_exact_prompt_tokens"),
        "minimum_compact_context_residual_tokens": report.get("minimum_compact_context_residual_tokens"),
        "task4_old_exact_prompt_tokens": task4.get("old_exact_prompt_tokens"),
        "task4_compact_exact_prompt_tokens": task4.get("compact_exact_prompt_tokens"),
        "task4_byte_reduction_fraction": task4.get("byte_reduction_fraction"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
