#!/usr/bin/env python3
"""Audit the statement-granularity repair without opening hidden evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_canary as canary
import theseus_semantic_ir_production_adequacy_runtime as historical
import theseus_semantic_ir_production_adequacy_runtime_v4 as repaired


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool_v3.json"
DEFAULT_INTERRUPTION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_v3_watchdog_interruption.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_statement_granularity_audit.json"
POLICY = "project_theseus_semantic_ir_statement_granularity_audit_v1"
MODEL_CONTEXT_TOKENS = 262_144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=p2a.rel(DEFAULT_POOL))
    parser.add_argument("--interruption", default=p2a.rel(DEFAULT_INTERRUPTION))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(
        p2a.resolve(args.pool),
        p2a.resolve(args.interruption),
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(pool_path: Path, interruption_path: Path) -> dict[str, Any]:
    pool = p2a.read_json(pool_path)
    interruption = p2a.read_json(interruption_path)
    rows = p2a.dicts(pool.get("rows"))
    faults: list[str] = []
    if pool.get("trigger_state") != "GREEN" or len(rows) != 18:
        faults.append("sealed_v3_pool_invalid")
    if interruption.get("scientific_status") not in {
        None,
        "INVALID_INFRASTRUCTURE_INCOMPLETE_DENOMINATOR",
    } and p2a.mapping(interruption.get("classification")).get(
        "scientific_status"
    ) != "INVALID_INFRASTRUCTURE_INCOMPLETE_DENOMINATOR":
        faults.append("v3_interruption_disposition_invalid")
    if p2a.mapping(interruption.get("classification")).get(
        "hidden_evaluation_opened"
    ) is not False:
        faults.append("hidden_evaluation_state_invalid")

    repaired_rows: list[dict[str, Any]] = []
    task4_comparison: dict[str, Any] = {}
    for pool_row in rows[3:]:
        index = int(pool_row.get("index") or 0)
        if index not in {4, *range(5, 19)}:
            continue
        task_path = p2a.resolve(str(pool_row.get("task_manifest") or ""))
        task = p2a.read_json(task_path)
        inventory_task = copy.deepcopy(task)
        contract = p2a.mapping(inventory_task.get("semantic_ir_contract"))
        contract["maximum_symbol_nodes"] = 1_000_000
        contract["maximum_semantic_scope_nodes"] = 1_000_000
        inventory_task["semantic_ir_contract"] = contract
        with tempfile.TemporaryDirectory(
            prefix=f"theseus-statement-granularity-{index:02d}-"
        ) as directory:
            root = Path(directory) / "source"
            p2a.extract_source_archive(
                p2a.resolve(str(task.get("source_archive") or "")),
                root,
                str(task.get("source_archive_root") or ""),
            )
            symbols = repaired.semantic_scope_symbol_table(root, inventory_task)
            common = canary.render_common_context(root, inventory_task, symbols)
            missing = p2a.strings(
                p2a.mapping(inventory_task.get("candidate_visible_context")).get(
                    "missing_allowed_effect_paths"
                )
            )
            if missing:
                common += "\n[MISSING_ALLOWED_EFFECT_PATHS]\n" + "\n".join(missing)
            prompt = repaired.render_prompt(inventory_task, common)
            prompt_bytes = len(prompt.encode("utf-8"))
            metrics = p2a.mapping(symbols.get("statement_scope_metrics"))
            row = {
                "index": index,
                "opaque_task_id": task.get("opaque_task_id"),
                "statement_node_count": len(p2a.dicts(symbols.get("nodes"))),
                "single_line_node_count": metrics.get("single_line_node_count"),
                "maximum_node_line_span": metrics.get("maximum_node_line_span"),
                "prompt_utf8_bytes": prompt_bytes,
                "conservative_context_residual_tokens": MODEL_CONTEXT_TOKENS
                - prompt_bytes,
                "inventory_complete": symbols.get("inventory_complete") is True,
                "inventory_truncated": symbols.get("inventory_truncated") is True,
                "project_selected_node_cap_applied": metrics.get(
                    "project_selected_node_cap_applied"
                ),
            }
            repaired_rows.append(row)
            if prompt_bytes >= MODEL_CONTEXT_TOKENS:
                faults.append(f"task_{index:02d}_physical_context_not_proven")
            if index == 4:
                historical_symbols = historical.semantic_scope_symbol_table(root, task)
                target = next(
                    (
                        node
                        for node in p2a.dicts(symbols.get("nodes"))
                        if node.get("path") == "skbio/alignment/_pair.py"
                        and node.get("node_type") == "Assign"
                        and int(node.get("start_line") or 0) == 424
                    ),
                    {},
                )
                container = next(
                    (
                        node
                        for node in p2a.dicts(historical_symbols.get("nodes"))
                        if node.get("path") == "skbio/alignment/_pair.py"
                        and node.get("node_type") == "FunctionDef"
                        and node.get("label") == "pair_align"
                    ),
                    {},
                )
                target_span = int(target.get("end_line") or 0) - int(
                    target.get("start_line") or 0
                ) + 1
                container_span = int(container.get("end_line") or 0) - int(
                    container.get("start_line") or 0
                ) + 1
                task4_comparison = {
                    "historical_scope_node_count": len(
                        p2a.dicts(historical_symbols.get("nodes"))
                    ),
                    "historical_nearest_container": container,
                    "historical_nearest_container_line_span": container_span,
                    "repaired_exact_statement_target": target,
                    "repaired_exact_statement_line_span": target_span,
                    "mutation_span_reduction_factor": (
                        round(container_span / target_span, 3) if target_span else None
                    ),
                    "historical_prompt_utf8_bytes": pool_row.get(
                        "serialized_prompt_utf8_bytes"
                    ),
                    "repaired_prompt_utf8_bytes": prompt_bytes,
                    "repaired_conservative_context_residual_tokens": MODEL_CONTEXT_TOKENS
                    - prompt_bytes,
                }
                if not target or not container or target_span != 1 or container_span <= 1:
                    faults.append("task_04_statement_target_repair_not_proven")

    if len(repaired_rows) != 15:
        faults.append("repaired_surface_audit_denominator_invalid")
    if any(row["inventory_truncated"] for row in repaired_rows):
        faults.append("statement_inventory_truncated")
    if any(
        row["project_selected_node_cap_applied"] is not False
        for row in repaired_rows
    ):
        faults.append("project_selected_node_cap_applied")

    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": (
            "STATEMENT_GRANULARITY_REPRESENTATION_REPAIR_GREEN"
            if not faults
            else "STATEMENT_GRANULARITY_REPAIR_INADEQUATE"
        ),
        "faults": sorted(set(faults)),
        "scope": "Candidate-visible representation and physical addressability audit only; no model call, hidden evaluation, task correctness, subsystem effect, D1, D2, training, serving, or book-support authority.",
        "historical_runtime": {
            "path": p2a.rel(ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime.py"),
            "sha256": p2a.sha256_file(
                ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime.py"
            ),
        },
        "repaired_runtime": {
            "path": p2a.rel(ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime_v4.py"),
            "sha256": p2a.sha256_file(
                ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime_v4.py"
            ),
        },
        "diagnostic_backend": {
            "path": p2a.rel(ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_backend_v2.py"),
            "sha256": p2a.sha256_file(
                ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_backend_v2.py"
            ),
        },
        "v3_interruption": {
            "path": p2a.rel(interruption_path),
            "sha256": p2a.sha256_file(interruption_path),
            "watchdog_generated_tokens": p2a.mapping(
                interruption.get("watchdog_backend_receipt")
            ).get("generated_tokens"),
            "project_selected_quality_token_cap": p2a.mapping(
                interruption.get("watchdog_backend_receipt")
            ).get("project_selected_quality_token_cap"),
        },
        "task_04_consumed_surface_mechanics_comparison": task4_comparison,
        "unexposed_and_consumed_parent_surface_addressability": repaired_rows,
        "fresh_task_requirements": {
            "consumed_tasks_01_through_04_may_not_be_rerun": True,
            "fresh_same_stratum_replacements_required": [1, 2, 3, 4],
            "tasks_05_through_18_require_prospective_v4_packet_rebinding": True,
            "candidate_generation_opened": False,
            "hidden_evaluation_opened": False,
        },
        "counters": p2a.zero_counters(),
        "maximum_inference": "GREEN establishes that complete statement inventory removes the observed whole-function representation wall while preserving physical context addressability on the audited parent surfaces. It does not establish model completion, candidate correctness, full implementation adequacy, causal subsystem utility, D1, D2, or ASI Stack support.",
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    comparison = p2a.mapping(
        report.get("task_04_consumed_surface_mechanics_comparison")
    )
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "faults": report.get("faults"),
        "task_04_mutation_span_reduction_factor": comparison.get(
            "mutation_span_reduction_factor"
        ),
        "audited_surface_count": len(
            p2a.dicts(
                report.get("unexposed_and_consumed_parent_surface_addressability")
            )
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
