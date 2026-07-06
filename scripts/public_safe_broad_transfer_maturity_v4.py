#!/usr/bin/env python3
"""Generate the v4 public-safe broad transfer maturity curriculum.

This is the successor after the v3 verifier-mismatch private repair. It uses
only local synthetic contracts and private tests. The rows intentionally avoid
public benchmark names, prompts, tests, solutions, task ids, score labels, and
candidate code while targeting the broad shape of the below-floor public cards:
stdin-style parsing, function contracts, return/type shape, algorithmic
planning, state machines, and multi-step utility transforms.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from broad_private_generalization_ladder_v1 import Template, template_bank  # noqa: E402
from code_lm_private_rows import training_data_path  # noqa: E402
from code_residual_curriculum import verify_private_solution_rows  # noqa: E402


POLICY = "project_theseus_public_safe_broad_transfer_maturity_v4"
CARD_ID = "public_safe_broad_transfer_maturity_v4"
EVIDENCE_LEVEL = "public_safe_broad_transfer_maturity_v4_generated_only"
CONTRACT_POLICY = "project_theseus_decoder_contract_v4_public_safe_broad_transfer_maturity"
TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "public_safe_broad_transfer_maturity_v4_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "public_safe_broad_transfer_maturity_v4_heldout_code_lm_tasks.jsonl",
)

FAMILIES = (
    "stdin_contest_contracts",
    "entrypoint_function_contracts",
    "algorithmic_planning_contracts",
    "return_type_shape_contracts",
    "stateful_edge_contracts",
    "multi_step_tool_contracts",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--heldout-rows", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument("--out", default="reports/public_safe_broad_transfer_maturity_v4.json")
    parser.add_argument("--markdown-out", default="reports/public_safe_broad_transfer_maturity_v4.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    templates = v4_template_bank()
    train_rows = build_rows(
        templates,
        row_count=max(2400, int(args.train_rows)),
        split="train",
        seed=int(args.seed),
        id_offset=0,
    )
    heldout_rows = build_rows(
        templates,
        row_count=max(1000, int(args.heldout_rows)),
        split="heldout",
        seed=int(args.seed) + 200_000,
        id_offset=2_000_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    train_check = verify_private_solution_rows(train_rows, max_failures=24)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=24)
    train_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in train_rows)
    heldout_family_counts = Counter(str(row.get("broad_private_family_v1")) for row in heldout_rows)
    train_categories = Counter(str(row.get("category")) for row in train_rows)
    heldout_categories = Counter(str(row.get("category")) for row in heldout_rows)
    leakage = public_leakage_scan(train_rows + heldout_rows)
    gates = [
        gate("private_train_rows_ge_2400", len(train_rows) >= 2400, len(train_rows)),
        gate("private_heldout_rows_ge_1000", len(heldout_rows) >= 1000, len(heldout_rows)),
        gate("required_family_count", set(train_family_counts) == set(FAMILIES), dict(train_family_counts)),
        gate("heldout_required_family_count", set(heldout_family_counts) == set(FAMILIES), dict(heldout_family_counts)),
        gate("category_diversity_ge_24", len(train_categories) >= 24 and len(heldout_categories) >= 24, {
            "train_categories": len(train_categories),
            "heldout_categories": len(heldout_categories),
        }),
        gate("private_train_solution_tests_pass", train_check["failure_count"] == 0, train_check),
        gate("private_heldout_solution_tests_pass", heldout_check["failure_count"] == 0, heldout_check),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Private-only v4 maturity pressure for broad public-safe transfer before any new bounded public calibration.",
        "inputs": {
            "seed": int(args.seed),
            "template_count": len(templates),
            "public_benchmark_inputs_read": False,
            "public_prompts_used": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_score_labels_used": False,
            "target_surfaces": [
                "stdin_style_io",
                "entrypoint_function_contracts",
                "algorithmic_planning",
                "return_type_shape",
                "stateful_edges",
                "multi_step_tool_transforms",
            ],
        },
        "outputs": {
            "private_train_jsonl": rel(train_path),
            "private_heldout_jsonl": rel(heldout_path),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "private_train_row_count": len(train_rows),
            "private_heldout_row_count": len(heldout_rows),
            "family_train_row_counts": dict(sorted(train_family_counts.items())),
            "family_heldout_row_counts": dict(sorted(heldout_family_counts.items())),
            "category_train_count": len(train_categories),
            "category_heldout_count": len(heldout_categories),
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "public_data_leakage_hit_count": leakage["hit_count"],
            "external_inference_calls": 0,
            "score_semantics": "private synthetic public-safe maturity pressure only; not public calibration",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "rebuild the Rust decoder so the private train token bridge can load the v4 train path",
            "fan out v4 heldout with STS-on and same-seed STS-off control",
            "score v4 with the private broad scorer and learned-distillation gate",
            "keep public calibration locked until readiness is refreshed and operator-approved",
        ],
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def v4_template_bank() -> list[Template]:
    by_family = {
        "stdin_algorithmic": "stdin_contest_contracts",
        "graph_search": "algorithmic_planning_contracts",
        "dynamic_programming": "algorithmic_planning_contracts",
        "intervals": "algorithmic_planning_contracts",
        "return_interface_fidelity": "return_type_shape_contracts",
        "data_structures": "entrypoint_function_contracts",
        "numeric_edge_cases": "entrypoint_function_contracts",
        "parsing_encoding": "return_type_shape_contracts",
        "state_machines": "stateful_edge_contracts",
        "adversarial_metamorphic": "stateful_edge_contracts",
        "tool_style_transforms": "multi_step_tool_contracts",
        "multi_step_contracts": "multi_step_tool_contracts",
    }
    templates: list[Template] = []
    for base in template_bank():
        maturity_family = by_family.get(base.family, "entrypoint_function_contracts")
        category = f"v4_{base.category}"
        semantic = f"v4_{base.semantic_family or base.category}"
        tags = tuple(sorted({
            "public_safe_broad_transfer_maturity_v4",
            maturity_family,
            *base.tags,
        }))
        visible_arg_count_hint = base.visible_arg_count_hint
        argument_roles = base.argument_roles
        if base.category == "bpg_lcs_length":
            visible_arg_count_hint = 2
            argument_roles = {"data": "left_string", "other": "right_string"}
        templates.append(
            replace(
                base,
                family=maturity_family,
                category=category,
                entry=category,
                prompt=f"Private maturity contract: {base.prompt}",
                tags=tags,
                semantic_family=semantic,
                visible_arg_count_hint=visible_arg_count_hint,
                argument_roles=argument_roles,
            )
        )
    return templates


def build_rows(
    templates: list[Template],
    *,
    row_count: int,
    split: str,
    seed: int,
    id_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        template = templates[(index + seed) % len(templates)]
        variant = seed + index * 19
        rows.append(row_from_template(template, split=split, task_index=id_offset + index, variant=variant))
    return rows


def row_from_template(template: Template, *, split: str, task_index: int, variant: int) -> dict[str, Any]:
    entry = f"{template.entry}_{task_index:07d}"
    tags = sorted({
        CARD_ID,
        split,
        template.family,
        template.category,
        *template.tags,
    })
    return {
        "task_id": f"{CARD_ID}_{template.family}_{task_index:07d}",
        "source_task_id": f"{CARD_ID}_{split}_{variant:07d}",
        "card_id": CARD_ID,
        "source_id": f"local_generated_{CARD_ID}",
        "split": "train" if split == "train" else "eval",
        "category": template.category,
        "prompt": template.prompt,
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template.body,
        "tests": normalize_test_source(template.tests(entry, variant)),
        "tags": tags,
        "broad_private_family_v1": template.family,
        "public_safe_maturity_family_v4": template.family,
        "targeted_private_residual_family_v3": "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum",
        "residual_concept": template.semantic_family or template.category,
        "concept_residual_label": template.category,
        "metamorphic_properties": metamorphic_properties(template),
        "decoder_contract": decoder_contract(template),
        "benchmark_evidence_level": EVIDENCE_LEVEL,
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "public_prompts_included": False,
        "public_score_labels_included": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": POLICY,
            "family": template.family,
            "category": template.category,
            "variant": variant,
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_score_labels_used": False,
            "semantics": "private synthetic public-safe broad transfer maturity pressure only",
        },
    }


def decoder_contract(template: Template) -> dict[str, Any]:
    return {
        "policy": CONTRACT_POLICY,
        "return_shape": template.return_shape,
        "type_family": template.type_family,
        "semantic_family": template.semantic_family or template.category,
        "visible_arg_count_hint": template.visible_arg_count_hint,
        "required_constructs": list(template.required_constructs),
        "residual_label_hint": template.category,
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": 1.7,
        "score_semantics": "private public-safe broad transfer maturity pressure only",
        "argument_roles": template.argument_roles or {"data": "primary_input"},
        "return_contract": {
            "shape": template.return_shape,
            "empty_or_invalid_behavior": "covered_by_private_v4_assertions",
            "must_preserve_container_shape": template.return_shape in {"list", "dict", "tuple"},
        },
        "generation_plan": {
            "policy": "private_train_solution_body -> semantic_family_token_decoder -> heldout_contract_body",
            "skeleton_bias": list(template.required_constructs),
            "repair_strategy": "prefer private-train-induced reusable semantic bodies over category-specific adapters",
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def normalize_test_source(source: str) -> str:
    text = source.replace("\\nassert ", "\nassert ")
    if text.endswith("\\n"):
        text = text[:-2] + "\n"
    return text


def metamorphic_properties(template: Template) -> list[str]:
    common = {
        "stdin_contest_contracts": ["stdin_whitespace_tolerance", "newline_output_contract"],
        "entrypoint_function_contracts": ["empty_input_boundary", "type_guard_boundary"],
        "algorithmic_planning_contracts": ["order_or_graph_invariance", "edge_boundary_cases"],
        "return_type_shape_contracts": ["exact_return_shape", "malformed_input_guard"],
        "stateful_edge_contracts": ["state_reset_boundary", "stream_order_sensitivity"],
        "multi_step_tool_contracts": ["pipeline_order_matters", "stable_projection"],
    }
    return common.get(template.family, ["private_contract_generalization"])


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for family in FAMILIES:
        train = [row for row in train_rows if row.get("broad_private_family_v1") == family]
        heldout = [row for row in heldout_rows if row.get("broad_private_family_v1") == family]
        reports.append(
            {
                "family": family,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "categories": sorted({str(row.get("category")) for row in train}),
                "decoder_contract_rows": sum(1 for row in train if isinstance(row.get("decoder_contract"), dict)),
            }
        )
    return reports


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = [
        "humaneval",
        "mbpp",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "canonical_solution",
        "public_test",
        "public prompt",
    ]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
        if len(hits) >= 20:
            break
    return {"hit_count": len(hits), "sample_hits": hits}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Public-Safe Broad Transfer Maturity V4",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Train categories: {summary.get('category_train_count')}",
        f"- Heldout categories: {summary.get('category_heldout_count')}",
        f"- Train solution failures: {summary.get('private_train_solution_failures')}",
        f"- Heldout solution failures: {summary.get('private_heldout_solution_failures')}",
        f"- Public-data leakage hits: {summary.get('public_data_leakage_hit_count')}",
        "",
        "## Families",
    ]
    for row in report.get("families", []):
        lines.append(
            f"- `{row.get('family')}`: train {row.get('train_rows')}, heldout {row.get('heldout_rows')}, categories {len(row.get('categories') or [])}"
        )
    lines.extend(["", "No public benchmark prompts, tests, solutions, score labels, or task ids are used."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
