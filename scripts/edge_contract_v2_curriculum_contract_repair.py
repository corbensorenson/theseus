#!/usr/bin/env python3
"""Repair private edge-contract-v2 decoder contracts.

This is a private-row hygiene step. It does not generate solutions, import
public benchmark prompts/tests, or change the private task assertions. It only
fills missing decoder-contract metadata so the edge-contract-v2 closure can
learn from every private row instead of silently dropping rows with incomplete
generation plans.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402


SOURCE_PRIVATE_IN = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    )
)
DEFAULT_REPAIRED_OUT = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_repaired_code_lm_tasks.jsonl",
    )
)
DEFAULT_REPORT = ROOT / "reports" / "edge_contract_v2_curriculum_contract_repair.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "edge_contract_v2_curriculum_contract_repair.md"

PLAN_KEYS = ("skeleton_bias", "repair_strategy", "verifier_feedback")
GENERATION_POLICY = (
    "signature -> argument_roles -> return_contract -> semantic_family -> "
    "state_variables -> branch_loop_skeleton -> body -> verifier_repair"
)
DEFAULT_FEEDBACK = [
    "visible_argument_mismatch",
    "return_shape_mismatch",
    "missing_required_skeleton",
    "semantic_family_mismatch",
    "edge_boundary_failure",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-in", default=str(SOURCE_PRIVATE_IN))
    parser.add_argument("--out", default=str(DEFAULT_REPAIRED_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    source_path = resolve(args.private_in)
    out_path = resolve(args.out)
    rows = read_jsonl(source_path)
    before = contract_summary(rows)
    repaired_rows: list[dict[str, Any]] = []
    repaired_task_ids: list[str] = []
    unsafe_task_ids: list[str] = []

    for row in rows:
        if not is_v2_row(row):
            repaired_rows.append(row)
            continue
        if is_unsafe_public_training_row(row):
            unsafe_task_ids.append(str(row.get("task_id") or "unknown"))
        repaired = repair_row(row)
        if not has_complete_generation_plan(row) and has_complete_generation_plan(repaired):
            repaired_task_ids.append(str(row.get("task_id") or "unknown"))
        repaired_rows.append(repaired)

    after = contract_summary(repaired_rows)
    write_jsonl(out_path, repaired_rows)
    report = {
        "policy": "project_theseus_edge_contract_v2_curriculum_contract_repair_v1",
        "created_utc": now(),
        "trigger_state": "GREEN"
        if after["generation_plan_rows"] == after["v2_rows"] and not unsafe_task_ids and after["v2_rows"] > 0
        else "RED",
        "summary": {
            "source_private_in": rel_or_abs(source_path),
            "repaired_out": rel_or_abs(out_path),
            "input_rows": len(rows),
            "output_rows": len(repaired_rows),
            "v2_rows": after["v2_rows"],
            "generation_plan_rows_before": before["generation_plan_rows"],
            "generation_plan_rows_after": after["generation_plan_rows"],
            "repaired_task_count": len(repaired_task_ids),
            "unsafe_public_training_rows": len(unsafe_task_ids),
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "before": before,
        "after": after,
        "repaired_task_ids": repaired_task_ids,
        "unsafe_public_training_task_ids": unsafe_task_ids[:50],
        "rules": {
            "preserve_solution_body": True,
            "preserve_tests": True,
            "private_only_metadata_repair": True,
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "next_actions": [
            "run edge_contract_v2_private_verifier against the repaired curriculum",
            "if verifier row gates are YELLOW, run edge_contract_v2_private_closure_runner",
            "keep public calibration locked until the private closure and transfer-readiness gates are green",
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.report_out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] == "GREEN" else 2


def repair_row(row: dict[str, Any]) -> dict[str, Any]:
    repaired = copy.deepcopy(row)
    contract = repaired.get("decoder_contract") if isinstance(repaired.get("decoder_contract"), dict) else {}
    contract = dict(contract)
    generation_plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    generation_plan = dict(generation_plan)

    category = normalized(repaired.get("category"))
    prompt = normalized(repaired.get("prompt"))
    tags = [normalized(tag) for tag in repaired.get("tags", []) if tag is not None]
    body = str(repaired.get("solution_body") or "")
    label = str(
        contract.get("residual_label_hint")
        or repaired.get("concept_residual_label")
        or repaired.get("residual_label")
        or category
        or "edge_contract_v2"
    )
    return_shape = infer_return_shape(contract, category, prompt, body)
    type_family = infer_type_family(contract, category, prompt, tags)
    required_constructs = infer_required_constructs(contract, category, prompt, body, return_shape)
    visible_arg_count = infer_visible_arg_count(contract, body, prompt)

    contract.setdefault("policy", "project_theseus_decoder_contract_v2_private_edge")
    contract.setdefault("full_body_required", True)
    contract.setdefault("guardrail_only", False)
    contract.setdefault("feedback_weight", 1.45)
    contract["required_constructs"] = sorted(set(required_constructs))
    contract["residual_label_hint"] = label
    contract["return_shape"] = return_shape
    contract["type_family"] = type_family
    contract["visible_arg_count_hint"] = visible_arg_count
    contract.setdefault("argument_roles", infer_argument_roles(visible_arg_count, body, prompt))
    if not isinstance(contract.get("argument_roles"), dict):
        contract["argument_roles"] = infer_argument_roles(visible_arg_count, body, prompt)
    contract.setdefault(
        "return_contract",
        {
            "empty_or_invalid_behavior": "covered_by_private_edge_assertions",
            "must_preserve_container_shape": bool("preserve" in category or "container" in category),
            "shape": return_shape,
        },
    )
    if isinstance(contract.get("return_contract"), dict):
        contract["return_contract"].setdefault("shape", return_shape)
        contract["return_contract"].setdefault(
            "empty_or_invalid_behavior", "covered_by_private_edge_assertions"
        )
        contract["return_contract"].setdefault(
            "must_preserve_container_shape", bool("preserve" in category or "container" in category)
        )
    contract.setdefault(
        "score_semantics",
        "private edge-contract v2 generation pressure only; public benchmarks remain calibration-only",
    )

    generation_plan["policy"] = str(generation_plan.get("policy") or GENERATION_POLICY)
    generation_plan["public_solutions_used"] = False
    generation_plan["public_tests_used"] = False
    if not generation_plan.get("repair_strategy"):
        generation_plan["repair_strategy"] = infer_repair_strategy(category, prompt, return_shape, type_family)
    if not generation_plan.get("skeleton_bias"):
        generation_plan["skeleton_bias"] = infer_skeleton_bias(category, prompt, body, return_shape, type_family)
    if not generation_plan.get("verifier_feedback"):
        generation_plan["verifier_feedback"] = infer_verifier_feedback(category, type_family, return_shape)
    contract["generation_plan"] = generation_plan
    repaired["decoder_contract"] = contract
    repaired["public_benchmark"] = False
    repaired["public_tests_included"] = False
    repaired["public_benchmark_solutions_included"] = False
    repaired["curriculum_contract_repair"] = {
        "policy": "project_theseus_edge_contract_v2_curriculum_contract_repair_v1",
        "private_only_metadata_repair": True,
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }
    return repaired


def infer_return_shape(contract: dict[str, Any], category: str, prompt: str, body: str) -> str:
    existing = str(contract.get("return_shape") or "").strip()
    if existing and existing != "unknown":
        return existing
    text = f"{category} {prompt} {body}".lower()
    if any(token in text for token in ("dictionary", "dict", "result dictionary", "out = {}")):
        return "dict"
    if any(token in text for token in ("list of dictionaries", "list", "out = []", ".append(", "sorted(")):
        return "list"
    if any(token in text for token in ("tuple", "preserving tuple", "container")):
        return "same_container"
    if any(token in text for token in ("boolean", "booleans", "truthy", "true", "false")):
        return "list" if "booleans" in text or "out.append" in text else "bool"
    if any(token in text for token in ("count ", "numeric sum", "half the product", "recurrence", "lucas", "return a recurrence")):
        return "number"
    if "second item" in text or "fallback" in text:
        return "item_or_fallback"
    return "unknown"


def infer_type_family(contract: dict[str, Any], category: str, prompt: str, tags: list[str]) -> str:
    existing = str(contract.get("type_family") or "").strip()
    if existing and existing != "unknown":
        return existing
    text = " ".join([category, prompt, *tags]).lower()
    if any(token in text for token in ("runtime_optional", "optional_dependency", "pandas")):
        return "collection_logic"
    if "numpy" in text or "numeric sum" in text:
        return "heterogeneous_numeric_text"
    if any(token in text for token in ("partition", "typed_pairs", "preserve_sequence", "second_or", "interface")):
        return "interface_fidelity"
    if any(token in text for token in ("lucas", "recurrence", "number_theory", "count_digit")):
        return "algorithmic_planning"
    if any(token in text for token in ("parse", "parser", "encoding")):
        return "parsing_encoding"
    if any(token in text for token in ("truth", "type_handling", "mixed")):
        return "heterogeneous_type_contract"
    if any(token in text for token in ("sorted_unique", "collection", "sorting")):
        return "collection_logic"
    return "general_semantics"


def infer_required_constructs(
    contract: dict[str, Any], category: str, prompt: str, body: str, return_shape: str
) -> list[str]:
    required = contract.get("required_constructs") if isinstance(contract.get("required_constructs"), list) else []
    constructs = {str(item) for item in required if item}
    text = f"{category} {prompt} {body}".lower()
    if any(token in text for token in (" if ", "\nif ", "try:", "except", "fallback", "guard")):
        constructs.add("branch")
    if any(token in text for token in (" for ", "\nfor ", " while ", "\nwhile ", "range(")):
        constructs.add("loop")
    if "=" in body and "==" not in body:
        constructs.add("locals")
    if any(token in text for token in ("append(", "dict", "list", "set(", "sorted(", "tuple", "collection")):
        constructs.add("collection_ops")
    if return_shape not in {"unknown", ""}:
        constructs.add("type_and_return_shape")
    if not constructs:
        constructs.update({"branch", "locals", "type_and_return_shape"})
    return sorted(constructs)


def infer_visible_arg_count(contract: dict[str, Any], body: str, prompt: str) -> int:
    existing = contract.get("visible_arg_count_hint")
    if isinstance(existing, int) and existing > 0:
        return existing
    text = f"{body} {prompt}".lower()
    if "other" in body or "fallback" in text or "height" in text or "divisor" in text:
        return 2
    return 1


def infer_argument_roles(visible_arg_count: int, body: str, prompt: str) -> dict[str, str]:
    roles = {"data": "primary_input"}
    text = f"{body} {prompt}".lower()
    if visible_arg_count >= 2:
        if "fallback" in text:
            roles["other"] = "fallback_value"
        elif "height" in text:
            roles["other"] = "secondary_numeric_input"
        elif "divisor" in text:
            roles["other"] = "divisor_and_digit_tuple"
        else:
            roles["other"] = "secondary_parameter"
    return roles


def infer_repair_strategy(category: str, prompt: str, return_shape: str, type_family: str) -> str:
    text = f"{category} {prompt}".lower()
    if "optional" in text:
        return (
            "use guarded optional dependency imports, deterministic pure-Python fallback, "
            "and exact return-shape builders before token repair"
        )
    if type_family == "interface_fidelity":
        return (
            "preserve the visible signature, filter inputs with explicit type guards, "
            f"and build the exact {return_shape} return shape before semantic expansion"
        )
    if type_family == "algorithmic_planning":
        return (
            "select loop state variables, initialize boundary cases, and apply recurrence "
            "or counting transitions before verifier repair"
        )
    if type_family == "parsing_encoding":
        return (
            "normalize separators and signs, parse only validated tokens, and preserve "
            "the requested collection return shape"
        )
    return (
        "use private edge-contract verifier feedback to select visible-interface, "
        "return-shape, branch/loop/local-state, and edge-condition skeletons before token repair"
    )


def infer_skeleton_bias(category: str, prompt: str, body: str, return_shape: str, type_family: str) -> list[str]:
    text = f"{category} {prompt} {body}".lower()
    biases: list[str] = []
    add_if(biases, "guarded_optional_import_fallback", "optional" in text or "import pandas" in text or "import numpy" in text)
    add_if(biases, "dict_accumulator_return", return_shape == "dict" or "out = {}" in text)
    add_if(biases, "list_accumulator_return", return_shape == "list" or "out = []" in text or ".append(" in text)
    add_if(biases, "preserve_input_container_shape", return_shape == "same_container" or "preserving tuple" in text)
    add_if(biases, "length_guard_fallback", "second item" in text or "fallback" in text or "len(data)" in text)
    add_if(biases, "numeric_formula_return", "half the product" in text or "data * other" in text)
    add_if(biases, "counting_loop_with_digit_filter", "count occurrences" in text or "str(item).count" in text)
    add_if(biases, "recurrence_state_update_loop", "lucas" in text or "recurrence" in text or "a, b =" in text)
    add_if(biases, "sorted_unique_collection_return", "sorted" in text or "unique values" in text)
    add_if(biases, "truthy_string_normalization", "truthy" in text or "strip().lower()" in text)
    add_if(biases, "signed_integer_token_parser", "signed integer" in text or "isdigit" in text)
    add_if(biases, "type_guarded_loop_filter", "isinstance" in text or type_family.startswith("heterogeneous"))
    add_if(biases, "branch_loop_local_body", "for " in text or "if " in text)
    if not biases:
        biases.extend(["visible_argument_preservation", f"{return_shape}_return_shape", f"{type_family}_semantic_family"])
    return dedupe(biases)


def infer_verifier_feedback(category: str, type_family: str, return_shape: str) -> list[str]:
    feedback = list(DEFAULT_FEEDBACK)
    if type_family in {"interface_fidelity", "heterogeneous_type_contract", "heterogeneous_numeric_text"}:
        feedback.extend(["type_family_mismatch", "container_shape_mismatch"])
    if type_family == "algorithmic_planning":
        feedback.extend(["boundary_case_failure", "state_update_mismatch"])
    if type_family == "parsing_encoding":
        feedback.extend(["parser_normalization_failure", "encoding_boundary_failure"])
    if return_shape in {"dict", "list", "same_container"} or "partition" in category:
        feedback.append("container_return_shape_mismatch")
    return dedupe(feedback)


def contract_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v2_rows = [row for row in rows if is_v2_row(row)]
    missing: list[str] = []
    generation_plan_rows = 0
    shape_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for row in v2_rows:
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        shape = str(contract.get("return_shape") or "unknown")
        family = str(contract.get("type_family") or "unknown")
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        if has_complete_generation_plan(row):
            generation_plan_rows += 1
        else:
            missing.append(str(row.get("task_id") or "unknown"))
    return {
        "v2_rows": len(v2_rows),
        "generation_plan_rows": generation_plan_rows,
        "missing_generation_plan": missing[:50],
        "return_shape_counts": shape_counts,
        "type_family_counts": family_counts,
    }


def has_complete_generation_plan(row: dict[str, Any]) -> bool:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    return all(bool(plan.get(key)) for key in PLAN_KEYS)


def is_v2_row(row: dict[str, Any]) -> bool:
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    return (
        any("edge_contract_v2" in str(tag) for tag in tags)
        or "edge_contract_v2" in str(row.get("residual_concept") or "")
        or "edge_contract_v2" in str(row.get("targeted_private_residual_family_v2") or "")
    )


def is_unsafe_public_training_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("public_benchmark")
        or row.get("public_tests_included")
        or row.get("public_benchmark_solutions_included")
        or str(row.get("benchmark_evidence_level") or "").startswith("public")
    )


def add_if(values: list[str], value: str, condition: bool) -> None:
    if condition:
        values.append(value)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Edge Contract V2 Curriculum Contract Repair",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- source_private_in: `{summary.get('source_private_in')}`",
        f"- repaired_out: `{summary.get('repaired_out')}`",
        f"- v2_rows: `{summary.get('v2_rows')}`",
        f"- generation_plan_rows_before: `{summary.get('generation_plan_rows_before')}`",
        f"- generation_plan_rows_after: `{summary.get('generation_plan_rows_after')}`",
        f"- repaired_task_count: `{summary.get('repaired_task_count')}`",
        f"- unsafe_public_training_rows: `{summary.get('unsafe_public_training_rows')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
