#!/usr/bin/env python3
"""Generate private-only post-v4 shadow transfer tasks.

This lane runs after the v4 private maturity gate is clean but the next public
calibration is still operator-locked. It uses only residual category labels and
abstract surface names as routing pressure. It must not copy public prompts,
tests, solutions, traces, score labels, or task ids into train/eval rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402
from code_residual_curriculum import verify_private_solution_rows  # noqa: E402
from public_safe_broad_transfer_maturity_v4 import normalize_test_source, v4_template_bank  # noqa: E402


POLICY = "project_theseus_post_v4_private_shadow_transfer_v1"
CARD_ID = "post_v4_private_shadow_transfer_v1"
EVIDENCE_LEVEL = "post_v4_private_shadow_transfer_v1_generated_only"
CONTRACT_POLICY = "project_theseus_decoder_contract_v6_post_v4_private_shadow_transfer"
TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "post_v4_private_shadow_transfer_v1_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "post_v4_private_shadow_transfer_v1_heldout_code_lm_tasks.jsonl",
)
DEFAULT_PACKET = ROOT / "reports" / "public_calibration_readiness_packet.json"
DEFAULT_RESIDUAL = ROOT / "reports" / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_LOCK = ROOT / "reports" / "public_calibration_operator_lock.flag"
DEFAULT_POST_V4_PUBLIC = ROOT / "reports" / "real_code_benchmark_graduation_post_v4_seed23_5x32.json"
DEFAULT_POST_V4_TRACES = ROOT / "reports" / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl"
DEFAULT_POST_V4_CANDIDATES = ROOT / "reports" / "student_code_candidates_post_v4_seed23_5x32.jsonl"
OPERATOR_APPROVAL = ROOT / "reports" / "public_calibration_operator_approval_post_v4_seed23_5x32.json"
OPERATOR_EXECUTE = ROOT / "reports" / "operator_bounded_public_calibration_execute.json"

RESIDUAL_FAMILIES = (
    "verifier_mismatch_shadow",
    "no_admissible_candidate_shadow",
    "return_shape_shadow",
    "algorithmic_planning_shadow",
    "interface_fidelity_shadow",
    "stateful_runtime_shadow",
)

SHADOW_SURFACES = (
    "function_contract_surface",
    "augmented_function_contract_surface",
    "library_style_contract_surface",
    "canonical_function_surface",
    "stdin_style_surface",
)

RESIDUAL_LABELS = (
    "verifier_mismatch",
    "no_admissible_candidate_regression",
    "return_shape",
    "algorithmic_planning",
    "interface_fidelity",
    "stateful_runtime",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=2400)
    parser.add_argument("--heldout-rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument("--packet", default=rel(DEFAULT_PACKET))
    parser.add_argument("--public-residual", default=rel(DEFAULT_RESIDUAL))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_LOCK))
    parser.add_argument("--post-v4-public-result", default=rel(DEFAULT_POST_V4_PUBLIC))
    parser.add_argument("--out", default="reports/post_v4_private_shadow_transfer_v1.json")
    parser.add_argument("--markdown-out", default="reports/post_v4_private_shadow_transfer_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = resolve(args.packet)
    residual_path = resolve(args.public_residual)
    lock_path = resolve(args.operator_lock)
    post_v4_public_path = resolve(args.post_v4_public_result)
    packet = read_json(packet_path, {})
    residual = read_json(residual_path, {})
    residual_summary = object_field(residual, "summary")
    templates = v4_template_bank()
    train_rows = build_rows(
        templates,
        row_count=max(1200, int(args.train_rows)),
        split="train",
        seed=int(args.seed),
        id_offset=0,
    )
    heldout_rows = build_rows(
        templates,
        row_count=max(160, int(args.heldout_rows)),
        split="heldout",
        seed=int(args.seed) + 400_000,
        id_offset=4_000_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    train_check = verify_private_solution_rows(train_rows, max_failures=24)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=24)
    train_families = Counter(str(row.get("broad_private_family_v1")) for row in train_rows)
    heldout_families = Counter(str(row.get("broad_private_family_v1")) for row in heldout_rows)
    train_surfaces = Counter(str(row.get("post_v4_shadow_surface_v1")) for row in train_rows)
    heldout_surfaces = Counter(str(row.get("post_v4_shadow_surface_v1")) for row in heldout_rows)
    train_categories = Counter(str(row.get("category")) for row in train_rows)
    heldout_categories = Counter(str(row.get("category")) for row in heldout_rows)
    leakage = public_leakage_scan(train_rows + heldout_rows)
    preflight = preflight_state(packet, residual_summary, lock_path, post_v4_public_path)
    gates = [
        gate("readiness_packet_green_but_operator_locked", preflight["packet_green_and_locked"], preflight),
        gate("post_v4_public_artifacts_approved_or_absent", preflight["post_v4_public_artifact_state"]["allowed"], preflight["post_v4_public_artifact_state"]),
        gate("private_train_rows_ge_1200", len(train_rows) >= 1200, len(train_rows)),
        gate("private_heldout_rows_ge_160", len(heldout_rows) >= 160, len(heldout_rows)),
        gate("shadow_family_coverage", set(train_families) == set(RESIDUAL_FAMILIES), dict(train_families)),
        gate("shadow_surface_coverage", set(train_surfaces) == set(SHADOW_SURFACES), dict(train_surfaces)),
        gate("heldout_shadow_surface_coverage", set(heldout_surfaces) == set(SHADOW_SURFACES), dict(heldout_surfaces)),
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
        "purpose": "Private-only shadow transfer pressure while the operator-reviewed public run remains locked.",
        "inputs": {
            "seed": int(args.seed),
            "template_count": len(templates),
            "packet": rel(packet_path),
            "public_residual_report": rel(residual_path),
            "operator_lock": rel(lock_path),
            "post_v4_public_result": rel(post_v4_public_path),
            "public_residual_summary_fields_only": True,
            "public_benchmark_inputs_read": False,
            "public_prompts_used": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_traces_used": False,
            "public_score_labels_used": False,
            "post_v4_public_artifact_state": preflight["post_v4_public_artifact_state"],
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
            "family_train_row_counts": dict(sorted(train_families.items())),
            "family_heldout_row_counts": dict(sorted(heldout_families.items())),
            "surface_train_row_counts": dict(sorted(train_surfaces.items())),
            "surface_heldout_row_counts": dict(sorted(heldout_surfaces.items())),
            "category_train_count": len(train_categories),
            "category_heldout_count": len(heldout_categories),
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "public_data_leakage_hit_count": leakage["hit_count"],
            "packet_ready": preflight["packet_ready"],
            "operator_lock_active": preflight["operator_lock_active"],
            "post_v4_public_result_exists": post_v4_public_path.exists(),
            "post_v4_public_artifacts_approved_or_absent": preflight["post_v4_public_artifact_state"]["allowed"],
            "residual_labels_used": list(RESIDUAL_LABELS),
            "shadow_surface_count": len(SHADOW_SURFACES),
            "external_inference_calls": 0,
            "score_semantics": "private synthetic shadow transfer pressure only; not public calibration",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "build the release binary so the v6 private train path is loaded",
            "fan out a 160-task shadow heldout smoke with explicit private-safe STS streams",
            "score against an empty-stream STS-off control before interpreting transfer",
            "keep public calibration locked unless the operator explicitly approves one bounded run",
        ],
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def build_rows(templates: list[Any], *, row_count: int, split: str, seed: int, id_offset: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(row_count):
        template = templates[(index + seed) % len(templates)]
        residual_index = (index + seed) % len(RESIDUAL_LABELS)
        surface_index = (index // max(1, len(RESIDUAL_LABELS)) + seed) % len(SHADOW_SURFACES)
        rows.append(
            row_from_template(
                template,
                split=split,
                task_index=id_offset + index,
                variant=seed + index * 29,
                residual_label=RESIDUAL_LABELS[residual_index],
                family=RESIDUAL_FAMILIES[residual_index],
                surface=SHADOW_SURFACES[surface_index],
            )
        )
    return rows


def row_from_template(
    template: Any,
    *,
    split: str,
    task_index: int,
    variant: int,
    residual_label: str,
    family: str,
    surface: str,
) -> dict[str, Any]:
    base_category = str(template.category).removeprefix("v4_")
    category = f"shadow_{residual_label}_{base_category}"
    semantic_family = f"shadow_{residual_label}_{template.semantic_family or base_category}"
    entry = f"{category}_{task_index:07d}"
    tags = sorted({
        CARD_ID,
        split,
        family,
        surface,
        residual_label,
        base_category,
        *[str(tag) for tag in template.tags if not public_name(str(tag))],
    })
    return {
        "task_id": f"{CARD_ID}_{family}_{surface}_{task_index:07d}",
        "source_task_id": f"{CARD_ID}_{split}_{variant:07d}",
        "card_id": CARD_ID,
        "source_id": f"local_generated_{CARD_ID}",
        "split": "train" if split == "train" else "eval",
        "category": category,
        "prompt": f"Private shadow transfer contract: {template.prompt}",
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template.body,
        "tests": normalize_test_source(template.tests(entry, variant)),
        "tags": tags,
        "broad_private_family_v1": family,
        "post_v4_shadow_surface_v1": surface,
        "post_v4_shadow_residual_v1": residual_label,
        "targeted_private_residual_family_v3": CARD_ID,
        "residual_concept": residual_label,
        "concept_residual_label": category,
        "decoder_contract": decoder_contract(template, residual_label, family, surface, semantic_family),
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
            "family": family,
            "surface": surface,
            "residual_label": residual_label,
            "base_category": base_category,
            "variant": variant,
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_traces_used": False,
            "public_score_labels_used": False,
            "semantics": "private synthetic post-v4 shadow transfer pressure only",
        },
    }


def decoder_contract(template: Any, residual_label: str, family: str, surface: str, semantic_family: str) -> dict[str, Any]:
    return {
        "policy": CONTRACT_POLICY,
        "return_shape": template.return_shape,
        "type_family": template.type_family,
        "semantic_family": semantic_family,
        "visible_arg_count_hint": template.visible_arg_count_hint,
        "required_constructs": list(template.required_constructs),
        "residual_label_hint": residual_label,
        "shadow_surface_hint": surface,
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": 1.9,
        "score_semantics": "private post-v4 shadow transfer pressure only",
        "argument_roles": template.argument_roles or {"data": "primary_input"},
        "return_contract": {
            "shape": template.return_shape,
            "empty_or_invalid_behavior": "covered_by_private_shadow_assertions",
            "must_preserve_container_shape": template.return_shape in {"list", "dict", "tuple"},
        },
        "generation_plan": {
            "policy": "private_shadow_train_body -> reusable_shadow_token_decoder -> heldout_contract_body",
            "skeleton_bias": list(template.required_constructs),
            "repair_strategy": f"learn reusable {family} behavior for {surface}",
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def preflight_state(packet: dict[str, Any], residual_summary: dict[str, Any], lock_path: Path, post_v4_public_path: Path) -> dict[str, Any]:
    packet_ready = bool(
        packet.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and packet.get("mode") == "post_distillation_v4_operator_review"
        and packet.get("trigger_state") == "GREEN"
        and packet.get("technical_ready_for_one_bounded_public_calibration") is True
        and packet.get("public_calibration_allowed") is False
    )
    post_v4_state = post_v4_public_artifact_state(post_v4_public_path, lock_path)
    return {
        "packet_ready": packet_ready,
        "operator_lock_active": lock_path.exists(),
        "post_v4_public_result_exists": post_v4_public_path.exists(),
        "packet_green_and_locked": packet_ready and lock_path.exists() and post_v4_state["allowed"],
        "post_v4_public_artifact_state": post_v4_state,
        "residual_summary_categories": residual_summary.get("adapter_adjusted_dominant_categories")
        or residual_summary.get("dominant_categories")
        or [],
    }


def post_v4_public_artifact_state(post_v4_public_path: Path, lock_path: Path) -> dict[str, Any]:
    post_v4_artifacts = [
        post_v4_public_path,
        DEFAULT_POST_V4_TRACES,
        DEFAULT_POST_V4_CANDIDATES,
        ROOT / "reports" / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
    ]
    present = [rel(path) for path in post_v4_artifacts if path.exists()]
    if not present:
        return {
            "allowed": True,
            "mode": "absent",
            "present_artifacts": [],
            "approval_valid": False,
            "execute_report_valid": False,
            "required_outputs_present": False,
            "operator_lock_active": lock_path.exists(),
        }
    approval = read_json(OPERATOR_APPROVAL, {})
    execute = read_json(OPERATOR_EXECUTE, {})
    execute_summary = object_field(execute, "summary")
    approval_valid = (
        approval.get("policy") == "project_theseus_public_calibration_operator_approval_v1"
        and approval.get("approved") is True
        and approval.get("proposed_slug") == "post_v4_seed23_5x32"
        and int(first_number(approval.get("max_runs"), 0)) == 1
    )
    execute_valid = (
        execute.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and execute.get("trigger_state") == "GREEN"
        and execute_summary.get("executed") is True
        and execute_summary.get("proposed_slug") == "post_v4_seed23_5x32"
        and execute_summary.get("output_exists_after") is True
        and execute_summary.get("operator_lock_present_after") is True
        and int(first_number(execute_summary.get("run_returncode"), -1)) == 0
    )
    required_outputs_present = all(path.exists() for path in (post_v4_public_path, DEFAULT_POST_V4_TRACES, DEFAULT_POST_V4_CANDIDATES))
    allowed = approval_valid and execute_valid and required_outputs_present and lock_path.exists()
    return {
        "allowed": allowed,
        "mode": "approved_spent_one_shot" if allowed else "unapproved_or_incomplete",
        "present_artifacts": present,
        "approval_valid": approval_valid,
        "execute_report_valid": execute_valid,
        "required_outputs_present": required_outputs_present,
        "operator_lock_active": lock_path.exists(),
        "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
    }


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in RESIDUAL_FAMILIES:
        train = [row for row in train_rows if row.get("broad_private_family_v1") == family]
        heldout = [row for row in heldout_rows if row.get("broad_private_family_v1") == family]
        rows.append(
            {
                "family": family,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "surfaces": sorted({str(row.get("post_v4_shadow_surface_v1")) for row in train}),
                "categories": sorted({str(row.get("category")) for row in train}),
            }
        )
    return rows


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


def public_name(value: str) -> bool:
    text = value.lower()
    return any(item in text for item in ("humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench"))


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Post-V4 Private Shadow Transfer V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Shadow surfaces: {summary.get('shadow_surface_count')}",
        f"- Train categories: {summary.get('category_train_count')}",
        f"- Heldout categories: {summary.get('category_heldout_count')}",
        f"- Public-data leakage hits: {summary.get('public_data_leakage_hit_count')}",
        f"- Operator lock active: {summary.get('operator_lock_active')}",
        "",
        "No public benchmark prompts, tests, solutions, traces, score labels, or task ids are used in generated rows.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
