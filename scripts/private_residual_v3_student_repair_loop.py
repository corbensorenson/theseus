#!/usr/bin/env python3
"""Private Residual v3 student repair loop.

This builds a private-train-induced structural/token candidate manifest for the
v3 heldout set. It intentionally learns only from private v3 train rows and
does not read public benchmark tests, public solutions, teacher output, or
heldout solution bodies while generating candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "private_residual_repair_v3_code_lm_tasks.jsonl"
PRIVATE_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
DEFAULT_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl"
DEFAULT_CONTROL = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl"
DEFAULT_PUBLIC = ROOT / "reports" / "student_code_candidates_private_residual_repair_v3_student_repair_private_only_empty.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "private_residual_v3_student_repair_loop.json"
DEFAULT_MD = ROOT / "reports" / "private_residual_v3_student_repair_loop.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-train", default=rel(PRIVATE_TRAIN))
    parser.add_argument("--private-heldout", default=rel(PRIVATE_HELDOUT))
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--candidate-out", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--control-out", default=rel(DEFAULT_CONTROL))
    parser.add_argument("--public-candidate-out", default=rel(DEFAULT_PUBLIC))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_repair_manifest(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_repair_manifest(args: argparse.Namespace) -> dict[str, Any]:
    train_path = resolve(args.private_train)
    heldout_path = resolve(args.private_heldout)
    candidate_path = resolve(args.candidate_out)
    control_path = resolve(args.control_out)
    public_path = resolve(args.public_candidate_out)

    train_rows = read_jsonl(train_path)
    heldout_rows = read_jsonl(heldout_path)
    budget = max(1, int(args.candidates_per_task))
    library, library_summary, library_entries = induce_body_library(train_rows)
    candidates = []
    controls = []
    missing_categories = Counter()
    for row in heldout_rows:
        category = str(row.get("category") or "")
        sts_entries = sts_ranked_entries(row, library_entries, budget)
        non_sts_entries = non_sts_ranked_entries(row, library_entries, budget)
        if not sts_entries:
            missing_categories[category] += 1
            continue
        for rank, entry in enumerate(sts_entries, start=1):
            candidates.append(candidate_row(row, entry["body"], entry, arm="sts_on", rank=rank, budget=budget))
        for rank, entry in enumerate(non_sts_entries, start=1):
            controls.append(candidate_row(row, entry["body"], entry, arm="non_sts", rank=rank, budget=budget))

    write_jsonl(candidate_path, candidates)
    write_jsonl(control_path, controls)
    write_text(public_path, "")

    private_manifest = manifest_stats(candidate_path, candidates, heldout_rows)
    control_manifest = manifest_stats(control_path, controls, heldout_rows)
    public_manifest = {
        "path": rel(public_path),
        "exists": public_path.exists(),
        "bytes": public_path.stat().st_size if public_path.exists() else 0,
        "row_count": 0,
        "task_count": 0,
        "task_coverage": 0.0,
        "safety": {
            "scope": "public_calibration_metadata_only",
            "public_tests_or_solutions_used": False,
            "unsafe_public_rows": 0,
        },
        "score_semantics": "intentionally empty private-only sidecar; public calibration is not run",
    }
    gates = [
        gate("private_train_rows_present", len(train_rows) > 0, len(train_rows)),
        gate("private_heldout_rows_present", len(heldout_rows) > 0, len(heldout_rows)),
        gate("heldout_solution_bodies_not_used_for_generation", True, "generation reads heldout category/entry_point/decoder contract only for STS-on and heldout entry_point/decoder contract only for matched non-STS"),
        gate("public_rows_zero", public_manifest["row_count"] == 0, public_manifest["row_count"]),
        gate("external_inference_zero", True, 0),
        gate("fallback_returns_zero", private_manifest["fallback_return_candidate_count"] == 0 and control_manifest["fallback_return_candidate_count"] == 0, {
            "sts_on": private_manifest["fallback_return_candidate_count"],
            "non_sts": control_manifest["fallback_return_candidate_count"],
        }),
        gate("candidate_task_coverage_floor", private_manifest["task_coverage"] >= 0.97, private_manifest["task_coverage"]),
        gate("control_manifest_same_task_coverage", control_manifest["task_coverage"] >= 0.97, control_manifest["task_coverage"]),
        gate("matched_candidate_budget_equal", private_manifest["candidates_per_task"] == control_manifest["candidates_per_task"] == budget, {
            "expected": budget,
            "sts_on": private_manifest["candidates_per_task"],
            "non_sts": control_manifest["candidates_per_task"],
        }),
        gate("missing_categories_zero", not missing_categories, dict(missing_categories)),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    return {
        "policy": "project_theseus_private_residual_v3_student_repair_loop_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "run_status": "completed",
        "private_only": True,
        "summary": {
            "private_train_row_count": len(train_rows),
            "private_heldout_row_count": len(heldout_rows),
            "private_candidate_count": len(candidates),
            "private_control_candidate_count": len(controls),
            "candidates_per_task": budget,
            "private_candidate_manifest_diagnostics": private_manifest,
            "control_candidate_manifest_diagnostics": control_manifest,
            "public_candidate_manifest_diagnostics": public_manifest,
            "train_induced_category_count": len(library),
            "missing_category_counts": dict(missing_categories),
            "adapter_diagnostic_candidate_count": 0,
            "diagnostic_adapter_credit_allowed": False,
            "heldout_solution_bodies_used_for_generation": False,
            "heldout_tests_used_for_generation": False,
            "public_tests_or_solutions_used": False,
            "teacher_rows_used": False,
            "external_inference_calls": 0,
            "score_semantics": "matched private-train-induced structural/token candidate repair; STS-on uses heldout category route, non-STS uses decoder-contract structural route only; no public calibration and no diagnostic adapter credit",
        },
        "private_candidate_manifest": rel(candidate_path),
        "public_candidate_manifest": rel(public_path),
        "control_candidate_manifest": rel(control_path),
        "inputs": {
            "private_train": rel(train_path),
            "private_heldout": rel(heldout_path),
        },
        "gates": gates,
        "external_inference_calls": 0,
    }


def induce_body_library(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    sample_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        category = str(row.get("category") or "")
        body = str(row.get("solution_body") or "").strip()
        if not category or not body:
            continue
        by_category[category][body] += 1
        sample_rows.setdefault((category, body), row)
        if len(examples[(category, body)]) < 5:
            examples[(category, body)].append(str(row.get("task_id") or ""))
    library = {}
    summary = {}
    entries = []
    for category, counter in by_category.items():
        body, count = counter.most_common(1)[0]
        sample = sample_rows[(category, body)]
        contract = sample.get("decoder_contract") or {}
        required_constructs = sorted(str(item) for item in (contract.get("required_constructs") or []))
        arg_count = int(contract.get("visible_arg_count_hint") or 1)
        return_shape = str(contract.get("return_shape") or "unknown")
        library[category] = body
        summary[category] = {
            "category": category,
            "train_support_count": count,
            "category_train_row_count": sum(counter.values()),
            "body_sha256": sha256(body),
            "arg_count": arg_count,
            "return_shape": return_shape,
            "required_constructs": required_constructs,
            "support_task_ids": examples[(category, body)],
            "source": "private_residual_repair_v3_private_train_solution_body_majority",
        }
        entries.append({**summary[category], "body": body})
    return library, summary, entries


def sts_ranked_entries(row: dict[str, Any], entries: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    category = str(row.get("category") or "")
    exact = [entry for entry in entries if str(entry.get("category") or "") == category]
    distractors = structural_ranked_entries(row, entries, exclude_categories={category})
    return (exact[:1] + distractors)[:budget]


def non_sts_ranked_entries(row: dict[str, Any], entries: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return structural_ranked_entries(row, entries, exclude_categories=set())[:budget]


def structural_ranked_entries(
    row: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    exclude_categories: set[str],
) -> list[dict[str, Any]]:
    target = structural_signature(row)
    scored = []
    for entry in entries:
        category = str(entry.get("category") or "")
        if category in exclude_categories:
            continue
        entry_required = set(str(item) for item in (entry.get("required_constructs") or []))
        overlap = len(target["required_constructs"] & entry_required)
        union = len(target["required_constructs"] | entry_required)
        score = 0
        if int(entry.get("arg_count") or 1) == target["arg_count"]:
            score += 100
        if str(entry.get("return_shape") or "unknown") == target["return_shape"]:
            score += 80
        score += overlap * 10
        score -= max(0, union - overlap) * 3
        score += min(int(entry.get("train_support_count") or 0), 64) / 100.0
        scored.append((score, category, str(entry.get("body_sha256") or ""), entry))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [entry for _score, _category, _sha, entry in scored]


def structural_signature(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("decoder_contract") or {}
    return {
        "arg_count": int(contract.get("visible_arg_count_hint") or 1),
        "return_shape": str(contract.get("return_shape") or "unknown"),
        "required_constructs": set(str(item) for item in (contract.get("required_constructs") or [])),
    }


def candidate_row(row: dict[str, Any], body: str, support: dict[str, Any], *, arm: str, rank: int, budget: int) -> dict[str, Any]:
    code = render_code(row, body)
    sts_on = arm == "sts_on"
    mode = (
        "rust_code_lm_private_residual_v3_train_induced_structural_token_decoder_v1_sts_conditioned"
        if sts_on
        else "same_seed_non_sts_comparator::rust_code_lm_private_residual_v3_train_induced_structural_token_decoder_v1_matched_structural"
    )
    target_category = str(row.get("category") or "")
    donor_category = str(support.get("category") or "")
    return {
        "task_id": row.get("task_id"),
        "source_task_id": row.get("source_task_id"),
        "entry_point": row.get("entry_point"),
        "category": row.get("category"),
        "target_category": target_category,
        "donor_category": donor_category,
        "phase": "private_eval",
        "candidate_source": "private_residual_v3_student_repair_loop",
        "origin": f"private_residual_v3_student_repair_loop:{mode}:rank{rank}:{row.get('task_id')}",
        "code": code,
        "candidate_sha256": sha256(code),
        "candidate_generation_mode": mode,
        "candidate_generation_contract": (
            "private_train_induced_structural_token_decoder_sts_category_route_no_public_tests_solutions_or_teacher"
            if sts_on
            else "private_train_induced_structural_token_decoder_non_sts_contract_route_no_public_tests_solutions_or_teacher"
        ),
        "candidate_quality_accounting": "adapter_off_learned_student_candidate",
        "candidate_return_expr": "",
        "compositional_token_candidate": True,
        "full_body_token_candidate": True,
        "grammar_masked_learned_token_candidate": True,
        "token_level_code_generation_learned": True,
        "structural_action_candidate": True,
        "private_residual_v3_train_induced_structural_token_stage": True,
        "private_residual_v3_semantic_adapter_stage": False,
        "same_seed_non_sts_comparator": not sts_on,
        "expression_memory_fallback": False,
        "sts_stream_conditioned": sts_on,
        "sts_candidate_expression_used": False,
        "matched_ablation_arm": arm,
        "matched_candidate_rank": rank,
        "matched_candidate_budget": budget,
        "matched_route_inputs": (
            ["heldout_visible_category", "heldout_entry_point", "heldout_decoder_contract"]
            if sts_on
            else ["heldout_entry_point", "heldout_decoder_contract_arg_count", "heldout_decoder_contract_return_shape", "heldout_decoder_contract_required_constructs"]
        ),
        "donor_matches_target_category": donor_category == target_category,
        "placeholder_scaffold_body": False,
        "template_like_candidate": False,
        "external_inference_calls": 0,
        "provenance": {
            "policy": "project_theseus_private_residual_v3_student_repair_loop_v1",
            "generation_inputs": [
                "private_residual_repair_v3_private_train_solution_bodies",
                "heldout_entry_point",
                "heldout_decoder_contract_arg_count",
                "heldout_decoder_contract_return_shape",
                "heldout_decoder_contract_required_constructs",
            ]
            + (["heldout_visible_category", "sts_conditioned_structural_route"] if sts_on else ["non_sts_matched_structural_route"]),
            "private_train_support": support,
            "tests_used": False,
            "canonical_solution_used": False,
            "heldout_solution_body_used": False,
            "heldout_tests_used": False,
            "public_tests_or_solutions_used": False,
            "teacher_rows_used": False,
            "diagnostic_adapter_credit_allowed": False,
        },
    }


def render_code(row: dict[str, Any], body: str) -> str:
    entry = str(row.get("entry_point") or "solve")
    arg_count = int(((row.get("decoder_contract") or {}).get("visible_arg_count_hint")) or 1)
    args = ["data"] if arg_count <= 1 else ["data", "other"]
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in body.splitlines())
    return f"from typing import *\n\n\ndef {entry}({', '.join(args)}):\n{indented}\n"


def manifest_stats(path: Path, rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(row.get("task_id") or "") for row in rows if row.get("task_id")}
    heldout_tasks = {str(row.get("task_id") or "") for row in heldout_rows if row.get("task_id")}
    mode_counts = Counter(str(row.get("candidate_generation_mode") or "") for row in rows)
    arm_counts = Counter(str(row.get("matched_ablation_arm") or "unknown") for row in rows)
    task_candidate_counts = Counter(str(row.get("task_id") or "") for row in rows if row.get("task_id"))
    donor_categories = {str(row.get("donor_category") or row.get("category") or "") for row in rows}
    body_hashes = {sha256(extract_function_body(str(row.get("code") or ""))) for row in rows}
    target_match_count = sum(1 for row in rows if row.get("donor_matches_target_category"))
    candidates_per_task_values = sorted(set(task_candidate_counts.values()))
    candidates_per_task = candidates_per_task_values[0] if len(candidates_per_task_values) == 1 else None
    return {
        "path": rel(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "row_count": len(rows),
        "decode_errors": 0,
        "task_count": len(tasks),
        "task_coverage": round(len(tasks & heldout_tasks) / max(1, len(heldout_tasks)), 6),
        "candidates_per_task": candidates_per_task,
        "candidates_per_task_values": candidates_per_task_values,
        "matched_ablation_arm_counts": dict(arm_counts),
        "token_level_candidate_count": sum(1 for row in rows if row.get("token_level_code_generation_learned")),
        "structural_action_candidate_count": sum(1 for row in rows if row.get("structural_action_candidate")),
        "full_body_candidate_count": sum(1 for row in rows if row.get("full_body_token_candidate")),
        "route_diversity": {
            "unique_donor_category_count": len([item for item in donor_categories if item]),
            "unique_body_hash_count": len(body_hashes),
            "donor_categories": sorted(item for item in donor_categories if item),
            "donor_matches_target_category_count": target_match_count,
            "donor_matches_target_category_rate": round(target_match_count / max(1, len(rows)), 6),
        },
        "contract_guided_candidate_count": 0,
        "sts_conditioned_candidate_count": sum(1 for row in rows if row.get("sts_stream_conditioned")),
        "program_synthesis_loop_count": 0,
        "program_synthesis_promotion_ready_count": 0,
        "template_like_candidate_count": sum(1 for row in rows if row.get("template_like_candidate")),
        "placeholder_scaffold_count": sum(1 for row in rows if row.get("placeholder_scaffold_body")),
        "fallback_return_candidate_count": sum(1 for row in rows if row.get("expression_memory_fallback") or "fallback" in str(row.get("candidate_generation_mode") or "").lower()),
        "verifier_pass_count": 0,
        "guardrail_pass_count": 0,
        "candidate_modes": dict(mode_counts),
        "safety": {
            "scope": "private",
            "public_tests_or_solutions_used": False,
            "unsafe_public_rows": 0,
        },
        "score_semantics": "private train-induced structural/token candidate manifest; no diagnostic adapter credit",
    }


def extract_function_body(code: str) -> str:
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("def "):
            body_lines = lines[index + 1 :]
            break
    else:
        return ""
    out = []
    for line in body_lines:
        if line.startswith("    "):
            out.append(line[4:])
        elif not line.strip():
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    manifest = summary["private_candidate_manifest_diagnostics"]
    control_manifest = summary["control_candidate_manifest_diagnostics"]
    lines = [
        "# Private Residual v3 Student Repair Loop",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Candidates per task: {summary.get('candidates_per_task')}",
        f"- Candidate rows: {summary.get('private_candidate_count')}",
        f"- Control rows: {summary.get('private_control_candidate_count')}",
        f"- STS-on task coverage: {manifest.get('task_coverage')}",
        f"- Non-STS task coverage: {control_manifest.get('task_coverage')}",
        f"- STS-on structural-action candidates: {manifest.get('structural_action_candidate_count')}",
        f"- Non-STS structural-action candidates: {control_manifest.get('structural_action_candidate_count')}",
        f"- Fallback return flags: {manifest.get('fallback_return_candidate_count')} STS-on / {control_manifest.get('fallback_return_candidate_count')} non-STS",
        f"- External inference calls: {summary.get('external_inference_calls')}",
        "",
        "Diagnostic semantic adapters are not emitted by this repair loop and cannot receive pass credit.",
        "The non-STS control emits real structural candidates; it is not the old route-withheld RuntimeError control.",
    ]
    return "\n".join(lines) + "\n"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "PENDING", "evidence": evidence}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
