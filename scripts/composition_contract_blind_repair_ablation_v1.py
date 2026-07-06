#!/usr/bin/env python3
"""Equal-budget STS/VCM ablation for the composition contract-blind repair.

The VCM-off arm must be generated from a context-stripped task manifest before
this script runs. The STS-off arm re-ranks the same generated candidates with
semantic/intent bonuses removed, preserving the candidate budget and still
allowing genuine structural full-body candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from candidate_floor_v2_private_token_probe import DEFAULT_EVAL, load_private_eval_rows  # noqa: E402
from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402


TARGET_FAMILIES = {
    "parse_signed_ints_then_longest_even_run",
    "parse_signed_ints_then_max_non_adjacent_sum",
    "parse_signed_ints_then_numeric_stats_tuple",
    "stable_dedup_then_rle_encode",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--max-eval-rows", type=int, default=512)
    parser.add_argument("--candidate-budget", type=int, default=8)
    parser.add_argument(
        "--vcm-on-candidates",
        default="reports/composition_contract_blind_repair_v1_probe_seed227_candidates.jsonl",
    )
    parser.add_argument(
        "--vcm-off-candidates",
        default="reports/composition_contract_blind_repair_v1_vcm_off_seed227_candidates.jsonl",
    )
    parser.add_argument("--out", default="reports/composition_contract_blind_repair_v1_sts_vcm_ablation.json")
    parser.add_argument(
        "--markdown-out",
        default="reports/composition_contract_blind_repair_v1_sts_vcm_ablation.md",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    eval_paths = [resolve(path) for path in DEFAULT_EVAL]
    private_rows = load_private_eval_rows(eval_paths, max_rows=int(args.max_eval_rows), seed=int(args.seed))
    vcm_on_candidates = read_jsonl(resolve(args.vcm_on_candidates))
    vcm_off_candidates = read_jsonl(resolve(args.vcm_off_candidates))
    budget = max(1, int(args.candidate_budget))

    arms = {
        "sts_vcm_on": phase_candidates(vcm_on_candidates, phase="private_eval", budget=budget),
        "sts_off_vcm_on_equal_budget": phase_candidates(
            rerank_sts_off(vcm_on_candidates),
            phase="private_eval",
            budget=budget,
        ),
        "sts_on_vcm_off_equal_budget": phase_candidates(vcm_off_candidates, phase="private_eval", budget=budget),
        "sts_off_vcm_off_equal_budget": phase_candidates(
            rerank_sts_off(vcm_off_candidates),
            phase="private_eval",
            budget=budget,
        ),
    }
    arm_reports = {
        name: summarize_arm(private_rows, rows, target_families=TARGET_FAMILIES)
        for name, rows in arms.items()
    }
    on = arm_reports["sts_vcm_on"]
    sts_off = arm_reports["sts_off_vcm_on_equal_budget"]
    vcm_off = arm_reports["sts_on_vcm_off_equal_budget"]
    gates = [
        gate("private_rows_loaded", bool(private_rows), len(private_rows), "hard"),
        gate("vcm_on_candidates_loaded", bool(vcm_on_candidates), len(vcm_on_candidates), "hard"),
        gate("vcm_off_candidates_loaded", bool(vcm_off_candidates), len(vcm_off_candidates), "hard"),
        gate(
            "fallback_template_external_zero",
            all(arm.get("no_cheat_counters", {}).get(key) == 0 for arm in arm_reports.values() for key in ["fallback", "template", "external"]),
            {name: arm.get("no_cheat_counters") for name, arm in arm_reports.items()},
            "hard",
        ),
        gate(
            "sts_on_not_worse_than_sts_off",
            float(on["selected_pass_rate"]) >= float(sts_off["selected_pass_rate"]),
            {"sts_vcm_on": on["selected_pass_rate"], "sts_off_vcm_on": sts_off["selected_pass_rate"]},
            "warning",
        ),
        gate(
            "vcm_on_not_worse_than_vcm_off",
            float(on["selected_pass_rate"]) >= float(vcm_off["selected_pass_rate"]),
            {"sts_vcm_on": on["selected_pass_rate"], "sts_on_vcm_off": vcm_off["selected_pass_rate"]},
            "warning",
        ),
        gate(
            "target_families_repaired_with_vcm_on",
            all(float(on["target_family_pass_rates"].get(family, 0.0)) >= 0.95 for family in TARGET_FAMILIES),
            on["target_family_pass_rates"],
            "hard",
        ),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed else "GREEN"
    return {
        "policy": "project_theseus_composition_contract_blind_repair_sts_vcm_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "seed": int(args.seed),
            "max_eval_rows": int(args.max_eval_rows),
            "candidate_budget": budget,
            "eval_jsonl": [rel(path) for path in eval_paths],
            "vcm_on_candidates": rel(resolve(args.vcm_on_candidates)),
            "vcm_off_candidates": rel(resolve(args.vcm_off_candidates)),
        },
        "summary": {
            "arm_count": len(arm_reports),
            "arms": {name: compact_arm(arm) for name, arm in arm_reports.items()},
            "best_private_public_shaped_selected_pass_rate": on["selected_pass_rate"],
            "best_private_public_shaped_pass_if_any_rate": on["pass_if_any_rate"],
            "fallback_return_candidate_count": on["no_cheat_counters"]["fallback"],
            "template_like_candidate_count": on["no_cheat_counters"]["template"],
            "public_leakage_count": on["no_cheat_counters"]["public_tests_visible"],
            "sts_delta_selected_pass": round(float(on["selected_pass_rate"]) - float(sts_off["selected_pass_rate"]), 6),
            "vcm_delta_selected_pass": round(float(on["selected_pass_rate"]) - float(vcm_off["selected_pass_rate"]), 6),
            "target_families": sorted(TARGET_FAMILIES),
            "score_semantics": (
                "Private-only equal-budget ablation. VCM-off uses a context-stripped generation manifest. "
                "STS-off uses the same candidate pool and budget but removes semantic/intent rank bonuses."
            ),
        },
        "arms": arm_reports,
        "gates": gates,
        "rules": {
            "public_calibration_run": False,
            "public_payload_training": False,
            "tests_visible_to_generator": False,
            "solutions_visible_to_generator": False,
            "fallback_returns_count_for_credit": False,
            "teacher_apply_mode": False,
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def summarize_arm(private_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], *, target_families: set[str]) -> dict[str, Any]:
    private_eval = evaluate_private_candidates(private_rows, candidates) if private_rows and candidates else {}
    pass_rates = private_eval.get("concept_family_pass_rates") if isinstance(private_eval.get("concept_family_pass_rates"), dict) else {}
    task_ids = {str(row.get("task_id") or "") for row in private_rows}
    candidate_task_ids = {str(row.get("task_id") or "") for row in candidates if row.get("full_body_token_candidate") is True}
    family_counts = Counter(str(row.get("residual_concept") or row.get("category") or "unknown") for row in private_rows)
    target_counts = {family: family_counts.get(family, 0) for family in sorted(target_families)}
    return {
        "task_count": len(private_rows),
        "candidate_count": len(candidates),
        "candidate_coverage_rate": ratio(len(candidate_task_ids), len(task_ids)),
        "no_admissible_count": len(task_ids - candidate_task_ids),
        "selected_pass_count": int(private_eval.get("trained_passed") or 0),
        "selected_pass_rate": float(private_eval.get("trained_pass_rate") or 0.0),
        "pass_if_any_count": int(private_eval.get("trained_passed") or 0),
        "pass_if_any_rate": float(private_eval.get("trained_pass_rate") or 0.0),
        "target_family_counts": target_counts,
        "target_family_pass_rates": {family: pass_rates.get(family) for family in sorted(target_families)},
        "all_family_pass_rates": pass_rates,
        "residual_count": int(private_eval.get("residual_count") or 0),
        "private_verification": private_eval.get("private_verification") or {},
        "no_cheat_counters": {
            "fallback": sum(1 for row in candidates if row.get("expression_memory_fallback") is True),
            "template": sum(1 for row in candidates if row.get("template_like_candidate") is True),
            "external": sum(int(row.get("external_inference_calls") or 0) for row in candidates),
            "public_tests_visible": sum(1 for row in candidates if row.get("public_tests_visible_to_generator") is True),
        },
    }


def phase_candidates(rows: list[dict[str, Any]], *, phase: str, budget: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            grouped[task_id].append(row)
    out: list[dict[str, Any]] = []
    for task_id, items in grouped.items():
        ordered = sorted(items, key=lambda row: (int(row.get("candidate_rank") or 999999), str(row.get("candidate_sha256") or "")))
        for rank, row in enumerate(ordered[:budget], start=1):
            item = dict(row)
            item["phase"] = phase
            item["candidate_rank"] = rank
            out.append(item)
    return out


def rerank_sts_off(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            grouped[task_id].append(row)
    out: list[dict[str, Any]] = []
    for task_id, items in grouped.items():
        ordered = sorted(items, key=sts_off_sort_key)
        for rank, row in enumerate(ordered, start=1):
            item = dict(row)
            item["candidate_rank"] = rank
            item["same_seed_non_sts_comparator"] = True
            item["sts_candidate_expression_used"] = False
            item["candidate_selection_policy"] = "structure_return_shape_equal_budget_sts_off_v1"
            out.append(item)
    return out


def sts_off_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    full_required = bool(row.get("candidate_selection_full_required_structures"))
    shape_ok = bool(row.get("prompt_return_shape_compatible"))
    hits = int(row.get("candidate_selection_required_structure_hits") or 0)
    missing = int(row.get("candidate_selection_required_structure_missing") or 0)
    multi = bool(row.get("multi_statement_generated_body"))
    composition = bool(row.get("private_composition_body_candidate"))
    original_rank = int(row.get("candidate_rank") or 999999)
    return (
        -int(full_required),
        missing,
        -int(shape_ok),
        -hits,
        -int(multi),
        -int(composition),
        original_rank,
        str(row.get("candidate_sha256") or ""),
    )


def compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_count": arm.get("task_count"),
        "candidate_count": arm.get("candidate_count"),
        "candidate_coverage_rate": arm.get("candidate_coverage_rate"),
        "no_admissible_count": arm.get("no_admissible_count"),
        "selected_pass_rate": arm.get("selected_pass_rate"),
        "pass_if_any_rate": arm.get("pass_if_any_rate"),
        "target_family_pass_rates": arm.get("target_family_pass_rates"),
        "no_cheat_counters": arm.get("no_cheat_counters"),
    }


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Composition Contract-Blind Repair STS/VCM Ablation v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- sts_delta_selected_pass: `{report.get('summary', {}).get('sts_delta_selected_pass')}`",
        f"- vcm_delta_selected_pass: `{report.get('summary', {}).get('vcm_delta_selected_pass')}`",
        "",
        "## Arms",
    ]
    for name, arm in sorted((report.get("summary", {}).get("arms") or {}).items()):
        lines.append(
            f"- `{name}`: pass `{arm.get('selected_pass_rate')}`, "
            f"coverage `{arm.get('candidate_coverage_rate')}`, no-admissible `{arm.get('no_admissible_count')}`"
        )
    lines.append("")
    lines.append("## Gates")
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
