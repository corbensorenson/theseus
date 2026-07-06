#!/usr/bin/env python3
"""Compact ablation report for the canonical transformer/hybrid survival lane."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TRANSFORMER = ROOT / "reports" / "private_candidate_replay_contract_audit_canonical_transformer_hybrid_clean64_v1.json"
DEFAULT_OLD_LEARNED = ROOT / "reports" / "private_candidate_replay_contract_audit_canonical_ablation_old_learned_template_like_clean64_v1.json"
DEFAULT_NGRAM = ROOT / "reports" / "private_candidate_replay_contract_audit_canonical_ablation_private_ngram_clean64_v1.json"
DEFAULT_STRUCTURAL = ROOT / "reports" / "private_candidate_replay_contract_audit_reality_harness_structural_adapter.json"
DEFAULT_OUT = ROOT / "reports" / "canonical_transformer_hybrid_survival_lane_ablation_v1.json"
DEFAULT_MD = ROOT / "reports" / "canonical_transformer_hybrid_survival_lane_ablation_v1.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-report", default=rel(DEFAULT_TRANSFORMER))
    parser.add_argument("--old-learned-report", default=rel(DEFAULT_OLD_LEARNED))
    parser.add_argument("--private-ngram-report", default=rel(DEFAULT_NGRAM))
    parser.add_argument("--structural-report", default=rel(DEFAULT_STRUCTURAL))
    parser.add_argument("--candidate-budget-per-task", type=int, default=8)
    parser.add_argument("--heldout-task-count", type=int, default=64)
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = {
        "transformer_hybrid_survival_lane": summarize_report(
            resolve(args.transformer_report),
            expected_tasks=args.heldout_task_count,
            expected_budget=args.candidate_budget_per_task,
            same_split=True,
            equal_budget=True,
        ),
        "old_learned_template_like": summarize_report(
            resolve(args.old_learned_report),
            expected_tasks=args.heldout_task_count,
            expected_budget=args.candidate_budget_per_task,
            same_split=True,
            equal_budget=True,
        ),
        "private_ngram_body": summarize_report(
            resolve(args.private_ngram_report),
            expected_tasks=args.heldout_task_count,
            expected_budget=args.candidate_budget_per_task,
            same_split=True,
            equal_budget=True,
        ),
        "structural_adapter_historical": summarize_report(
            resolve(args.structural_report),
            expected_tasks=args.heldout_task_count,
            expected_budget=args.candidate_budget_per_task,
            same_split=False,
            equal_budget=False,
        ),
    }
    strict_rows = {
        key: row
        for key, row in rows.items()
        if row["present"] and row["same_heldout_split"] and row["equal_candidate_budget"]
    }
    best_key = max(
        strict_rows,
        key=lambda key: (
            strict_rows[key]["selected_pass_rate"],
            strict_rows[key]["pass_if_any_rate"],
            strict_rows[key]["selected_compile_rate"],
        ),
    )
    gates = [
        gate("strict_same_split_rows_present", set(strict_rows) == {
            "transformer_hybrid_survival_lane",
            "old_learned_template_like",
            "private_ngram_body",
        }, sorted(strict_rows)),
        gate("transformer_hybrid_best_strict_row", best_key == "transformer_hybrid_survival_lane", best_key),
        gate("no_public_calibration", all(row["public_calibration_run"] is False for row in rows.values() if row["present"]), None),
        gate("external_inference_zero", sum(row["external_inference_calls"] for row in rows.values() if row["present"]) == 0, None),
        gate("fallback_and_constant_zero", all(
            row["fallback_return_candidate_count"] == 0
            and row["unconditional_constant_return_candidate_count"] == 0
            for row in strict_rows.values()
        ), None),
    ]
    failed = [row for row in gates if not row["passed"]]
    return {
        "policy": "project_theseus_canonical_survival_lane_ablation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN" if not failed else "YELLOW",
        "inputs": {
            "candidate_budget_per_task": args.candidate_budget_per_task,
            "heldout_task_count": args.heldout_task_count,
            "transformer_report": rel(resolve(args.transformer_report)),
            "old_learned_report": rel(resolve(args.old_learned_report)),
            "private_ngram_report": rel(resolve(args.private_ngram_report)),
            "structural_report": rel(resolve(args.structural_report)),
        },
        "summary": {
            "strict_comparison_rows": sorted(strict_rows),
            "best_strict_row": best_key,
            "strict_equal_budget_semantics": (
                "transformer_hybrid, old_learned_template_like, and private_ngram_body use the same 64-row "
                "private heldout split with candidate budget 8; structural_adapter_historical is reported as "
                "context only because no same-split clean64 structural-only manifest is present"
            ),
            "rows": rows,
        },
        "gates": gates,
        "external_inference_calls": sum(row["external_inference_calls"] for row in rows.values() if row["present"]),
        "public_calibration_run": any(row["public_calibration_run"] for row in rows.values() if row["present"]),
        "score_semantics": "private replay ablation only; no public calibration, no training, no threshold tuning",
    }


def summarize_report(
    path: Path,
    *,
    expected_tasks: int,
    expected_budget: int,
    same_split: bool,
    equal_budget: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": rel(path),
            "present": False,
            "same_heldout_split": same_split,
            "equal_candidate_budget": False,
            "missing_reason": "report_not_found",
        }
    report = json.loads(path.read_text())
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
    task_count = int(summary.get("task_count") or 0)
    candidate_row_count = int(summary.get("candidate_row_count") or 0)
    selected_pass = int(summary.get("selected_intended_behavior_pass_count") or 0)
    pass_if_any = int(summary.get("pass_if_any_count") or 0)
    selected_compile = int(summary.get("selected_compile_pass_count") or 0)
    selected_runtime = int(summary.get("selected_runtime_load_count") or 0)
    return {
        "path": rel(path),
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "same_heldout_split": same_split and task_count == expected_tasks,
        "equal_candidate_budget": bool(equal_budget),
        "requested_candidate_budget_per_task": expected_budget if equal_budget else None,
        "task_count": task_count,
        "candidate_row_count": candidate_row_count,
        "eligible_candidate_count": int(summary.get("eligible_candidate_count") or 0),
        "replayed_candidate_count": int(summary.get("replayed_candidate_count") or 0),
        "selected_compile_count": selected_compile,
        "selected_compile_rate": ratio(selected_compile, task_count),
        "selected_runtime_load_count": selected_runtime,
        "selected_runtime_load_rate": ratio(selected_runtime, task_count),
        "selected_behavior_pass_count": selected_pass,
        "selected_pass_fraction": fraction(selected_pass, task_count),
        "selected_pass_rate": ratio(selected_pass, task_count),
        "pass_if_any_count": pass_if_any,
        "pass_if_any_fraction": fraction(pass_if_any, task_count),
        "pass_if_any_rate": ratio(pass_if_any, task_count),
        "functional_promotion_count": int(summary.get("functional_promotion_count") or 0),
        "functional_promotion_by_family": summary.get("functional_promotion_by_family") or {},
        "candidate_family_counts": summary.get("candidate_family_counts") or {},
        "candidate_integrity_mismatch_count": int(summary.get("candidate_integrity_mismatch_count") or 0),
        "fallback_return_candidate_count": int(summary.get("fallback_return_candidate_count") or 0),
        "unconditional_constant_return_candidate_count": int(summary.get("unconditional_constant_return_candidate_count") or 0),
        "public_boundary_violation_count": int(summary.get("public_boundary_violation_count") or 0),
        "public_calibration_run": bool(report.get("public_calibration_run")),
        "external_inference_calls": int(report.get("external_inference_calls") or 0),
        "family_filter": inputs.get("family_filter", ""),
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["summary"]["rows"]
    lines = [
        "# Canonical Survival Lane Ablation v1",
        "",
        f"State: **{report['trigger_state']}**",
        "",
        report["summary"]["strict_equal_budget_semantics"],
        "",
        "| Lane | Same split | Equal budget | Selected pass | Pass-if-any | Compile | Families |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for key, row in rows.items():
        families = ", ".join(f"{name}:{count}" for name, count in row.get("candidate_family_counts", {}).items())
        lines.append(
            "| {key} | {same} | {budget} | {selected} | {pia} | {compile_rate:.3f} | {families} |".format(
                key=key,
                same=row.get("same_heldout_split"),
                budget=row.get("equal_candidate_budget"),
                selected=row.get("selected_pass_fraction", "n/a"),
                pia=row.get("pass_if_any_fraction", "n/a"),
                compile_rate=float(row.get("selected_compile_rate") or 0.0),
                families=families or "n/a",
            )
        )
    lines.extend([
        "",
        f"Best strict row: `{report['summary']['best_strict_row']}`",
        "",
        "No public calibration was run. No training rows were written.",
    ])
    return "\n".join(lines) + "\n"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def fraction(count: int, denominator: int) -> str:
    return f"{count}/{denominator}" if denominator else "0/0"


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    raise SystemExit(main())
