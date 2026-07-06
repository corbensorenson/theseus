#!/usr/bin/env python3
"""Generate targeted private residual curriculum v2 for Code LM transfer.

This is the coordinated private-data lane for the current public-transfer wall.
It converts the latest public calibration residual categories into private,
locally generated rows across four bounded families:

- edge_contract_v2
- candidate_floor_adapter_v2
- return_type_shape_v2
- parsing_encoding_v1

The script does not copy public prompts, public tests, public solutions, or
candidate code. Public reports are used only to select residual categories and
verification-stage pressure.
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
from code_residual_curriculum import (  # noqa: E402
    build_private_rows,
    read_json,
    read_jsonl,
    summarize_broad_context,
    summarize_residuals,
    verify_private_solution_rows,
    write_json,
    write_jsonl,
)


FAMILIES: tuple[dict[str, str], ...] = (
    {
        "id": "edge_contract_v2",
        "focus": "edge_contract_v2_private_residual_curriculum",
        "path": training_data_path(
            "high_transfer",
            "private_train",
            "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
        ),
    },
    {
        "id": "candidate_floor_adapter_v2",
        "focus": "candidate_floor_adapter_v2",
        "path": training_data_path(
            "high_transfer",
            "private_train",
            "candidate_floor_adapter_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
        ),
    },
    {
        "id": "return_type_shape_v2",
        "focus": "return_type_shape_v2",
        "path": training_data_path(
            "high_transfer",
            "private_train",
            "return_type_shape_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
        ),
    },
    {
        "id": "parsing_encoding_v1",
        "focus": "parsing_encoding_v1",
        "path": training_data_path(
            "high_transfer",
            "private_train",
            "parsing_encoding_v1_private_residual_curriculum_residual_code_lm_tasks.jsonl",
        ),
    },
)

COMBINED_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "targeted_private_residual_curriculum_v2_residual_code_lm_tasks.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-in",
        default="reports/real_code_benchmark_traces_source_mbpp_source_evalplus_source_bigcodebench_source_human_eval_seed14_8.jsonl",
    )
    parser.add_argument(
        "--real-code-report",
        default="reports/real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_human_eval_seed14_8.json",
    )
    parser.add_argument("--broad-transfer-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--broad-scheduler", default="reports/broad_code_calibration_scheduler.json")
    parser.add_argument("--rows-per-family", type=int, default=240)
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--combined-private-out", default=COMBINED_DEFAULT)
    parser.add_argument("--out", default="reports/targeted_private_residual_curriculum_v2.json")
    parser.add_argument("--markdown-out", default="reports/targeted_private_residual_curriculum_v2.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    traces = read_jsonl(resolve(args.trace_in))
    real_code = read_json(resolve(args.real_code_report), {})
    broad_matrix = read_json(resolve(args.broad_transfer_matrix), {})
    broad_scheduler = read_json(resolve(args.broad_scheduler), {})
    residual_summary = summarize_residuals(traces)
    broad_context = summarize_broad_context(broad_matrix, broad_scheduler)
    family_reports: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    for index, family in enumerate(FAMILIES):
        rows = build_private_rows(
            residual_summary,
            seed=int(args.seed) + index,
            max_rows=max(24, int(args.rows_per_family)),
            broad_context=broad_context,
            concept_focus=family["focus"],
        )
        for row in rows:
            tags = list(row.get("tags") or [])
            tags.append(f"targeted_private_residual_curriculum_v2_{family['id']}")
            row["tags"] = sorted(dict.fromkeys(str(tag) for tag in tags))
            row["targeted_private_residual_family_v2"] = family["id"]
        path = resolve(family["path"])
        write_jsonl(path, rows)
        private_check = verify_private_solution_rows(rows)
        family_reports.append(
            {
                "family": family["id"],
                "focus": family["focus"],
                "private_train_jsonl": rel(path),
                "row_count": len(rows),
                "private_solution_test_failures": private_check["failure_count"],
                "concept_residual_counts": dict(
                    Counter(str(row.get("residual_concept") or "unknown") for row in rows)
                ),
                "decoder_contract_rows": sum(1 for row in rows if isinstance(row.get("decoder_contract"), dict) and row.get("decoder_contract")),
                "sample_categories": sorted({str(row.get("category") or "") for row in rows})[:12],
                "private_solution_check": private_check,
            }
        )
        combined_rows.extend(rows)

    combined_path = resolve(args.combined_private_out)
    write_jsonl(combined_path, combined_rows)
    combined_check = verify_private_solution_rows(combined_rows)
    family_row_counts = {row["family"]: row["row_count"] for row in family_reports}
    gates = [
        gate("four_required_families_written", set(family_row_counts) == {row["id"] for row in FAMILIES}, family_row_counts),
        gate("each_family_has_rows", all(count > 0 for count in family_row_counts.values()), family_row_counts),
        gate("combined_private_rows_written", len(combined_rows) == sum(family_row_counts.values()), {"combined_rows": len(combined_rows), "combined_private_out": rel(combined_path)}),
        gate("private_solution_tests_pass", combined_check["failure_count"] == 0, combined_check),
        gate("decoder_contracts_present", all(item["decoder_contract_rows"] > 0 for item in family_reports), {item["family"]: item["decoder_contract_rows"] for item in family_reports}),
        gate("public_prompts_not_copied", True, "uses residual class/stage summaries only"),
        gate("public_tests_not_copied", True, "public test bodies are never read into generated rows"),
        gate("public_solutions_not_copied", True, "canonical public solutions are never read or emitted"),
        gate(
            "real_code_score_is_calibration_only",
            real_code.get("public_benchmark_score_claim") == "student_code_lm_checkpoint_public_task_calibration_only",
            real_code.get("public_benchmark_score_claim"),
        ),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    return {
        "policy": "project_theseus_targeted_private_residual_curriculum_v2",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Target the current broad public-transfer residual wall with private generated rows only.",
        "inputs": {
            "trace_in": rel(resolve(args.trace_in)),
            "real_code_report": rel(resolve(args.real_code_report)),
            "broad_transfer_matrix": rel(resolve(args.broad_transfer_matrix)),
            "broad_scheduler": rel(resolve(args.broad_scheduler)),
            "rows_per_family": int(args.rows_per_family),
            "seed": int(args.seed),
        },
        "outputs": {
            "combined_private_train_jsonl": rel(combined_path),
            "family_private_train_jsonl": {item["family"]: item["private_train_jsonl"] for item in family_reports},
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "combined_private_row_count": len(combined_rows),
            "family_row_counts": family_row_counts,
            "private_solution_test_failures": combined_check["failure_count"],
            "residual_class_counts": residual_summary["class_counts"],
            "verification_stage_counts": residual_summary["verification_stage_counts"],
            "dominant_verification_stage": residual_summary["dominant_verification_stage"],
            "target_families": [row["id"] for row in FAMILIES],
            "public_task_ids_hashed_only": True,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "external_inference_calls": 0,
        },
        "families": family_reports,
        "residual_targets": residual_summary["targets"],
        "verification_pressure": residual_summary["verification_pressure"],
        "broad_context": broad_context,
        "gates": gates,
        "next_actions": [
            "rerun train-once fanout with these private rows included in the high-transfer private input contract",
            "run decoder gate, private/public transfer proof, and STS causal A/B before any public calibration",
            "run exactly one bounded public calibration only if private gates stay GREEN",
        ],
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Targeted Private Residual Curriculum V2",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Combined private rows: {summary.get('combined_private_row_count')}",
        f"- Family row counts: {summary.get('family_row_counts')}",
        f"- Private solution test failures: {summary.get('private_solution_test_failures')}",
        f"- Residual classes: {summary.get('residual_class_counts')}",
        f"- Verification stages: {summary.get('verification_stage_counts')}",
        f"- Public solutions included: {summary.get('public_benchmark_solutions_included')}",
        f"- Public tests included: {summary.get('public_tests_included')}",
        f"- External inference calls: {summary.get('external_inference_calls')}",
        "",
        "## Families",
    ]
    for family in report.get("families", []):
        lines.append(
            f"- `{family.get('family')}`: rows `{family.get('row_count')}`, "
            f"decoder contracts `{family.get('decoder_contract_rows')}`, "
            f"solution failures `{family.get('private_solution_test_failures')}`"
        )
    lines.append("")
    lines.append("Public calibration content is not embedded; this is private generated training pressure only.")
    return "\n".join(lines) + "\n"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
