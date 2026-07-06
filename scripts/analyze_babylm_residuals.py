"""Residual analysis for local BabyLM/BLIMP SymLiquid reports.

This joins a SymLiquid eval report back to the JSONL benchmark metadata and
summarizes failures by field, linguistic term, and rule. It is local-only and
does not call model providers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--eval-input", required=True)
    parser.add_argument("--out", default="reports/babylm_residual_analysis.json")
    parser.add_argument("--min-cases", type=int, default=8)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    eval_rows = read_jsonl(Path(args.eval_input))
    results = report.get("eval", {}).get("results", [])
    row_alignment = "exact"
    if len(results) < len(eval_rows):
        eval_rows = eval_rows[: len(results)]
        row_alignment = "prefix_limited"
    elif len(results) != len(eval_rows):
        raise SystemExit(
            f"result/eval row mismatch: results={len(results)} eval_rows={len(eval_rows)}"
        )

    groups: dict[str, dict[str, dict[str, Any]]] = {
        "field": defaultdict(new_group),
        "linguistics_term": defaultdict(new_group),
        "rule": defaultdict(new_group),
    }
    failures = []
    for idx, (row, result) in enumerate(zip(eval_rows, results)):
        correct = bool(result.get("correct"))
        for group_name in groups:
            key = str(row.get(group_name) or row.get("UID") or "unknown")
            update_group(groups[group_name][key], correct)
        if not correct:
            failures.append(
                {
                    "index": idx,
                    "case_id": result.get("case_id"),
                    "field": row.get("field", "unknown"),
                    "linguistics_term": row.get("linguistics_term", "unknown"),
                    "rule": row.get("rule") or row.get("UID") or "unknown",
                    "expected": result.get("expected"),
                    "output": result.get("output"),
                    "sentence_good": row.get("sentence_good"),
                    "sentence_bad": row.get("sentence_bad"),
                }
            )

    payload = {
        "policy": "local_only_no_external_inference",
        "methodology": "benchmaxxing_residual_analysis",
        "report": args.report,
        "eval_input": args.eval_input,
        "row_alignment": row_alignment,
        "analyzed_rows": len(results),
        "summary": report.get("eval", {}).get("summary", {}),
        "worst_by_field": ranked(groups["field"], args.min_cases, args.limit),
        "worst_by_linguistics_term": ranked(
            groups["linguistics_term"], args.min_cases, args.limit
        ),
        "worst_by_rule": ranked(groups["rule"], args.min_cases, args.limit),
        "failure_examples": failures[: args.limit],
        "recommendation": recommend(groups["rule"], groups["linguistics_term"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def new_group() -> dict[str, Any]:
    return {"cases": 0, "correct": 0}


def update_group(group: dict[str, Any], correct: bool) -> None:
    group["cases"] += 1
    if correct:
        group["correct"] += 1


def ranked(
    groups: dict[str, dict[str, Any]],
    min_cases: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, group in groups.items():
        cases = int(group["cases"])
        if cases < min_cases:
            continue
        correct = int(group["correct"])
        accuracy = correct / max(1, cases)
        rows.append(
            {
                "name": name,
                "cases": cases,
                "correct": correct,
                "accuracy": accuracy,
                "residual": 1.0 - accuracy,
            }
        )
    rows.sort(key=lambda row: (-row["residual"], -row["cases"], row["name"]))
    return rows[:limit]


def recommend(
    rules: dict[str, dict[str, Any]],
    terms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    worst_rules = ranked(rules, min_cases=8, limit=6)
    worst_terms = ranked(terms, min_cases=8, limit=6)
    return {
        "wall_type": "architecture_training_wall",
        "target_rules": [row["name"] for row in worst_rules],
        "target_terms": [row["name"] for row in worst_terms],
        "next_intervention": (
            "Build learned sequence-state features for the highest-residual BLIMP "
            "families, then validate on mutated/private minimal pairs before "
            "treating the public score as real frontier progress."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
