#!/usr/bin/env python3
"""Mine existing code-proposer gap rows for the next private residual pressure.

This is a cheap diagnostic pass over the already-generated private gap report.
It does not create new benchmark suites or run training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAP = ROOT / "reports" / "neural_seed_code_proposer_gap_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-report", default=str(DEFAULT_GAP.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/neural_seed_residual_mining.json")
    parser.add_argument("--markdown-out", default="reports/neural_seed_residual_mining.md")
    args = parser.parse_args()

    report = mine(read_json(resolve(args.gap_report)), args.gap_report)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def mine(gap: dict[str, Any], gap_path: str) -> dict[str, Any]:
    rows = gap.get("task_rows") if isinstance(gap.get("task_rows"), list) else []
    sym_only = [row for row in rows if row.get("gap_status") == "symliquid_only_win"]
    transformer_only = [row for row in rows if row.get("gap_status") == "transformer_only_win"]
    both_fail = [row for row in rows if row.get("gap_status") == "both_fail"]
    report = {
        "policy": "project_theseus_neural_seed_residual_mining_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if rows else "RED",
        "source_gap_report": gap_path,
        "summary": {
            "task_rows": len(rows),
            "symliquid_only_win_count": len(sym_only),
            "transformer_only_win_count": len(transformer_only),
            "both_fail_count": len(both_fail),
            "both_pass_count": sum(1 for row in rows if row.get("gap_status") == "both_pass"),
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
        },
        "symliquid_only_wins": mine_group(sym_only),
        "transformer_only_wins": mine_group(transformer_only),
        "both_failures": mine_group(both_fail),
        "next_private_pressure": next_pressure(sym_only, both_fail),
        "score_semantics": (
            "Private diagnostic residual mining only. This reads existing private gap rows, does not create "
            "new suites, does not train, does not call a teacher, and does not touch public calibration."
        ),
        "external_inference_calls": 0,
    }
    return report


def mine_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter(str(row.get("family") or "unknown") for row in rows)
    sym_templates = Counter(str(get_path(row, ["arms", "symliquid_style", "top_template_id"], "")) for row in rows)
    tx_templates = Counter(str(get_path(row, ["arms", "transformer_control", "top_template_id"], "")) for row in rows)
    causes = Counter(
        f"sym={get_path(row, ['arms', 'symliquid_style', 'failure_cause'], 'passed')};"
        f"tx={get_path(row, ['arms', 'transformer_control', 'failure_cause'], 'passed')}"
        for row in rows
    )
    examples = [
        {
            "task_id": row.get("task_id"),
            "family": row.get("family"),
            "entry_point": row.get("entry_point"),
            "symliquid_top_template_id": get_path(row, ["arms", "symliquid_style", "top_template_id"], None),
            "transformer_top_template_id": get_path(row, ["arms", "transformer_control", "top_template_id"], None),
            "symliquid_stage": get_path(row, ["arms", "symliquid_style", "sts_on_stage"], None),
            "transformer_stage": get_path(row, ["arms", "transformer_control", "sts_on_stage"], None),
        }
        for row in rows[:16]
    ]
    return {
        "count": len(rows),
        "family_counts": dict(families.most_common(12)),
        "symliquid_top_template_counts": dict(sym_templates.most_common(8)),
        "transformer_top_template_counts": dict(tx_templates.most_common(8)),
        "cause_pair_counts": dict(causes.most_common(8)),
        "examples": examples,
    }


def next_pressure(sym_only: list[dict[str, Any]], both_fail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"symliquid_only": 0, "both_fail": 0})
    for row in sym_only:
        by_family[str(row.get("family") or "unknown")]["symliquid_only"] += 1
    for row in both_fail:
        by_family[str(row.get("family") or "unknown")]["both_fail"] += 1
    ranked = sorted(
        by_family.items(),
        key=lambda item: (item[1]["symliquid_only"], item[1]["both_fail"], item[0]),
        reverse=True,
    )
    pressure = []
    for family, counts in ranked[:12]:
        if counts["symliquid_only"]:
            action = "preserve_and_interrogate_symliquid_unique_win_pattern"
        elif counts["both_fail"]:
            action = "repair_shared_ranker_or_body_coverage_for_both_fail_family"
        else:
            action = "monitor"
        pressure.append({"family": family, **counts, "recommended_action": action})
    return pressure


def get_path(obj: Any, path: list[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Neural Seed Residual Mining",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- task_rows: `{summary.get('task_rows')}`",
        f"- symliquid_only_win_count: `{summary.get('symliquid_only_win_count')}`",
        f"- transformer_only_win_count: `{summary.get('transformer_only_win_count')}`",
        f"- both_fail_count: `{summary.get('both_fail_count')}`",
        "",
        "## Next Private Pressure",
        "",
    ]
    for row in report.get("next_private_pressure", []):
        lines.append(f"- `{row.get('family')}`: symliquid_only=`{row.get('symliquid_only')}`, both_fail=`{row.get('both_fail')}`, action=`{row.get('recommended_action')}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
