"""Hard public-memory benchmark admission audit for Theseus VCM.

This is a source/readiness audit, not a public payload scorer. It records which
hard memory and long-context benchmarks can be admitted, which are only metadata
ready, and which are blocked by license/evaluator/source constraints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CARDS = ROOT / "benchmarks" / "cards"
DEFAULT_SOURCE_ROOT = ROOT / "data" / "public_benchmarks" / "vcm_official_sources"
DEFAULT_OUT = REPORTS / "vcm_hard_memory_benchmark_readiness.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_hard_memory_benchmark_readiness.md"
DEFAULT_PUBLIC_ROW_TARGET = 2000
DEFAULT_PRIVATE_ROW_TARGET = 1000

CARD_IDS = [
    "source_ruler",
    "source_babilong",
    "source_longmemeval",
    "source_longmemeval_v2",
    "source_needlebench_opencompass",
    "source_helmet",
    "source_longbench_v2",
    "source_infinitebench",
    "source_nolima",
    "source_michelangelo_lsq",
    "source_lveval",
    "source_loft",
    "source_mtrag",
    "source_mtrag_un",
    "source_facts_grounding",
    "source_locomoplus",
    "source_locomo",
]

PROMPT_READY_SOURCE_IDS = {"ruler", "babilong", "longmemeval", "longbench_v2", "needlebench_opencompass", "infinitebench"}
HF_PROMPT_READY_SOURCE_IDS = {"longbench_v2", "needlebench_opencompass"}
GENERATED_PROMPT_READY_SOURCE_IDS = {"infinitebench"}
METADATA_SOURCE_IDS = {
    "needlebench_opencompass",
    "longmemeval_v2",
    "helmet",
    "longbench_v2",
    "infinitebench",
    "lveval",
}

PUBLIC_PAYLOAD_COUNTERS = {
    "public_prompt_chars_loaded": 0,
    "public_context_chars_loaded": 0,
    "public_answer_chars_loaded": 0,
    "public_trace_chars_loaded": 0,
    "public_solution_chars_loaded": 0,
    "public_template_chars_loaded": 0,
    "public_tests_loaded": 0,
    "public_training_rows_written": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=rel(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--public-row-target", type=int, default=DEFAULT_PUBLIC_ROW_TARGET)
    parser.add_argument("--private-row-target", type=int, default=DEFAULT_PRIVATE_ROW_TARGET)
    parser.add_argument("--cards", nargs="*", default=CARD_IDS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        source_root=resolve(args.source_root),
        card_ids=args.cards,
        public_row_target=max(1, args.public_row_target),
        private_row_target=max(1, args.private_row_target),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if args.summary_only:
        print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"], "blockers": report["blockers"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, source_root: Path, card_ids: list[str], public_row_target: int, private_row_target: int, started: float) -> dict[str, Any]:
    cards = [load_card(card_id) for card_id in card_ids]
    rows = [audit_card(card, source_root=source_root) for card in cards]
    prompt_report = read_json(REPORTS / "vcm_public_memory_prompt_calibration.json")
    hard_private = read_json(REPORTS / "vcm_hard_memory_private_analogues.json")
    private_summary = dict_value(hard_private.get("summary"))
    summary = summarize(rows, prompt_report=prompt_report, hard_private=hard_private, public_row_target=public_row_target, private_row_target=private_row_target)
    blockers = readiness_blockers(rows, summary)
    trigger_state = "GREEN" if not blockers else "YELLOW"
    return {
        "policy": "project_theseus_vcm_hard_memory_benchmark_readiness_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "calibration_mode": "source_admission_and_hard_memory_readiness",
        "source_root": rel(source_root),
        "public_row_target": public_row_target,
        "private_row_target": private_row_target,
        "summary": {
            **summary,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "external_inference_calls": 0,
            "teacher_solving_calls": 0,
            "fallback_return_count": 0,
            "public_payload_counters": dict(PUBLIC_PAYLOAD_COUNTERS),
        },
        "rows": rows,
        "blockers": blockers,
        "private_hard_analogue_report": rel(REPORTS / "vcm_hard_memory_private_analogues.json"),
        "private_hard_analogue_state": hard_private.get("trigger_state", "missing"),
        "private_hard_analogue_summary": private_summary,
        "recommendation": recommendation(summary, blockers),
        "public_boundary": {
            "public_payloads_loaded": False,
            "public_training_use_allowed": False,
            "external_inference_allowed": False,
            "fallback_returns_allowed": False,
            "model_judge_allowed": False,
        },
    }


def audit_card(card: dict[str, Any], *, source_root: Path) -> dict[str, Any]:
    source_id = str(card.get("source_id") or card.get("id") or "")
    status = str(card.get("status") or "")
    license_allowed = bool(card.get("license_allowed"))
    staged = bool(card.get("staged"))
    source_path = source_root / source_id
    source_present = source_path.exists()
    commit = git_head(source_path)
    if card.get("load_error"):
        admission_state = "blocked_missing_card"
        blocker = str(card.get("path") or "")
    elif source_id in PROMPT_READY_SOURCE_IDS:
        if source_present or source_id in HF_PROMPT_READY_SOURCE_IDS or source_id in GENERATED_PROMPT_READY_SOURCE_IDS:
            admission_state = "admitted_prompt_ready"
            blocker = ""
        else:
            admission_state = "blocked_missing_official_source"
            blocker = f"official source not staged under {rel(source_path)}"
    elif source_id in METADATA_SOURCE_IDS and license_allowed:
        admission_state = "metadata_ready_prompt_adapter_pending"
        blocker = "metadata/card admitted; deterministic public payload adapter not implemented yet"
    elif "noncommercial" in status or not license_allowed:
        admission_state = "blocked_license_or_terms"
        blocker = card_blocker(card)
    elif "model_judge" in status:
        admission_state = "blocked_model_judge"
        blocker = "benchmark requires model-judge/private leaderboard path not admitted locally"
    elif "queued" in status or "review" in status or "needed" in status:
        admission_state = "queued_admission_review"
        blocker = card_blocker(card)
    else:
        admission_state = "metadata_ready_prompt_adapter_pending" if license_allowed else "queued_admission_review"
        blocker = card_blocker(card)
    return {
        "card_id": card.get("id"),
        "source_id": source_id,
        "name": card.get("name"),
        "status": status,
        "admission_state": admission_state,
        "url": card.get("url"),
        "paper_url": card.get("paper_url", ""),
        "license_spdx": card.get("license_spdx"),
        "license_allowed": license_allowed,
        "decision": card.get("decision", ""),
        "source_present": source_present,
        "source_commit": commit,
        "staged": staged,
        "prompt_level_adapter_ready": admission_state == "admitted_prompt_ready",
        "metadata_ready": admission_state in {"admitted_prompt_ready", "metadata_ready_prompt_adapter_pending"},
        "private_analogue_required": True,
        "public_training_allowed": False,
        "external_inference_allowed": False,
        "fallback_returns_allowed": False,
        "blocker": blocker,
    }


def card_blocker(card: dict[str, Any]) -> str:
    status = str(card.get("status") or "")
    if "noncommercial" in status:
        return "license blocks active public payload staging under current policy"
    if "model_judge" in status:
        return "model-judge or private leaderboard scoring is not admitted"
    if "license" in status or "review" in status:
        return "license/evaluator review required before public payload staging"
    if "locator" in status or "source" in status:
        return "official source or payload locator still needs verification"
    if not bool(card.get("license_allowed")):
        return "license not currently allowed or unknown"
    return "deterministic prompt-level adapter pending"


def summarize(rows: list[dict[str, Any]], *, prompt_report: dict[str, Any], hard_private: dict[str, Any], public_row_target: int, private_row_target: int) -> dict[str, Any]:
    states = count_by(rows, "admission_state")
    prompt_summary = dict_value(prompt_report.get("summary"))
    prompt_items = int(prompt_summary.get("item_count") or prompt_summary.get("scored_item_count") or 0)
    if not prompt_items:
        per_benchmark = dict_value(prompt_summary.get("per_benchmark"))
        prompt_items = sum(int(dict_value(row).get("items") or dict_value(row).get("item_count") or 0) for row in per_benchmark.values())
    hard_summary = dict_value(hard_private.get("summary"))
    private_case_count = int(hard_summary.get("case_count") or 0)
    families = sorted(dict_value(hard_summary.get("family_summary")).keys())
    return {
        "candidate_count": len(rows),
        "admitted_prompt_ready_count": states.get("admitted_prompt_ready", 0),
        "metadata_ready_count": states.get("metadata_ready_prompt_adapter_pending", 0) + states.get("admitted_prompt_ready", 0),
        "blocked_or_queued_count": len(rows) - states.get("admitted_prompt_ready", 0) - states.get("metadata_ready_prompt_adapter_pending", 0),
        "admission_states": states,
        "current_public_prompt_rows_scored": prompt_items,
        "public_row_target": public_row_target,
        "public_row_target_met": prompt_items >= public_row_target,
        "private_hard_case_count": private_case_count,
        "private_row_target": private_row_target,
        "private_row_target_met": private_case_count >= private_row_target,
        "private_hard_family_count": len(families),
        "private_hard_families": families,
        "private_hard_vcm_pass_rate": hard_summary.get("vcm_on_pass_rate"),
        "private_hard_best_single_non_vcm_pass_rate": hard_summary.get("best_single_non_vcm_pass_rate"),
        "private_hard_delta": hard_summary.get("vcm_over_best_single_non_vcm_delta"),
        "private_hard_min_family_pass_rate": hard_summary.get("minimum_family_pass_rate"),
    }


def readiness_blockers(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if int(summary.get("metadata_ready_count") or 0) < 5:
        blockers.append({"severity": "blocker", "kind": "insufficient_admitted_or_metadata_ready_families", "detail": f"metadata_ready={summary.get('metadata_ready_count')}"})
    if int(summary.get("private_hard_case_count") or 0) < int(summary.get("private_row_target") or 0):
        blockers.append({"severity": "blocker", "kind": "private_hard_analogue_target_not_met", "detail": f"private_cases={summary.get('private_hard_case_count')} target={summary.get('private_row_target')}"})
    if not bool(summary.get("public_row_target_met")):
        blockers.append({"severity": "warning", "kind": "public_row_target_not_met", "detail": f"public_rows={summary.get('current_public_prompt_rows_scored')} target={summary.get('public_row_target')}; needs official source/payload admission and exact-run unlock, not synthetic substitution"})
    for row in rows:
        if row["admission_state"].startswith("blocked") or row["admission_state"].startswith("queued"):
            blockers.append({"severity": "warning", "kind": "source_admission_blocked", "source_id": row["source_id"], "detail": row["blocker"]})
    return blockers


def recommendation(summary: dict[str, Any], blockers: list[dict[str, Any]]) -> str:
    blocker_kinds = {str(row.get("kind")) for row in blockers}
    if "private_hard_analogue_target_not_met" in blocker_kinds:
        return "Run scripts/vcm_hard_memory_private_analogues.py before any new public hard-memory spend."
    if not bool(summary.get("public_row_target_met")):
        return "Admit official deterministic public adapters/source payloads next; do not claim 2000-row public hard-memory coverage yet."
    return "Hard-memory readiness is sufficient for a governed exact-once public confirmation manifest, provided operator policy unlock exists."


def load_card(card_id: str) -> dict[str, Any]:
    path = CARDS / f"{card_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"id": card_id, "source_id": card_id.replace("source_", ""), "load_error": True, "path": rel(path)}
    if isinstance(value, dict):
        return value
    return {"id": card_id, "source_id": card_id.replace("source_", ""), "load_error": True, "path": rel(path)}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Hard Memory Benchmark Readiness",
        "",
        f"State: `{report['trigger_state']}`",
        f"Candidates: `{summary['candidate_count']}`",
        f"Prompt-ready public sources: `{summary['admitted_prompt_ready_count']}`",
        f"Metadata-ready sources: `{summary['metadata_ready_count']}`",
        f"Public prompt rows scored: `{summary['current_public_prompt_rows_scored']}` / `{summary['public_row_target']}`",
        f"Private hard analogue rows: `{summary['private_hard_case_count']}` / `{summary['private_row_target']}`",
        f"Private hard analogue VCM: `{summary['private_hard_vcm_pass_rate']}`",
        f"Private hard analogue best non-VCM: `{summary['private_hard_best_single_non_vcm_pass_rate']}`",
        "",
        "## Source Admission",
        "",
    ]
    for row in report["rows"]:
        lines.append(
            f"- `{row['source_id']}`: `{row['admission_state']}` license `{row.get('license_spdx')}` staged `{row['source_present']}` blocker `{row['blocker']}`"
        )
    lines.extend(["", "## Blockers", ""])
    for row in report["blockers"]:
        source = f" `{row.get('source_id')}`" if row.get("source_id") else ""
        lines.append(f"- `{row['severity']}`{source}: `{row['kind']}` - {row['detail']}")
    lines.extend(["", "## Recommendation", "", report["recommendation"]])
    return "\n".join(lines) + "\n"


def git_head(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
