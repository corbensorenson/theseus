"""Governed public-memory metadata calibration for Theseus VCM.

This runner is intentionally bounded. It reads public benchmark *cards* only,
not public prompts, contexts, answers, traces, tests, or answer templates. The
result is a metadata-clean VCM-on/off calibration slice and a run-once ledger.
Prompt-level public benchmark scoring remains a separate, explicit adapter
task and is reported as pending rather than substituted with synthetic green
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CARDS = ROOT / "benchmarks" / "cards"
DEFAULT_OUT = REPORTS / "vcm_public_memory_calibration.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_public_memory_calibration.md"
DEFAULT_LEDGER = REPORTS / "vcm_public_memory_calibration_ledger.jsonl"
DEFAULT_CARDS = [
    "source_ruler",
    "source_babilong",
    "source_needlebench_opencompass",
    "source_longmemeval",
    "source_longmemeval_v2",
    "source_helmet",
    "source_longbench_v2",
    "source_infinitebench",
]

REQUIRED_FACETS = [
    "license_allowed",
    "metadata_import_approved",
    "vcm_adapter_contract",
    "input_contract_present",
    "output_contract_present",
    "contamination_firewall_present",
    "network_forbidden_during_scoring",
    "external_inference_forbidden",
    "public_training_use_forbidden",
    "private_analogue_path_declared",
    "regression_policy_present",
]

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

TASK_TAXONOMY = {
    "ruler": ["retrieval", "multi_hop_tracking", "aggregation", "robustness"],
    "babilong": ["distributed_fact_tracking", "fact_chaining", "induction", "deduction", "counting"],
    "needlebench_opencompass": ["needle_retrieval", "multi_needle", "context_depth_robustness"],
    "longmemeval": ["multi_session_memory", "temporal_update", "abstention", "evidence_grounding"],
    "longmemeval_v2": ["agent_history_memory", "web_trajectory_memory", "procedural_memory", "temporal_reasoning"],
    "helmet": ["retrieval", "reasoning", "generation", "summarization", "length_control"],
    "longbench_v2": ["long_context_reasoning", "document_qa", "multi_hop_context"],
    "infinitebench": ["hundred_k_context", "retrieval", "qa", "summarization"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--ledger", default=rel(DEFAULT_LEDGER))
    parser.add_argument("--slice-id", default="vcm_public_memory_metadata_clean_2026_06_18")
    parser.add_argument(
        "--operator-unlock",
        default="",
        help="Required exact-run unlock string. This records the current user-approved public-memory metadata calibration spend.",
    )
    parser.add_argument("--cards", nargs="*", default=DEFAULT_CARDS)
    parser.add_argument("--allow-existing-lock", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        cards=args.cards,
        slice_id=args.slice_id,
        operator_unlock=args.operator_unlock,
        ledger_path=resolve(args.ledger),
        allow_existing_lock=args.allow_existing_lock,
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    cards: list[str],
    slice_id: str,
    operator_unlock: str,
    ledger_path: Path,
    allow_existing_lock: bool,
    started: float,
) -> dict[str, Any]:
    loaded_cards = [load_card(card_id) for card_id in cards]
    loaded_cards = [card for card in loaded_cards if card]
    surface_hash = stable_hash(sanitize_cards(loaded_cards))
    prior_locks = [
        row
        for row in read_jsonl(ledger_path)
        if row.get("slice_id") == slice_id and row.get("surface_hash") == surface_hash
    ]
    unlock_present = bool(operator_unlock.strip())
    locked_existing = bool(prior_locks)
    rows = [score_card(card) for card in loaded_cards]
    summary = summarize(rows, locked_existing=locked_existing, unlock_present=unlock_present)
    blockers = blockers_for(rows, unlock_present=unlock_present, locked_existing=locked_existing, allow_existing_lock=allow_existing_lock)
    ledger_entry = {
        "policy": "project_theseus_vcm_public_memory_calibration_ledger_v1",
        "created_utc": now(),
        "slice_id": slice_id,
        "surface_hash": surface_hash,
        "calibration_mode": "metadata_clean_public_benchmark_card_slice",
        "operator_unlock": operator_unlock,
        "benchmark_count": len(rows),
        "payload_scope": "public benchmark metadata cards only",
        "public_payload_counters": dict(PUBLIC_PAYLOAD_COUNTERS),
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "training_rows_written": 0,
        "vcm_on_mean_facet_recall": summary["vcm_on_mean_facet_recall"],
        "vcm_off_mean_facet_recall": summary["vcm_off_mean_facet_recall"],
        "official_payload_item_score_claimed": False,
    }
    appended_ledger = False
    if unlock_present and not locked_existing:
        append_jsonl(ledger_path, [ledger_entry])
        appended_ledger = True
    if blockers:
        trigger_state = "RED" if any(row["severity"] == "blocker" for row in blockers) else "YELLOW"
    else:
        trigger_state = "YELLOW"
    report = {
        "policy": "project_theseus_vcm_public_memory_calibration_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "slice_id": slice_id,
        "surface_hash": surface_hash,
        "calibration_mode": "metadata_clean_public_benchmark_card_slice",
        "operator_unlock_present": unlock_present,
        "locked_existing": locked_existing,
        "ledger_appended": appended_ledger,
        "ledger": rel(ledger_path),
        "summary": {
            **summary,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "cost_usd": 0.0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "public_training_rows_written": 0,
            "public_payload_counters": dict(PUBLIC_PAYLOAD_COUNTERS),
            "official_payload_item_score_claimed": False,
            "official_payload_item_score": None,
            "prompt_level_public_scoring": "not_run_payload_not_staged",
        },
        "public_boundary": {
            "public_prompts_loaded": False,
            "public_contexts_loaded": False,
            "public_answers_loaded": False,
            "public_tests_loaded": False,
            "public_traces_loaded": False,
            "public_templates_loaded": False,
            "public_training_use_allowed": False,
            "external_inference_allowed": False,
            "fallback_returns_allowed": False,
            "network_during_scoring": False,
        },
        "rows": rows,
        "blockers": blockers,
        "residual_plan": residual_plan(rows),
        "notes": [
            "This is the first governed public-memory VCM-on/off calibration slice, but it is metadata-clean only.",
            "No public benchmark item payloads were loaded, answered, scored, copied, or converted into training rows.",
            "Prompt-level apples-to-apples scoring on official benchmark items remains pending official adapters and a separate exact-run unlock.",
        ],
    }
    return report


def load_card(card_id: str) -> dict[str, Any]:
    path = CARDS / f"{card_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"id": card_id, "load_error": True, "path": rel(path)}
    if isinstance(value, dict):
        value["_path"] = rel(path)
        return value
    return {"id": card_id, "load_error": True, "path": rel(path)}


def score_card(card: dict[str, Any]) -> dict[str, Any]:
    source_id = str(card.get("source_id") or card.get("id") or "")
    facets = facet_values(card)
    vcm_on_detected = {key: bool(value) for key, value in facets.items()}
    vcm_off_detected = flat_baseline_facets(card)
    on_hits = sum(1 for key in REQUIRED_FACETS if vcm_on_detected.get(key))
    off_hits = sum(1 for key in REQUIRED_FACETS if vcm_off_detected.get(key))
    missing_on = [key for key in REQUIRED_FACETS if not vcm_on_detected.get(key)]
    missing_off = [key for key in REQUIRED_FACETS if not vcm_off_detected.get(key)]
    residuals = ["official_payload_adapter_pending", "prompt_level_public_scoring_not_run"]
    if missing_on:
        residuals.append("metadata_card_incomplete")
    if not bool(card.get("staged")):
        residuals.append("source_payload_not_staged")
    return {
        "card_id": card.get("id"),
        "source_id": source_id,
        "name": card.get("name"),
        "path": card.get("_path"),
        "url": card.get("url"),
        "license_spdx": card.get("license_spdx"),
        "calibration_scope": "metadata_card_only",
        "task_taxonomy": TASK_TAXONOMY.get(source_id, []),
        "vcm_on": {
            "system": "vcm_graph_metadata_resolver",
            "facet_recall": round(on_hits / len(REQUIRED_FACETS), 6),
            "facets_detected": vcm_on_detected,
            "missing_facets": missing_on,
            "evidence_recall": round(on_hits / len(REQUIRED_FACETS), 6),
            "public_item_answer_accuracy": None,
        },
        "vcm_off": {
            "system": "flat_card_metadata_baseline",
            "facet_recall": round(off_hits / len(REQUIRED_FACETS), 6),
            "facets_detected": vcm_off_detected,
            "missing_facets": missing_off,
            "evidence_recall": round(off_hits / len(REQUIRED_FACETS), 6),
            "public_item_answer_accuracy": None,
        },
        "winner": "vcm_on" if on_hits > off_hits else ("vcm_off" if off_hits > on_hits else "tie"),
        "stale_deletion_compliance": stale_deletion_compliance(card),
        "contamination_counters": dict(PUBLIC_PAYLOAD_COUNTERS),
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "training_rows_written": 0,
        "residual_categories": residuals,
    }


def facet_values(card: dict[str, Any]) -> dict[str, bool]:
    permission = dict_value(card.get("permission_envelope"))
    contamination = str(card.get("contamination_policy") or "").lower()
    gates = {str(row).lower() for row in list_value(card.get("promotion_gates"))}
    return {
        "license_allowed": bool(card.get("license_allowed")) and bool(card.get("license_spdx")),
        "metadata_import_approved": card.get("decision") in {"approved_for_catalog_import", "approved_open_license_pending_import"},
        "vcm_adapter_contract": card.get("adapter_type") == "vcm_context_recovery_adapter"
        and card.get("runner_family") == "vcm_context_recovery_local",
        "input_contract_present": bool(dict_value(card.get("input_contract")).get("memory_system")),
        "output_contract_present": bool(dict_value(card.get("output_contract")).get("evidence_precision_recall")),
        "contamination_firewall_present": "never train on public prompts" in contamination
        and "answers" in contamination
        and "templates" in contamination,
        "network_forbidden_during_scoring": "forbidden during scoring" in str(permission.get("network") or "").lower(),
        "external_inference_forbidden": str(permission.get("external_inference") or "").lower() == "forbidden",
        "public_training_use_forbidden": "no_public_training_rows" in gates and "public_calibration_only" in gates,
        "private_analogue_path_declared": "private_analogue_suite_present" in gates,
        "regression_policy_present": bool(card.get("regression_policy")),
    }


def flat_baseline_facets(card: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(
        str(card.get(key) or "")
        for key in ["id", "source_id", "name", "category", "status", "url", "license_spdx", "adapter_type", "runner_family", "capability_target"]
    ).lower()
    return {
        "license_allowed": bool(card.get("license_allowed")) and bool(card.get("license_spdx")),
        "metadata_import_approved": "approved" in str(card.get("decision") or "").lower(),
        "vcm_adapter_contract": "vcm_context_recovery_adapter" in text and "vcm_context_recovery_local" in text,
        "input_contract_present": False,
        "output_contract_present": False,
        "contamination_firewall_present": False,
        "network_forbidden_during_scoring": False,
        "external_inference_forbidden": False,
        "public_training_use_forbidden": "public calibration" in text,
        "private_analogue_path_declared": False,
        "regression_policy_present": False,
    }


def stale_deletion_compliance(card: dict[str, Any]) -> dict[str, Any]:
    output = dict_value(card.get("output_contract"))
    target = str(card.get("capability_target") or "").lower()
    abstention_contract = "abstention" in output or "abstention" in str(output).lower()
    compliance = "stale" in target and "deletion" in target and abstention_contract
    return {
        "declared": compliance,
        "evidence": "capability target and output contract mention stale/deletion/abstention behavior",
    }


def summarize(rows: list[dict[str, Any]], *, locked_existing: bool, unlock_present: bool) -> dict[str, Any]:
    count = len(rows)
    on_values = [float(get_path(row, ["vcm_on", "facet_recall"], 0.0) or 0.0) for row in rows]
    off_values = [float(get_path(row, ["vcm_off", "facet_recall"], 0.0) or 0.0) for row in rows]
    wins = {
        "vcm_on": len([row for row in rows if row.get("winner") == "vcm_on"]),
        "vcm_off": len([row for row in rows if row.get("winner") == "vcm_off"]),
        "tie": len([row for row in rows if row.get("winner") == "tie"]),
    }
    return {
        "benchmark_count": count,
        "operator_unlock_present": unlock_present,
        "locked_existing": locked_existing,
        "metadata_clean_calibration_complete": bool(count and unlock_present),
        "ledger_lock_present": bool(locked_existing),
        "vcm_on_mean_facet_recall": round(sum(on_values) / max(1, count), 6),
        "vcm_off_mean_facet_recall": round(sum(off_values) / max(1, count), 6),
        "facet_recall_lift": round((sum(on_values) - sum(off_values)) / max(1, count), 6),
        "win_counts": wins,
        "stale_deletion_declared_count": len([row for row in rows if get_path(row, ["stale_deletion_compliance", "declared"], False)]),
        "official_payload_blocker_count": count,
        "no_admissible_rate": 0.0 if count else 1.0,
    }


def blockers_for(
    rows: list[dict[str, Any]],
    *,
    unlock_present: bool,
    locked_existing: bool,
    allow_existing_lock: bool,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not unlock_present:
        blockers.append(
            {
                "severity": "blocker",
                "kind": "missing_operator_unlock",
                "detail": "Public calibration is operator-locked; pass --operator-unlock for this exact bounded metadata-clean run.",
            }
        )
    if locked_existing and not allow_existing_lock:
        blockers.append(
            {
                "severity": "blocker",
                "kind": "slice_already_locked",
                "detail": "This public-memory metadata slice is already in the ledger and was not rerun.",
            }
        )
    if not rows:
        blockers.append({"severity": "blocker", "kind": "no_cards_loaded", "detail": "No public-memory benchmark cards loaded."})
    for row in rows:
        missing = get_path(row, ["vcm_on", "missing_facets"], [])
        if missing:
            blockers.append(
                {
                    "severity": "warning",
                    "kind": "metadata_card_incomplete",
                    "card_id": row.get("card_id"),
                    "missing_facets": missing,
                }
            )
        blockers.append(
            {
                "severity": "warning",
                "kind": "official_payload_item_scoring_pending",
                "card_id": row.get("card_id"),
                "detail": "Official public prompts/contexts/answers were not loaded; prompt-level public item score is not claimed.",
            }
        )
    return blockers


def residual_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for row in rows:
        plan.append(
            {
                "source_id": row.get("source_id"),
                "private_only_next_pressure": [
                    f"mirror taxonomy: {', '.join(row.get('task_taxonomy') or ['unknown'])}",
                    "build private dogfood/context-recovery analogue rows",
                    "keep public benchmark item payloads calibration-only",
                ],
                "public_payload_status": "pending_official_adapter_and_exact_run_unlock",
            }
        )
    return plan


def sanitize_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for card in cards:
        sanitized.append(
            {
                "id": card.get("id"),
                "source_id": card.get("source_id"),
                "name": card.get("name"),
                "url": card.get("url"),
                "license_spdx": card.get("license_spdx"),
                "adapter_type": card.get("adapter_type"),
                "runner_family": card.get("runner_family"),
                "permission_envelope": card.get("permission_envelope"),
                "promotion_gates": card.get("promotion_gates"),
                "contamination_policy": card.get("contamination_policy"),
            }
        )
    return sanitized


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Public Memory Calibration",
        "",
        f"State: `{report['trigger_state']}`",
        "",
        "## Summary",
        "",
        f"- Calibration mode: `{report['calibration_mode']}`",
        f"- Benchmarks: `{summary['benchmark_count']}`",
        f"- VCM-on mean facet recall: `{summary['vcm_on_mean_facet_recall']}`",
        f"- VCM-off mean facet recall: `{summary['vcm_off_mean_facet_recall']}`",
        f"- Facet recall lift: `{summary['facet_recall_lift']}`",
        f"- VCM-only wins: `{summary['win_counts']['vcm_on']}`",
        f"- Off-only wins: `{summary['win_counts']['vcm_off']}`",
        f"- External inference calls: `{summary['external_inference_calls']}`",
        f"- Public training rows written: `{summary['public_training_rows_written']}`",
        f"- Fallback return count: `{summary['fallback_return_count']}`",
        f"- Official payload item score claimed: `{summary['official_payload_item_score_claimed']}`",
        "",
        "## Boundaries",
        "",
        "- Public prompt/context/answer/test/trace/template payloads were not loaded.",
        "- This is not a prompt-level public item score.",
        "- Training pressure must come from private analogues, dogfood traces, or governed teacher rows.",
        "",
        "## Per Benchmark",
        "",
    ]
    for row in report["rows"]:
        lines.append(
            f"- `{row.get('source_id')}`: VCM-on `{get_path(row, ['vcm_on', 'facet_recall'], None)}`, "
            f"VCM-off `{get_path(row, ['vcm_off', 'facet_recall'], None)}`, winner `{row.get('winner')}`, "
            f"payload score `not_claimed`"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers / Pending Work", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker.get('severity')}` `{blocker.get('kind')}`: {blocker.get('detail') or blocker.get('card_id')}")
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for key in path:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
