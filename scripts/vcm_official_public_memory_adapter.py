"""Prompt-level public-memory calibration adapters for VCM.

The script stages a bounded prompt-level slice from public memory benchmark
sources into an ignored quarantine area, runs VCM-on versus VCM-off and local
memory-system baselines under the same context budget and scorer path, and
writes only aggregate/private residual repair fixtures. Public prompts,
contexts, answers, traces, tests, and templates are never written to training
rows.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcm_public_memory_adapter_support import (
    aggregate_residuals,
    annotate_resolver_result,
    append_jsonl,
    before_location_query,
    build_private_residuals,
    compact_space,
    dict_value,
    evidence_metrics,
    file_hash,
    first_capitalized_name,
    forbidden_item_overlaps,
    git_head,
    item_length_metrics,
    item_manifest_row,
    length_bucket,
    list_value,
    now,
    object_from_question,
    parse_int_csv,
    payload_row,
    queued_row,
    read_json,
    read_jsonl,
    rel,
    render_markdown,
    residual_categories,
    resolve,
    ruler_query_keys,
    run,
    safe_id,
    score_prediction,
    stable_hash,
    stale_update_deletion_eval,
    summarize_rows,
    unscored_item_row,
    write_json,
    write_jsonl,
    write_text,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_SOURCE_ROOT = ROOT / "data" / "public_benchmarks" / "vcm_official_sources"
DEFAULT_QUARANTINE_ROOT = ROOT / "data" / "public_benchmarks" / "vcm_memory_quarantine"
DEFAULT_OUT = REPORTS / "vcm_public_memory_prompt_calibration.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_public_memory_prompt_calibration.md"
DEFAULT_LEDGER = REPORTS / "vcm_public_memory_prompt_calibration_ledger.jsonl"
DEFAULT_PRIVATE_RESIDUALS = REPORTS / "vcm_public_memory_private_residual_fixtures.jsonl"
DEFAULT_PRIVATE_REPAIR = REPORTS / "vcm_public_memory_private_residual_repair.json"

OFFICIAL_REPOS = {
    "ruler": {
        "url": "https://github.com/NVIDIA/RULER.git",
        "license": "apache-2.0",
        "card": "source_ruler",
    },
    "babilong": {
        "url": "https://github.com/booydar/babilong.git",
        "license": "apache-2.0",
        "card": "source_babilong",
    },
    "longmemeval": {
        "url": "https://github.com/xiaowu0162/longmemeval.git",
        "license": "mit",
        "card": "source_longmemeval",
    },
}

OFFICIAL_HF_SOURCES = {
    "longbench_v2": {
        "dataset": "THUDM/LongBench-v2",
        "split": "train",
        "license": "mit",
        "card": "source_longbench_v2",
    },
    "needlebench_opencompass": {
        "dataset": "opencompass/needlebench",
        "split": "test",
        "license": "apache-2.0",
        "card": "source_needlebench_opencompass",
    },
    "longmemeval_v2": {
        "dataset": "xiaowu0162/LongMemEval-V2",
        "split": "train",
        "license": "apache-2.0",
        "card": "source_longmemeval_v2",
        "status": "blocked_image_payload_until_text_evaluator_staged",
    },
}

BABILONG_TASKS = [
    ("qa1_single-supporting-fact", "qa1"),
    ("qa2_two-supporting-facts", "qa2"),
    ("qa3_three-supporting-facts", "qa3"),
    ("qa6_yes-no-questions", "qa6"),
]

LONGMEMEVAL_DATA_CANDIDATES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "longmemeval_oracle.json",
]


@dataclass
class PublicMemoryItem:
    item_id: str
    benchmark: str
    task: str
    prompt: str
    context: str
    question: str
    answers: list[str]
    oracle_evidence: list[dict[str, str]]
    metadata: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=rel(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--quarantine-root", default=rel(DEFAULT_QUARANTINE_ROOT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--ledger", default=rel(DEFAULT_LEDGER))
    parser.add_argument("--private-residuals-out", default=rel(DEFAULT_PRIVATE_RESIDUALS))
    parser.add_argument("--private-repair-out", default=rel(DEFAULT_PRIVATE_REPAIR))
    parser.add_argument("--slice-id", default="vcm_public_memory_prompt_slice_2026_06_18")
    parser.add_argument("--operator-unlock", default="")
    parser.add_argument("--allow-existing-lock", action="store_true")
    parser.add_argument("--fetch-sources", action="store_true")
    parser.add_argument("--context-budget-chars", type=int, default=900)
    parser.add_argument("--max-items-per-benchmark", type=int, default=64)
    parser.add_argument("--item-offset-per-benchmark", type=int, default=0)
    parser.add_argument("--ruler-max-items", type=int, default=0)
    parser.add_argument("--babilong-max-items", type=int, default=0)
    parser.add_argument("--longmemeval-max-items", type=int, default=0)
    parser.add_argument("--longbench-v2-max-items", type=int, default=0)
    parser.add_argument("--needlebench-max-items", type=int, default=0)
    parser.add_argument("--infinitebench-max-items", type=int, default=0)
    parser.add_argument("--longmemeval-v2-max-items", type=int, default=0)
    parser.add_argument("--ruler-offset", type=int, default=-1)
    parser.add_argument("--babilong-offset", type=int, default=-1)
    parser.add_argument("--longmemeval-offset", type=int, default=-1)
    parser.add_argument("--longbench-v2-offset", type=int, default=-1)
    parser.add_argument("--needlebench-offset", type=int, default=-1)
    parser.add_argument("--infinitebench-offset", type=int, default=-1)
    parser.add_argument("--longmemeval-v2-offset", type=int, default=-1)
    parser.add_argument("--disable-longmemeval", action="store_true")
    parser.add_argument("--hf-stream-sources", action="store_true")
    parser.add_argument("--ruler-source-token-buckets", default="8000,32000,128000")
    parser.add_argument("--needlebench-source-token-buckets", default="8000,32000,128000")
    parser.add_argument("--infinitebench-source-token-buckets", default="8000,32000,128000")
    parser.add_argument("--forbid-overlap-slice-id", action="append", default=[])
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    max_items_by_benchmark = {
        "ruler": max(1, args.ruler_max_items or args.max_items_per_benchmark),
        "babilong": max(1, args.babilong_max_items or args.max_items_per_benchmark),
        "longmemeval": 0 if args.disable_longmemeval else max(1, args.longmemeval_max_items or args.max_items_per_benchmark),
        "longbench_v2": max(0, args.longbench_v2_max_items),
        "needlebench": max(0, args.needlebench_max_items),
        "infinitebench": max(0, args.infinitebench_max_items),
        "longmemeval_v2": max(0, args.longmemeval_v2_max_items),
    }
    offsets_by_benchmark = {
        "ruler": max(0, args.ruler_offset if args.ruler_offset >= 0 else args.item_offset_per_benchmark),
        "babilong": max(0, args.babilong_offset if args.babilong_offset >= 0 else args.item_offset_per_benchmark),
        "longmemeval": max(0, args.longmemeval_offset if args.longmemeval_offset >= 0 else args.item_offset_per_benchmark),
        "longbench_v2": max(0, args.longbench_v2_offset if args.longbench_v2_offset >= 0 else args.item_offset_per_benchmark),
        "needlebench": max(0, args.needlebench_offset if args.needlebench_offset >= 0 else args.item_offset_per_benchmark),
        "infinitebench": max(0, args.infinitebench_offset if args.infinitebench_offset >= 0 else args.item_offset_per_benchmark),
        "longmemeval_v2": max(0, args.longmemeval_v2_offset if args.longmemeval_v2_offset >= 0 else args.item_offset_per_benchmark),
    }
    report = build_report(
        source_root=resolve(args.source_root),
        quarantine_root=resolve(args.quarantine_root),
        ledger_path=resolve(args.ledger),
        private_residuals_path=resolve(args.private_residuals_out),
        private_repair_path=resolve(args.private_repair_out),
        slice_id=args.slice_id,
        operator_unlock=args.operator_unlock,
        allow_existing_lock=args.allow_existing_lock,
        fetch_sources=args.fetch_sources,
        context_budget_chars=max(128, args.context_budget_chars),
        max_items_per_benchmark=max(1, args.max_items_per_benchmark),
        item_offset_per_benchmark=max(0, args.item_offset_per_benchmark),
        max_items_by_benchmark=max_items_by_benchmark,
        offsets_by_benchmark=offsets_by_benchmark,
        ruler_source_token_buckets=parse_int_csv(args.ruler_source_token_buckets, default=[8000, 32000, 128000]),
        needlebench_source_token_buckets=parse_int_csv(args.needlebench_source_token_buckets, default=[8000, 32000, 128000]),
        infinitebench_source_token_buckets=parse_int_csv(args.infinitebench_source_token_buckets, default=[8000, 32000, 128000]),
        hf_stream_sources=args.hf_stream_sources,
        forbid_overlap_slice_ids=[value for value in args.forbid_overlap_slice_id if value.strip()],
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if args.summary_only:
        print(json.dumps({"trigger_state": report["trigger_state"], "slice_id": report["slice_id"], "summary": report["summary"], "blockers": report["blockers"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    source_root: Path,
    quarantine_root: Path,
    ledger_path: Path,
    private_residuals_path: Path,
    private_repair_path: Path,
    slice_id: str,
    operator_unlock: str,
    allow_existing_lock: bool,
    fetch_sources: bool,
    context_budget_chars: int,
    max_items_per_benchmark: int,
    item_offset_per_benchmark: int,
    max_items_by_benchmark: dict[str, int],
    offsets_by_benchmark: dict[str, int],
    ruler_source_token_buckets: list[int],
    needlebench_source_token_buckets: list[int],
    infinitebench_source_token_buckets: list[int],
    hf_stream_sources: bool,
    forbid_overlap_slice_ids: list[str],
    started: float,
) -> dict[str, Any]:
    if fetch_sources:
        fetch_official_sources(source_root)

    source_records = inspect_sources(source_root)
    items, stage_blockers = stage_public_items(
        source_root=source_root,
        max_items_per_benchmark=max_items_per_benchmark,
        item_offset_per_benchmark=item_offset_per_benchmark,
        max_items_by_benchmark=max_items_by_benchmark,
        offsets_by_benchmark=offsets_by_benchmark,
        ruler_source_token_buckets=ruler_source_token_buckets,
        needlebench_source_token_buckets=needlebench_source_token_buckets,
        infinitebench_source_token_buckets=infinitebench_source_token_buckets,
        hf_stream_sources=hf_stream_sources,
    )
    item_manifest_rows = [item_manifest_row(item) for item in items]
    surface_hash = stable_hash(
        {
            "slice_id": slice_id,
            "source_records": source_records,
            "item_ids": [item.item_id for item in items],
            "item_manifest_hash": stable_hash(item_manifest_rows),
            "context_budget_chars": context_budget_chars,
            "item_offset_per_benchmark": item_offset_per_benchmark,
            "offsets_by_benchmark": offsets_by_benchmark,
            "max_items_by_benchmark": max_items_by_benchmark,
            "ruler_source_token_buckets": ruler_source_token_buckets,
            "needlebench_source_token_buckets": needlebench_source_token_buckets,
            "infinitebench_source_token_buckets": infinitebench_source_token_buckets,
            "hf_stream_sources": hf_stream_sources,
        }
    )
    prior_locks = [
        row
        for row in read_jsonl(ledger_path)
        if row.get("slice_id") == slice_id and row.get("surface_hash") == surface_hash
    ]
    unlock_present = bool(operator_unlock.strip())
    locked_existing = bool(prior_locks)
    blockers = list(stage_blockers)
    if not unlock_present:
        blockers.append(
            {
                "severity": "blocker",
                "kind": "missing_operator_unlock",
                "detail": "Prompt-level public calibration is exact-run locked; pass --operator-unlock for this slice.",
            }
        )
    if locked_existing and not allow_existing_lock:
        blockers.append(
            {
                "severity": "blocker",
                "kind": "slice_already_locked",
                "detail": "This prompt-level public-memory slice is already in the ledger and was not rerun.",
            }
        )
    overlap_by_slice = forbidden_item_overlaps(
        ledger_path=ledger_path,
        quarantine_root=quarantine_root,
        slice_ids=forbid_overlap_slice_ids,
        current_item_ids=[item.item_id for item in items],
    )
    for prior_slice_id, overlaps in overlap_by_slice.items():
        if overlaps:
            blockers.append(
                {
                    "severity": "blocker",
                    "kind": "forbidden_public_surface_overlap",
                    "detail": f"{slice_id} overlaps {prior_slice_id} on {len(overlaps)} item ids; first overlaps={overlaps[:10]}",
                }
            )

    slice_dir = quarantine_root / slice_id
    payload_path = slice_dir / "payloads.jsonl"
    manifest_path = slice_dir / "manifest.json"
    item_manifest_path = slice_dir / "item_manifest.json"
    rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    if unlock_present and items and not any(row["severity"] == "blocker" for row in blockers):
        for item in items:
            scored = score_item(item, context_budget_chars=context_budget_chars)
            rows.append(scored)
            payload_rows.append(payload_row(item))
        write_jsonl(payload_path, payload_rows)
        write_json(item_manifest_path, {
            "policy": "project_theseus_vcm_public_memory_item_manifest_v1",
            "created_utc": now(),
            "slice_id": slice_id,
            "surface_hash": surface_hash,
            "item_count": len(item_manifest_rows),
            "items": item_manifest_rows,
            "public_prompt_chars": 0,
            "public_context_chars": 0,
            "public_answer_chars": 0,
            "private_training_allowed": False,
        })
        write_json(
            manifest_path,
            {
                "policy": "project_theseus_vcm_public_memory_quarantine_manifest_v1",
                "created_utc": now(),
                "slice_id": slice_id,
                "surface_hash": surface_hash,
                "payload_path": rel(payload_path),
                "payload_hash": file_hash(payload_path),
                "item_manifest": rel(item_manifest_path),
                "item_manifest_hash": file_hash(item_manifest_path),
                "item_count": len(payload_rows),
                "private_training_allowed": False,
                "commit_payloads_to_git": False,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            },
        )
    elif items:
        for item in items:
            rows.append(unscored_item_row(item))

    summary = summarize_rows(rows, items, context_budget_chars)
    residuals, repair_report = build_private_residuals(rows, private_residuals_path)
    write_jsonl(private_residuals_path, residuals)
    write_json(private_repair_path, repair_report)

    ledger_appended = False
    if unlock_present and rows and not locked_existing and not any(row["severity"] == "blocker" for row in blockers):
        append_jsonl(
            ledger_path,
            [
                {
                    "policy": "project_theseus_vcm_public_memory_prompt_calibration_ledger_v1",
                    "created_utc": now(),
                    "slice_id": slice_id,
                    "surface_hash": surface_hash,
                    "operator_unlock": operator_unlock,
                    "calibration_mode": "prompt_level_public_memory_quarantined_slice",
                    "item_count": len(rows),
                    "benchmarks": sorted({row.get("benchmark") for row in rows}),
                    "context_budget_chars": context_budget_chars,
                    "item_offset_per_benchmark": item_offset_per_benchmark,
                    "offsets_by_benchmark": offsets_by_benchmark,
                    "max_items_by_benchmark": max_items_by_benchmark,
                    "ruler_source_token_buckets": ruler_source_token_buckets,
                    "needlebench_source_token_buckets": needlebench_source_token_buckets,
                    "infinitebench_source_token_buckets": infinitebench_source_token_buckets,
                    "hf_stream_sources": hf_stream_sources,
                    "forbid_overlap_slice_ids": forbid_overlap_slice_ids,
                    "payload_manifest": rel(manifest_path),
                    "payload_hash": file_hash(payload_path) if payload_path.exists() else "",
                    "item_manifest": rel(item_manifest_path),
                    "item_manifest_hash": file_hash(item_manifest_path) if item_manifest_path.exists() else "",
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                    "public_training_rows_written": 0,
                }
            ],
        )
        ledger_appended = True

    if any(row["severity"] == "blocker" for row in blockers):
        trigger_state = "RED"
    elif not rows:
        trigger_state = "RED"
    elif any(row.get("benchmark") == "longmemeval" and row.get("status") == "queued" for row in rows):
        trigger_state = "YELLOW"
    else:
        trigger_state = "GREEN"

    longmemeval_scored = any(row.get("benchmark") == "longmemeval" and row.get("status") == "scored" for row in rows)
    notes = [
        "Prompt-level public payloads are staged only under ignored quarantine with no-training labels.",
        "VCM-on and VCM-off use the same context budget and local scorer path; VCM-on differs only in evidence selection.",
    ]
    if longmemeval_scored:
        notes.append("LongMemEval is scored through a local deterministic extractor over staged official JSON; no model judge or external inference is used.")
    else:
        notes.append("LongMemEval is queued until its Hugging Face data package and evaluator can be staged without external model judging.")

    return {
        "policy": "project_theseus_vcm_public_memory_prompt_calibration_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "slice_id": slice_id,
        "surface_hash": surface_hash,
        "calibration_mode": "prompt_level_public_memory_quarantined_slice",
        "operator_unlock_present": unlock_present,
        "locked_existing": locked_existing,
        "ledger_appended": ledger_appended,
        "source_records": source_records,
        "quarantine": {
            "root": rel(quarantine_root),
            "slice_dir": rel(slice_dir),
            "payload_path": rel(payload_path),
            "payload_manifest": rel(manifest_path),
            "payload_hash": file_hash(payload_path) if payload_path.exists() else "",
            "item_manifest": rel(item_manifest_path),
            "item_manifest_hash": file_hash(item_manifest_path) if item_manifest_path.exists() else "",
            "private_training_allowed": False,
            "commit_payloads_to_git": False,
        },
        "summary": {
            **summary,
            "item_offset_per_benchmark": item_offset_per_benchmark,
            "offsets_by_benchmark": offsets_by_benchmark,
            "max_items_by_benchmark": max_items_by_benchmark,
            "ruler_source_token_buckets": ruler_source_token_buckets,
            "needlebench_source_token_buckets": needlebench_source_token_buckets,
            "infinitebench_source_token_buckets": infinitebench_source_token_buckets,
            "hf_stream_sources": hf_stream_sources,
            "forbidden_overlap_slices": forbid_overlap_slice_ids,
            "forbidden_overlap_counts": {key: len(value) for key, value in overlap_by_slice.items()},
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "cost_usd": 0.0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "public_training_rows_written": 0,
            "teacher_solving_calls": 0,
            "private_residual_fixture_count": len(residuals),
            "private_residuals": rel(private_residuals_path),
            "private_repair_report": rel(private_repair_path),
        },
        "public_boundary": {
            "public_payloads_loaded": bool(payload_rows),
            "public_payloads_quarantined": bool(payload_rows) and payload_path.exists() and manifest_path.exists(),
            "public_training_use_allowed": False,
            "public_rows_admitted_to_training": 0,
            "external_inference_allowed": False,
            "fallback_returns_allowed": False,
            "teacher_solving_allowed": False,
        },
        "rows": rows,
        "aggregate_residuals": aggregate_residuals(rows),
        "blockers": blockers,
        "notes": notes,
    }


def fetch_official_sources(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    for name, spec in OFFICIAL_REPOS.items():
        target = source_root / name
        if (target / ".git").exists():
            run(["git", "fetch", "--depth", "1", "origin"], cwd=target)
            run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=target)
        else:
            run(["git", "clone", "--depth", "1", spec["url"], str(target)], cwd=source_root)


def inspect_sources(source_root: Path) -> list[dict[str, Any]]:
    records = []
    for name, spec in OFFICIAL_REPOS.items():
        path = source_root / name
        license_path = path / "LICENSE"
        records.append(
            {
                "source_id": name,
                "url": spec["url"],
                "card": spec["card"],
                "path": rel(path),
                "present": path.exists(),
                "commit": git_head(path),
                "license_spdx": spec["license"],
                "license_present": license_path.exists(),
                "private_training_allowed": False,
            }
        )
    for name, spec in OFFICIAL_HF_SOURCES.items():
        records.append(
            {
                "source_id": name,
                "source_kind": "huggingface_dataset",
                "url": f"https://huggingface.co/datasets/{spec['dataset']}",
                "dataset": spec["dataset"],
                "split": spec["split"],
                "card": spec["card"],
                "path": "",
                "present": False,
                "commit": "",
                "license_spdx": spec["license"],
                "license_present": True,
                "status": spec.get("status", "streaming_slice_adapter_ready"),
                "private_training_allowed": False,
            }
        )
    return records


def stage_public_items(
    *,
    source_root: Path,
    max_items_per_benchmark: int,
    item_offset_per_benchmark: int,
    max_items_by_benchmark: dict[str, int],
    offsets_by_benchmark: dict[str, int],
    ruler_source_token_buckets: list[int],
    needlebench_source_token_buckets: list[int],
    infinitebench_source_token_buckets: list[int],
    hf_stream_sources: bool,
) -> tuple[list[PublicMemoryItem], list[dict[str, Any]]]:
    items: list[PublicMemoryItem] = []
    blockers: list[dict[str, Any]] = []
    ruler_path = source_root / "ruler"
    babilong_path = source_root / "babilong"
    longmemeval_path = source_root / "longmemeval"
    if not ruler_path.exists():
        blockers.append({"severity": "blocker", "kind": "missing_official_source", "source_id": "ruler", "detail": rel(ruler_path)})
    else:
        items.extend(
            build_ruler_items(
                max_items=max_items_by_benchmark.get("ruler", max_items_per_benchmark),
                offset=offsets_by_benchmark.get("ruler", item_offset_per_benchmark),
                source_token_buckets=ruler_source_token_buckets,
            )
        )
    if not babilong_path.exists():
        blockers.append({"severity": "blocker", "kind": "missing_official_source", "source_id": "babilong", "detail": rel(babilong_path)})
    else:
        built, babilong_blockers = build_babilong_items(
            babilong_path,
            max_items=max_items_by_benchmark.get("babilong", max_items_per_benchmark),
            offset_per_task=offsets_by_benchmark.get("babilong", item_offset_per_benchmark),
        )
        items.extend(built)
        blockers.extend(babilong_blockers)
    if max_items_by_benchmark.get("longmemeval", max_items_per_benchmark) <= 0:
        pass
    elif longmemeval_path.exists():
        built, longmemeval_blockers = build_longmemeval_items(
            longmemeval_path,
            max_items=max_items_by_benchmark.get("longmemeval", max_items_per_benchmark),
            offset=offsets_by_benchmark.get("longmemeval", item_offset_per_benchmark),
        )
        if built:
            items.extend(built)
        else:
            items.append(
                PublicMemoryItem(
                    item_id="longmemeval_queued_cleaned_data_pending",
                    benchmark="longmemeval",
                    task="queued",
                    prompt="",
                    context="",
                    question="",
                    answers=[],
                    oracle_evidence=[],
                    metadata={
                        "status": "queued",
                        "reason": "official LongMemEval JSON data is not staged locally; fetch longmemeval_s_cleaned.json or longmemeval_oracle.json into the ignored source data directory before prompt-level scoring",
                        "source_commit": git_head(longmemeval_path),
                    },
                )
            )
        blockers.extend(longmemeval_blockers)
    if max_items_by_benchmark.get("longbench_v2", 0) > 0:
        built, longbench_blockers = build_longbench_v2_items(
            max_items=max_items_by_benchmark.get("longbench_v2", 0),
            offset=offsets_by_benchmark.get("longbench_v2", item_offset_per_benchmark),
            hf_stream_sources=hf_stream_sources,
        )
        items.extend(built)
        blockers.extend(longbench_blockers)
    if max_items_by_benchmark.get("needlebench", 0) > 0:
        built, needlebench_blockers = build_needlebench_items(
            max_items=max_items_by_benchmark.get("needlebench", 0),
            offset=offsets_by_benchmark.get("needlebench", item_offset_per_benchmark),
            source_token_buckets=needlebench_source_token_buckets,
            hf_stream_sources=hf_stream_sources,
        )
        items.extend(built)
        blockers.extend(needlebench_blockers)
    if max_items_by_benchmark.get("infinitebench", 0) > 0:
        items.extend(
            build_infinitebench_retrievekv_items(
                max_items=max_items_by_benchmark.get("infinitebench", 0),
                offset=offsets_by_benchmark.get("infinitebench", item_offset_per_benchmark),
                source_token_buckets=infinitebench_source_token_buckets,
            )
        )
    if max_items_by_benchmark.get("longmemeval_v2", 0) > 0:
        blockers.append(
            {
                "severity": "warning",
                "kind": "longmemeval_v2_text_adapter_blocked",
                "source_id": "longmemeval_v2",
                "detail": "The loadable Hugging Face surface currently exposes image rows in this environment; text trajectories and deterministic evaluator need staging before scoring.",
            }
        )
    return items, blockers


def build_ruler_items(*, max_items: int, offset: int = 0, source_token_buckets: list[int] | None = None) -> list[PublicMemoryItem]:
    source_token_buckets = source_token_buckets or [8000, 32000, 128000]
    distractors = [
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
        "A quiet paragraph about weather, tools, and shelves contains no special magic numbers.",
        "Several ordinary notes mention workshops, airports, laptops, and batteries without any requested key.",
        "The archive log repeats harmless filler so tail windows can lose earlier needles.",
    ]
    keys = [
        "quiet-harbor",
        "silver-ember",
        "green-canyon",
        "amber-lattice",
        "violet-signal",
        "orange-keystone",
        "cobalt-ridge",
        "crimson-anchor",
        "opal-falcon",
        "jade-circuit",
        "brass-lantern",
        "indigo-socket",
        "white-bastion",
        "black-orbit",
        "golden-relay",
        "scarlet-matrix",
    ]
    items = []
    task_cycle = ["niah_single_1", "niah_multiquery", "niah_multivalue", "niah_multikey_1"]
    for item_no in range(offset, offset + max_items):
        task = task_cycle[item_no % len(task_cycle)]
        key_a = keys[item_no % len(keys)]
        key_b = keys[(item_no * 5 + 3) % len(keys)]
        key_c = keys[(item_no * 7 + 5) % len(keys)]
        value_a = str(1_000_000 + item_no * 37 + 11)
        value_b = str(2_000_000 + item_no * 41 + 17)
        value_c = str(3_000_000 + item_no * 43 + 19)
        if task == "niah_single_1":
            needles = [(key_a, value_a)]
            query_keys = [key_a]
        elif task == "niah_multiquery":
            needles = [(key_a, value_a), (key_b, value_b), (key_c, value_c)]
            query_keys = [key_a, key_c]
        elif task == "niah_multivalue":
            needles = [(key_a, value_a), (key_a, value_b), (key_b, value_c)]
            query_keys = [key_a]
        else:
            needles = [(key_a, value_a), (key_b, value_b), (key_c, value_c)]
            query_keys = [key_b]
        target_tokens = max(128, source_token_buckets[(item_no - offset) % len(source_token_buckets)])
        target_chars = target_tokens * 4
        filler_line_chars = 96
        filler_count = max(36, min(8000, target_chars // filler_line_chars))
        context_lines = [
            f"filler-{item_no:05d}-{idx:05d}: {distractors[(item_no + idx) % len(distractors)]}"
            for idx in range(filler_count)
        ]
        evidence = []
        for idx, (key, value) in enumerate(needles, start=1):
            text = f"One of the special magic number for {key} is: {value}."
            evidence.append({"id": f"ruler_{task}_{item_no:04d}:needle:{idx}", "text": text, "key": key})
            insert_at = ((item_no + 1) * (idx * 7 + 3)) % max(1, len(context_lines) - 4)
            context_lines.insert(insert_at, text)
        context = "\n".join(context_lines)
        query = ", ".join(query_keys[:-1]) + (", and " + query_keys[-1] if len(query_keys) > 1 else query_keys[0])
        question = f"What are all the special magic number for {query} mentioned in the provided text?"
        answers = [value for key, value in needles if key in set(query_keys)]
        oracle_evidence = [
            {"id": row["id"], "text": row["text"]}
            for row in evidence
            if row.get("key") in set(query_keys)
        ]
        prompt = f"Some special magic numbers are hidden within the following text. Make sure to memorize it.\n{context}\n{question}"
        items.append(
            PublicMemoryItem(
                item_id=f"ruler_{task}_{item_no:04d}",
                benchmark="ruler",
                task=task,
                prompt=prompt,
                context=context,
                question=question,
                answers=answers,
                oracle_evidence=oracle_evidence,
                metadata={
                    "official_family": "RULER synthetic NIAH",
                    "source_task_config": task,
                    "generated_official_format": True,
                    "score_semantics": "local exact set match against generated official-format outputs",
                    "hidden_needle_count": len(evidence),
                    "queried_needle_count": len(oracle_evidence),
                    "target_source_tokens": target_tokens,
                    "target_source_token_bucket": length_bucket(target_tokens),
                },
            )
        )
    return items


def build_babilong_items(
    source_path: Path,
    *,
    max_items: int,
    offset_per_task: int = 0,
) -> tuple[list[PublicMemoryItem], list[dict[str, Any]]]:
    zip_path = source_path / "data" / "tasks_1-20_v1-2.zip"
    if not zip_path.exists():
        return [], [{"severity": "blocker", "kind": "missing_babilong_task_zip", "detail": rel(zip_path)}]
    items: list[PublicMemoryItem] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            per_task = max(1, max_items // len(BABILONG_TASKS))
            for task_name, task_id in BABILONG_TASKS:
                path = f"tasks_1-20_v1-2/en-10k/{task_name}_test.txt"
                raw = archive.read(path).decode("utf-8", errors="replace").splitlines()
                items.extend(
                    parse_babi_task(
                        raw,
                        benchmark_path=path,
                        task_id=task_id,
                        task_name=task_name,
                        limit=per_task,
                        offset=offset_per_task,
                    )
                )
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        return [], [{"severity": "blocker", "kind": "babilong_parse_failed", "detail": f"{type(exc).__name__}: {exc}"}]
    return items[:max_items], []


def build_longmemeval_items(source_path: Path, *, max_items: int, offset: int = 0) -> tuple[list[PublicMemoryItem], list[dict[str, Any]]]:
    data_dir = source_path / "data"
    data_path = next((data_dir / name for name in LONGMEMEVAL_DATA_CANDIDATES if (data_dir / name).exists()), None)
    if data_path is None:
        return [], []
    try:
        raw = json.loads(data_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [{"severity": "warning", "kind": "longmemeval_parse_failed", "detail": f"{type(exc).__name__}: {exc}"}]
    rows = raw if isinstance(raw, list) else list_value(raw.get("data") if isinstance(raw, dict) else None)
    items: list[PublicMemoryItem] = []
    for row in rows[offset: offset + max_items]:
        if not isinstance(row, dict):
            continue
        item = longmemeval_item(row, source_name=data_path.name)
        if item:
            items.append(item)
    return items, []


def build_longbench_v2_items(*, max_items: int, offset: int = 0, hf_stream_sources: bool) -> tuple[list[PublicMemoryItem], list[dict[str, Any]]]:
    if not hf_stream_sources:
        return [], [
            {
                "severity": "warning",
                "kind": "hf_stream_sources_disabled",
                "source_id": "longbench_v2",
                "detail": "Pass --hf-stream-sources to stream the official THUDM/LongBench-v2 rows into quarantine for deterministic local scoring.",
            }
        ]
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        return [], [{"severity": "warning", "kind": "datasets_import_failed", "source_id": "longbench_v2", "detail": f"{type(exc).__name__}: {exc}"}]
    try:
        dataset = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
    except Exception as exc:
        return [], [{"severity": "warning", "kind": "longbench_v2_stream_failed", "source_id": "longbench_v2", "detail": f"{type(exc).__name__}: {exc}"}]
    items: list[PublicMemoryItem] = []
    eligible_seen = 0
    for row in dataset:
        if not isinstance(row, dict):
            continue
        if eligible_seen < offset:
            eligible_seen += 1
            continue
        eligible_seen += 1
        item = longbench_v2_item(row)
        if item:
            items.append(item)
        if len(items) >= max_items:
            break
    if not items:
        return [], [{"severity": "warning", "kind": "longbench_v2_no_rows", "source_id": "longbench_v2", "detail": f"offset={offset} max_items={max_items} yielded no valid rows"}]
    return items, []


def longbench_v2_item(row: dict[str, Any]) -> PublicMemoryItem | None:
    row_id = str(row.get("_id") or "")
    question = compact_space(str(row.get("question") or ""))
    answer = compact_space(str(row.get("answer") or "")).upper()
    raw_context = str(row.get("context") or "")
    choices = {
        "A": compact_space(str(row.get("choice_A") or "")),
        "B": compact_space(str(row.get("choice_B") or "")),
        "C": compact_space(str(row.get("choice_C") or "")),
        "D": compact_space(str(row.get("choice_D") or "")),
    }
    if not row_id or not question or answer not in choices or not raw_context or not all(choices.values()):
        return None
    context = chunk_public_context(raw_context, chunk_chars=1200)
    choice_lines = "\n".join(f"{letter}. {text}" for letter, text in choices.items())
    prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}\n{choice_lines}"
    oracle_evidence = longbench_oracle_evidence(context, choices[answer], row_id=row_id)
    return PublicMemoryItem(
        item_id=f"longbench_v2_{safe_id(row_id)}",
        benchmark="longbench_v2",
        task=str(row.get("sub_domain") or row.get("domain") or "mcq"),
        prompt=prompt,
        context=context,
        question=f"{question}\n{choice_lines}",
        answers=[answer],
        oracle_evidence=oracle_evidence,
        metadata={
            "official_family": "LongBench v2",
            "hf_dataset": "THUDM/LongBench-v2",
            "split": "train",
            "row_id": row_id,
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
            "length": row.get("length"),
            "choices": choices,
            "score_semantics": "local deterministic multiple-choice selection over selected evidence context; no model judge",
        },
    )


def longbench_oracle_evidence(context: str, answer_choice: str, *, row_id: str) -> list[dict[str, str]]:
    if not answer_choice:
        return []
    answer_terms = meaningful_terms(answer_choice)
    if not answer_terms:
        return []
    rows: list[tuple[int, int, str]] = []
    for idx, line in enumerate(context.splitlines(), start=1):
        terms = meaningful_terms(line)
        overlap = len(terms & answer_terms)
        if overlap > 0:
            rows.append((overlap, idx, line))
    evidence = []
    for _, idx, line in sorted(rows, key=lambda value: (-value[0], value[1]))[:4]:
        evidence.append({"id": f"longbench_v2:{row_id}:chunk{idx}", "text": line})
    return evidence


def build_needlebench_items(
    *,
    max_items: int,
    offset: int = 0,
    source_token_buckets: list[int] | None = None,
    hf_stream_sources: bool,
) -> tuple[list[PublicMemoryItem], list[dict[str, Any]]]:
    source_token_buckets = source_token_buckets or [8000, 32000, 128000]
    if not hf_stream_sources:
        return [], [
            {
                "severity": "warning",
                "kind": "hf_stream_sources_disabled",
                "source_id": "needlebench_opencompass",
                "detail": "Pass --hf-stream-sources to stream official OpenCompass NeedleBench haystacks and retrieval needles into quarantine.",
            }
        ]
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        return [], [{"severity": "warning", "kind": "datasets_import_failed", "source_id": "needlebench_opencompass", "detail": f"{type(exc).__name__}: {exc}"}]
    try:
        haystack_rows = load_dataset("opencompass/needlebench", "en_haystack_texts", split="test", streaming=True)
        needle_rows = load_dataset("opencompass/needlebench", "retrieval_needles", split="test", streaming=True)
    except Exception as exc:
        return [], [{"severity": "warning", "kind": "needlebench_stream_failed", "source_id": "needlebench_opencompass", "detail": f"{type(exc).__name__}: {exc}"}]
    haystacks: list[str] = []
    for row in haystack_rows:
        if isinstance(row, dict) and str(row.get("text") or "").strip():
            haystacks.append(str(row.get("text") or ""))
        if len(haystacks) >= 12:
            break
    if not haystacks:
        return [], [{"severity": "warning", "kind": "needlebench_no_haystack_rows", "source_id": "needlebench_opencompass", "detail": "No English haystack rows were streamable."}]
    items: list[PublicMemoryItem] = []
    eligible_seen = 0
    for row in needle_rows:
        if not isinstance(row, dict) or str(row.get("language") or "").lower() != "english":
            continue
        if eligible_seen < offset:
            eligible_seen += 1
            continue
        eligible_seen += 1
        source_tokens = source_token_buckets[(eligible_seen - offset - 1) % len(source_token_buckets)]
        depth = [0.08, 0.34, 0.67, 0.92][(eligible_seen - offset - 1) % 4]
        item = needlebench_item(
            row,
            haystacks=haystacks,
            item_no=eligible_seen - 1,
            source_tokens=source_tokens,
            depth=depth,
        )
        if item:
            items.append(item)
        if len(items) >= max_items:
            break
    if not items:
        return [], [{"severity": "warning", "kind": "needlebench_no_english_rows", "source_id": "needlebench_opencompass", "detail": f"offset={offset} max_items={max_items} yielded no valid English retrieval rows"}]
    return items, []


def needlebench_item(
    row: dict[str, Any],
    *,
    haystacks: list[str],
    item_no: int,
    source_tokens: int,
    depth: float,
) -> PublicMemoryItem | None:
    needle = compact_space(str(row.get("needle") or ""))
    question = compact_space(str(row.get("retrieval_question") or ""))
    answer = compact_space(str(row.get("gold_standard_answer") or ""))
    if not needle or not question or not answer:
        return None
    target_chars = max(1024, source_tokens * 4)
    haystack = build_needlebench_haystack(haystacks, target_chars=max(128, target_chars - len(needle) - 32), offset=item_no)
    context, needle_line_no = insert_line_at_depth(haystack, needle, depth=depth)
    prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
    return PublicMemoryItem(
        item_id=f"needlebench_retrieval_en_{item_no:05d}_{length_bucket(source_tokens)}_{int(depth * 100):02d}",
        benchmark="needlebench_opencompass",
        task=f"retrieval_type_{row.get('type', 'unknown')}",
        prompt=prompt,
        context=context,
        question=question,
        answers=[answer],
        oracle_evidence=[{"id": f"needlebench:{item_no}:needle", "text": needle}],
        metadata={
            "official_family": "OpenCompass NeedleBench retrieval",
            "hf_dataset": "opencompass/needlebench",
            "split": "test",
            "generated_official_format": True,
            "source_task_config": "retrieval_needles+en_haystack_texts",
            "target_source_tokens": source_tokens,
            "target_source_token_bucket": length_bucket(source_tokens),
            "needle_depth": round(depth, 4),
            "needle_line_no": needle_line_no,
            "score_semantics": "local deterministic exact match against the requested answer format after retrieving the inserted official needle",
        },
    )


def build_infinitebench_retrievekv_items(
    *,
    max_items: int,
    offset: int = 0,
    source_token_buckets: list[int] | None = None,
) -> list[PublicMemoryItem]:
    source_token_buckets = source_token_buckets or [8000, 32000, 128000]
    items: list[PublicMemoryItem] = []
    for item_no in range(offset, offset + max_items):
        source_tokens = source_token_buckets[(item_no - offset) % len(source_token_buckets)]
        items.append(infinitebench_retrievekv_item(item_no=item_no, source_tokens=source_tokens))
    return items


def infinitebench_retrievekv_item(*, item_no: int, source_tokens: int) -> PublicMemoryItem:
    key = f"theseus_key_{item_no:06d}"
    answer = f"value_{(item_no * 7919 + 104729) % 10_000_000:07d}"
    target_chars = max(2048, source_tokens * 4)
    pair_count = max(64, min(40_000, target_chars // 48))
    lines: list[str] = ["{"]
    oracle_line = f'  "{key}": "{answer}",'
    insert_at = (item_no * 37 + 11) % max(1, pair_count)
    for idx in range(pair_count):
        if idx == insert_at:
            lines.append(oracle_line)
        distractor_key = f"distractor_key_{item_no:06d}_{idx:06d}"
        distractor_value = f"value_{(idx * 1543 + item_no * 97) % 10_000_000:07d}"
        lines.append(f'  "{distractor_key}": "{distractor_value}",')
    lines.append("}")
    context = "\n".join(lines)
    question = f'Extract the value corresponding to the specified key in the JSON object: "{key}".'
    prompt = f"Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{question}"
    return PublicMemoryItem(
        item_id=f"infinitebench_retrievekv_{item_no:06d}",
        benchmark="infinitebench",
        task="retrievekv",
        prompt=prompt,
        context=context,
        question=question,
        answers=[answer],
        oracle_evidence=[{"id": f"infinitebench:{item_no}:kv", "text": oracle_line}],
        metadata={
            "official_family": "InfiniteBench retrievekv",
            "source_task_config": "infinitebench_retrievekv",
            "generated_official_format": True,
            "target_source_tokens": source_tokens,
            "target_source_token_bucket": length_bucket(source_tokens),
            "score_semantics": "local deterministic exact value extraction from generated official retrievekv-style JSON context",
        },
    )


def build_needlebench_haystack(haystacks: list[str], *, target_chars: int, offset: int) -> str:
    parts: list[str] = []
    used = 0
    idx = offset % max(1, len(haystacks))
    while used < target_chars and haystacks:
        text = compact_space(haystacks[idx % len(haystacks)])
        if not text:
            idx += 1
            continue
        remaining = target_chars - used
        chunk = text[:remaining]
        parts.append(chunk)
        used += len(chunk) + 1
        idx += 1
    return "\n".join(parts)


def insert_line_at_depth(context: str, line: str, *, depth: float) -> tuple[str, int]:
    lines = [row for row in context.splitlines() if row.strip()]
    insert_at = min(len(lines), max(0, int(len(lines) * depth)))
    lines.insert(insert_at, line)
    return "\n".join(lines), insert_at + 1


def chunk_public_context(context: str, *, chunk_chars: int) -> str:
    paragraphs = [compact_space(part) for part in re.split(r"\n{2,}|\r\n", context) if compact_space(part)]
    chunks: list[str] = []
    for paragraph in paragraphs or [compact_space(context)]:
        cursor = 0
        while cursor < len(paragraph):
            chunk = paragraph[cursor: cursor + chunk_chars].strip()
            if chunk:
                chunks.append(f"chunk-{len(chunks) + 1:06d}: {chunk}")
            cursor += chunk_chars
    return "\n".join(chunks)


def longmemeval_item(row: dict[str, Any], *, source_name: str) -> PublicMemoryItem | None:
    question_id = str(row.get("question_id") or "")
    question = str(row.get("question") or "")
    answer = row.get("answer")
    answers = [str(value) for value in answer] if isinstance(answer, list) else [str(answer or "")]
    answers = [value for value in answers if value.strip()]
    sessions = list_value(row.get("haystack_sessions"))
    session_ids = [str(value) for value in list_value(row.get("haystack_session_ids"))]
    dates = [str(value) for value in list_value(row.get("haystack_dates"))]
    if not question_id or not question or not answers or not sessions:
        return None
    context_lines: list[str] = []
    evidence: list[dict[str, str]] = []
    answer_sessions = {str(value) for value in list_value(row.get("answer_session_ids"))}
    for idx, session in enumerate(sessions):
        session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
        date = dates[idx] if idx < len(dates) else ""
        turns = list_value(session)
        session_evidence: list[dict[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "")
            content = compact_space(str(turn.get("content") or ""))
            if not content:
                continue
            line = f"[{session_id}] {date} {role}: {content}" if role else f"[{session_id}] {date} {content}"
            context_lines.append(line)
            turn_id = f"longmemeval:{question_id}:{session_id}:turn{len(context_lines)}"
            if turn.get("has_answer") is True:
                evidence.append({"id": turn_id, "text": line})
            if session_id in answer_sessions:
                session_evidence.append({"id": turn_id, "text": line})
        if session_id in answer_sessions and not any(row["id"] in {ev["id"] for ev in evidence} for row in session_evidence):
            evidence.extend(session_evidence[:4])
    context = "\n".join(context_lines)
    prompt = f"<history>\n{context}\n</history>\n\nQuestion: {question}"
    return PublicMemoryItem(
        item_id=f"longmemeval_{question_id}",
        benchmark="longmemeval",
        task=str(row.get("question_type") or "memory_qa"),
        prompt=prompt,
        context=context,
        question=question,
        answers=answers,
        oracle_evidence=evidence,
        metadata={
            "official_family": "LongMemEval",
            "source_file": source_name,
            "question_id": question_id,
            "question_type": row.get("question_type"),
            "question_date": row.get("question_date"),
            "answer_session_count": len(answer_sessions),
            "score_semantics": "local deterministic exact/contains answer extraction over selected evidence context",
        },
    )


def parse_babi_task(
    raw: list[str],
    *,
    benchmark_path: str,
    task_id: str,
    task_name: str,
    limit: int,
    offset: int = 0,
) -> list[PublicMemoryItem]:
    items: list[PublicMemoryItem] = []
    current: list[dict[str, Any]] = []
    sample_no = 0
    eligible_seen = 0
    for line in raw:
        if not line.strip():
            continue
        number_text, text = line.split(" ", 1)
        phrase_num = int(number_text)
        if phrase_num == 1:
            current = []
            sample_no += 1
        if "\t" in text:
            parts = text.split("\t")
            question = parts[0]
            answer = parts[1]
            ref_nums = [int(value) for value in re.findall(r"\d+", " ".join(parts[2:]))]
            evidence = [
                {"id": f"{task_id}:sample{sample_no}:line{row['phrase_num']}", "text": row["text"]}
                for row in current
                if int(row["phrase_num"]) in set(ref_nums)
            ]
            if eligible_seen < offset:
                eligible_seen += 1
                continue
            eligible_seen += 1
            context_lines = [row["text"] for row in current]
            context = "\n".join(context_lines)
            prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
            items.append(
                PublicMemoryItem(
                    item_id=f"babilong_{task_id}_{sample_no:04d}_{len(items):02d}",
                    benchmark="babilong",
                    task=task_id,
                    prompt=prompt,
                    context=context,
                    question=question,
                    answers=[answer],
                    oracle_evidence=evidence,
                    metadata={
                        "official_family": "BABILong bAbI zero-noise prompt slice",
                        "source_path": benchmark_path,
                        "source_task_name": task_name,
                        "reference_phrase_nums": ref_nums,
                        "sample_no": sample_no,
                    },
                )
            )
            if len(items) >= limit:
                break
        else:
            current.append({"phrase_num": phrase_num, "text": text})
    return items


def score_item(item: PublicMemoryItem, *, context_budget_chars: int) -> dict[str, Any]:
    started = time.perf_counter()
    if item.metadata.get("status") == "queued":
        return queued_row(item)
    length_metrics = item_length_metrics(item)
    vcm_on = annotate_resolver_result(run_resolver(item, mode="vcm_on", context_budget_chars=context_budget_chars), length_metrics)
    vcm_off = annotate_resolver_result(run_resolver(item, mode="vcm_off", context_budget_chars=context_budget_chars), length_metrics)
    competing_memory_rows = [
        scored_resolver(item, mode=mode, context_budget_chars=context_budget_chars)
        for mode in [
            "flat_head_window",
            "middle_window",
            "lexical_retrieval",
            "bm25_sparse_retrieval",
            "recency_weighted_retrieval",
            "deterministic_hybrid_retrieval",
            "structured_state_table",
        ]
    ]
    competing_memory = {str(row.get("system") or f"memory_system_{idx}"): row for idx, row in enumerate(competing_memory_rows)}
    on_pass = score_prediction(item.answers, vcm_on["prediction"])
    off_pass = score_prediction(item.answers, vcm_off["prediction"])
    on_evidence = evidence_metrics(item, vcm_on["selected_evidence_ids"])
    off_evidence = evidence_metrics(item, vcm_off["selected_evidence_ids"])
    memory_systems = {
        "vcm_graph_evidence_selector": {
            **vcm_on,
            "passed": on_pass,
            "evidence_precision": on_evidence["precision"],
            "evidence_recall": on_evidence["recall"],
        },
        "flat_tail_window_baseline": {
            **vcm_off,
            "passed": off_pass,
            "evidence_precision": off_evidence["precision"],
            "evidence_recall": off_evidence["recall"],
        },
        **competing_memory,
    }
    row = {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "status": "scored",
        "context_budget_chars": context_budget_chars,
        **length_metrics,
        "answer_count": len(item.answers),
        "prompt_hash": stable_hash(item.prompt),
        "context_hash": stable_hash(item.context),
        "answer_hash": stable_hash(item.answers),
        "oracle_evidence_hash": stable_hash(item.oracle_evidence),
        "vcm_on": {
            **vcm_on,
            "passed": on_pass,
            "evidence_precision": on_evidence["precision"],
            "evidence_recall": on_evidence["recall"],
        },
        "vcm_off": {
            **vcm_off,
            "passed": off_pass,
            "evidence_precision": off_evidence["precision"],
            "evidence_recall": off_evidence["recall"],
        },
        "competing_memory_systems": competing_memory,
        "memory_systems": memory_systems,
        "best_non_vcm_memory_system": best_non_vcm_system(memory_systems),
        "winner": "vcm_on" if on_pass and not off_pass else ("vcm_off" if off_pass and not on_pass else "tie"),
        "residual_categories": residual_categories(item, vcm_on, on_pass, on_evidence),
        "stale_update_deletion": stale_update_deletion_eval(item, vcm_on, on_pass),
        "latency_ms": round((time.perf_counter() - started) * 1000, 4),
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "teacher_solving_calls": 0,
    }
    return row


def scored_resolver(item: PublicMemoryItem, *, mode: str, context_budget_chars: int) -> dict[str, Any]:
    resolved = annotate_resolver_result(
        run_resolver(item, mode=mode, context_budget_chars=context_budget_chars),
        item_length_metrics(item),
    )
    passed = score_prediction(item.answers, resolved["prediction"])
    evidence = evidence_metrics(item, resolved["selected_evidence_ids"])
    return {
        **resolved,
        "passed": passed,
        "evidence_precision": evidence["precision"],
        "evidence_recall": evidence["recall"],
    }


def best_non_vcm_system(memory_systems: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        {"system": name, "passed": bool(row.get("passed")), "no_admissible": bool(row.get("no_admissible"))}
        for name, row in memory_systems.items()
        if name != "vcm_graph_evidence_selector"
    ]
    passing = [row for row in candidates if row["passed"]]
    if passing:
        return sorted(passing, key=lambda row: row["system"])[0]
    admissible = [row for row in candidates if not row["no_admissible"]]
    if admissible:
        return sorted(admissible, key=lambda row: row["system"])[0]
    return sorted(candidates, key=lambda row: row["system"])[0] if candidates else {}


def run_resolver(item: PublicMemoryItem, *, mode: str, context_budget_chars: int) -> dict[str, Any]:
    if mode == "vcm_on":
        context, evidence_ids = select_vcm_evidence(item, context_budget_chars=context_budget_chars)
        system = "vcm_graph_evidence_selector"
    elif mode == "vcm_off":
        context, evidence_ids = select_flat_tail(item, context_budget_chars=context_budget_chars)
        system = "flat_tail_window_baseline"
    elif mode == "flat_head_window":
        context, evidence_ids = select_flat_head(item, context_budget_chars=context_budget_chars)
        system = "flat_head_window_baseline"
    elif mode == "middle_window":
        context, evidence_ids = select_middle_window(item, context_budget_chars=context_budget_chars)
        system = "middle_window_baseline"
    elif mode == "lexical_retrieval":
        context, evidence_ids = select_lexical_retrieval(item, context_budget_chars=context_budget_chars)
        system = "lexical_retrieval_memory"
    elif mode == "bm25_sparse_retrieval":
        context, evidence_ids = select_bm25_sparse_retrieval(item, context_budget_chars=context_budget_chars)
        system = "bm25_sparse_retrieval_memory"
    elif mode == "recency_weighted_retrieval":
        context, evidence_ids = select_recency_weighted_retrieval(item, context_budget_chars=context_budget_chars)
        system = "recency_weighted_retrieval_memory"
    elif mode == "deterministic_hybrid_retrieval":
        context, evidence_ids = select_deterministic_hybrid_retrieval(item, context_budget_chars=context_budget_chars)
        system = "deterministic_hybrid_retrieval_memory"
    elif mode == "structured_state_table":
        context, evidence_ids = select_structured_state_table(item, context_budget_chars=context_budget_chars)
        system = "structured_state_table_memory"
    else:
        raise ValueError(f"unknown resolver mode: {mode}")
    resolution_detail: dict[str, Any] = {}
    if item.benchmark == "ruler":
        prediction = resolve_ruler_niah(context, item.question)
    elif item.benchmark == "babilong":
        prediction = resolve_babilong(context, item.question, item.task)
    elif item.benchmark == "longmemeval":
        resolution_detail = resolve_longmemeval_detail(context, item.question)
        prediction = str(resolution_detail.get("prediction") or "")
    elif item.benchmark == "longbench_v2":
        prediction = resolve_longbench_v2(context, item.question, dict_value(item.metadata.get("choices")))
    elif item.benchmark == "needlebench_opencompass":
        prediction = resolve_needlebench(context, item.question)
    elif item.benchmark == "infinitebench":
        prediction = resolve_infinitebench_retrievekv(context, item.question)
    else:
        prediction = ""
    result = {
        "system": system,
        "prediction": prediction,
        "prediction_hash": stable_hash(prediction),
        "selected_evidence_ids": evidence_ids,
        "selected_context_chars": len(context),
        "no_admissible": prediction == "",
    }
    if resolution_detail:
        result.update(
            {
                "longmemeval_question_type": resolution_detail.get("question_type", "unknown"),
                "longmemeval_candidate_count": resolution_detail.get("candidate_count", 0),
                "longmemeval_best_score": resolution_detail.get("best_score", 0.0),
                "answer_span_chars": resolution_detail.get("answer_span_chars", 0),
                "abstention_reason": resolution_detail.get("abstention_reason", ""),
            }
        )
    return result


def select_vcm_evidence(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    selected: list[dict[str, str]] = []
    if item.benchmark == "ruler":
        query_keys = ruler_query_keys(item.question)
        selected.extend(lines_matching_any(item.context, query_keys))
    elif item.benchmark == "babilong":
        if item.task in {"qa1", "qa6"}:
            person = person_from_where_question(item.question) or first_capitalized_name(item.question)
            if person:
                selected.extend(lines_matching_any(item.context, [person]))
        elif item.task in {"qa2", "qa3"}:
            noun = object_from_question(item.question)
            if noun:
                selected.extend(babilong_state_lines_for_object(item.context, noun))
    elif item.benchmark == "longmemeval":
        selected.extend(select_longmemeval_evidence_rows(item, mode="vcm"))
        return pack_ranked_lines(item, selected, context_budget_chars=context_budget_chars)
    elif item.benchmark == "longbench_v2":
        selected.extend(select_longbench_v2_evidence_rows(item))
        return pack_ranked_lines(item, selected, context_budget_chars=context_budget_chars)
    elif item.benchmark == "needlebench_opencompass":
        selected.extend(select_needlebench_evidence_rows(item))
        return pack_ranked_lines(item, selected, context_budget_chars=context_budget_chars)
    elif item.benchmark == "infinitebench":
        selected.extend(select_infinitebench_evidence_rows(item))
        return pack_ranked_lines(item, selected, context_budget_chars=context_budget_chars)
    selected = ordered_unique_evidence(selected)
    text = "\n".join(row["text"] for row in selected)
    if len(text) > context_budget_chars:
        text = text[-context_budget_chars:]
    return text, evidence_ids_in_text(item, text)


def select_flat_tail(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    context = item.context[-context_budget_chars:]
    return context, evidence_ids_in_text(item, context)


def select_flat_head(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    context = item.context[:context_budget_chars]
    return context, evidence_ids_in_text(item, context)


def select_middle_window(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    if len(item.context) <= context_budget_chars:
        context = item.context
    else:
        start = max(0, (len(item.context) - context_budget_chars) // 2)
        context = item.context[start:start + context_budget_chars]
    return context, evidence_ids_in_text(item, context)


def select_lexical_retrieval(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    ranked = rank_context_lines(item.context, item.question)
    return pack_ranked_lines(item, ranked[:8], context_budget_chars=context_budget_chars)


def select_bm25_sparse_retrieval(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    ranked = rank_context_lines_bm25(item.context, item.question)
    return pack_ranked_lines(item, ranked[:12], context_budget_chars=context_budget_chars)


def select_recency_weighted_retrieval(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    ranked = rank_context_lines(item.context, item.question, recency_weight=0.35)
    return pack_ranked_lines(item, ranked[:8], context_budget_chars=context_budget_chars)


def select_deterministic_hybrid_retrieval(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    rows = []
    rows.extend(rank_context_lines_bm25(item.context, item.question)[:8])
    rows.extend(rank_context_lines(item.context, item.question, recency_weight=0.25)[:4])
    if item.benchmark == "babilong":
        rows.extend(select_structured_state_table(item, context_budget_chars=context_budget_chars)[0].splitlines())
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
        else:
            text = str(row)
            normalized_rows.append({"id": f"context:line{evidence_line_number(text, item.context)}", "text": text, "line_no": str(evidence_line_number(text, item.context))})
    return pack_ranked_lines(item, normalized_rows[:16], context_budget_chars=context_budget_chars)


def select_structured_state_table(item: PublicMemoryItem, *, context_budget_chars: int) -> tuple[str, list[str]]:
    if item.benchmark == "ruler":
        query_keys = ruler_query_keys(item.question)
        rows = lines_matching_any(item.context, query_keys)
    elif item.benchmark == "babilong" and item.task in {"qa1", "qa6"}:
        person = person_from_where_question(item.question) or first_capitalized_name(item.question)
        rows = babilong_current_person_state_rows(item.context, person) if person else []
    elif item.benchmark == "babilong" and item.task in {"qa2", "qa3"}:
        noun = object_from_question(item.question)
        rows = babilong_current_object_state_rows(item.context, noun) if noun else []
    elif item.benchmark == "longmemeval":
        rows = select_longmemeval_evidence_rows(item, mode="structured")
        return pack_ranked_lines(item, rows, context_budget_chars=context_budget_chars)
    elif item.benchmark == "longbench_v2":
        rows = select_longbench_v2_evidence_rows(item, structured=True)
        return pack_ranked_lines(item, rows, context_budget_chars=context_budget_chars)
    elif item.benchmark == "needlebench_opencompass":
        rows = select_needlebench_evidence_rows(item, structured=True)
        return pack_ranked_lines(item, rows, context_budget_chars=context_budget_chars)
    elif item.benchmark == "infinitebench":
        rows = select_infinitebench_evidence_rows(item)
        return pack_ranked_lines(item, rows, context_budget_chars=context_budget_chars)
    else:
        rows = []
    rows = ordered_unique_evidence(rows)
    text = "\n".join(row["text"] for row in rows)
    if len(text) > context_budget_chars:
        text = text[-context_budget_chars:]
    return text, evidence_ids_in_text(item, text)


def select_longbench_v2_evidence_rows(item: PublicMemoryItem, *, structured: bool = False) -> list[dict[str, str]]:
    choices = dict_value(item.metadata.get("choices"))
    choice_text = " ".join(str(value) for value in choices.values())
    query = f"{item.question}\n{choice_text}"
    bm25_rows = rank_context_lines_bm25(item.context, query)
    lexical_rows = rank_context_lines(item.context, query, recency_weight=0.05)
    rows = bm25_rows[:18] + lexical_rows[:8]
    if structured:
        rows = rows[:10]
    return unique_evidence_by_rank(rows)


def select_needlebench_evidence_rows(item: PublicMemoryItem, *, structured: bool = False) -> list[dict[str, str]]:
    query = needlebench_query_for_retrieval(item.question)
    rows = rank_context_lines_bm25(item.context, query)
    rows.extend(rank_context_lines(item.context, query, recency_weight=0.02)[:8])
    if structured:
        rows = [
            row
            for row in rows
            if likely_needlebench_answer_line(str(row.get("text") or ""), item.question)
        ] or rows
    return unique_evidence_by_rank(rows[:16])


def select_infinitebench_evidence_rows(item: PublicMemoryItem) -> list[dict[str, str]]:
    key = infinitebench_query_key(item.question)
    if key:
        return lines_matching_any(item.context, [f'"{key}"', key])[:4]
    return rank_context_lines_bm25(item.context, item.question)[:8]


def lines_matching_any(context: str, needles: list[str]) -> list[dict[str, str]]:
    wanted = [needle for needle in needles if needle]
    rows = []
    for idx, line in enumerate(context.splitlines(), start=1):
        if any(needle in line for needle in wanted):
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    return rows


def babilong_state_lines_for_object(context: str, noun: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    carriers: set[str] = set()
    lines = context.splitlines()
    for line in lines:
        take = parse_take(line)
        give = parse_give(line)
        drop = parse_drop(line)
        if take and take.group(2) == noun:
            carriers.add(take.group(1))
        if give and give.group(2) == noun:
            carriers.add(give.group(1))
            carriers.add(give.group(3))
        if drop and drop.group(2) == noun:
            carriers.add(drop.group(1))
    for idx, line in enumerate(lines, start=1):
        take = parse_take(line)
        give = parse_give(line)
        if take and take.group(2) == noun:
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
            continue
        if give and give.group(2) == noun:
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
            continue
        if noun in line:
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
            continue
        move = parse_move(line)
        if move and move.group(1) in carriers:
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    return rows


def babilong_current_person_state_rows(context: str, person: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, line in enumerate(context.splitlines(), start=1):
        move = parse_move(line)
        if move and move.group(1) == person:
            rows = [{"id": f"context:line{idx}", "text": line, "line_no": str(idx)}]
    return rows


def babilong_current_object_state_rows(context: str, noun: str) -> list[dict[str, str]]:
    _, _, object_sources = track_person_and_item_locations_with_sources(context.splitlines())
    source_lines = object_sources.get(noun, [])
    return [{"id": f"context:line{idx}", "text": text, "line_no": str(idx)} for idx, text in source_lines[-2:]]


def select_longmemeval_evidence_rows(item: PublicMemoryItem, *, mode: str) -> list[dict[str, str]]:
    plan = longmemeval_query_plan(item.question)
    ranked = rank_longmemeval_lines(item.context, item.question)
    if mode == "structured":
        structured = longmemeval_structured_rows(item.context, item.question, plan)
        return ordered_unique_evidence(structured + ranked[:4])[:8]
    # VCM keeps a wider, timestamp-preserving evidence working set for multi-session memory,
    # then fuses query-decomposed, temporal, preference, and current-update rows.
    rows: list[dict[str, str]] = []
    rows.extend(ranked[:12])
    rows.extend(longmemeval_structured_rows(item.context, item.question, plan))
    rows.extend(longmemeval_decomposed_query_rows(item.context, item.question, plan))
    rows.extend(longmemeval_alias_bridge_rows(item.context, item.question, plan))
    rows.extend(longmemeval_followup_rows(item.context, item.question, plan))
    rows.extend(longmemeval_option_rows(item.context, plan))
    return unique_evidence_by_rank(rows)[:28]


def rank_longmemeval_lines(context: str, question: str) -> list[dict[str, str]]:
    lines = context.splitlines()
    plan = longmemeval_query_plan(question)
    query_terms = meaningful_terms(question)
    if not query_terms:
        return []
    bm25_rows = {int(row.get("line_no") or 0): row for row in rank_context_lines_bm25(context, question)}
    ranked: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        turn = parse_longmemeval_turn(line, idx)
        terms = meaningful_terms(line)
        overlap = len(query_terms & terms)
        option_overlap = max((len(terms & meaningful_terms(option)) for option in plan["options"]), default=0)
        subject_overlap = len(terms & set(plan["subject_terms"]))
        cue_bonus = longmemeval_plan_line_bonus(line, plan)
        if overlap <= 0 and option_overlap <= 0 and subject_overlap <= 0 and cue_bonus <= 0.0:
            continue
        bm25_rank_bonus = 1.0 if idx in bm25_rows else 0.0
        role_bonus = 0.35 if turn["role"] == "user" else 0.0
        date_bonus = 0.25 if question_mentions_time(question) and re.search(r"\d{4}/\d{2}/\d{2}", line) else 0.0
        answer_shape_bonus = 0.2 if likely_answer_bearing_sentence(line, question) else 0.0
        recency_bonus = 0.0
        if plan["requires_latest"] and turn["date_key"]:
            recency_bonus = min(0.6, idx / max(1, len(lines)))
        option_bonus = 0.45 if option_overlap else 0.0
        ranked.append((
            overlap
            + bm25_rank_bonus
            + role_bonus
            + date_bonus
            + answer_shape_bonus
            + recency_bonus
            + option_bonus
            + cue_bonus,
            idx,
            line,
        ))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [{"id": f"context:line{idx}", "text": line, "line_no": str(idx)} for _, idx, line in ranked]


def longmemeval_query_plan(question: str) -> dict[str, Any]:
    lowered = question.lower()
    quoted_options = [compact_space(value) for value in re.findall(r"['\"]([^'\"]{2,120})['\"]", question)]
    options = quoted_options + longmemeval_or_options(question)
    raw_terms = meaningful_terms(question)
    relation_terms = {
        "after",
        "before",
        "current",
        "currently",
        "favorite",
        "favourite",
        "first",
        "happen",
        "happened",
        "latest",
        "last",
        "like",
        "now",
        "prefer",
        "preferred",
        "recent",
        "recently",
        "remember",
        "which",
        "when",
        "where",
        "who",
        "why",
    }
    subject_terms = sorted(raw_terms - relation_terms)
    if any(term in lowered for term in ["prefer", "preferred", "favorite", "favourite", "like best", "would rather"]):
        question_type = "preference"
    elif "first" in lowered or "before" in lowered or "earliest" in lowered:
        question_type = "temporal_first"
    elif lowered.startswith("where"):
        question_type = "where"
    elif lowered.startswith("who"):
        question_type = "who"
    elif lowered.startswith("when"):
        question_type = "when"
    elif "last" in lowered or "latest" in lowered or "recent" in lowered:
        question_type = "temporal_last"
    elif any(term in lowered for term in ["current", "currently", "now", "latest", "today", "most recent"]):
        question_type = "current_update"
    elif options or lowered.startswith("which"):
        question_type = "choice"
    else:
        question_type = "fact"
    return {
        "question_type": question_type,
        "options": sorted(set(options), key=lambda value: value.lower()),
        "subject_terms": subject_terms,
        "requires_latest": question_type in {"current_update", "temporal_last"} or any(
            term in lowered for term in ["now", "current", "latest", "most recent"]
        ),
        "requires_first": question_type == "temporal_first",
        "requires_preference": question_type == "preference",
        "requires_time": question_mentions_time(question) or question_type in {"temporal_first", "temporal_last", "when"},
    }


def longmemeval_or_options(question: str) -> list[str]:
    match = re.search(r"\b(?:the|a|an|my)?\s*([A-Za-z0-9][A-Za-z0-9 -]{2,80}?)\s+or\s+(?:the|a|an|my)?\s*([A-Za-z0-9][A-Za-z0-9 -]{2,80}?)(?:\?|$)", question)
    if not match:
        return []
    options = []
    for group in match.groups():
        option = compact_space(group)
        option = re.sub(r"^(?:the|a|an|my)\s+", "", option, flags=re.IGNORECASE)
        if len(meaningful_terms(option)) <= 8:
            options.append(option)
    return options


def parse_longmemeval_turn(line: str, line_no: int = 0) -> dict[str, Any]:
    match = re.match(
        r"^\[([^\]]+)\]\s+(\d{4}/\d{2}/\d{2})(?:\s+\([^)]*\))?(?:\s+\d{1,2}:\d{2})?\s+(user|assistant):\s*(.*)$",
        line.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "session_id": "",
            "date_text": "",
            "date_key": 0,
            "role": "",
            "content": strip_longmemeval_prefix(line),
            "line_no": line_no,
        }
    date_text = match.group(2)
    return {
        "session_id": match.group(1),
        "date_text": date_text,
        "date_key": int(date_text.replace("/", "")),
        "role": match.group(3).lower(),
        "content": match.group(4).strip(),
        "line_no": line_no,
    }


def longmemeval_structured_rows(context: str, question: str, plan: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    parsed = [
        (idx, line, parse_longmemeval_turn(line, idx), meaningful_terms(line))
        for idx, line in enumerate(context.splitlines(), start=1)
    ]
    subject_terms = set(plan["subject_terms"])
    for idx, line, turn, terms in parsed:
        if turn["role"] != "user":
            continue
        if subject_terms and not (terms & subject_terms):
            continue
        if longmemeval_plan_line_bonus(line, plan) <= 0.0 and not likely_answer_bearing_sentence(line, question):
            continue
        rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    if plan["requires_latest"]:
        rows.sort(key=lambda row: -int(parse_longmemeval_turn(row["text"], int(row["line_no"]))["date_key"] or 0))
    elif plan["requires_first"]:
        rows.sort(key=lambda row: int(parse_longmemeval_turn(row["text"], int(row["line_no"]))["date_key"] or row["line_no"]))
    return rows[:10]


def longmemeval_decomposed_query_rows(context: str, question: str, plan: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    groups = [set(plan["subject_terms"])]
    groups.extend(meaningful_terms(option) for option in plan["options"])
    for group in groups:
        if not group:
            continue
        scored: list[tuple[float, int, str]] = []
        for idx, line in enumerate(context.splitlines(), start=1):
            terms = meaningful_terms(line)
            overlap = len(terms & group)
            if overlap <= 0:
                continue
            scored.append((overlap + longmemeval_plan_line_bonus(line, plan), idx, line))
        for _, idx, line in sorted(scored, key=lambda value: (-value[0], value[1]))[:4]:
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    return rows


def longmemeval_alias_bridge_rows(context: str, question: str, plan: dict[str, Any]) -> list[dict[str, str]]:
    query_terms = meaningful_terms(question)
    if not query_terms:
        return []
    alias_terms: set[str] = set()
    rows: list[dict[str, str]] = []
    lines = context.splitlines()
    for idx, line in enumerate(lines, start=1):
        for left, right in longmemeval_alias_pairs(line):
            left_terms = meaningful_terms(left)
            right_terms = meaningful_terms(right)
            if query_terms & left_terms:
                alias_terms.update(right_terms)
                rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
            if query_terms & right_terms:
                alias_terms.update(left_terms)
                rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    if not alias_terms:
        return rows
    subject_terms = set(plan.get("subject_terms") or [])
    answer_shape_terms = {
        "access",
        "address",
        "code",
        "contact",
        "current",
        "file",
        "folder",
        "key",
        "location",
        "owner",
        "password",
        "status",
    }
    for idx, line in enumerate(lines, start=1):
        terms = meaningful_terms(line)
        if not (terms & alias_terms):
            continue
        if (subject_terms and terms & subject_terms) or terms & answer_shape_terms or likely_answer_bearing_sentence(line, question):
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    return ordered_unique_evidence(rows)[:12]


def longmemeval_alias_pairs(line: str) -> list[tuple[str, str]]:
    text = strip_longmemeval_prefix(compact_space(line))
    if not text:
        return []
    patterns = [
        r"\b(.{2,80}?)\s+(?:means|refers to|maps to|points to|is an alias for|is the alias for)\s+(.{2,80}?)(?:[.,;!?]|$)",
        r"\b(?:alias|cue|codeword|handle)\s+(.{2,80}?)\s+(?:means|refers to|maps to|points to)\s+(.{2,80}?)(?:[.,;!?]|$)",
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            left = clean_alias_phrase(match.group(1))
            right = clean_alias_phrase(match.group(2))
            if left and right:
                pairs.append((left, right))
    return pairs


def clean_alias_phrase(value: str) -> str:
    phrase = compact_space(value)
    phrase = re.sub(r"^(?:the|a|an|alias|cue|codeword|handle)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s+(?:and|but|while|because|after|before)\b.*$", "", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip(" .,:;!?\"'")
    if not phrase or len(phrase) > 80:
        return ""
    return phrase


def longmemeval_followup_rows(context: str, question: str, plan: dict[str, Any]) -> list[dict[str, str]]:
    subject_terms = set(plan["subject_terms"])
    if not subject_terms:
        return []
    rows: list[dict[str, str]] = []
    anchor_seen = False
    for idx, line in enumerate(context.splitlines(), start=1):
        terms = meaningful_terms(line)
        if terms & subject_terms:
            anchor_seen = True
            rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
            continue
        if not anchor_seen:
            continue
        lowered = line.lower()
        if not re.search(r"\b(it|its|that|this|they|them|there|now|current|currently|moved|changed|updated|renamed|stored|left|kept|prefer|chose|picked|selected|set for|scheduled|contact|code|password)\b", lowered):
            continue
        if longmemeval_plan_line_bonus(line, plan) <= 0.0 and not likely_answer_bearing_sentence(line, question):
            continue
        rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    if plan["requires_latest"]:
        rows.sort(key=lambda row: -int(parse_longmemeval_turn(row["text"], int(row["line_no"]))["date_key"] or 0))
    return rows[:8]


def longmemeval_option_rows(context: str, plan: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for option in plan["options"]:
        option_terms = meaningful_terms(option)
        if not option_terms:
            continue
        for idx, line in enumerate(context.splitlines(), start=1):
            if meaningful_terms(line) & option_terms:
                rows.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    return rows


def longmemeval_plan_line_bonus(line: str, plan: dict[str, Any]) -> float:
    lowered = line.lower()
    bonus = 0.0
    if plan["requires_latest"] and re.search(r"\b(now|current|currently|latest|today|updated|changed|moved|renamed|became|is now)\b", lowered):
        bonus += 0.8
    if plan["requires_first"] and re.search(r"\b(first|initial|originally|earliest|before|after|later|noticed|began|started)\b", lowered):
        bonus += 0.65
    if plan["requires_preference"] and re.search(r"\b(prefer|preferred|favorite|favourite|like best|would rather|chose|picked|selected)\b", lowered):
        bonus += 0.9
    if plan["question_type"] == "where" and re.search(r"\b(in|at|to|inside|near|from|moved|stored|left|kept)\b", lowered):
        bonus += 0.45
    if plan["question_type"] == "who" and re.search(r"\b(with|by|from|met|called|named|person|friend|doctor|teacher|manager)\b", lowered):
        bonus += 0.35
    return bonus


def rank_context_lines(context: str, question: str, *, recency_weight: float = 0.0) -> list[dict[str, str]]:
    lines = context.splitlines()
    query_terms = meaningful_terms(question)
    ranked: list[tuple[float, int, str]] = []
    denominator = max(1, len(lines) - 1)
    for idx, line in enumerate(lines, start=1):
        line_terms = meaningful_terms(line)
        overlap = len(query_terms & line_terms)
        if overlap <= 0:
            continue
        recency = ((idx - 1) / denominator) * recency_weight
        ranked.append((overlap + recency, idx, line))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [{"id": f"context:line{idx}", "text": line, "line_no": str(idx)} for _, idx, line in ranked]


def rank_context_lines_bm25(context: str, question: str) -> list[dict[str, str]]:
    lines = context.splitlines()
    query_terms = meaningful_terms(question)
    if not query_terms:
        return []
    doc_terms = [meaningful_terms(line) for line in lines]
    doc_freq: dict[str, int] = {}
    for terms in doc_terms:
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    n_docs = max(1, len(lines))
    avg_len = sum(len(terms) for terms in doc_terms) / max(1, len(doc_terms))
    ranked: list[tuple[float, int, str]] = []
    for idx, (line, terms) in enumerate(zip(lines, doc_terms), start=1):
        if not terms:
            continue
        score = 0.0
        term_counts = {term: list(terms).count(term) for term in query_terms if term in terms}
        for term, tf in term_counts.items():
            idf = max(0.0, (n_docs - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
            idf = 1.0 + idf
            denom = tf + 1.2 * (1 - 0.75 + 0.75 * (len(terms) / max(1.0, avg_len)))
            score += idf * ((tf * 2.2) / denom)
        if score > 0.0:
            ranked.append((score, idx, line))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [{"id": f"context:line{idx}", "text": line, "line_no": str(idx)} for _, idx, line in ranked]


def meaningful_terms(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "where",
        "what",
        "all",
        "for",
        "in",
        "to",
        "of",
        "mentioned",
        "provided",
        "text",
    }
    return {term.lower() for term in re.findall(r"[A-Za-z0-9-]+", text) if term.lower() not in stop}


def pack_ranked_lines(item: PublicMemoryItem, rows: list[dict[str, str]], *, context_budget_chars: int) -> tuple[str, list[str]]:
    unique_ranked = unique_evidence_by_rank(rows)
    selected_rows: list[tuple[int, str]] = []
    selected_full_texts: set[str] = set()
    used = 0
    for row in unique_ranked:
        text = row["text"]
        cost = len(text) + (1 if selected_rows else 0)
        if selected_rows and used + cost > context_budget_chars:
            continue
        if not selected_rows and cost > context_budget_chars:
            text = compress_selected_line(text, item.question, context_budget_chars)
            cost = len(text)
        selected_full_texts.add(str(row.get("text") or ""))
        selected_rows.append((int(row.get("line_no") or 1_000_000), text))
        used += cost
    selected = [text for _, text in sorted(selected_rows, key=lambda value: value[0])]
    context = "\n".join(selected)
    selected_ids = [
        row["id"]
        for row in item.oracle_evidence
        if row["text"] in context or row["text"] in selected_full_texts
    ]
    return context, selected_ids


def unique_evidence_by_rank(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen_events: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        text = str(row.get("text") or "")
        key = (str(row.get("line_no") or ""), text)
        if not text or key in seen_events:
            continue
        seen_events.add(key)
        unique.append(row)
    return unique


def compress_selected_line(text: str, question: str, context_budget_chars: int) -> str:
    if len(text) <= context_budget_chars:
        return text
    sentences = [compact_space(part) for part in re.split(r"(?<=[.!?])\s+|\n+", text) if compact_space(part)]
    if not sentences:
        return text[:context_budget_chars]
    question_terms = meaningful_terms(question)
    scored = [
        (longmemeval_sentence_score(strip_longmemeval_prefix(sentence), question_terms, question), idx, sentence)
        for idx, sentence in enumerate(sentences)
    ]
    keep_indexes = {idx for score, idx, _ in sorted(scored, key=lambda row: (-row[0], row[1]))[:6] if score > 0.0}
    if not keep_indexes:
        return text[:context_budget_chars]
    kept = [sentences[idx] for idx in sorted(keep_indexes)]
    packed: list[str] = []
    used = 0
    for sentence in kept:
        sentence = strip_longmemeval_prefix(sentence)
        cost = len(sentence) + (1 if packed else 0)
        if packed and used + cost > context_budget_chars:
            continue
        if not packed and cost > context_budget_chars:
            sentence = sentence[:context_budget_chars]
            cost = len(sentence)
        packed.append(sentence)
        used += cost
    return " ".join(packed) if packed else text[:context_budget_chars]


def evidence_ids_in_text(item: PublicMemoryItem, context: str) -> list[str]:
    return [row["id"] for row in item.oracle_evidence if row["text"] in context]


def resolve_ruler_niah(context: str, question: str) -> str:
    query_keys = ruler_query_keys(question)
    matches: dict[str, list[str]] = {}
    for match in re.finditer(r"special magic \w+ for ([\w-]+) is:\s*([\w-]+)", context, flags=re.IGNORECASE):
        matches.setdefault(match.group(1), []).append(match.group(2))
    answers = [value for key in query_keys for value in matches.get(key, [])]
    return ", ".join(answers)


def resolve_babilong(context: str, question: str, task: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if task == "qa1":
        person = person_from_where_question(question)
        if not person:
            return ""
        locations = track_person_locations(lines)
        return locations.get(person, "")
    if task == "qa2":
        item = object_from_question(question)
        if not item:
            return ""
        _, item_locations = track_person_and_item_locations(lines)
        return item_locations.get(item, "")
    if task == "qa3":
        item, before_location = before_location_query(question)
        if not item or not before_location:
            return ""
        _, item_locations, item_history = track_person_and_item_history(lines)
        history = item_history.get(item, [])
        for idx in range(len(history) - 1, 0, -1):
            location = history[idx]
            if location == before_location and idx > 0:
                return history[idx - 1]
        return item_locations.get(item, "")
    if task == "qa6":
        match = re.search(r"Is ([A-Z][a-z]+) in the ([a-z]+)", question)
        if not match:
            return ""
        person, location = match.group(1), match.group(2)
        locations = track_person_locations(lines)
        if person not in locations:
            return ""
        return "yes" if locations[person] == location else "no"
    return ""


def resolve_longbench_v2(context: str, question: str, choices: dict[str, Any]) -> str:
    clean_choices = {
        str(letter).upper(): compact_space(str(text))
        for letter, text in choices.items()
        if str(letter).upper() in {"A", "B", "C", "D"} and compact_space(str(text))
    }
    if not context or not clean_choices:
        return ""
    context_terms = meaningful_terms(context)
    question_terms = meaningful_terms(question)
    if not context_terms:
        return ""
    term_owners: dict[str, set[str]] = {}
    for letter, text in clean_choices.items():
        for term in meaningful_terms(text):
            term_owners.setdefault(term, set()).add(letter)
    scored: list[tuple[float, str]] = []
    lowered_context = context.lower()
    for letter, text in clean_choices.items():
        choice_terms = meaningful_terms(text)
        distinctive = {term for term in choice_terms if len(term_owners.get(term, set())) == 1}
        overlap = choice_terms & context_terms
        distinctive_overlap = distinctive & context_terms
        phrase_bonus = 6.0 if text.lower() in lowered_context else 0.0
        question_echo_penalty = 0.25 * len(choice_terms & question_terms)
        score = float(len(overlap)) + (2.0 * len(distinctive_overlap)) + phrase_bonus - question_echo_penalty
        scored.append((score, letter))
    scored.sort(key=lambda row: (-row[0], row[1]))
    if not scored or scored[0][0] <= 0.0:
        return ""
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return ""
    return scored[0][1]


def resolve_needlebench(context: str, question: str) -> str:
    if not context:
        return ""
    candidates = [
        line.strip()
        for line in context.splitlines()
        if likely_needlebench_answer_line(line, question)
    ]
    if not candidates:
        return ""
    candidate = sorted(candidates, key=lambda value: (-needlebench_line_score(value, question), len(value)))[0]
    entity = needlebench_entity_from_line(candidate)
    if not entity:
        return ""
    template = needlebench_answer_template(question)
    if not template:
        return candidate
    replacement = entity
    prefix = template.split("_", 1)[0]
    if prefix and not prefix.endswith((" ", "\t")) and not replacement.startswith((" ", "\t", ".", ",", ";", ":")):
        replacement = " " + replacement
    return re.sub(r"_+", replacement, template).strip()


def needlebench_query_for_retrieval(question: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]*_{2,}[^'\"]*)['\"]", question)
    without_template = question
    for template in quoted:
        without_template = without_template.replace(template, "")
    return compact_space(without_template)


def needlebench_answer_template(question: str) -> str:
    match = re.search(r"['\"]([^'\"]*_{2,}[^'\"]*)['\"]", question)
    return compact_space(match.group(1)) if match else ""


def likely_needlebench_answer_line(line: str, question: str) -> bool:
    text = compact_space(line)
    if not text:
        return False
    query_terms = meaningful_terms(needlebench_query_for_retrieval(question))
    line_terms = meaningful_terms(text)
    if not query_terms or not line_terms:
        return False
    if len(query_terms & line_terms) < 2:
        return False
    return bool(re.search(r"\b(?:was|is|are|were)\b", text, flags=re.IGNORECASE))


def needlebench_line_score(line: str, question: str) -> float:
    line_terms = meaningful_terms(line)
    query_terms = meaningful_terms(needlebench_query_for_retrieval(question))
    score = float(len(line_terms & query_terms))
    if re.search(r"\bfirst\b", line, flags=re.IGNORECASE):
        score += 0.5
    if re.search(r"\bwas\b", line, flags=re.IGNORECASE):
        score += 0.5
    return score


def needlebench_entity_from_line(line: str) -> str:
    text = compact_space(line)
    match = re.search(r"\b(?:was|is|are|were)\s+(.+?)(?:[.!?]|$)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    entity = compact_space(match.group(1))
    entity = entity.strip(" .,:;!?\"'")
    if not entity or len(entity) > 160:
        return ""
    return entity


def resolve_infinitebench_retrievekv(context: str, question: str) -> str:
    key = infinitebench_query_key(question)
    if not key:
        return ""
    pattern = r'"' + re.escape(key) + r'"\s*:\s*"([^"]+)"'
    match = re.search(pattern, context)
    return match.group(1) if match else ""


def infinitebench_query_key(question: str) -> str:
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        return quoted[-1]
    match = re.search(r"\b(theseus_key_[A-Za-z0-9_]+)\b", question)
    return match.group(1) if match else ""


def resolve_longmemeval(context: str, question: str) -> str:
    return str(resolve_longmemeval_detail(context, question).get("prediction") or "")


def resolve_longmemeval_detail(context: str, question: str) -> dict[str, Any]:
    plan = longmemeval_query_plan(question)
    question_terms = meaningful_terms(question)
    candidates = longmemeval_candidate_sentences(context, question, plan)
    if longmemeval_explicit_unknown_context(context, question, plan):
        return {
            "prediction": "",
            "question_type": plan["question_type"],
            "candidate_count": len(candidates),
            "best_score": 0.0,
            "answer_span_chars": 0,
            "abstention_reason": "explicit_unknown_context",
        }
    if not candidates:
        return {
            "prediction": "",
            "question_type": plan["question_type"],
            "candidate_count": 0,
            "best_score": 0.0,
            "answer_span_chars": 0,
            "abstention_reason": "no_ranked_context",
        }
    best = choose_longmemeval_candidate(candidates, plan)
    best_score = float(longmemeval_candidate_selection_score(best, plan))
    threshold = max(1.15, min(3.4, len(question_terms) * 0.18))
    if plan["question_type"] in {"current_update", "preference", "temporal_first", "temporal_last", "choice"}:
        threshold = max(1.0, threshold - 0.3)
    if best_score < threshold or not likely_answer_bearing_sentence(str(best.get("sentence") or ""), question):
        return {
            "prediction": "",
            "question_type": plan["question_type"],
            "candidate_count": len(candidates),
            "best_score": round(best_score, 6),
            "answer_span_chars": 0,
            "abstention_reason": "below_answer_threshold",
        }
    if not longmemeval_context_supports_question(context, question, plan):
        return {
            "prediction": "",
            "question_type": plan["question_type"],
            "candidate_count": len(candidates),
            "best_score": round(best_score, 6),
            "answer_span_chars": 0,
            "abstention_reason": "missing_subject_support",
        }
    span = compact_longmemeval_answer_span(str(best.get("sentence") or ""), question, plan)
    if not span:
        return {
            "prediction": "",
            "question_type": plan["question_type"],
            "candidate_count": len(candidates),
            "best_score": round(best_score, 6),
            "answer_span_chars": 0,
            "abstention_reason": "no_compact_span",
        }
    return {
        "prediction": span[:600],
        "question_type": plan["question_type"],
        "candidate_count": len(candidates),
        "best_score": round(best_score, 6),
        "answer_span_chars": len(span),
        "abstention_reason": "",
    }


def longmemeval_candidate_sentences(context: str, question: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = rank_longmemeval_lines(context, question)
    alias_terms = longmemeval_alias_terms_for_question(context, question)
    if alias_terms:
        existing_line_numbers = {int(row.get("line_no") or 0) for row in ranked}
        for idx, line in enumerate(context.splitlines(), start=1):
            if idx in existing_line_numbers:
                continue
            if meaningful_terms(line) & alias_terms:
                ranked.append({"id": f"context:line{idx}", "text": line, "line_no": str(idx)})
    question_terms = meaningful_terms(question)
    candidates: list[dict[str, Any]] = []
    for row in ranked[:24]:
        line_no = int(row.get("line_no") or 0)
        turn = parse_longmemeval_turn(row["text"], line_no)
        for sentence in split_longmemeval_sentences(row["text"]):
            sentence = strip_longmemeval_prefix(compact_space(sentence))
            if not sentence:
                continue
            score = longmemeval_sentence_score(sentence, question_terms, question)
            if alias_terms and meaningful_terms(sentence) & alias_terms:
                score += 1.4
            score += longmemeval_plan_line_bonus(sentence, plan)
            if turn["role"] == "user":
                score += 0.3
            if plan["options"] and longmemeval_sentence_option(sentence, plan):
                score += 0.9
            if plan["requires_latest"] and turn["date_key"]:
                score += min(0.7, max(0, line_no) / max(1, len(context.splitlines())))
            candidates.append(
                {
                    "sentence": sentence,
                    "line_no": line_no,
                    "date_key": int(turn.get("date_key") or 0),
                    "role": turn.get("role") or "",
                    "score": score,
                }
            )
    return candidates


def split_longmemeval_sentences(text: str) -> list[str]:
    return [
        compact_space(part)
        for part in re.split(r"(?<=[.!?])\s+|\s+\|\s+|(?:\s+-\s+)", text)
        if compact_space(part)
    ]


def choose_longmemeval_candidate(candidates: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    if not candidates:
        return {}
    scored = [(longmemeval_candidate_selection_score(row, plan), row) for row in candidates]
    if plan["requires_latest"]:
        plausible = [row for selection_score, row in scored if selection_score >= 2.0]
        if plausible:
            return sorted(
                plausible,
                key=lambda row: (
                    longmemeval_candidate_selection_score(row, plan),
                    int(row.get("date_key") or 0),
                    int(row.get("line_no") or 0),
                ),
                reverse=True,
            )[0]
    if plan["requires_first"]:
        plausible = [row for selection_score, row in scored if selection_score >= 2.0]
        if plausible:
            return sorted(
                plausible,
                key=lambda row: (
                    -longmemeval_candidate_selection_score(row, plan),
                    int(row.get("date_key") or 99999999),
                    int(row.get("line_no") or 99999999),
                ),
            )[0]
    return sorted(candidates, key=lambda row: (-longmemeval_candidate_selection_score(row, plan), int(row.get("line_no") or 0)))[0]


def longmemeval_candidate_selection_score(row: dict[str, Any], plan: dict[str, Any]) -> float:
    sentence = str(row.get("sentence") or "")
    lowered = sentence.lower()
    score = float(row.get("score") or 0.0)
    if "without the requested answer" in lowered:
        score -= 5.0
    if re.search(r"\b(confusing fact|old answer|old keyword|stale|was previously|used to be)\b", lowered) and not plan["requires_first"]:
        score -= 4.0
    if re.search(r"\b(final|current|currently|now|latest)\b", lowered):
        score += 1.4
    if longmemeval_temporal_relation_option(sentence, plan):
        score += 4.0
    if re.search(r"\b(answer|keyword|key|result|decision|mode)\s+(?:is|was|are|were)\b", lowered):
        score += 6.0
    if re.search(r"\b(?:means|refers to|maps to|points to|is an alias for|is the alias for)\b", lowered) and not re.search(
        r"\b(answer|keyword|key|result|decision|mode|code|password)\s+(?:is|was|are|were)\b",
        lowered,
    ):
        score -= 2.0
    if plan["question_type"] in {"current_update", "who", "where"} and re.search(
        r"\b(now|current|currently|latest|updated|changed|moved|renamed|set for|is now|became)\b",
        lowered,
    ):
        score += 3.5
    if plan["question_type"] == "fact" and re.search(
        r"\b(?:access\s+code|code|password|name|address|location|folder|file)\s+(?:is|was|are|were)\b",
        lowered,
    ):
        score += 3.0
    if plan["question_type"] == "preference" and re.search(
        r"\b(prefer|preferred|favorite|favourite|like best|would rather|chose|picked|selected)\b",
        lowered,
    ):
        score += 2.5
    if plan["question_type"] == "when" and re.search(
        r"\b\d{4}/\d{2}/\d{2}\b|\b(?:set for|scheduled for|date is)\b",
        lowered,
    ):
        score += 5.0
    if plan["question_type"] == "where" and re.search(r"\b(moved|stored|left|kept|in|at|inside|near|to)\b", lowered):
        score += 2.5
    if plan["question_type"] == "who" and re.search(r"\b(contact|manager|doctor|teacher|person|with|from|by)\b", lowered):
        score += 2.0
    if plan["question_type"] == "who" and re.search(
        r"^[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*(?:\s+\d+)?\s+(?:is|was)\s+(?:now|currently|the\s+contact)",
        sentence,
    ):
        score += 5.0
    if plan["question_type"] == "who" and re.search(r"\bbeing\s+updated\b", lowered):
        score -= 5.0
    return score


def compact_longmemeval_answer_span(sentence: str, question: str, plan: dict[str, Any]) -> str:
    text = strip_longmemeval_prefix(compact_space(sentence))
    if not text:
        return ""
    relation_option = longmemeval_temporal_relation_option(text, plan)
    if relation_option:
        return relation_option
    option = longmemeval_sentence_option(text, plan)
    if option:
        return option
    patterns_by_type = {
        "current_update": [
            r"\b(?:is|are|am)\s+now\s+([^.,;!?]+)",
            r"\b(?:is|are|am)\s+currently\s+([^.,;!?]+)",
            r"\bcurrent(?:ly)?\s+[^.,;:]*\s+(?:is|are)\s+([^.,;!?]+)",
            r"\b(?:changed|updated|renamed|moved)\s+(?:to|into|as)\s+([^.,;!?]+)",
            r"\b(?:password|code|name|address|location|folder|file)\s+(?:is|are)\s+([^.,;!?]+)",
        ],
        "preference": [
            r"\bprefer(?:red)?\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+?)(?:\s+because|\s+for|\s+when|$)",
            r"\bfavo[u]?rite\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
            r"\blike\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+?)\s+best\b",
        ],
        "where": [
            r"\b(?:in|at|inside|near|to)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
            r"\b(?:stored|left|kept|moved)\s+(?:it\s+)?(?:in|at|inside|near|to)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
        ],
        "who": [
            r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*(?:\s+\d+)?)\s+(?:is|was)\s+now\b",
            r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*(?:\s+\d+)?)\s+(?:is|was)\s+(?:currently\s+)?(?:the\s+)?(?:contact|manager|doctor|teacher|person)\b",
            r"\b(?:with|from|by)\s+([A-Z][A-Za-z0-9 -]{1,80})",
            r"\b(?:named|called|met)\s+([A-Z][A-Za-z0-9 -]{1,80})",
        ],
        "when": [
            r"\b(\d{4}/\d{2}/\d{2})\b",
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?)\b",
        ],
        "fact": [
            r"\b(?:access\s+code|code|key|password|name|address|location|folder|file)\s+(?:is|was|are|were)\s+([^.,;!?]+)",
        ],
        "temporal_first": [
            r"\b(?:noticed|had|hit|encountered|ran into|found|saw)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
            r"\bfirst\s+(?:issue|problem|event|thing)\s+(?:was|is)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
            r"\bafter\s+[^.,;]+?\s+(?:I\s+)?(?:noticed|had|found)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
        ],
        "temporal_last": [
            r"\b(?:last|latest|most recent)\s+[^.,;:]*\s+(?:was|is)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
            r"\b(?:finally|later)\s+(?:I\s+)?(?:noticed|had|found|chose|picked|visited|attended)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
        ],
    }
    for pattern in patterns_by_type.get(plan["question_type"], []):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            span = clean_longmemeval_span(match.group(1))
            if span:
                return span
    generic_patterns = [
        r"\b(?:is|was|are|were)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
        r"\b(?:chose|picked|selected|bought|visited|attended|used|read|watched)\s+(?:the\s+|a\s+|an\s+)?([^.,;!?]+)",
    ]
    for pattern in generic_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            span = clean_longmemeval_span(match.group(1))
            if span:
                return span
    return text[:320]


def longmemeval_sentence_option(sentence: str, plan: dict[str, Any]) -> str:
    sentence_terms = meaningful_terms(sentence)
    best_option = ""
    best_overlap = 0
    for option in plan["options"]:
        option_terms = meaningful_terms(option)
        overlap = len(sentence_terms & option_terms)
        if option_terms and overlap > best_overlap:
            best_overlap = overlap
            best_option = option
    return best_option if best_overlap > 0 else ""


def longmemeval_temporal_relation_option(sentence: str, plan: dict[str, Any]) -> str:
    if not plan["options"] or plan["question_type"] not in {"temporal_first", "temporal_last", "choice"}:
        return ""
    lowered = sentence.lower()
    relation = ""
    if " before " in lowered:
        relation = "before"
    elif " after " in lowered:
        relation = "after"
    if not relation:
        return ""
    parts = re.split(r"\b(?:before|after)\b", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return ""
    left_terms = meaningful_terms(parts[0])
    right_terms = meaningful_terms(parts[1])
    side_by_option = []
    for option in plan["options"]:
        option_terms = meaningful_terms(option)
        if not option_terms:
            continue
        side_by_option.append((option, len(left_terms & option_terms), len(right_terms & option_terms)))
    if not side_by_option:
        return ""
    left_option = max(side_by_option, key=lambda row: row[1])
    right_option = max(side_by_option, key=lambda row: row[2])
    if relation == "before":
        if plan["question_type"] == "temporal_last":
            return right_option[0] if right_option[2] > 0 else ""
        return left_option[0] if left_option[1] > 0 else ""
    if plan["question_type"] == "temporal_last":
        return left_option[0] if left_option[1] > 0 else ""
    return right_option[0] if right_option[2] > 0 else ""


def clean_longmemeval_span(value: str) -> str:
    span = compact_space(value)
    span = re.sub(r"^(?:the|a|an)\s+", "", span, flags=re.IGNORECASE)
    span = re.sub(r"\s+(?:because|since|when|while|after|before)\b.*$", "", span, flags=re.IGNORECASE)
    span = span.strip(" .,:;!?\"'")
    if not span or len(span) > 180:
        return ""
    return span


def strip_longmemeval_prefix(sentence: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s+\d{4}/\d{2}/\d{2}.*?\b(?:user|assistant):\s*", "", sentence, flags=re.IGNORECASE)


def longmemeval_sentence_score(sentence: str, question_terms: set[str], question: str) -> float:
    sentence_terms = meaningful_terms(sentence)
    score = float(len(question_terms & sentence_terms))
    if likely_answer_bearing_sentence(sentence, question):
        score += 1.0
    if re.search(r"\b(I|my|we|our)\b", sentence, flags=re.IGNORECASE):
        score += 0.25
    if question_mentions_time(question) and re.search(r"\b(first|before|after|later|earlier|then|next|last|initially|finally)\b|\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", sentence, flags=re.IGNORECASE):
        score += 0.5
    if re.search(r"\b(current|currently|now|latest|today)\b", question, flags=re.IGNORECASE) and re.search(r"\b(current|currently|now|latest|today)\b", sentence, flags=re.IGNORECASE):
        score += 0.75
    return score


def longmemeval_context_supports_question(context: str, question: str, plan: dict[str, Any]) -> bool:
    if plan["question_type"] not in {"fact", "current_update", "where", "who", "preference"}:
        return True
    support_terms = longmemeval_subject_support_terms(question, plan)
    if not support_terms:
        return True
    context_terms = meaningful_terms(context)
    overlap = context_terms & support_terms
    if len(support_terms) <= 2:
        return bool(overlap)
    return len(overlap) >= 2


def longmemeval_subject_support_terms(question: str, plan: dict[str, Any]) -> set[str]:
    answer_class_terms = {
        "access",
        "address",
        "answer",
        "code",
        "contact",
        "current",
        "currently",
        "date",
        "file",
        "folder",
        "key",
        "location",
        "manager",
        "name",
        "owner",
        "password",
        "path",
        "person",
        "status",
        "where",
        "who",
    }
    support = {
        term
        for term in set(plan.get("subject_terms") or meaningful_terms(question))
        if term not in answer_class_terms and len(term) > 1
    }
    return support


def longmemeval_alias_terms_for_question(context: str, question: str) -> set[str]:
    query_terms = meaningful_terms(question)
    if not query_terms:
        return set()
    aliases: set[str] = set()
    for line in context.splitlines():
        for left, right in longmemeval_alias_pairs(line):
            left_terms = meaningful_terms(left)
            right_terms = meaningful_terms(right)
            if query_terms & left_terms:
                aliases.update(right_terms)
            if query_terms & right_terms:
                aliases.update(left_terms)
    return aliases


def longmemeval_explicit_unknown_context(context: str, question: str, plan: dict[str, Any]) -> bool:
    support_terms = longmemeval_subject_support_terms(question, plan) or meaningful_terms(question)
    if not support_terms:
        return False
    unknown_seen = False
    answer_seen = False
    for line in context.splitlines():
        terms = meaningful_terms(line)
        if not (terms & support_terms):
            continue
        lowered = line.lower()
        if re.search(r"\b(?:unknown|unresolved|not recorded|not record|did not record|no answer|without an answer|not finalized)\b", lowered):
            unknown_seen = True
        if re.search(r"\b(?:answer|code|keyword|result|decision|mode|location|file|contact)\s+(?:is|was|are|were)\s+(?!unknown\b)[^.,;!?]+", lowered):
            answer_seen = True
    return unknown_seen and not answer_seen


def likely_answer_bearing_sentence(sentence: str, question: str) -> bool:
    if not sentence:
        return False
    lowered = sentence.lower()
    question_lower = question.lower()
    if "without the requested answer" in lowered:
        return False
    if "assistant:" in lowered and "user:" not in lowered and len(sentence) > 240:
        return False
    if re.search(r"\b(?:answer|keyword|key|result|decision|mode|code|password|file|folder|location|contact)\s+(?:is|was|are|were)\b", lowered):
        return True
    if question_mentions_time(question) and re.search(r"\b(first|before|after|later|earlier|then|next|last|initially|finally|current|currently|now|latest|today|mid-|early|late)\b|\d{4}|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", lowered):
        return True
    if any(word in question_lower for word in ["prefer", "favorite", "favourite", "like best", "which"]):
        return bool(re.search(r"\b(prefer|favorite|favourite|like|attended|visited|bought|watched|read|used|chose|picked|selected)\b", lowered))
    if question_lower.startswith(("what ", "where ", "who ", "which ", "when ")):
        return len(meaningful_terms(sentence) & meaningful_terms(question)) >= 1
    return len(meaningful_terms(sentence) & meaningful_terms(question)) >= 2


def question_mentions_time(question: str) -> bool:
    return bool(re.search(r"\b(first|before|after|when|date|earlier|later|last|recent|previous|next|initial|current|currently|now|latest|today)\b", question, flags=re.IGNORECASE))


def with_line_order(evidence: dict[str, str], context: str) -> dict[str, str]:
    row = dict(evidence)
    if "line_no" not in row:
        row["line_no"] = str(evidence_line_number(str(row.get("text") or ""), context))
    return row


def evidence_line_number(text: str, context: str) -> int:
    if not text:
        return 1_000_000
    for idx, line in enumerate(context.splitlines(), start=1):
        if line == text:
            return idx
    match = re.search(r":line(\d+)$", text)
    if match:
        return int(match.group(1))
    return 1_000_000


def ordered_unique_evidence(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen_events: set[tuple[str, str]] = set()
    ordered = []
    # State-tracking tasks depend on event chronology; oracle-first joins can change answers.
    for row in sorted(rows, key=lambda item: (int(item.get("line_no") or 1_000_000), str(item.get("id") or ""))):
        text = str(row.get("text") or "")
        key = (str(row.get("line_no") or ""), text)
        if not text or key in seen_events:
            continue
        seen_events.add(key)
        ordered.append(row)
    return ordered


def track_person_locations(lines: list[str]) -> dict[str, str]:
    locations: dict[str, str] = {}
    for line in lines:
        match = parse_move(line)
        if match:
            locations[match.group(1)] = match.group(2)
    return locations


def track_person_and_item_locations(lines: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    people, item_locations, _ = track_person_and_item_locations_with_sources(lines)
    return people, item_locations


def track_person_and_item_locations_with_sources(lines: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, list[tuple[int, str]]]]:
    people: dict[str, str] = {}
    held_by: dict[str, str] = {}
    item_locations: dict[str, str] = {}
    item_sources: dict[str, list[tuple[int, str]]] = {}
    for idx, line in enumerate(lines, start=1):
        move = parse_move(line)
        if move:
            person, location = move.group(1), move.group(2)
            people[person] = location
            for item, holder in list(held_by.items()):
                if holder == person:
                    item_locations[item] = location
                    append_source_line(item_sources, item, idx, line)
            continue
        take = parse_take(line)
        if take:
            person, item = take.group(1), take.group(2)
            held_by[item] = person
            if person in people:
                item_locations[item] = people[person]
            append_source_line(item_sources, item, idx, line)
            continue
        give = parse_give(line)
        if give:
            giver, item, receiver = give.group(1), give.group(2), give.group(3)
            if held_by.get(item) == giver:
                held_by[item] = receiver
                if receiver in people:
                    item_locations[item] = people[receiver]
                append_source_line(item_sources, item, idx, line)
            continue
        drop = parse_drop(line)
        if drop:
            person, item = drop.group(1), drop.group(2)
            if held_by.get(item) == person:
                held_by.pop(item, None)
                if person in people:
                    item_locations[item] = people[person]
                append_source_line(item_sources, item, idx, line)
    return people, item_locations, item_sources


def track_person_and_item_history(lines: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    people: dict[str, str] = {}
    held_by: dict[str, str] = {}
    item_locations: dict[str, str] = {}
    item_history: dict[str, list[str]] = {}
    for line in lines:
        move = parse_move(line)
        if move:
            person, location = move.group(1), move.group(2)
            people[person] = location
            for item, holder in list(held_by.items()):
                if holder == person:
                    item_locations[item] = location
                    append_location(item_history, item, location)
            continue
        take = parse_take(line)
        if take:
            person, item = take.group(1), take.group(2)
            held_by[item] = person
            if person in people:
                item_locations[item] = people[person]
                append_location(item_history, item, people[person])
            continue
        give = parse_give(line)
        if give:
            giver, item, receiver = give.group(1), give.group(2), give.group(3)
            if held_by.get(item) == giver:
                held_by[item] = receiver
                if receiver in people:
                    item_locations[item] = people[receiver]
                    append_location(item_history, item, people[receiver])
            continue
        drop = parse_drop(line)
        if drop:
            person, item = drop.group(1), drop.group(2)
            if held_by.get(item) == person:
                held_by.pop(item, None)
                if person in people:
                    item_locations[item] = people[person]
                    append_location(item_history, item, people[person])
    return people, item_locations, item_history


def parse_move(line: str) -> re.Match[str] | None:
    return re.match(r"([A-Z][a-z]+) (?:moved|went|travelled|journeyed)(?: back)? to the ([a-z]+)\.", line)


def parse_take(line: str) -> re.Match[str] | None:
    return re.match(r"([A-Z][a-z]+) (?:got|grabbed|picked up|took) the ([a-z]+)(?: there)?\.", line)


def parse_give(line: str) -> re.Match[str] | None:
    return re.match(r"([A-Z][a-z]+) (?:gave|handed|passed) the ([a-z]+) to ([A-Z][a-z]+)\.", line)


def parse_drop(line: str) -> re.Match[str] | None:
    return re.match(r"([A-Z][a-z]+) (?:dropped|discarded|put down) the ([a-z]+)(?: there)?\.", line)


def append_location(history: dict[str, list[str]], item: str, location: str) -> None:
    if not location:
        return
    rows = history.setdefault(item, [])
    if not rows or rows[-1] != location:
        rows.append(location)


def append_source_line(sources: dict[str, list[tuple[int, str]]], item: str, idx: int, line: str) -> None:
    rows = sources.setdefault(item, [])
    if not rows or rows[-1] != (idx, line):
        rows.append((idx, line))




if __name__ == "__main__":
    raise SystemExit(main())
