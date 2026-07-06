"""VCM public-memory readiness audit.

This audit is the no-cheat bridge between private VCM repair work and any
future prompt-level public-memory calibration. It does not load public prompt
payloads for training or repair. Public reports are read only as calibration
evidence; private readiness pressure comes from local analogue fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))

import vcm_official_public_memory_adapter as prompt_adapter  # noqa: E402


DEFAULT_OUT = REPORTS / "vcm_public_memory_readiness_audit.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_public_memory_readiness_audit.md"
DEFAULT_SOURCE_ROOT = ROOT / "data" / "public_benchmarks" / "vcm_official_sources"
RECOMMENDED_PUBLIC_SLICE_ID = "vcm_public_memory_prompt_slice_2026_06_19_lme_private_residual_confirmation_fresh"
LOCKED_LONGMEMEVAL_BASELINE_VCM = 0.055

HISTORICAL_FAILING_PROMPT_SLICE = {
    "slice_id": "vcm_public_memory_prompt_slice_2026_06_18",
    "vcm_on_pass_rate": 0.666667,
    "vcm_off_pass_rate": 0.833333,
    "off_only_wins": 2,
    "vcm_only_wins": 0,
    "residual_categories": [
        "no_admissible",
        "state_tracking_failure",
        "temporal_update_failure",
    ],
    "root_cause": (
        "QA1 shared person-question parser misses plus QA3 VCM evidence chronology loss; "
        "the repairs are proven only through private analogues before any fresh public spend."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=rel(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--context-budget-chars", type=int, default=900)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        source_root=resolve(args.source_root),
        context_budget_chars=max(128, args.context_budget_chars),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, source_root: Path, context_budget_chars: int, started: float) -> dict[str, Any]:
    metadata_report = read_json(REPORTS / "vcm_public_memory_calibration.json")
    prompt_report = read_json(REPORTS / "vcm_public_memory_prompt_calibration.json")
    private_context = read_json(REPORTS / "vcm_context_recovery_benchmark.json")
    private_ablation = read_json(REPORTS / "vcm_on_off_ablation.json")
    private_repair = read_json(REPORTS / "vcm_public_memory_private_residual_repair.json")
    private_lme_residual = read_json(REPORTS / "vcm_longmemeval_private_residual_curriculum.json")

    surfaces = audit_surfaces(
        source_root=source_root,
        metadata_report=metadata_report,
        prompt_report=prompt_report,
    )
    private_rows = run_private_analogue_tests(context_budget_chars=context_budget_chars)
    private_summary = summarize_private_rows(private_rows)
    coverage = coverage_summary(private_context, private_ablation, private_repair)
    coverage = merge_private_analogue_coverage(coverage, private_summary)
    longmemeval_residual_summary = longmemeval_private_residual_summary(private_lme_residual)
    contamination = contamination_summary(private_rows, private_context, private_ablation, private_repair, private_lme_residual)
    postmortem = postmortem_summary(prompt_report, private_summary)
    latest_long_context = latest_long_context_summary(prompt_report)
    gates = readiness_gates(
        surfaces=surfaces,
        private_summary=private_summary,
        coverage=coverage,
        contamination=contamination,
        postmortem=postmortem,
        latest_long_context=latest_long_context,
        longmemeval_residual_summary=longmemeval_residual_summary,
    )
    hard_failures = [row for row in gates if not row["passed"] and row["severity"] == "blocker"]
    public_calibration_allowed = not hard_failures
    longmemeval_status = next((row.get("status") for row in surfaces if row.get("source_id") == "longmemeval"), "")
    notes = [
        "Public benchmark content remains calibration-only and is never admitted to training.",
        "This audit uses private analogue rows to prove adapter repairs before any fresh public calibration.",
    ]
    if longmemeval_status == "ready_prompt_scoring":
        notes.append("LongMemEval official JSON is staged and has a deterministic local prompt-level adapter; no model judge is required in this lane.")
    else:
        notes.append("LongMemEval remains queued unless data and evaluator can be staged with deterministic local scoring.")
    return {
        "policy": "project_theseus_vcm_public_memory_readiness_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if public_calibration_allowed else "YELLOW",
        "public_calibration_allowed": public_calibration_allowed,
        "recommended_public_slice_id": RECOMMENDED_PUBLIC_SLICE_ID if public_calibration_allowed else "",
        "surface_audit": surfaces,
        "locked_prompt_slice_postmortem": postmortem,
        "latest_long_context_prompt_slice": latest_long_context,
        "private_analogue_summary": private_summary,
        "private_longmemeval_residual_curriculum": longmemeval_residual_summary,
        "private_analogue_rows": private_rows,
        "private_vcm_coverage": coverage,
        "contamination_counters": contamination,
        "gates": gates,
        "hard_failures": hard_failures,
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "notes": notes,
    }


def audit_surfaces(
    *,
    source_root: Path,
    metadata_report: dict[str, Any],
    prompt_report: dict[str, Any],
) -> list[dict[str, Any]]:
    records = {row["source_id"]: row for row in prompt_adapter.inspect_sources(source_root)}
    prompt_benchmarks = set(dict_value(get_path(prompt_report, ["summary", "per_benchmark"], {})))
    queued_rows = {
        row.get("benchmark"): row
        for row in list_value(prompt_report.get("rows"))
        if isinstance(row, dict) and row.get("status") == "queued"
    }
    rows: list[dict[str, Any]] = []
    for source_id in ["ruler", "babilong", "longmemeval"]:
        source = records.get(source_id, {})
        if source_id in {"ruler", "babilong"}:
            status = "ready_prompt_scoring"
            blocker = ""
            if not source.get("present"):
                status = "blocked_missing_source"
                blocker = "official source clone is missing"
            elif source_id == "babilong" and not (source_root / "babilong" / "data" / "tasks_1-20_v1-2.zip").exists():
                status = "blocked_missing_official_task_zip"
                blocker = "BABILong official task zip is missing"
        else:
            if source.get("present") and longmemeval_data_present(source_root):
                status = "ready_prompt_scoring"
                blocker = ""
            else:
                status = "queued_deterministic_scorer_needed"
                blocker = str(queued_rows.get("longmemeval", {}).get("reason") or "LongMemEval data/evaluator staging is not clean yet")
        rows.append(
            {
                "source_id": source_id,
                "status": status,
                "source_present": bool(source.get("present")),
                "license_present": bool(source.get("license_present")),
                "license_spdx": source.get("license_spdx"),
                "source_commit": source.get("commit"),
                "prompt_adapter": source_id in {"ruler", "babilong", "longmemeval"} and status == "ready_prompt_scoring",
                "latest_prompt_scored": source_id in prompt_benchmarks,
                "blocker": blocker,
                "public_training_allowed": False,
            }
        )

    metadata_rows = list_value(metadata_report.get("rows"))
    known = {row["source_id"] for row in rows}
    for row in metadata_rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in known:
            continue
        rows.append(
            {
                "source_id": source_id,
                "status": "metadata_only_prompt_adapter_blocked",
                "source_present": False,
                "license_present": bool(row.get("license_spdx")),
                "license_spdx": row.get("license_spdx"),
                "source_commit": "",
                "prompt_adapter": False,
                "latest_prompt_scored": False,
                "blocker": "catalogued memory/context card has no official prompt-level deterministic adapter in this lane yet",
                "public_training_allowed": False,
                "taxonomy": row.get("task_taxonomy", []),
            }
        )
    return rows


def longmemeval_data_present(source_root: Path) -> bool:
    data_dir = source_root / "longmemeval" / "data"
    return any((data_dir / name).exists() for name in prompt_adapter.LONGMEMEVAL_DATA_CANDIDATES)


def private_lme_noise(prefix: str, start_day: int, count: int) -> list[str]:
    rows = []
    for idx in range(count):
        day = ((start_day + idx - 1) % 28) + 1
        rows.append(
            f"[{prefix}_{idx:03d}] 2026/04/{day:02d} (Mon) 07:00 user: "
            f"Routine note {idx} mentions schedule, errands, and a generic storage reminder without the requested answer."
        )
    return rows


def run_private_analogue_tests(*, context_budget_chars: int) -> list[dict[str, Any]]:
    items = [
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa1_person_parser",
            benchmark="babilong",
            task="qa1",
            prompt="",
            context="Mira travelled to the workshop.\nSol went to the hallway.",
            question="Where is Mira? ",
            answers=["workshop"],
            oracle_evidence=[{"id": "qa1:private:line1", "text": "Mira travelled to the workshop."}],
            metadata={"private_analogue": "person_question_parser"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa3_chronology",
            benchmark="babilong",
            task="qa3",
            prompt="",
            context="\n".join(
                [
                    "Mira moved to the bedroom.",
                    "Mira grabbed the token there.",
                    "Mira moved to the lab.",
                    "Mira dropped the token.",
                    "Mira grabbed the token.",
                    "Mira went to the office.",
                    "Mira journeyed to the kitchen.",
                    "Mira discarded the token there.",
                ]
            ),
            question="Where was the token before the kitchen? ",
            answers=["office"],
            oracle_evidence=[
                {"id": "qa3:private:line6", "text": "Mira went to the office."},
                {"id": "qa3:private:line7", "text": "Mira journeyed to the kitchen."},
                {"id": "qa3:private:line8", "text": "Mira discarded the token there."},
            ],
            metadata={"private_analogue": "chronological_state_tracking"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa3_repeated_event_chronology",
            benchmark="babilong",
            task="qa3",
            prompt="",
            context="\n".join(
                [
                    "Mira moved to the lab.",
                    "Mira got the token.",
                    "Mira moved to the hall.",
                    "Mira moved to the office.",
                    "Mira moved to the hall.",
                    "Mira discarded the token there.",
                ]
            ),
            question="Where was the token before the hall? ",
            answers=["office"],
            oracle_evidence=[
                {"id": "qa3:private:line2", "text": "Mira got the token."},
                {"id": "qa3:private:line4", "text": "Mira moved to the office."},
                {"id": "qa3:private:line5", "text": "Mira moved to the hall."},
                {"id": "qa3:private:line6", "text": "Mira discarded the token there."},
            ],
            metadata={"private_analogue": "repeated_event_chronology"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa2_give_transfer",
            benchmark="babilong",
            task="qa2",
            prompt="",
            context="\n".join(
                [
                    "Mira moved to the lab.",
                    "Mira picked up the token.",
                    "Sol went to the office.",
                    "Mira gave the token to Sol.",
                    "Sol moved to the garden.",
                ]
            ),
            question="Where is the token? ",
            answers=["garden"],
            oracle_evidence=[
                {"id": "qa2:private:line2", "text": "Mira picked up the token."},
                {"id": "qa2:private:line4", "text": "Mira gave the token to Sol."},
                {"id": "qa2:private:line5", "text": "Sol moved to the garden."},
            ],
            metadata={"private_analogue": "give_transfer_tracking"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa2_drop_transfer",
            benchmark="babilong",
            task="qa2",
            prompt="",
            context="\n".join(
                [
                    "Mira journeyed to the bedroom.",
                    "Mira grabbed the apple.",
                    "Mira went to the kitchen.",
                    "Mira dropped the apple there.",
                    "Mira moved to the hallway.",
                ]
            ),
            question="Where is the apple? ",
            answers=["kitchen"],
            oracle_evidence=[
                {"id": "qa2:private:line2", "text": "Mira grabbed the apple."},
                {"id": "qa2:private:line3", "text": "Mira went to the kitchen."},
                {"id": "qa2:private:line4", "text": "Mira dropped the apple there."},
            ],
            metadata={"private_analogue": "drop_transfer_tracking"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_babilong_qa2_distractor_rejection",
            benchmark="babilong",
            task="qa2",
            prompt="",
            context="\n".join(
                [
                    "Mira went to the hallway.",
                    "Mira picked up the apple.",
                    "Sol went to the kitchen.",
                    "Sol picked up the milk.",
                    "Mira travelled to the office.",
                    "Sol journeyed to the garden.",
                ]
            ),
            question="Where is the apple? ",
            answers=["office"],
            oracle_evidence=[
                {"id": "qa2:private:line1", "text": "Mira went to the hallway."},
                {"id": "qa2:private:line2", "text": "Mira picked up the apple."},
                {"id": "qa2:private:line5", "text": "Mira travelled to the office."},
            ],
            metadata={"private_analogue": "distractor_rejection"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_temporal_first_issue",
            benchmark="longmemeval",
            task="temporal-reasoning",
            prompt="",
            context="\n".join(
                [
                    "[private_session_1] 2026/02/10 (Tue) 09:00 user: I took the rover in for its first service and later noticed the GPS system not functioning correctly.",
                    "[private_session_2] 2026/02/12 (Thu) 18:00 user: The tire pressure warning showed up after the GPS issue.",
                    "[private_session_3] 2026/02/15 (Sun) 10:00 assistant: Generic maintenance tips do not contain the remembered issue.",
                ]
            ),
            question="What was the first issue I had after the rover's first service?",
            answers=["GPS system not functioning correctly"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:temporal:line1",
                    "text": "[private_session_1] 2026/02/10 (Tue) 09:00 user: I took the rover in for its first service and later noticed the GPS system not functioning correctly.",
                }
            ],
            metadata={"private_analogue": "longmemeval_temporal_answer_shape"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_current_update",
            benchmark="longmemeval",
            task="knowledge-update",
            prompt="",
            context="\n".join(
                [
                    "[private_session_1] 2026/03/01 (Sun) 11:00 user: The workshop router password used to be alpha-four.",
                    "[private_session_2] 2026/03/08 (Sun) 11:00 user: The workshop router password is now delta-seven.",
                    "[private_session_3] 2026/03/09 (Mon) 11:00 assistant: I can suggest ways to store passwords safely.",
                ]
            ),
            question="What is the current workshop router password?",
            answers=["delta-seven"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:update:line2",
                    "text": "[private_session_2] 2026/03/08 (Sun) 11:00 user: The workshop router password is now delta-seven.",
                }
            ],
            metadata={"private_analogue": "longmemeval_knowledge_update"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_preference",
            benchmark="longmemeval",
            task="single-session-preference",
            prompt="",
            context="\n".join(
                [
                    "[private_session_1] 2026/04/01 (Wed) 08:00 user: For travel days I prefer the smaller black backpack because it fits under the seat.",
                    "[private_session_2] 2026/04/02 (Thu) 08:00 assistant: A packing checklist may help.",
                ]
            ),
            question="Which backpack do I prefer for travel days?",
            answers=["smaller black backpack"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:preference:line1",
                    "text": "[private_session_1] 2026/04/01 (Wed) 08:00 user: For travel days I prefer the smaller black backpack because it fits under the seat.",
                }
            ],
            metadata={"private_analogue": "longmemeval_preference_recall"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_no_answer_abstention",
            benchmark="longmemeval",
            task="abstention",
            prompt="",
            context="\n".join(
                [
                    "[private_session_1] 2026/05/01 (Fri) 08:00 user: The storage shelf has a blue bin and a green bin.",
                    "[private_session_2] 2026/05/02 (Sat) 08:00 assistant: I can help label the bins.",
                ]
            ),
            question="What color is the hidden cable?",
            answers=["__NO_ANSWER__"],
            oracle_evidence=[],
            metadata={"private_analogue": "longmemeval_no_answer_abstention"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_temporal_choice_first",
            benchmark="longmemeval",
            task="temporal-reasoning",
            prompt="",
            context="\n".join(
                private_lme_noise("private_choice_front", 1, 10)
                + [
                    "[private_choice_session_1] 2026/04/10 (Fri) 10:00 user: I attended the calibration clinic before the packet audit workshop.",
                    "[private_choice_session_2] 2026/04/12 (Sun) 10:00 user: The packet audit workshop happened later and had more people.",
                ]
                + private_lme_noise("private_choice_tail", 14, 12)
            ),
            question="Which event did I attend first, the calibration clinic or the packet audit workshop?",
            answers=["calibration clinic"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:choice:line11",
                    "text": "[private_choice_session_1] 2026/04/10 (Fri) 10:00 user: I attended the calibration clinic before the packet audit workshop.",
                }
            ],
            metadata={"private_analogue": "longmemeval_multi_session_query_decomposition"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_pronoun_current_location",
            benchmark="longmemeval",
            task="knowledge-update",
            prompt="",
            context="\n".join(
                [
                    "[private_location_session_1] 2026/05/01 (Fri) 09:00 user: The folding projector started out in the lab cabinet.",
                ]
                + private_lme_noise("private_location_middle", 2, 14)
                + [
                    "[private_location_session_9] 2026/05/18 (Mon) 18:00 user: I moved it to studio B after the room reshuffle.",
                    "[private_location_session_10] 2026/05/19 (Tue) 18:00 assistant: A room label can make future lookup easier.",
                ]
            ),
            question="Where is the folding projector currently?",
            answers=["studio B"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:pronoun:update:line16",
                    "text": "[private_location_session_9] 2026/05/18 (Mon) 18:00 user: I moved it to studio B after the room reshuffle.",
                }
            ],
            metadata={"private_analogue": "longmemeval_structured_recency_fusion"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_exact_span_compaction",
            benchmark="longmemeval",
            task="single-session-assistant",
            prompt="",
            context="\n".join(
                [
                    "[private_span_session_1] 2026/06/01 (Mon) 08:00 user: For the field kit I prefer the matte silver caliper because the digital one drains batteries and the black one sticks.",
                    "[private_span_session_2] 2026/06/02 (Tue) 08:00 assistant: I can keep a tool preference note.",
                ]
            ),
            question="Which caliper do I prefer for the field kit?",
            answers=["matte silver caliper"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:span:line1",
                    "text": "[private_span_session_1] 2026/06/01 (Mon) 08:00 user: For the field kit I prefer the matte silver caliper because the digital one drains batteries and the black one sticks.",
                }
            ],
            metadata={"private_analogue": "longmemeval_answer_span_compaction"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_longmemeval_distractor_assistant_rejection",
            benchmark="longmemeval",
            task="knowledge-update",
            prompt="",
            context="\n".join(
                [
                    "[private_distractor_session_1] 2026/06/04 (Thu) 09:00 assistant: You might call the NAS archive either Atlas or Harbor.",
                    "[private_distractor_session_2] 2026/06/05 (Fri) 09:00 user: I decided the NAS archive is now called Harbor.",
                ]
            ),
            question="What is the current name of the NAS archive?",
            answers=["Harbor"],
            oracle_evidence=[
                {
                    "id": "longmemeval:private:distractor:line2",
                    "text": "[private_distractor_session_2] 2026/06/05 (Fri) 09:00 user: I decided the NAS archive is now called Harbor.",
                }
            ],
            metadata={"private_analogue": "longmemeval_user_update_over_assistant_distractor"},
        ),
        prompt_adapter.PublicMemoryItem(
            item_id="private_ruler_multivalue_answer_shape",
            benchmark="ruler",
            task="niah_multivalue",
            prompt="",
            context="\n".join(
                [
                    "The grass is green.",
                    "One of the special magic number for cobalt-ridge is: 113.",
                    "The sky is blue.",
                    "One of the special magic number for cobalt-ridge is: 217.",
                    "The sun is yellow.",
                ]
            ),
            question="What are all the special magic number for cobalt-ridge mentioned in the provided text?",
            answers=["113", "217"],
            oracle_evidence=[
                {"id": "ruler:private:line2", "text": "One of the special magic number for cobalt-ridge is: 113."},
                {"id": "ruler:private:line4", "text": "One of the special magic number for cobalt-ridge is: 217."},
            ],
            metadata={"private_analogue": "answer_shape_multivalue"},
        ),
    ]
    rows = []
    for item in items:
        scored = prompt_adapter.score_item(item, context_budget_chars=context_budget_chars)
        memory_systems = {
            name: {
                "passed": bool(row.get("passed")),
                "no_admissible": bool(row.get("no_admissible")),
                "evidence_precision": row.get("evidence_precision"),
                "evidence_recall": row.get("evidence_recall"),
                "answer_span_chars": row.get("answer_span_chars", 0),
            }
            for name, row in prompt_adapter.dict_value(scored.get("memory_systems")).items()
            if isinstance(row, dict)
        }
        best_non_vcm = prompt_adapter.best_non_vcm_system(prompt_adapter.dict_value(scored.get("memory_systems")))
        rows.append(
            {
                "item_id": item.item_id,
                "private_analogue": item.metadata.get("private_analogue"),
                "benchmark_family": item.benchmark,
                "task": item.task,
                "vcm_on_passed": bool(get_path(scored, ["vcm_on", "passed"], False)),
                "vcm_off_passed": bool(get_path(scored, ["vcm_off", "passed"], False)),
                "best_non_vcm_system": best_non_vcm.get("system", ""),
                "best_non_vcm_passed": bool(best_non_vcm.get("passed")),
                "memory_systems": memory_systems,
                "winner": scored.get("winner"),
                "vcm_on_prediction_hash": get_path(scored, ["vcm_on", "prediction_hash"], ""),
                "vcm_off_prediction_hash": get_path(scored, ["vcm_off", "prediction_hash"], ""),
                "vcm_on_evidence_precision": get_path(scored, ["vcm_on", "evidence_precision"], 0.0),
                "vcm_on_evidence_recall": get_path(scored, ["vcm_on", "evidence_recall"], 0.0),
                "vcm_off_evidence_precision": get_path(scored, ["vcm_off", "evidence_precision"], 0.0),
                "vcm_off_evidence_recall": get_path(scored, ["vcm_off", "evidence_recall"], 0.0),
                "vcm_on_no_admissible": bool(get_path(scored, ["vcm_on", "no_admissible"], False)),
                "vcm_off_no_admissible": bool(get_path(scored, ["vcm_off", "no_admissible"], False)),
                "external_inference_calls": 0,
                "teacher_solving_calls": 0,
                "fallback_return_count": 0,
                "public_training_rows_written": 0,
                "public_prompt_chars_loaded": 0,
                "public_context_chars_loaded": 0,
                "public_answer_chars_loaded": 0,
            }
        )
    return rows


def summarize_private_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vcm_on = [1.0 if row["vcm_on_passed"] else 0.0 for row in rows]
    vcm_off = [1.0 if row["vcm_off_passed"] else 0.0 for row in rows]
    best_non_vcm = [1.0 if row.get("best_non_vcm_passed") else 0.0 for row in rows]
    longmemeval_rows = [row for row in rows if row.get("benchmark_family") == "longmemeval"]
    longmemeval_vcm = [1.0 if row["vcm_on_passed"] else 0.0 for row in longmemeval_rows]
    longmemeval_best = [1.0 if row.get("best_non_vcm_passed") else 0.0 for row in longmemeval_rows]
    memory_system_names = sorted({
        name
        for row in rows
        for name in prompt_adapter.dict_value(row.get("memory_systems"))
    })
    memory_system_pass_rates = {
        name: mean([
            1.0
            if get_path(row, ["memory_systems", name, "passed"], False)
            else 0.0
            for row in rows
        ])
        for name in memory_system_names
    }
    return {
        "case_count": len(rows),
        "vcm_on_pass_rate": mean(vcm_on),
        "vcm_off_pass_rate": mean(vcm_off),
        "best_non_vcm_pass_rate": mean(best_non_vcm),
        "vcm_over_best_non_vcm_delta": round(mean(vcm_on) - mean(best_non_vcm), 6),
        "vcm_only_wins": sum(1 for row in rows if row["vcm_on_passed"] and not row["vcm_off_passed"]),
        "off_only_wins": sum(1 for row in rows if row["vcm_off_passed"] and not row["vcm_on_passed"]),
        "best_non_vcm_only_wins": sum(1 for row in rows if row.get("best_non_vcm_passed") and not row["vcm_on_passed"]),
        "both_pass": sum(1 for row in rows if row["vcm_on_passed"] and row["vcm_off_passed"]),
        "memory_system_pass_rates": memory_system_pass_rates,
        "longmemeval_case_count": len(longmemeval_rows),
        "longmemeval_vcm_on_pass_rate": mean(longmemeval_vcm),
        "longmemeval_best_non_vcm_pass_rate": mean(longmemeval_best),
        "longmemeval_vcm_over_best_non_vcm_delta": round(mean(longmemeval_vcm) - mean(longmemeval_best), 6),
        "longmemeval_best_non_vcm_only_wins": sum(1 for row in longmemeval_rows if row.get("best_non_vcm_passed") and not row["vcm_on_passed"]),
        "vcm_on_evidence_precision": mean([float(row["vcm_on_evidence_precision"] or 0.0) for row in rows]),
        "vcm_on_evidence_recall": mean([float(row["vcm_on_evidence_recall"] or 0.0) for row in rows]),
        "covered_analogues": sorted({str(row["private_analogue"]) for row in rows}),
    }


def coverage_summary(
    private_context: dict[str, Any],
    private_ablation: dict[str, Any],
    private_repair: dict[str, Any],
) -> dict[str, Any]:
    context_summary = dict_value(private_context.get("summary"))
    ablation_summary = dict_value(private_ablation.get("summary"))
    scope = dict_value(private_context.get("benchmark_scope"))
    categories = set(scope.get("supplemental_private_residual_categories") or [])
    current_repair_required = {
        f"public_memory_private_residual_{category}"
        for category in list_value(private_repair.get("residual_categories"))
        if str(category).strip()
    }
    historical_required = {
        "public_memory_private_residual_no_admissible",
        "public_memory_private_residual_state_tracking_failure",
        "public_memory_private_residual_temporal_update_failure",
    }
    required = current_repair_required or historical_required
    return {
        "context_recovery_state": private_context.get("trigger_state"),
        "ablation_state": private_ablation.get("trigger_state"),
        "private_repair_state": private_repair.get("trigger_state"),
        "context_case_count": scope.get("case_count"),
        "supplemental_private_residual_cases": scope.get("supplemental_private_residual_cases"),
        "supplemental_categories": sorted(categories),
        "required_public_memory_residual_categories": sorted(required),
        "required_public_memory_residual_categories_present": required.issubset(categories),
        "vcm_answer_accuracy": context_summary.get("vcm_answer_accuracy"),
        "best_baseline_answer_accuracy": context_summary.get("best_baseline_answer_accuracy"),
        "ablation_answer_lift": ablation_summary.get("answer_accuracy_lift"),
        "ablation_off_only_wins": get_path(ablation_summary, ["win_counts", "off_only"], None),
        "ablation_fallback_return_count": ablation_summary.get("fallback_return_count"),
    }


def merge_private_analogue_coverage(coverage: dict[str, Any], private_summary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(coverage)
    covered = set(list_value(merged.get("supplemental_categories")))
    analogue_map = {
        "longmemeval_answer_span_compaction": "public_memory_private_residual_longmemeval_answer_span_compaction",
        "longmemeval_multi_session_query_decomposition": "public_memory_private_residual_longmemeval_query_decomposition",
        "longmemeval_no_answer_abstention": "public_memory_private_residual_longmemeval_abstention_thresholding",
        "longmemeval_structured_recency_fusion": "public_memory_private_residual_longmemeval_structured_recency_fusion",
    }
    for analogue in list_value(private_summary.get("covered_analogues")):
        category = analogue_map.get(str(analogue))
        if category:
            covered.add(category)
    required = set(list_value(merged.get("required_public_memory_residual_categories")))
    merged["supplemental_categories"] = sorted(covered)
    merged["required_public_memory_residual_categories_present"] = required.issubset(covered)
    return merged


def contamination_summary(*reports_or_rows: Any) -> dict[str, int]:
    counters = {
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "public_prompt_chars_loaded": 0,
        "public_context_chars_loaded": 0,
        "public_answer_chars_loaded": 0,
    }
    for value in reports_or_rows:
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict):
                continue
            summary = dict_value(row.get("summary"))
            firewall = dict_value(row.get("contamination_firewall"))
            for key in counters:
                counters[key] += int(row.get(key) or summary.get(key) or firewall.get(key.replace("_chars_loaded", "s_loaded")) or 0)
    return counters


def postmortem_summary(prompt_report: dict[str, Any], private_summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in list_value(prompt_report.get("rows")) if isinstance(row, dict)]
    qa1_ties = [
        row
        for row in rows
        if row.get("benchmark") == "babilong"
        and row.get("task") == "qa1"
        and not get_path(row, ["vcm_on", "passed"], False)
        and not get_path(row, ["vcm_off", "passed"], False)
    ]
    qa3_off_wins = [
        row
        for row in rows
        if row.get("benchmark") == "babilong"
        and row.get("task") == "qa3"
        and row.get("winner") == "vcm_off"
    ]
    return {
        "historical_failing_prompt_slice": HISTORICAL_FAILING_PROMPT_SLICE,
        "latest_prompt_slice_id": prompt_report.get("slice_id"),
        "latest_vcm_on_pass_rate": get_path(prompt_report, ["summary", "vcm_on_pass_rate"], None),
        "latest_vcm_off_pass_rate": get_path(prompt_report, ["summary", "vcm_off_pass_rate"], None),
        "latest_off_only_wins": get_path(prompt_report, ["summary", "win_counts", "vcm_off"], None),
        "latest_qa1_shared_parser_failures": len(qa1_ties),
        "latest_qa3_chronology_off_only_wins": len(qa3_off_wins),
        "root_cause": HISTORICAL_FAILING_PROMPT_SLICE["root_cause"],
        "private_analogues_pass": private_summary.get("off_only_wins") == 0
        and private_summary.get("vcm_on_pass_rate") == 1.0,
    }


def latest_long_context_summary(prompt_report: dict[str, Any]) -> dict[str, Any]:
    summary = dict_value(prompt_report.get("summary"))
    per_benchmark = dict_value(summary.get("per_benchmark"))
    longmemeval = dict_value(per_benchmark.get("longmemeval"))
    longmemeval_systems = dict_value(longmemeval.get("memory_systems"))
    longmemeval_best_single_non_vcm = max(
        [
            float(row.get("pass_rate") or 0.0)
            for name, row in longmemeval_systems.items()
            if name != "vcm_graph_evidence_selector" and isinstance(row, dict)
        ],
        default=0.0,
    )
    return {
        "slice_id": prompt_report.get("slice_id"),
        "surface_hash": prompt_report.get("surface_hash"),
        "scored_item_count": summary.get("scored_item_count"),
        "item_manifest": get_path(prompt_report, ["quarantine", "item_manifest"], ""),
        "item_manifest_hash": get_path(prompt_report, ["quarantine", "item_manifest_hash"], ""),
        "source_context_token_distribution": summary.get("source_context_token_distribution"),
        "per_length_bucket": summary.get("per_length_bucket"),
        "vcm_on_pass_rate": summary.get("vcm_on_pass_rate"),
        "vcm_off_pass_rate": summary.get("vcm_off_pass_rate"),
        "vcm_over_flat_tail_delta": summary.get("vcm_over_flat_tail_delta"),
        "vcm_over_best_non_vcm_delta": summary.get("vcm_over_best_non_vcm_delta"),
        "longmemeval_item_count": longmemeval.get("items"),
        "longmemeval_vcm_on_pass_rate": longmemeval.get("vcm_on_pass_rate"),
        "longmemeval_vcm_off_pass_rate": longmemeval.get("vcm_off_pass_rate"),
        "longmemeval_best_non_vcm": longmemeval_best_single_non_vcm,
        "longmemeval_vcm_over_best_single_non_vcm_delta": round(
            float(longmemeval.get("vcm_on_pass_rate") or 0.0) - longmemeval_best_single_non_vcm,
            6,
        ),
        "longmemeval_vcm_over_flat_tail_delta": round(
            float(longmemeval.get("vcm_on_pass_rate") or 0.0) - float(longmemeval.get("vcm_off_pass_rate") or 0.0),
            6,
        ),
        "longmemeval_question_type": longmemeval.get("longmemeval_question_type"),
        "forbidden_overlap_counts": summary.get("forbidden_overlap_counts"),
        "external_inference_calls": summary.get("external_inference_calls"),
        "fallback_return_count": summary.get("fallback_return_count"),
        "public_training_rows_written": summary.get("public_training_rows_written"),
    }


def longmemeval_private_residual_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict_value(report.get("summary"))
    proposal = dict_value(report.get("future_public_calibration_proposal"))
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "private_only": report.get("private_only"),
        "case_count": summary.get("case_count"),
        "vcm_on_pass_rate": summary.get("vcm_on_pass_rate"),
        "best_single_non_vcm_pass_rate": summary.get("best_single_non_vcm_pass_rate"),
        "vcm_over_best_single_non_vcm_delta": summary.get("vcm_over_best_single_non_vcm_delta"),
        "minimum_major_question_type_pass_rate": summary.get("minimum_major_question_type_pass_rate"),
        "vcm_on_evidence_recall": summary.get("vcm_on_evidence_recall"),
        "abstention": summary.get("abstention"),
        "diagnostic_counts": summary.get("diagnostic_counts"),
        "future_public_proposal_state": proposal.get("proposal_state"),
        "run_public_automatically": proposal.get("run_public_automatically"),
        "hard_failure_count": len(list_value(report.get("hard_failures"))),
        "all_blocker_gates_passed": all(
            row.get("passed") is True
            for row in list_value(report.get("gates"))
            if isinstance(row, dict) and row.get("severity") == "blocker"
        ),
        "external_inference_calls": report.get("external_inference_calls"),
        "teacher_solving_calls": report.get("teacher_solving_calls"),
        "fallback_return_count": report.get("fallback_return_count"),
        "public_training_rows_written": report.get("public_training_rows_written"),
        "public_prompt_chars_loaded": report.get("public_prompt_chars_loaded"),
        "public_context_chars_loaded": report.get("public_context_chars_loaded"),
        "public_answer_chars_loaded": report.get("public_answer_chars_loaded"),
    }


def readiness_gates(
    *,
    surfaces: list[dict[str, Any]],
    private_summary: dict[str, Any],
    coverage: dict[str, Any],
    contamination: dict[str, int],
    postmortem: dict[str, Any],
    latest_long_context: dict[str, Any],
    longmemeval_residual_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    ready_surfaces = {row["source_id"] for row in surfaces if row["status"] == "ready_prompt_scoring"}
    queued_surfaces = {row["source_id"] for row in surfaces if row["status"].startswith("queued")}
    gates = [
        gate("official_surface_audit_complete", {"ruler", "babilong", "longmemeval"}.issubset({row["source_id"] for row in surfaces}), "blocker", f"surfaces={[row['source_id'] for row in surfaces]}"),
        gate("ruler_and_babilong_prompt_ready", {"ruler", "babilong"}.issubset(ready_surfaces), "blocker", f"ready={sorted(ready_surfaces)}"),
        gate("longmemeval_ready_or_cleanly_queued", "longmemeval" in ready_surfaces or "longmemeval" in queued_surfaces, "blocker", f"ready={sorted(ready_surfaces)} queued={sorted(queued_surfaces)}"),
        gate("locked_slice_postmortem_done", bool(postmortem.get("private_analogues_pass")), "blocker", str(postmortem)),
        gate("private_adapter_no_off_only_regressions", int(private_summary.get("off_only_wins") or 0) == 0, "blocker", f"off_only={private_summary.get('off_only_wins')}"),
        gate("private_adapter_all_vcm_pass", float(private_summary.get("vcm_on_pass_rate") or 0.0) >= 1.0, "blocker", f"vcm_on={private_summary.get('vcm_on_pass_rate')}"),
        gate(
            "private_longmemeval_vcm_not_below_best_non_vcm",
            float(private_summary.get("longmemeval_vcm_over_best_non_vcm_delta") or 0.0) >= 0.0
            and int(private_summary.get("longmemeval_best_non_vcm_only_wins") or 0) == 0,
            "blocker",
            f"delta={private_summary.get('longmemeval_vcm_over_best_non_vcm_delta')} best_only={private_summary.get('longmemeval_best_non_vcm_only_wins')}",
        ),
        gate(
            "private_longmemeval_residual_curriculum_green",
            longmemeval_residual_summary.get("policy") == "project_theseus_vcm_longmemeval_private_residual_curriculum_v1"
            and longmemeval_residual_summary.get("trigger_state") == "GREEN"
            and longmemeval_residual_summary.get("private_only") is True
            and int(longmemeval_residual_summary.get("case_count") or 0) >= 150
            and float(longmemeval_residual_summary.get("vcm_on_pass_rate") or 0.0) >= 0.85
            and float(longmemeval_residual_summary.get("minimum_major_question_type_pass_rate") or 0.0) >= 0.75
            and float(longmemeval_residual_summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
            and float(longmemeval_residual_summary.get("vcm_on_evidence_recall") or 0.0) >= 0.85
            and longmemeval_residual_summary.get("all_blocker_gates_passed") is True
            and longmemeval_residual_summary.get("future_public_proposal_state") == "READY_TO_PROPOSE_EXACT_ONCE_PUBLIC_CONFIRMATION"
            and longmemeval_residual_summary.get("run_public_automatically") is False
            and all(
                int(longmemeval_residual_summary.get(key) or 0) == 0
                for key in [
                    "external_inference_calls",
                    "teacher_solving_calls",
                    "fallback_return_count",
                    "public_training_rows_written",
                    "public_prompt_chars_loaded",
                    "public_context_chars_loaded",
                    "public_answer_chars_loaded",
                ]
            ),
            "blocker",
            (
                f"state={longmemeval_residual_summary.get('trigger_state')} cases={longmemeval_residual_summary.get('case_count')} "
                f"vcm={longmemeval_residual_summary.get('vcm_on_pass_rate')} min_type={longmemeval_residual_summary.get('minimum_major_question_type_pass_rate')} "
                f"delta={longmemeval_residual_summary.get('vcm_over_best_single_non_vcm_delta')} recall={longmemeval_residual_summary.get('vcm_on_evidence_recall')} "
                f"proposal={longmemeval_residual_summary.get('future_public_proposal_state')}"
            ),
        ),
        gate("private_context_recovery_green", coverage.get("context_recovery_state") == "GREEN", "blocker", f"state={coverage.get('context_recovery_state')}"),
        gate("private_vcm_lift_positive", float(coverage.get("ablation_answer_lift") or 0.0) > 0.0, "blocker", f"lift={coverage.get('ablation_answer_lift')}"),
        gate("private_ablation_no_off_only", int(coverage.get("ablation_off_only_wins") or 0) == 0, "blocker", f"off_only={coverage.get('ablation_off_only_wins')}"),
        gate("private_public_memory_residual_categories_covered", bool(coverage.get("required_public_memory_residual_categories_present")), "blocker", f"categories={coverage.get('supplemental_categories')}"),
        gate(
            "latest_prompt_slice_has_item_manifest",
            bool(latest_long_context.get("item_manifest")) and bool(latest_long_context.get("item_manifest_hash")),
            "blocker",
            f"manifest={latest_long_context.get('item_manifest')} hash={latest_long_context.get('item_manifest_hash')}",
        ),
        gate(
            "latest_prompt_slice_long_context_or_lme_confirmation",
            (
                int(latest_long_context.get("scored_item_count") or 0) >= 1000
                and {"8k_to_32k", "32k_to_128k", "128k_plus"}.issubset(set(dict_value(latest_long_context.get("per_length_bucket"))))
            )
            or (
                int(latest_long_context.get("scored_item_count") or 0) >= 600
                and int(latest_long_context.get("longmemeval_item_count") or 0) >= 200
                and float(latest_long_context.get("longmemeval_vcm_on_pass_rate") or 0.0) > LOCKED_LONGMEMEVAL_BASELINE_VCM
                and float(latest_long_context.get("longmemeval_vcm_over_flat_tail_delta") or 0.0) > 0.0
                and float(latest_long_context.get("longmemeval_vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.0
            ),
            "blocker",
            (
                f"scored={latest_long_context.get('scored_item_count')} buckets={sorted(dict_value(latest_long_context.get('per_length_bucket')).keys())} "
                f"lme_items={latest_long_context.get('longmemeval_item_count')} lme_vcm={latest_long_context.get('longmemeval_vcm_on_pass_rate')} "
                f"lme_flat_delta={latest_long_context.get('longmemeval_vcm_over_flat_tail_delta')} "
                f"lme_best_delta={latest_long_context.get('longmemeval_vcm_over_best_single_non_vcm_delta')}"
            ),
        ),
        gate(
            "latest_prompt_slice_no_forbidden_overlap",
            all(int(value or 0) == 0 for value in dict_value(latest_long_context.get("forbidden_overlap_counts")).values()),
            "blocker",
            f"overlap_counts={latest_long_context.get('forbidden_overlap_counts')}",
        ),
        gate("no_cheat_counters_zero", all(int(value or 0) == 0 for value in contamination.values()), "blocker", str(contamination)),
    ]
    return gates


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VCM Public Memory Readiness Audit",
        "",
        f"State: `{report['trigger_state']}`",
        f"Public calibration allowed: `{report['public_calibration_allowed']}`",
        f"Recommended public slice: `{report['recommended_public_slice_id']}`",
        "",
        "## Surfaces",
        "",
    ]
    for row in report["surface_audit"]:
        lines.append(f"- `{row['source_id']}`: `{row['status']}` - {row.get('blocker') or 'ready'}")
    lines.extend(["", "## Private Analogues", ""])
    summary = report["private_analogue_summary"]
    lines.append(f"- VCM-on pass rate: `{summary['vcm_on_pass_rate']}`")
    lines.append(f"- VCM-off pass rate: `{summary['vcm_off_pass_rate']}`")
    lines.append(f"- Best non-VCM pass rate: `{summary.get('best_non_vcm_pass_rate')}`")
    lines.append(f"- LongMemEval VCM over best non-VCM: `{summary.get('longmemeval_vcm_over_best_non_vcm_delta')}`")
    lines.append(f"- Off-only wins: `{summary['off_only_wins']}`")
    lines.append(f"- Covered analogues: `{summary['covered_analogues']}`")
    residual = report.get("private_longmemeval_residual_curriculum", {})
    lines.extend(["", "## Private LongMemEval Residual Curriculum", ""])
    lines.append(f"- State: `{residual.get('trigger_state')}`")
    lines.append(f"- Cases: `{residual.get('case_count')}`")
    lines.append(f"- VCM pass rate: `{residual.get('vcm_on_pass_rate')}`")
    lines.append(f"- Best non-VCM pass rate: `{residual.get('best_single_non_vcm_pass_rate')}`")
    lines.append(f"- VCM over best non-VCM: `{residual.get('vcm_over_best_single_non_vcm_delta')}`")
    lines.append(f"- Minimum major question type pass rate: `{residual.get('minimum_major_question_type_pass_rate')}`")
    lines.append(f"- Evidence recall: `{residual.get('vcm_on_evidence_recall')}`")
    lines.append(f"- Future public proposal state: `{residual.get('future_public_proposal_state')}`")
    lines.extend(["", "## Latest Long Context Slice", ""])
    latest = report.get("latest_long_context_prompt_slice", {})
    lines.append(f"- Slice: `{latest.get('slice_id')}`")
    lines.append(f"- Scored items: `{latest.get('scored_item_count')}`")
    lines.append(f"- VCM-on / VCM-off: `{latest.get('vcm_on_pass_rate')}` / `{latest.get('vcm_off_pass_rate')}`")
    lines.append(f"- VCM over best non-VCM: `{latest.get('vcm_over_best_non_vcm_delta')}`")
    lines.append(f"- LongMemEval VCM / flat-tail: `{latest.get('longmemeval_vcm_on_pass_rate')}` / `{latest.get('longmemeval_vcm_off_pass_rate')}`")
    lines.append(f"- LongMemEval VCM over best single non-VCM: `{latest.get('longmemeval_vcm_over_best_single_non_vcm_delta')}`")
    lines.append(f"- Source token distribution: `{latest.get('source_context_token_distribution')}`")
    lines.append(f"- Item manifest hash: `{latest.get('item_manifest_hash')}`")
    lines.extend(["", "## Gates", ""])
    for row in report["gates"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: `{row['gate']}` - {row['evidence']}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


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
    return cursor if cursor is not None else default


def mean(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
