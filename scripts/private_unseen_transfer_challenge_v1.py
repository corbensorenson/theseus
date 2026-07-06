#!/usr/bin/env python3
"""Private unseen-transfer challenger.

This runner builds a private-only OOD slice from already-generated private
ecology heldout rows. It preserves hidden private tests and solution bodies but
changes task ids, entry points, categories, prompts, tags, and semantic-family
keys so exact semantic lookup is not enough. The expected path is the learned
fingerprint/prototype inference route.

It never reads or executes public benchmark data and never runs public
calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_residual_curriculum import verify_private_solution_rows  # noqa: E402


REPORTS = ROOT / "reports"
SOURCE_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl"
SOURCE_TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "private_ecology_generalization_v5_code_lm_tasks.jsonl"
CHALLENGE_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_unseen_transfer_challenge_v1_code_lm_tasks.jsonl"
STS_STREAMS = REPORTS / "private_unseen_transfer_challenge_v1_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "private_unseen_transfer_challenge_v1_private_safe_sts_streams.json"
EMPTY_PUBLIC = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_public.jsonl"
EMPTY_STS = REPORTS / "public_safe_broad_transfer_maturity_v4_empty_sts_streams.jsonl"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_unseen_transfer_challenge_v1.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_unseen_transfer_challenge_v1_empty_public.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_private_unseen_transfer_challenge_v1_fanout.json"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_unseen_transfer_challenge_v1_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_unseen_transfer_challenge_v1_sts_off_empty_public.jsonl"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_private_unseen_transfer_challenge_v1_sts_off_fanout.json"
SCORE_REPORT = REPORTS / "private_unseen_transfer_challenge_v1_score.json"
SCORE_MD = REPORTS / "private_unseen_transfer_challenge_v1_score.md"
LEARNED_ONLY_CANDIDATES = REPORTS / "code_lm_private_candidates_private_unseen_transfer_challenge_v1_learned_only.jsonl"
LEARNED_ONLY_SCORE = REPORTS / "private_unseen_transfer_challenge_v1_learned_only_score.json"
LEARNED_ONLY_SCORE_MD = REPORTS / "private_unseen_transfer_challenge_v1_learned_only_score.md"
LEARNED_GATE = REPORTS / "private_unseen_transfer_challenge_v1_learned_distillation_gate.json"
LEARNED_GATE_MD = REPORTS / "private_unseen_transfer_challenge_v1_learned_distillation_gate.md"
REPORT = REPORTS / "private_unseen_transfer_challenge_v1.json"
MD = REPORTS / "private_unseen_transfer_challenge_v1.md"
QUEUE = REPORTS / "private_unseen_transfer_challenge_v1_queue.jsonl"
LEDGER = REPORTS / "private_unseen_transfer_challenge_v1_ledger.jsonl"
HEARTBEAT = REPORTS / "private_unseen_transfer_challenge_v1_heartbeat.json"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
OPERATOR_DRY_RUN = REPORTS / "operator_bounded_public_calibration_dry_run.json"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]

ALIAS_BY_SEMANTIC = {
    "memory_state_tracking": ("recent_project_note_state", "Find the newest saved note text for each project."),
    "action_memory_rollup": ("owner_open_action_index", "Group currently open action labels by responsible owner."),
    "tool_status_parsing": ("command_transcript_outcome_parse", "Convert command transcript lines into structured command status entries."),
    "tool_error_clustering": ("runtime_failure_bucket_rollup", "Count runtime failure messages by timeout, permission, network, or other bucket."),
    "storage_selection": ("priority_quota_file_pick", "Choose file names that fit a byte budget using stable priority ordering."),
    "storage_sync_plan": ("manifest_delta_transfer_plan", "Create upload, download, or delete-style sync operations from two manifests."),
    "capability_latency_routing": ("capability_aware_node_pick", "Pick the best node that satisfies capability needs while respecting latency and battery constraints."),
    "voice_following_route": ("room_speaker_handoff_choice", "Choose the best speaker node for a room-aware voice response."),
    "dependency_planning": ("unblocked_work_item_order", "Return unblocked work item ids after dependency and priority filtering."),
    "project_progress_digest": ("project_work_state_summary", "Summarize open, blocked, done, and owner state for project work records."),
    "room_capability_summary": ("room_device_capability_counts", "Summarize device, mic, and speaker counts per room."),
    "media_preview_retrieval": ("filtered_media_preview_ids", "Return media ids matching album/tag filters ordered newest first."),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--checkpoint-in", default=rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=rel(REPORT))
    parser.add_argument("--markdown-out", default=rel(MD))
    parser.add_argument("--queue-out", default=rel(QUEUE))
    args = parser.parse_args()

    run_id = f"private_unseen_transfer_challenge_v1_{int(time.time())}"
    started = time.time()
    phases: list[dict[str, Any]] = []
    append_event(run_id, "run", "start", {"execute": bool(args.execute), "args": vars(args)})
    write_heartbeat(run_id, "preflight", "running", started, args, phases, {})

    preflight = preflight_report(args)
    phases.append(phase_record("preflight", 0, preflight, started, time.time()))
    append_event(run_id, "preflight", "finish", preflight)
    completion = "dry_run_ready"
    blocker: dict[str, Any] = {}
    if not preflight["ready"]:
        completion = "precise_blocker"
        blocker = {"kind": "preflight", "detail": preflight["blockers"][0] if preflight["blockers"] else "unknown"}
    elif args.execute:
        completion, blocker = execute_phases(run_id, args, phases, started)

    report = build_report(run_id, args, started, phases, preflight, completion, blocker)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), queue_rows(report, args))
    write_heartbeat(run_id, "complete", str(report["trigger_state"]).lower(), started, args, phases, blocker)
    append_event(run_id, "run", "finish", {"trigger_state": report["trigger_state"], "completion": completion, "blocker": blocker})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    packet = read_json(READINESS_PACKET, {})
    dry_run = read_json(OPERATOR_DRY_RUN, {})
    dry_summary = object_field(dry_run, "summary")
    disk = shutil.disk_usage(ROOT)
    free_gb = disk.free / (1024**3)
    battery = battery_state()
    checkpoint = resolve(args.checkpoint_in)
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    gates = [
        gate("source_private_heldout_present", SOURCE_HELDOUT.exists(), rel(SOURCE_HELDOUT)),
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE)),
        gate("checkpoint_present", checkpoint.exists(), rel(checkpoint)),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("public_calibration_disallowed", packet.get("public_calibration_allowed") is False, packet.get("public_calibration_allowed")),
        gate("operator_dry_run_not_executed", dry_summary.get("executed") is False, dry_summary.get("executed")),
        gate("forbidden_post_v4_public_artifacts_absent", not forbidden_present, forbidden_present),
        gate("free_disk_ge_min", free_gb >= float(args.min_free_gb), {"free_gb": round(free_gb, 3), "min_free_gb": float(args.min_free_gb)}),
        gate("battery_allowed_or_ac_power", bool(args.allow_battery or not battery.get("on_battery")), battery),
    ]
    blockers = [row for row in gates if not row["passed"]]
    return {
        "ready": not blockers,
        "blockers": blockers,
        "gates": gates,
        "free_gb": round(free_gb, 3),
        "battery": battery,
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def execute_phases(run_id: str, args: argparse.Namespace, phases: list[dict[str, Any]], started: float) -> tuple[str, dict[str, Any]]:
    ensure_sidecars()
    deadline = started + max(0.1, float(args.max_hours)) * 3600.0
    for name, cmd, env in phase_commands(args):
        if time.time() > deadline:
            return "precise_blocker", {"kind": "time_budget_exhausted", "detail": f"stopped before {name}"}
        write_heartbeat(run_id, name, "running", started, args, phases, {})
        append_event(run_id, name, "start", {"cmd": cmd, "env": sorted(env)})
        phase_start = time.time()
        result = run_command(cmd, env=env, timeout=max(60, int(deadline - time.time())))
        phases.append(phase_record(name, result["returncode"], result, phase_start, time.time()))
        append_event(run_id, name, "finish", result)
        if result["returncode"] != 0:
            return "precise_blocker", {"kind": "phase_failed", "phase": name, "returncode": result["returncode"], "stderr_tail": result["stderr_tail"]}
    score = read_json(SCORE_REPORT, {})
    learned = read_json(LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned, "summary")
    if (
        score.get("trigger_state") in {"GREEN", "YELLOW"}
        and learned.get("trigger_state") in {"GREEN", "YELLOW"}
        and first_number(score_summary.get("pass_rate"), 0.0) >= 0.70
        and first_number(learned_summary.get("learned_only_pass_rate"), 0.0) >= 0.70
        and int(first_number(learned_summary.get("prototype_pass_count"), 999)) == 0
    ):
        return "private_unseen_transfer_challenge_ready", {}
    return "precise_blocker", {
        "kind": "unseen_transfer_below_floor",
        "score_trigger_state": score.get("trigger_state"),
        "learned_trigger_state": learned.get("trigger_state"),
        "score_summary": score_summary,
        "learned_summary": learned_summary,
    }


def phase_commands(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, str]]]:
    rows = max(24, int(args.rows))
    return [
        (
            "generate_private_unseen_transfer_challenge",
            [
                sys.executable,
                "scripts/private_unseen_transfer_challenge_v1.py",
                "--rows",
                str(rows),
                "--seed",
                str(int(args.seed)),
                "--out",
                rel(REPORT),
                "--markdown-out",
                rel(MD),
                "--queue-out",
                rel(QUEUE),
            ],
            {},
        ),
        (
            "build_private_safe_sts_streams",
            [
                sys.executable,
                "scripts/private_task_sts_streams.py",
                "--tasks",
                rel(CHALLENGE_HELDOUT),
                "--out",
                rel(STS_STREAMS),
                "--report-out",
                rel(STS_STREAMS_REPORT),
                "--task-limit",
                str(rows),
            ],
            {},
        ),
        (
            "fanout_sts_on",
            fanout_command(args, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, STS_STREAMS, rows),
            fanout_env(sts_on=True),
        ),
        (
            "fanout_sts_off_control",
            fanout_command(args, CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, EMPTY_STS, rows),
            fanout_env(sts_on=False),
        ),
        (
            "score_private_unseen_transfer",
            [
                sys.executable,
                "scripts/broad_private_generalization_score_v1.py",
                "--heldout",
                rel(CHALLENGE_HELDOUT),
                "--candidates",
                rel(PRIVATE_CANDIDATES),
                "--control-candidates",
                rel(CONTROL_PRIVATE_CANDIDATES),
                "--timeout-seconds",
                str(max(1, int(args.score_timeout_seconds))),
                "--task-limit",
                str(rows),
                "--min-heldout-rows",
                str(rows),
                "--out",
                rel(SCORE_REPORT),
                "--markdown-out",
                rel(SCORE_MD),
            ],
            {},
        ),
        (
            "learned_only_distillation_gate",
            [
                sys.executable,
                "scripts/broad_private_learned_distillation_gate_v1.py",
                "--heldout",
                rel(CHALLENGE_HELDOUT),
                "--candidates",
                rel(PRIVATE_CANDIDATES),
                "--control-candidates",
                rel(CONTROL_PRIVATE_CANDIDATES),
                "--score",
                rel(SCORE_REPORT),
                "--private-train",
                rel(SOURCE_TRAIN),
                "--learned-only-candidates-out",
                rel(LEARNED_ONLY_CANDIDATES),
                "--learned-only-score-out",
                rel(LEARNED_ONLY_SCORE),
                "--learned-only-score-markdown-out",
                rel(LEARNED_ONLY_SCORE_MD),
                "--timeout-seconds",
                str(max(1, int(args.score_timeout_seconds))),
                "--task-limit",
                str(rows),
                "--min-heldout-rows",
                str(rows),
                "--out",
                rel(LEARNED_GATE),
                "--markdown-out",
                rel(LEARNED_GATE_MD),
            ],
            {},
        ),
    ]


def build_report(
    run_id: str,
    args: argparse.Namespace,
    started: float,
    phases: list[dict[str, Any]],
    preflight: dict[str, Any],
    completion: str,
    blocker: dict[str, Any],
) -> dict[str, Any]:
    if preflight.get("ready") and (not args.execute or not CHALLENGE_HELDOUT.exists()):
        generate_challenge_rows(max(24, int(args.rows)), int(args.seed))
    rows = read_jsonl(CHALLENGE_HELDOUT)
    row_count = len(rows)
    challenge_summary = challenge_row_summary(rows)
    streams = read_json(STS_STREAMS_REPORT, {})
    fanout = read_json(FANOUT_REPORT, {})
    control_fanout = read_json(CONTROL_FANOUT_REPORT, {})
    score = read_json(SCORE_REPORT, {})
    learned = read_json(LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned, "summary")
    streams_summary = object_field(streams, "summary")
    fanout_summary = object_field(fanout, "summary")
    control_summary = object_field(control_fanout, "summary")
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    freshness = freshness_after_rows()
    gates = [
        gate("preflight_ready", preflight.get("ready") is True, preflight.get("blockers")),
        gate("challenge_rows_written", row_count >= max(24, int(args.rows)), row_count),
        gate("exact_semantic_keys_withheld", challenge_summary["exact_semantic_key_replay_count"] == 0, challenge_summary),
        gate("private_solution_tests_pass", challenge_summary["private_solution_failure_count"] == 0, challenge_summary),
        gate("public_data_leakage_zero", challenge_summary["public_data_leakage_hit_count"] == 0, challenge_summary),
        gate("sts_streams_green", streams.get("trigger_state") in {"GREEN", None}, streams.get("trigger_state")),
        gate("fanout_sts_on_private_only", public_candidate_rows(PUBLIC_CANDIDATES) == 0, public_candidate_rows(PUBLIC_CANDIDATES)),
        gate("fanout_sts_off_private_only", public_candidate_rows(CONTROL_PUBLIC_CANDIDATES) == 0, public_candidate_rows(CONTROL_PUBLIC_CANDIDATES)),
        gate("score_private_floor", first_number(score_summary.get("pass_rate"), 0.0) >= 0.70 if score_summary else not bool(args.execute), score_summary.get("pass_rate")),
        gate("learned_only_floor", first_number(learned_summary.get("learned_only_pass_rate"), 0.0) >= 0.70 if learned_summary else not bool(args.execute), learned_summary.get("learned_only_pass_rate")),
        gate("prototype_pass_count_zero", int(first_number(learned_summary.get("prototype_pass_count"), 0)) == 0, learned_summary.get("prototype_pass_count")),
        gate("freshness_after_challenge_rows", freshness["fresh"] if args.execute else True, freshness),
        gate("public_lock_still_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("forbidden_post_v4_public_artifacts_absent", not forbidden_present, forbidden_present),
        gate("external_inference_zero", external_inference_zero(streams, fanout, control_fanout, score, learned), 0),
    ]
    hard_failed = [row for row in gates if not row["passed"] and row["gate"] in {"preflight_ready", "public_data_leakage_zero", "public_lock_still_active", "forbidden_post_v4_public_artifacts_absent", "external_inference_zero"}]
    failed = [row for row in gates if not row["passed"]]
    green = completion == "private_unseen_transfer_challenge_ready" and not failed
    trigger_state = "GREEN" if green else ("RED" if hard_failed else "YELLOW")
    return {
        "policy": "project_theseus_private_unseen_transfer_challenge_v1",
        "created_utc": now(),
        "run_id": run_id,
        "trigger_state": trigger_state,
        "public_calibration_allowed": False,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "inputs": {
            "execute": bool(args.execute),
            "rows": max(24, int(args.rows)),
            "seed": int(args.seed),
            "candidates_per_task": max(1, int(args.candidates_per_task)),
            "checkpoint_in": rel(resolve(args.checkpoint_in)),
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "summary": {
            "completion_evidence_status": completion,
            "elapsed_seconds": round(time.time() - started, 3),
            "phase_count": len(phases),
            "challenge_row_count": row_count,
            **challenge_summary,
            "sts_stream_task_count": streams_summary.get("sts_stream_task_count"),
            "sts_on_candidate_count": fanout_summary.get("private_candidate_count"),
            "sts_off_candidate_count": control_summary.get("private_candidate_count"),
            "pass_count": score_summary.get("pass_count"),
            "pass_rate": score_summary.get("pass_rate"),
            "control_pass_count": score_summary.get("control_pass_count"),
            "control_pass_rate": score_summary.get("control_pass_rate"),
            "sts_delta": score_summary.get("sts_delta"),
            "sts_regressions": score_summary.get("sts_regressions"),
            "learned_only_pass_count": learned_summary.get("learned_only_pass_count"),
            "learned_only_pass_rate": learned_summary.get("learned_only_pass_rate"),
            "learned_token_pass_count": learned_summary.get("learned_token_pass_count"),
            "prototype_pass_count": learned_summary.get("prototype_pass_count"),
            "freshness": freshness,
            "score_semantics": "private OOD transfer challenge only; not public calibration or promotion evidence",
            "external_inference_calls": 0,
        },
        "preflight": preflight,
        "gates": gates,
        "blocker": blocker,
        "phases": phases,
        "artifacts": artifacts(),
        "queue": queue_rows_from_state(trigger_state),
        "next_actions": next_actions(trigger_state, score, learned),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def generate_challenge_rows(row_count: int, seed: int) -> dict[str, Any]:
    source = read_jsonl(SOURCE_HELDOUT)
    selected = deterministic_sample(source, row_count, seed)
    rows = [challenge_row(row, index) for index, row in enumerate(selected)]
    write_jsonl(CHALLENGE_HELDOUT, rows)
    check = verify_private_solution_rows(rows, max_failures=24)
    return {"rows": len(rows), "private_solution_check": check}


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    keyed = [(stable_hash(f"{seed}:{row.get('task_id')}"), row) for row in rows]
    keyed.sort(key=lambda pair: pair[0])
    selected = [row for _, row in keyed[: min(count, len(keyed))]]
    selected.sort(key=lambda row: str(row.get("task_id") or ""))
    return selected


def challenge_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    contract = object_field(out, "decoder_contract")
    original_entry = str(out.get("entry_point") or "")
    original_category = str(out.get("category") or "")
    original_semantic = str(contract.get("semantic_family") or original_category)
    alias_semantic, alias_prompt = ALIAS_BY_SEMANTIC.get(original_semantic, (f"ood_{original_semantic}", str(out.get("prompt") or "")))
    new_category = f"private_ood_{alias_semantic}_{index:04d}"
    new_entry = f"{new_category}_entry"
    out["task_id"] = f"private_unseen_transfer_challenge_v1_{index:04d}"
    out["source_task_id"] = str(row.get("task_id") or "")
    out["entry_point"] = new_entry
    out["category"] = new_category
    out["prompt"] = f"Private OOD transfer contract: {alias_prompt}"
    out["benchmark_evidence_level"] = "private_ecology_generalization_v5_generated_only;private_unseen_transfer_challenge_v1_ood_alias"
    out["tags"] = challenge_tags(row, alias_semantic)
    out["private_unseen_transfer_challenge_v1"] = {
        "source_category": original_category,
        "source_semantic_family": original_semantic,
        "alias_semantic_family": alias_semantic,
        "exact_semantic_key_withheld": alias_semantic != original_semantic and new_category != original_category,
        "public_benchmark_inputs_read": False,
    }
    out["public_benchmark"] = False
    out["public_tests_included"] = False
    out["public_benchmark_solutions_included"] = False
    if original_entry:
        out["tests"] = str(out.get("tests") or "").replace(original_entry, new_entry)
    contract["semantic_family"] = alias_semantic
    contract["residual_label_hint"] = alias_semantic
    contract["score_semantics"] = "private OOD transfer challenge; exact private semantic key withheld"
    contract["generation_plan"] = object_field(contract, "generation_plan")
    contract["generation_plan"]["repair_strategy"] = "infer reusable behavior from private contract fingerprints, not exact semantic-family replay"
    contract["generation_plan"]["public_tests_used"] = False
    contract["generation_plan"]["public_solutions_used"] = False
    out["decoder_contract"] = contract
    return out


def challenge_tags(row: dict[str, Any], alias_semantic: str) -> list[str]:
    raw = [str(tag) for tag in row.get("tags", []) if isinstance(tag, str)]
    filtered = [
        tag
        for tag in raw
        if not tag.startswith("v5_")
        and tag not in {"private_ecology_generalization_v5", str(row.get("category") or "")}
    ]
    filtered.extend(["private_unseen_transfer_challenge_v1", "ood_alias", alias_semantic])
    return sorted(set(filtered))


def fanout_command(args: argparse.Namespace, private_out: Path, public_out: Path, report_out: Path, sts_streams: Path, eval_limit: int) -> list[str]:
    return [
        rel(RELEASE),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(CHALLENGE_HELDOUT),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC),
        "--checkpoint-in",
        rel(resolve(args.checkpoint_in)),
        "--seed",
        str(int(args.seed)),
        "--candidates-per-task",
        str(max(1, int(args.candidates_per_task))),
        "--private-candidate-out",
        rel(private_out),
        "--public-candidate-out",
        rel(public_out),
        "--report-out",
        rel(report_out),
        "--public-task-limit",
        "0",
        "--private-eval-limit",
        str(eval_limit),
        "--sts-streams",
        rel(sts_streams),
    ]


def fanout_env(*, sts_on: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
    if not sts_on:
        env["THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY"] = "1"
    return env


def challenge_row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    check = verify_private_solution_rows(rows, max_failures=24) if rows else {"failure_count": 0, "sample_failures": []}
    families = Counter(str(row.get("broad_private_family_v1") or "unknown") for row in rows)
    aliases = Counter(str(object_field(row, "decoder_contract").get("semantic_family") or "") for row in rows)
    exact_replay = 0
    for row in rows:
        marker = object_field(row, "private_unseen_transfer_challenge_v1")
        if not marker.get("exact_semantic_key_withheld"):
            exact_replay += 1
    leakage = public_leakage_scan(rows)
    return {
        "family_counts": dict(sorted(families.items())),
        "alias_semantic_family_count": len(aliases),
        "exact_semantic_key_replay_count": exact_replay,
        "private_solution_failure_count": int(check.get("failure_count") or 0),
        "private_solution_sample_failures": check.get("sample_failures") or [],
        "public_data_leakage_hit_count": leakage["hit_count"],
    }


def freshness_after_rows() -> dict[str, Any]:
    paths = {
        "challenge_rows": CHALLENGE_HELDOUT,
        "sts_streams": STS_STREAMS_REPORT,
        "fanout": FANOUT_REPORT,
        "control_fanout": CONTROL_FANOUT_REPORT,
        "score": SCORE_REPORT,
        "learned_gate": LEARNED_GATE,
    }
    mtimes = {name: int(path.stat().st_mtime) if path.exists() else 0 for name, path in paths.items()}
    fresh = bool(
        mtimes["challenge_rows"]
        and mtimes["sts_streams"] >= mtimes["challenge_rows"]
        and mtimes["fanout"] >= mtimes["sts_streams"]
        and mtimes["control_fanout"] >= mtimes["sts_streams"]
        and mtimes["score"] >= max(mtimes["fanout"], mtimes["control_fanout"])
        and mtimes["learned_gate"] >= mtimes["score"]
    )
    return {"fresh": fresh, "mtimes": mtimes}


def queue_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return report.get("queue") if isinstance(report.get("queue"), list) else queue_rows_from_state(str(report.get("trigger_state")))


def queue_rows_from_state(trigger_state: str) -> list[dict[str, Any]]:
    if trigger_state == "GREEN":
        return [
            queue_item("refresh_generalization_governor", [sys.executable, "scripts/theseus_generalization_governor_v1.py"], 90),
        ]
    return [
        queue_item("inspect_private_unseen_transfer_residuals", [], 20, safe=False),
        queue_item("rerun_private_unseen_transfer_challenge", [sys.executable, "scripts/private_unseen_transfer_challenge_v1.py", "--execute"], 30),
    ]


def queue_item(kind: str, command: list[str], priority: int, *, safe: bool = True) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_unseen_transfer_challenge_queue_item_v1",
        "queue": "private_unseen_transfer_challenge_v1",
        "kind": kind,
        "priority": priority,
        "status": "pending",
        "command": command,
        "public_calibration_allowed": False,
        "requires_operator_public_unlock": False,
        "safe_to_execute_without_operator_public_approval": safe,
    }


def next_actions(trigger_state: str, score: dict[str, Any], learned: dict[str, Any]) -> list[str]:
    if trigger_state == "GREEN":
        return [
            "treat this as stronger private OOD transfer evidence, not public promotion evidence",
            "refresh the generalization governor and keep public calibration locked",
        ]
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned, "summary")
    return [
        "cluster private unseen-transfer failures and convert them into a new private residual curriculum",
        f"observed pass_rate={score_summary.get('pass_rate')} learned_only={learned_summary.get('learned_only_pass_rate')}",
    ]


def artifacts() -> dict[str, str]:
    return {
        "challenge_rows": rel(CHALLENGE_HELDOUT),
        "sts_streams": rel(STS_STREAMS_REPORT),
        "sts_on_fanout": rel(FANOUT_REPORT),
        "sts_off_fanout": rel(CONTROL_FANOUT_REPORT),
        "score": rel(SCORE_REPORT),
        "learned_gate": rel(LEARNED_GATE),
        "report": rel(REPORT),
        "queue": rel(QUEUE),
        "ledger": rel(LEDGER),
        "heartbeat": rel(HEARTBEAT),
    }


def run_command(cmd: list[str], *, env: dict[str, str], timeout: int) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env.update(env)
    started = time.time()
    try:
        result = subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=max(1, int(timeout)))
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout_seconds": timeout,
        }


def ensure_sidecars() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    EMPTY_PUBLIC.write_text("", encoding="utf-8")
    EMPTY_STS.write_text("", encoding="utf-8")


def battery_state() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "on_battery": False, "reason": "not_macos"}
    try:
        result = subprocess.run(["pmset", "-g", "batt"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except Exception as exc:
        return {"available": False, "on_battery": False, "reason": f"{type(exc).__name__}: {exc}"}
    text = result.stdout.lower()
    return {"available": result.returncode == 0, "on_battery": "battery power" in text, "raw": result.stdout.strip()[:500]}


def write_heartbeat(run_id: str, phase: str, state: str, started: float, args: argparse.Namespace, phases: list[dict[str, Any]], blocker: dict[str, Any]) -> None:
    write_json(
        HEARTBEAT,
        {
            "policy": "project_theseus_private_unseen_transfer_challenge_heartbeat_v1",
            "run_id": run_id,
            "updated_utc": now(),
            "phase": phase,
            "state": state,
            "elapsed_seconds": round(time.time() - started, 3),
            "execute": bool(args.execute),
            "phase_count": len(phases),
            "blocker": blocker,
        },
    )


def append_event(run_id: str, phase: str, event: str, payload: Any) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"policy": "project_theseus_private_unseen_transfer_challenge_ledger_v1", "run_id": run_id, "created_utc": now(), "phase": phase, "event": event, "payload": payload}, sort_keys=True) + "\n")


def phase_record(name: str, returncode: int, evidence: Any, started: float, finished: float) -> dict[str, Any]:
    return {
        "phase": name,
        "returncode": int(returncode),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "elapsed_seconds": round(finished - started, 3),
        "evidence": evidence,
    }


def public_candidate_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def external_inference_zero(*reports: dict[str, Any]) -> bool:
    for report in reports:
        if int(first_number(report.get("external_inference_calls"), 0)) != 0:
            return False
        if int(first_number(object_field(report, "summary").get("external_inference_calls"), 0)) != 0:
            return False
    return True


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = ["humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench", "canonical_solution", "public_test"]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
    return {"hit_count": len(hits), "sample_hits": hits[:8]}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    lines = [
        "# Private Unseen Transfer Challenge v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Completion: `{summary.get('completion_evidence_status')}`",
        f"- Challenge rows: `{summary.get('challenge_row_count')}`",
        f"- Exact semantic replay count: `{summary.get('exact_semantic_key_replay_count')}`",
        f"- Pass rate: `{summary.get('pass_rate')}`",
        f"- Learned-only pass rate: `{summary.get('learned_only_pass_rate')}`",
        f"- Prototype pass count: `{summary.get('prototype_pass_count')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Failed gates: `{len(failed)}`",
        "",
        "## Artifacts",
    ]
    for key, value in object_field(report, "artifacts").items():
        lines.append(f"- `{key}`: `{value}`")
    if failed:
        lines.extend(["", "## Failed Gates"])
        for row in failed:
            lines.append(f"- `{row.get('gate')}`: `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
