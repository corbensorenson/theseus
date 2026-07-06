#!/usr/bin/env python3
"""Private contract-blind transfer gate.

This gate rewrites already-private heldout tasks so prompt, category, tags,
semantic-family, residual-label, and broad-family names are intentionally
opaque. Hidden private tests and solution bodies stay local. The goal is to
measure whether the STS-conditioned learned-token decoder can infer reusable
behavior from contract shape, argument roles, return shape, and required
constructs rather than from exact semantic names.

It never reads public benchmark tests or solutions and never runs public
calibration.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
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

from code_residual_curriculum import verify_private_solution_rows  # noqa: E402


REPORTS = ROOT / "reports"
SOURCE_HELDOUTS = [
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_eval"
    / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl",
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_eval"
    / "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl",
]
SOURCE_TRAINS = [
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl",
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "public_safe_broad_transfer_maturity_v4_code_lm_tasks.jsonl",
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "private_ecology_generalization_v5_code_lm_tasks.jsonl",
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "post_v4_private_shadow_transfer_v1_code_lm_tasks.jsonl",
]
HELDOUT = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_eval"
    / "private_contract_blind_transfer_v1_code_lm_tasks.jsonl"
)
TRAIN_REFERENCE = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "private_contract_blind_transfer_v1_train_reference_code_lm_tasks.jsonl"
)

EMPTY_PUBLIC = REPORTS / "private_contract_blind_transfer_v1_empty_public.jsonl"
EMPTY_STS = REPORTS / "private_contract_blind_transfer_v1_empty_sts_streams.jsonl"
STS_STREAMS = REPORTS / "private_contract_blind_transfer_v1_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "private_contract_blind_transfer_v1_private_safe_sts_streams.json"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_contract_blind_transfer_v1.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_contract_blind_transfer_v1_empty_public.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_private_contract_blind_transfer_v1_fanout.json"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_contract_blind_transfer_v1_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_contract_blind_transfer_v1_sts_off_empty_public.jsonl"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_private_contract_blind_transfer_v1_sts_off_fanout.json"
SCORE = REPORTS / "private_contract_blind_transfer_v1_score.json"
SCORE_MD = REPORTS / "private_contract_blind_transfer_v1_score.md"
LEARNED_CANDIDATES = REPORTS / "code_lm_private_candidates_private_contract_blind_transfer_v1_learned_only.jsonl"
LEARNED_SCORE = REPORTS / "private_contract_blind_transfer_v1_learned_only_score.json"
LEARNED_SCORE_MD = REPORTS / "private_contract_blind_transfer_v1_learned_only_score.md"
STRICT_LEARNED_CANDIDATES = REPORTS / "code_lm_private_candidates_private_contract_blind_transfer_v1_strict_novel_learned_only.jsonl"
STRICT_LEARNED_SCORE = REPORTS / "private_contract_blind_transfer_v1_strict_novel_learned_only_score.json"
STRICT_LEARNED_SCORE_MD = REPORTS / "private_contract_blind_transfer_v1_strict_novel_learned_only_score.md"
LEARNED_GATE = REPORTS / "private_contract_blind_transfer_v1_learned_distillation_gate.json"
LEARNED_GATE_MD = REPORTS / "private_contract_blind_transfer_v1_learned_distillation_gate.md"
REPORT = REPORTS / "private_contract_blind_transfer_v1.json"
MD = REPORTS / "private_contract_blind_transfer_v1.md"
QUEUE = REPORTS / "private_contract_blind_transfer_v1_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]

OPAQUE_PROMPT = "Private contract-blind transfer task. Implement the required transformation from the visible arguments and decoder contract."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=307)
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

    started = time.time()
    preflight = preflight_report(args)
    build = write_contract_blind_heldout(max_rows=max(1, int(args.rows)), seed=int(args.seed))
    commands: list[dict[str, Any]] = []
    completion = "dry_run_ready"
    blocker: dict[str, Any] = {}
    if not preflight["ready"]:
        completion = "precise_blocker"
        blocker = {"kind": "preflight", "detail": preflight["blockers"][0] if preflight["blockers"] else "unknown"}
    elif args.execute:
        completion, blocker, commands = execute(args, started)

    report = build_report(args, started, preflight, build, commands, completion, blocker)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), queue_rows(report, args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    packet = read_json(READINESS_PACKET, {})
    checkpoint = resolve(args.checkpoint_in)
    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    battery = battery_state()
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    gates = [
        gate("source_private_heldouts_present", all(path.exists() for path in SOURCE_HELDOUTS), [rel(path) for path in SOURCE_HELDOUTS]),
        gate("source_private_trains_present", all(path.exists() for path in SOURCE_TRAINS), [rel(path) for path in SOURCE_TRAINS]),
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE)),
        gate("checkpoint_present", checkpoint.exists(), rel(checkpoint)),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("public_calibration_disallowed", packet.get("public_calibration_allowed") is False, packet.get("public_calibration_allowed")),
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


def write_contract_blind_heldout(*, max_rows: int, seed: int) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for path in SOURCE_HELDOUTS:
        rows = read_jsonl(path)
        source_counts[rel(path)] = len(rows)
        source_rows.extend(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if private_row_allowed(row):
            grouped[str(row.get("category") or "unknown")].append(row)
    selected = round_robin_select(grouped, max_rows=max_rows, seed=seed)
    rewritten = [contract_blind_row(row, index) for index, row in enumerate(selected)]
    write_jsonl(HELDOUT, rewritten)
    train_reference: list[dict[str, Any]] = []
    for path in SOURCE_TRAINS:
        train_reference.extend(read_jsonl(path))
    write_jsonl(TRAIN_REFERENCE, train_reference)
    solution_check = verify_private_solution_rows(rewritten)
    leakage = public_leakage_scan(rewritten)
    contract_summary = contract_blind_summary(rewritten)
    return {
        "source_counts": dict(source_counts),
        "source_eligible_category_count": len(grouped),
        "row_count": len(rewritten),
        "heldout": rel(HELDOUT),
        "train_reference": rel(TRAIN_REFERENCE),
        "category_counts": dict(Counter(str(row.get("category") or "") for row in rewritten)),
        "source_category_counts": dict(Counter(str(row.get("private_contract_blind_transfer_v1", {}).get("source_category") or "") for row in rewritten)),
        "private_solution_failure_count": len(solution_check.get("failures", [])),
        "private_solution_sample_failures": solution_check.get("failures", [])[:5],
        "public_data_leakage_hit_count": leakage["hit_count"],
        "public_data_leakage_sample_hits": leakage["sample_hits"],
        **contract_summary,
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def private_row_allowed(row: dict[str, Any]) -> bool:
    if row.get("public_benchmark") is True:
        return False
    if row.get("public_tests_included") or row.get("public_benchmark_solutions_included"):
        return False
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    if not contract.get("policy") or not contract.get("return_shape") or not contract.get("type_family"):
        return False
    return bool(row.get("task_id") and row.get("tests") and row.get("solution_body"))


def round_robin_select(grouped: dict[str, list[dict[str, Any]]], *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    keys = sorted(grouped)
    if not keys:
        return []
    rotated = keys[seed % len(keys) :] + keys[: seed % len(keys)]
    positions = {key: 0 for key in rotated}
    selected: list[dict[str, Any]] = []
    while len(selected) < max_rows:
        progressed = False
        for key in rotated:
            rows = grouped[key]
            pos = positions[key]
            if pos >= len(rows):
                continue
            selected.append(rows[pos])
            positions[key] = pos + 1
            progressed = True
            if len(selected) >= max_rows:
                break
        if not progressed:
            break
    return selected


def contract_blind_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    contract = object_field(out, "decoder_contract")
    original_entry = str(out.get("entry_point") or "")
    original_category = str(out.get("category") or "")
    original_family = str(out.get("broad_private_family_v1") or "")
    original_semantic = str(contract.get("semantic_family") or original_category)
    opaque = f"contract_blind_unit_{index:04d}"
    entry = f"{opaque}_entry"
    out["task_id"] = f"private_contract_blind_transfer_v1_{index:04d}"
    out["source_task_id"] = str(row.get("task_id") or "")
    out["entry_point"] = entry
    out["category"] = opaque
    out["concept_residual_label"] = opaque
    out["residual_concept"] = opaque
    out["prompt"] = OPAQUE_PROMPT
    out["tags"] = ["contract_blind_transfer_v1", "heldout", f"shape_{safe_token(contract.get('return_shape'))}"]
    out["broad_private_family_v1"] = "contract_blind_transfer_v1"
    out["benchmark_evidence_level"] = f"{row.get('benchmark_evidence_level', '')};private_contract_blind_transfer_v1_generated_only"
    out["public_benchmark"] = False
    out["public_tests_included"] = False
    out["public_benchmark_solutions_included"] = False
    if original_entry:
        out["tests"] = str(out.get("tests") or "").replace(original_entry, entry)
    contract["semantic_family"] = opaque
    contract["residual_label_hint"] = opaque
    contract["score_semantics"] = "private contract-blind transfer; semantic names withheld"
    contract["generation_plan"] = object_field(contract, "generation_plan")
    contract["generation_plan"]["policy"] = "contract_shape_and_argument_roles -> learned reusable token body"
    contract["generation_plan"]["repair_strategy"] = "infer reusable behavior from contract fingerprints, not category, prompt, tag, semantic-family, or broad-family names"
    contract["generation_plan"]["public_tests_used"] = False
    contract["generation_plan"]["public_solutions_used"] = False
    preserve_visible_interface_shape(contract, out)
    out["decoder_contract"] = contract
    out["private_contract_blind_transfer_v1"] = {
        "source_task_id": str(row.get("task_id") or ""),
        "source_category": original_category,
        "source_semantic_family": original_semantic,
        "source_broad_private_family_v1": original_family,
        "opaque_category": opaque,
        "semantic_names_withheld": True,
        "public_benchmark_inputs_read": False,
    }
    out.pop("novel_composition_v1", None)
    return out


def preserve_visible_interface_shape(contract: dict[str, Any], row: dict[str, Any]) -> None:
    entry = str(row.get("entry_point") or "")
    visible_count = max_visible_call_arg_count(str(row.get("tests") or ""), entry)
    existing_count = int(first_number(contract.get("visible_arg_count_hint"), 0))
    if visible_count > existing_count:
        contract["visible_arg_count_hint"] = visible_count
    roles = object_field(contract, "argument_roles")
    if visible_count >= 1 and "data" not in roles:
        roles["data"] = "primary_input"
    if visible_count >= 2 and "other" not in roles:
        roles["other"] = "secondary_input"
    for index in range(max(2, len(roles)), visible_count):
        roles.setdefault(f"extra_{index - 2}", f"extra_input_{index - 2}")
    contract["argument_roles"] = roles


def max_visible_call_arg_count(tests: str, entry: str) -> int:
    if not entry:
        return 0
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return 0
    max_args = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == entry:
            max_args = max(max_args, len(node.args))
    return max_args


def execute(args: argparse.Namespace, started: float) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    ensure_sidecars()
    deadline = started + max(0.1, float(args.max_hours)) * 3600.0
    commands: list[dict[str, Any]] = []
    for name, command, env in phase_commands(args):
        if time.time() > deadline:
            return "precise_blocker", {"kind": "time_budget_exhausted", "detail": f"stopped before {name}"}, commands
        result = run_command(name, command, env=env, timeout=max(60, int(deadline - time.time())))
        commands.append(result)
        if result["returncode"] != 0:
            return "precise_blocker", {"kind": "phase_failed", "phase": name, "returncode": result["returncode"], "stderr_tail": result["stderr_tail"]}, commands
    score = read_json(SCORE, {})
    learned_gate = read_json(LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned_gate, "summary")
    learned_pass_inventory = object_field(learned_summary, "pass_inventory")
    diagnostic_adapter_pass_count = int(first_number(
        learned_summary.get("diagnostic_adapter_pass_count"),
        learned_pass_inventory.get("diagnostic_adapter_pass_count"),
        999,
    ))
    prototype_pass_count = int(first_number(
        learned_summary.get("prototype_pass_count"),
        learned_pass_inventory.get("prototype_pass_count"),
        999,
    ))
    if (
        score.get("trigger_state") in {"GREEN", "YELLOW"}
        and learned_gate.get("trigger_state") in {"GREEN", "YELLOW"}
        and first_number(score_summary.get("pass_rate"), 0.0) >= 0.70
        and first_number(learned_summary.get("strict_novel_learned_only_pass_rate"), 0.0) >= 0.70
        and prototype_pass_count == 0
        and diagnostic_adapter_pass_count == 0
    ):
        return "private_contract_blind_transfer_ready", {}, commands
    return "precise_blocker", {
        "kind": "contract_blind_transfer_below_floor",
        "score_trigger_state": score.get("trigger_state"),
        "learned_gate_trigger_state": learned_gate.get("trigger_state"),
        "score_summary": score_summary,
        "learned_summary": learned_summary,
    }, commands


def phase_commands(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, str]]]:
    eval_limit = max(1, int(args.rows))
    min_rows = max(1, int(args.min_rows))
    return [
        (
            "private_safe_sts_streams",
            [
                sys.executable,
                "scripts/private_task_sts_streams.py",
                "--tasks",
                rel(HELDOUT),
                "--out",
                rel(STS_STREAMS),
                "--report-out",
                rel(STS_STREAMS_REPORT),
                "--task-limit",
                str(eval_limit),
            ],
            {},
        ),
        (
            "fanout_sts_on",
            fanout_command(args, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, STS_STREAMS, eval_limit),
            fanout_env(sts_on=True),
        ),
        (
            "fanout_sts_off_control",
            fanout_command(args, CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, EMPTY_STS, eval_limit),
            fanout_env(sts_on=False),
        ),
        (
            "score_all_candidates",
            score_command(args, PRIVATE_CANDIDATES, CONTROL_PRIVATE_CANDIDATES, SCORE, SCORE_MD, min_rows),
            {},
        ),
        (
            "learned_distillation_gate",
            [
                sys.executable,
                "scripts/broad_private_learned_distillation_gate_v1.py",
                "--heldout",
                rel(HELDOUT),
                "--candidates",
                rel(PRIVATE_CANDIDATES),
                "--control-candidates",
                rel(CONTROL_PRIVATE_CANDIDATES),
                "--score",
                rel(SCORE),
                "--private-train",
                rel(TRAIN_REFERENCE),
                "--learned-only-candidates-out",
                rel(LEARNED_CANDIDATES),
                "--learned-only-score-out",
                rel(LEARNED_SCORE),
                "--learned-only-score-markdown-out",
                rel(LEARNED_SCORE_MD),
                "--strict-novel-learned-only-candidates-out",
                rel(STRICT_LEARNED_CANDIDATES),
                "--strict-novel-learned-only-score-out",
                rel(STRICT_LEARNED_SCORE),
                "--strict-novel-learned-only-score-markdown-out",
                rel(STRICT_LEARNED_SCORE_MD),
                "--timeout-seconds",
                str(max(1, int(args.score_timeout_seconds))),
                "--min-heldout-rows",
                str(min_rows),
                "--out",
                rel(LEARNED_GATE),
                "--markdown-out",
                rel(LEARNED_GATE_MD),
            ],
            {},
        ),
    ]


def fanout_command(
    args: argparse.Namespace,
    private_out: Path,
    public_out: Path,
    report_out: Path,
    sts_streams: Path,
    eval_limit: int,
) -> list[str]:
    return [
        rel(RELEASE),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(HELDOUT),
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


def score_command(args: argparse.Namespace, candidates: Path, control: Path, out: Path, markdown: Path, min_rows: int) -> list[str]:
    return [
        sys.executable,
        "scripts/broad_private_generalization_score_v1.py",
        "--heldout",
        rel(HELDOUT),
        "--candidates",
        rel(candidates),
        "--control-candidates",
        rel(control),
        "--timeout-seconds",
        str(max(1, int(args.score_timeout_seconds))),
        "--min-heldout-rows",
        str(min_rows),
        "--out",
        rel(out),
        "--markdown-out",
        rel(markdown),
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


def build_report(
    args: argparse.Namespace,
    started: float,
    preflight: dict[str, Any],
    build: dict[str, Any],
    commands: list[dict[str, Any]],
    completion: str,
    blocker: dict[str, Any],
) -> dict[str, Any]:
    score = read_json(SCORE, {})
    learned = read_json(LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    learned_summary = object_field(learned, "summary")
    learned_pass_inventory = object_field(learned_summary, "pass_inventory")
    diagnostic_adapter_pass_count = int(first_number(
        learned_summary.get("diagnostic_adapter_pass_count"),
        learned_pass_inventory.get("diagnostic_adapter_pass_count"),
        0,
    ))
    prototype_pass_count = int(first_number(
        learned_summary.get("prototype_pass_count"),
        learned_pass_inventory.get("prototype_pass_count"),
        0,
    ))
    candidates = read_jsonl(PRIVATE_CANDIDATES)
    inventory = candidate_inventory(candidates)
    hard_gates = [
        gate("preflight_ready", preflight["ready"], preflight["blockers"]),
        gate("private_solution_failures_zero", build["private_solution_failure_count"] == 0, build["private_solution_sample_failures"]),
        gate("public_data_leakage_zero", build["public_data_leakage_hit_count"] == 0, build["public_data_leakage_sample_hits"]),
        gate("semantic_names_withheld", build["semantic_names_withheld_count"] == build["row_count"], build),
        gate("public_calibration_locked", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("forbidden_post_v4_public_artifacts_absent", not [path for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()], []),
    ]
    quality_gates = [
        gate("score_pass_rate_ge_070", first_number(score_summary.get("pass_rate"), 0.0) >= 0.70 if args.execute else completion == "dry_run_ready", score_summary.get("pass_rate")),
        gate("strict_novel_learned_only_pass_rate_ge_070", first_number(learned_summary.get("strict_novel_learned_only_pass_rate"), 0.0) >= 0.70 if args.execute else completion == "dry_run_ready", learned_summary.get("strict_novel_learned_only_pass_rate")),
        gate("prototype_pass_count_zero", prototype_pass_count == 0, prototype_pass_count),
        gate("diagnostic_adapter_pass_count_zero", diagnostic_adapter_pass_count == 0, diagnostic_adapter_pass_count),
        gate("sts_control_lower_than_learned", first_number(score_summary.get("pass_rate"), 0.0) > first_number(score_summary.get("control_pass_rate"), -1.0) if args.execute else completion == "dry_run_ready", {"pass_rate": score_summary.get("pass_rate"), "control_pass_rate": score_summary.get("control_pass_rate")}),
    ]
    hard_failed = [row for row in hard_gates if not row["passed"]]
    quality_failed = [row for row in quality_gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else ("GREEN" if args.execute and not quality_failed and completion == "private_contract_blind_transfer_ready" else "YELLOW")
    return {
        "policy": "project_theseus_private_contract_blind_transfer_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "completion": completion,
        "blocker": blocker,
        "inputs": {
            "execute": bool(args.execute),
            "rows": int(args.rows),
            "min_rows": int(args.min_rows),
            "seed": int(args.seed),
            "public_calibration_allowed": False,
        },
        "summary": {
            "completion_evidence_status": completion,
            "row_count": build["row_count"],
            "source_eligible_category_count": build["source_eligible_category_count"],
            "pass_count": score_summary.get("pass_count"),
            "pass_rate": score_summary.get("pass_rate"),
            "control_pass_count": score_summary.get("control_pass_count"),
            "control_pass_rate": score_summary.get("control_pass_rate"),
            "strict_novel_learned_only_pass_count": learned_summary.get("strict_novel_learned_only_pass_count"),
            "strict_novel_learned_only_pass_rate": learned_summary.get("strict_novel_learned_only_pass_rate"),
            "learned_token_pass_count": learned_summary.get("learned_token_pass_count"),
            "prototype_pass_count": prototype_pass_count,
            "diagnostic_adapter_pass_count": diagnostic_adapter_pass_count,
            "exact_train_body_memory_pass_count": learned_summary.get("exact_train_body_memory_pass_count"),
            "semantic_names_withheld_count": build["semantic_names_withheld_count"],
            "candidate_row_count": len(candidates),
            "semantic_alias_inferred_candidate_rows": inventory["semantic_alias_inferred_candidate_rows"],
            "body_memory_replay_candidate_rows": inventory["body_memory_replay_candidate_rows"],
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
            "score_semantics": "private contract-blind transfer only; not promotion evidence and not public calibration",
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "artifacts": artifacts(),
        "build": build,
        "candidate_inventory": inventory,
        "commands": command_evidence(commands),
        "gates": hard_gates + quality_gates,
        "queue": queue_rows_for_report(trigger_state, args),
        "rules": {
            "public_boundary": "No public benchmark tests, solutions, prompts, traces, or score labels are used.",
            "operator_lock": "This gate never removes reports/public_calibration_operator_lock.flag.",
            "score_semantics": "Private contract-blind evidence can strengthen readiness but cannot promote the model by itself.",
        },
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def contract_blind_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    withheld = 0
    role_counts: Counter[str] = Counter()
    return_shapes: Counter[str] = Counter()
    type_families: Counter[str] = Counter()
    for row in rows:
        marker = object_field(row, "private_contract_blind_transfer_v1")
        contract = object_field(row, "decoder_contract")
        if marker.get("semantic_names_withheld") and str(contract.get("semantic_family") or "").startswith("contract_blind_unit_"):
            withheld += 1
        return_shapes[str(contract.get("return_shape") or "unknown")] += 1
        type_families[str(contract.get("type_family") or "unknown")] += 1
        roles = object_field(contract, "argument_roles")
        for key, value in roles.items():
            role_counts[f"{key}:{value}"] += 1
    return {
        "semantic_names_withheld_count": withheld,
        "return_shape_counts": dict(return_shapes),
        "type_family_counts": dict(type_families),
        "argument_role_counts_top20": dict(role_counts.most_common(20)),
    }


def candidate_inventory(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates)
    return {
        "candidate_rows": len(candidates),
        "learned_token_rows": sum(true(row.get("token_level_code_generation_learned")) for row in candidates),
        "semantic_alias_inferred_candidate_rows": sum("semantic_alias_inferred" in str(row.get("candidate_generation_mode") or "") for row in candidates),
        "body_memory_replay_candidate_rows": sum("body_memory_replay" in str(row.get("candidate_generation_mode") or "") for row in candidates),
        "prototype_rows": sum(true(row.get("broad_private_train_prototype_stage")) for row in candidates),
        "diagnostic_adapter_rows": sum(true(row.get("broad_private_generalization_semantic_adapter_stage")) or true(row.get("private_residual_v3_semantic_adapter_stage")) for row in candidates),
        "top_modes": dict(modes.most_common(20)),
    }


def queue_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return report.get("queue", queue_rows_for_report(str(report.get("trigger_state") or ""), args))


def queue_rows_for_report(trigger_state: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if trigger_state == "GREEN":
        return []
    return [
        {
            "policy": "project_theseus_private_contract_blind_transfer_queue_item_v1",
            "queue": "private_contract_blind_transfer_v1",
            "kind": "run_private_contract_blind_transfer_v1",
            "title": "Run private contract-blind transfer to test learned decoder behavior without semantic names.",
            "priority": 55,
            "command": [
                sys.executable,
                "scripts/private_contract_blind_transfer_v1.py",
                "--execute",
                "--rows",
                str(max(1, int(args.rows))),
                "--min-rows",
                str(max(1, int(args.min_rows))),
                "--seed",
                str(int(args.seed)),
                "--max-hours",
                str(float(args.max_hours)),
            ],
            "safe_to_execute_without_operator_public_approval": True,
            "requires_operator_public_unlock": False,
            "public_calibration_allowed": False,
            "status": "pending",
        }
    ]


def ensure_sidecars() -> None:
    for path in [EMPTY_PUBLIC, EMPTY_STS, PUBLIC_CANDIDATES, CONTROL_PUBLIC_CANDIDATES]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")


def run_command(name: str, command: list[str], *, env: dict[str, str] | None = None, timeout: int = 3600) -> dict[str, Any]:
    started = time.time()
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=proc_env, timeout=max(60, timeout))
        return {
            "name": name,
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "timeout",
        }


def artifacts() -> dict[str, str]:
    return {
        "heldout": rel(HELDOUT),
        "train_reference": rel(TRAIN_REFERENCE),
        "sts_streams": rel(STS_STREAMS),
        "private_candidates": rel(PRIVATE_CANDIDATES),
        "control_private_candidates": rel(CONTROL_PRIVATE_CANDIDATES),
        "score": rel(SCORE),
        "learned_gate": rel(LEARNED_GATE),
    }


def command_evidence(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": row.get("name"), "returncode": row.get("returncode"), "elapsed_seconds": row.get("elapsed_seconds")} for row in commands]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Contract-Blind Transfer v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Completion: `{report.get('completion')}`",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Pass rate: `{summary.get('pass_rate')}`",
        f"- Strict novel learned-only pass rate: `{summary.get('strict_novel_learned_only_pass_rate')}`",
        f"- Control pass rate: `{summary.get('control_pass_rate')}`",
        f"- Prototype passes: `{summary.get('prototype_pass_count')}`",
        f"- Diagnostic adapter passes: `{summary.get('diagnostic_adapter_pass_count')}`",
        f"- Semantic names withheld: `{summary.get('semantic_names_withheld_count')}`",
        f"- Public calibration allowed: `false`",
        "",
        "This is private-only contract-blind transfer evidence, not promotion evidence and not public calibration.",
    ]
    return "\n".join(lines) + "\n"


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    bad_tokens = ["humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench"]
    for row in rows:
        blob = json.dumps(row, sort_keys=True).lower()
        found = [token for token in bad_tokens if token in blob]
        if found:
            hits.append({"task_id": row.get("task_id"), "tokens": found})
    return {"hit_count": len(hits), "sample_hits": hits[:8]}


def battery_state() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"platform": sys.platform, "on_battery": False, "source": "non_macos_default"}
    try:
        proc = subprocess.run(["pmset", "-g", "batt"], text=True, capture_output=True, timeout=10)
        text = proc.stdout.lower()
        return {"platform": sys.platform, "on_battery": "battery power" in text, "raw": proc.stdout.strip()[:500]}
    except Exception as exc:
        return {"platform": sys.platform, "on_battery": False, "error": str(exc)}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"})


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def safe_token(value: Any) -> str:
    text = str(value or "unknown").lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return token or "unknown"


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
