#!/usr/bin/env python3
"""Private novel-composition stress gate for broad transfer.

This gate creates private heldout tasks that compose two previously learned
broad-private semantic bodies. Passing requires a learned token decoder to
combine reusable private-train behaviors, not just recover one exact semantic
family or semantic alias.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_novel_composition_v1_heldout_code_lm_tasks.jsonl"
EMPTY_PUBLIC = REPORTS / "code_lm_public_tasks_broad_private_novel_composition_private_only_empty.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_novel_composition_private_only_empty.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_novel_composition_sts_off_private_only_empty.jsonl"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_novel_composition_heldout.jsonl"
COMPOSITION_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_novel_composition_heldout_composition_only.jsonl"
CONTROL_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_novel_composition_heldout_sts_off.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_novel_composition_heldout_fanout.json"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_novel_composition_heldout_sts_off_fanout.json"
STS_STREAMS = REPORTS / "broad_private_novel_composition_heldout_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "broad_private_novel_composition_heldout_private_safe_sts_streams.json"
EMPTY_STS = REPORTS / "broad_private_novel_composition_empty_sts_streams.jsonl"
SCORE = REPORTS / "broad_private_novel_composition_score_v1.json"
SCORE_MD = REPORTS / "broad_private_novel_composition_score_v1.md"
COMPOSITION_SCORE = REPORTS / "broad_private_novel_composition_score_v1_composition_only.json"
COMPOSITION_SCORE_MD = REPORTS / "broad_private_novel_composition_score_v1_composition_only.md"
PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rows", type=int, default=1008)
    parser.add_argument("--min-rows", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--candidates-per-task", type=int, default=2)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--floor", type=float, default=0.70)
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--heldout-out", default=rel(HELDOUT))
    parser.add_argument("--out", default="reports/broad_private_novel_composition_gate_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_novel_composition_gate_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    heldout_path = resolve(args.heldout_out)
    bodies = private_train_bodies()
    heldout_report = write_composition_heldout(heldout_path, bodies, rows=max(1, int(args.rows)), seed=int(args.seed))
    ensure_private_sidecars()
    preflight = preflight_report(args, heldout_report)
    commands: list[dict[str, Any]] = []

    if args.execute and preflight["ready"]:
        commands.append(run_command("build_private_safe_sts_streams", private_safe_sts_stream_command(heldout_path)))
        commands.append(run_command("fanout_sts_on", fanout_command(args, heldout_path, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, sts_streams=STS_STREAMS), env=fanout_env(enabled=True)))
        commands.append(run_command("fanout_sts_off_control", fanout_command(args, heldout_path, CONTROL_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, sts_streams=EMPTY_STS), env=fanout_env(enabled=False)))
        candidates = read_jsonl(PRIVATE_CANDIDATES)
        composition = [row for row in candidates if composition_token_candidate(row)]
        write_jsonl(COMPOSITION_CANDIDATES, composition)
        commands.append(run_command("score_all_candidates", score_command(args, heldout_path, PRIVATE_CANDIDATES, SCORE, SCORE_MD)))
        commands.append(run_command("score_composition_only", score_command(args, heldout_path, COMPOSITION_CANDIDATES, COMPOSITION_SCORE, COMPOSITION_SCORE_MD)))

    candidates = read_jsonl(PRIVATE_CANDIDATES) if PRIVATE_CANDIDATES.exists() else []
    composition_candidates = read_jsonl(COMPOSITION_CANDIDATES) if COMPOSITION_CANDIDATES.exists() else []
    score = read_json(SCORE, {})
    composition_score = read_json(COMPOSITION_SCORE, {})
    score_summary = object_field(score, "summary")
    composition_summary = object_field(composition_score, "summary")
    inventory = candidate_inventory(candidates, composition_candidates)
    pass_inventory = pass_inventory_summary(score, candidates)
    composition_pass_rate = numeric(composition_summary.get("pass_rate"), 0.0)
    pass_rate = numeric(score_summary.get("pass_rate"), 0.0)
    floor = float(args.floor)
    gates = [
        gate("public_calibration_operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("heldout_rows_ge_minimum", int(heldout_report["row_count"]) >= int(args.min_rows), heldout_report),
        gate("composition_spec_diversity_ge_6", int(heldout_report["composition_spec_count"]) >= 6, heldout_report),
        gate("private_solution_tests_pass", heldout_report["private_solution_failure_count"] == 0, heldout_report),
        gate("preflight_ready", preflight["ready"], preflight),
        gate("execute_requested", bool(args.execute), bool(args.execute)),
        gate("fanout_commands_succeeded", commands_succeeded(commands, ["build_private_safe_sts_streams", "fanout_sts_on", "fanout_sts_off_control"]), command_evidence(commands)),
        gate("score_commands_succeeded", commands_succeeded(commands, ["score_all_candidates", "score_composition_only"]), command_evidence(commands)),
        gate("pass_rate_floor", pass_rate >= floor, {"observed": pass_rate, "minimum": floor}),
        gate("composition_only_pass_rate_floor", composition_pass_rate >= floor, {"observed": composition_pass_rate, "minimum": floor}),
        gate("composition_token_rows_present", int(inventory["composition_token_rows"]) > 0, inventory),
        gate("composition_token_passes_present", int(pass_inventory["composition_token_pass_count"]) > 0, pass_inventory),
        gate("diagnostic_adapter_pass_count_zero", int(pass_inventory["diagnostic_adapter_pass_count"]) == 0, pass_inventory),
        gate("prototype_pass_count_zero", int(pass_inventory["prototype_pass_count"]) == 0, pass_inventory),
        gate("public_candidate_manifests_empty", file_empty(PUBLIC_CANDIDATES) and file_empty(CONTROL_PUBLIC_CANDIDATES) and file_empty(EMPTY_PUBLIC), {
            "public_candidates": file_size(PUBLIC_CANDIDATES),
            "control_public_candidates": file_size(CONTROL_PUBLIC_CANDIDATES),
            "public_manifest": file_size(EMPTY_PUBLIC),
        }),
        gate("public_data_not_used", true(score_summary.get("public_tests_used")) is False and true(score_summary.get("public_solutions_used")) is False, {
            "public_tests_used": score_summary.get("public_tests_used"),
            "public_solutions_used": score_summary.get("public_solutions_used"),
        }),
        gate("external_inference_zero", int(score_summary.get("external_inference_calls") or 0) == 0, score_summary.get("external_inference_calls")),
    ]
    hard = {
        "public_calibration_operator_lock_active",
        "private_solution_tests_pass",
        "preflight_ready",
        "public_candidate_manifests_empty",
        "public_data_not_used",
        "external_inference_zero",
    }
    failed = [row for row in gates if not row["passed"]]
    hard_failed = [row for row in failed if row["gate"] in hard]
    trigger_state = "RED" if hard_failed else "YELLOW" if failed else "GREEN"
    return {
        "policy": "project_theseus_broad_private_novel_composition_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "execute": bool(args.execute),
            "rows": int(args.rows),
            "min_rows": int(args.min_rows),
            "seed": int(args.seed),
            "candidates_per_task": int(args.candidates_per_task),
            "public_calibration": "locked",
        },
        "summary": {
            "row_count": heldout_report["row_count"],
            "composition_spec_count": heldout_report["composition_spec_count"],
            "candidate_row_count": len(candidates),
            "composition_candidate_row_count": len(composition_candidates),
            "pass_rate": score_summary.get("pass_rate"),
            "pass_count": score_summary.get("pass_count"),
            "composition_only_pass_rate": composition_summary.get("pass_rate"),
            "composition_only_pass_count": composition_summary.get("pass_count"),
            "composition_token_rows": inventory["composition_token_rows"],
            "composition_token_pass_count": pass_inventory["composition_token_pass_count"],
            "diagnostic_adapter_pass_count": pass_inventory["diagnostic_adapter_pass_count"],
            "prototype_pass_count": pass_inventory["prototype_pass_count"],
            "hard_failed_gate_count": len(hard_failed),
            "failed_gate_count": len(failed),
            "elapsed_seconds": round(time.time() - started, 3),
            "score_semantics": "private novel composition stress only; not promotion evidence and not public calibration",
        },
        "heldout_report": heldout_report,
        "preflight": preflight,
        "candidate_inventory": inventory,
        "pass_inventory": pass_inventory,
        "gates": gates,
        "blockers": failed,
        "commands": commands,
        "artifacts": artifacts(heldout_path),
        "next_actions": next_actions(trigger_state, failed, pass_rate, composition_pass_rate, floor),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def composition_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "parse_signed_ints_then_longest_even_run",
            "steps": ["bpg_parse_signed_ints", "bpg_longest_even_run"],
            "prompt": "Extract signed integers from text, then return the longest contiguous run of even integers.",
            "cases": [("a -2 4 7 8 10 x", 2), ("1 2 4 6 9", 3), ("none", 0)],
        },
        {
            "id": "parse_signed_ints_then_max_non_adjacent_sum",
            "steps": ["bpg_parse_signed_ints", "bpg_max_non_adjacent_sum"],
            "prompt": "Extract signed integers from text, then return the maximum non-adjacent non-negative sum.",
            "cases": [("1 2 9 4", 10), ("-5 7 6 5", 12), ("0 0 0", 0)],
        },
        {
            "id": "parse_signed_ints_then_numeric_stats_tuple",
            "steps": ["bpg_parse_signed_ints", "bpg_numeric_stats_tuple"],
            "prompt": "Extract signed integers from text, then return (min, max, count).",
            "cases": [("z -5 12 0", (-5, 12, 3)), ("none", (None, None, 0)), ("4 +8 -2", (-2, 8, 3))],
        },
        {
            "id": "parse_signed_ints_then_gcd_positive",
            "steps": ["bpg_parse_signed_ints", "bpg_gcd_positive"],
            "prompt": "Extract signed integers from text, then return the gcd of positive absolute integer values.",
            "cases": [("a -12 b 18 0 c", 6), ("5 10 25", 5), ("noise", 0)],
        },
        {
            "id": "merge_intervals_then_interval_coverage",
            "steps": ["bpg_merge_intervals", "bpg_interval_coverage"],
            "prompt": "Merge half-open intervals, then return the total covered length.",
            "cases": [([(1, 3), (2, 5), (10, 12)], 6), ([(0, 1), (1, 4)], 4), ([], 0)],
        },
        {
            "id": "stable_dedup_then_rle_encode",
            "steps": ["bpg_stable_dedup", "bpg_rle_encode"],
            "prompt": "Normalize and stable-deduplicate tokens, then run-length encode the resulting sequence.",
            "cases": [([" A ", "a", "B", "b", "b"], [("a", 1), ("b", 1)]), (["x", "Y", "x", "z"], [("x", 1), ("y", 1), ("z", 1)]), ([], [])],
        },
    ]


def write_composition_heldout(out: Path, bodies: dict[str, str], *, rows: int, seed: int) -> dict[str, Any]:
    specs = composition_specs()
    output = []
    failures = []
    for index in range(rows):
        spec = specs[(index + seed) % len(specs)]
        case_input, expected = spec["cases"][(index // len(specs)) % len(spec["cases"])]
        task_index = 2_000_000 + index
        entry = f"bpg_novel_composition_{spec['id']}_{task_index:07d}"
        body = render_solution_body(spec["steps"], bodies)
        row = {
            "task_id": f"broad_private_novel_composition_v1_{spec['id']}_{task_index:07d}",
            "source_task_id": f"broad_private_novel_composition_v1_private_{seed}_{task_index:07d}",
            "card_id": "broad_private_generalization_ladder_v1",
            "source_id": "local_generated_broad_private_novel_composition_v1",
            "split": "eval",
            "category": f"bpg_novel_composition_{spec['id']}",
            "prompt": spec["prompt"],
            "entry_point": entry,
            "solution_expr": "",
            "solution_body": body,
            "tests": f"assert {entry}({case_input!r}) == {expected!r}\n",
            "tags": ["broad_private_generalization_ladder_v1", "novel_composition_v1", *spec["steps"]],
            "broad_private_family_v1": "novel_composition",
            "targeted_private_residual_family_v3": "novel_composition",
            "residual_concept": spec["id"],
            "concept_residual_label": spec["id"],
            "novel_composition_v1": {
                "policy": "project_theseus_broad_private_novel_composition_gate_v1",
                "steps": [{"semantic_family": step, "source": "private_train_broad_private_generalization_ladder_v1"} for step in spec["steps"]],
                "public_tests_used": False,
                "public_solutions_used": False,
            },
            "decoder_contract": {
                "policy": "project_theseus_decoder_contract_v1_broad_private_generalization",
                "return_shape": return_shape(expected),
                "type_family": "novel_composition_pipeline",
                "semantic_family": f"novel_composition_{spec['id']}",
                "composition_steps": [{"semantic_family": step} for step in spec["steps"]],
                "visible_arg_count_hint": 1,
                "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
                "residual_label_hint": spec["id"],
                "full_body_required": True,
                "guardrail_only": False,
                "feedback_weight": 1.45,
                "score_semantics": "private novel composition stress only",
                "generation_plan": {
                    "policy": "signature -> composition_steps -> private_train_bodies -> composed_body",
                    "public_tests_used": False,
                    "public_solutions_used": False,
                    "repair_strategy": "compose reusable private-train token bodies instead of exact semantic lookup",
                },
            },
            "benchmark_evidence_level": "broad_private_generalization_ladder_v1_generated_only",
            "public_benchmark": False,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "public_prompts_included": False,
            "public_score_labels_included": False,
            "license_spdx": "CC0-1.0",
            "candidate_expression_eligible": False,
            "provenance": {
                "policy": "project_theseus_broad_private_novel_composition_gate_v1",
                "composition_id": spec["id"],
                "public_benchmark_answers_used": False,
                "public_tests_used": False,
                "public_prompts_used": False,
                "public_score_labels_used": False,
            },
        }
        ok, error = verify_row(row)
        if not ok:
            failures.append({"task_id": row["task_id"], "error": error})
        output.append(row)
    write_jsonl(out, output)
    counts = Counter(row["category"] for row in output)
    return {
        "heldout": rel(out),
        "row_count": len(output),
        "composition_spec_count": len(counts),
        "category_counts": dict(sorted(counts.items())),
        "private_solution_failure_count": len(failures),
        "private_solution_failures": failures[:12],
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def render_solution_body(steps: list[str], bodies: dict[str, str]) -> str:
    lines = []
    for index, step in enumerate(steps):
        if index > 0:
            lines.append(f"data = _theseus_value_{index - 1}")
        last = index + 1 == len(steps)
        step_lines = nonempty_body_lines(bodies[step])
        if last:
            lines.extend(step_lines)
        else:
            lines.extend(render_intermediate_step(step_lines, index))
    return "\n".join(lines)


def render_intermediate_step(step_lines: list[str], index: int) -> list[str]:
    if (
        len(step_lines) >= 3
        and step_lines[0].startswith("if ")
        and step_lines[0].endswith(":")
        and step_lines[1].startswith("    return ")
    ):
        out = [
            step_lines[0],
            f"    _theseus_value_{index} = {step_lines[1].strip().removeprefix('return ').strip()}",
            "else:",
        ]
        for line in step_lines[2:]:
            if line.startswith("return "):
                out.append(f"    _theseus_value_{index} = {line.removeprefix('return ').strip()}")
            else:
                out.append(f"    {line}")
        return out
    out = []
    for line in step_lines:
        if line.startswith("return "):
            out.append(f"_theseus_value_{index} = {line.removeprefix('return ').strip()}")
        else:
            out.append(line)
    return out


def nonempty_body_lines(body: str) -> list[str]:
    return [line.rstrip() for line in body.splitlines() if line.strip()]


def private_train_bodies() -> dict[str, str]:
    bodies = {}
    for row in read_jsonl(TRAIN):
        category = str(row.get("category") or "")
        if category and category not in bodies:
            bodies[category] = str(row.get("solution_body") or "")
    missing = sorted({step for spec in composition_specs() for step in spec["steps"]} - set(bodies))
    if missing:
        raise SystemExit(f"missing private train bodies for {missing}")
    return bodies


def verify_row(row: dict[str, Any]) -> tuple[bool, str]:
    namespace: dict[str, Any] = {}
    code = f"def {row['entry_point']}(*args):\n    data = args[0] if len(args) > 0 else None\n"
    for line in str(row["solution_body"]).splitlines():
        code += f"    {line}\n"
    try:
        exec(code, namespace, namespace)
        exec(str(row["tests"]), namespace, namespace)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def return_shape(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, str):
        return "str"
    return "unknown"


def preflight_report(args: argparse.Namespace, heldout_report: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    if not PUBLIC_LOCK.exists():
        blockers.append(f"public calibration lock missing at {rel(PUBLIC_LOCK)}")
    if bool(args.execute) and not release_binary().exists():
        blockers.append(f"release binary missing at {rel(release_binary())}; run cargo build --release -p symliquid-cli")
    checkpoint = checkpoint_default(args)
    if bool(args.execute) and not checkpoint.exists():
        blockers.append(f"checkpoint missing at {rel(checkpoint)}")
    if int(heldout_report.get("row_count") or 0) <= 0:
        blockers.append("composition heldout is empty")
    if int(heldout_report.get("private_solution_failure_count") or 0) > 0:
        blockers.append("composition heldout private solution tests failed")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "public_lock_active": PUBLIC_LOCK.exists(),
        "release_binary": {"path": rel(release_binary()), "exists": release_binary().exists()},
        "checkpoint": {"path": rel(checkpoint), "exists": checkpoint.exists()},
    }


def fanout_command(args: argparse.Namespace, heldout_path: Path, private_out: Path, public_out: Path, report_out: Path, *, sts_streams: Path) -> list[str]:
    return [
        str(release_binary()),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(heldout_path),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC),
        "--checkpoint-in",
        rel(checkpoint_default(args)),
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
        "--sts-streams",
        rel(sts_streams),
    ]


def private_safe_sts_stream_command(heldout_path: Path) -> list[str]:
    return [sys.executable, "scripts/private_task_sts_streams.py", "--tasks", rel(heldout_path), "--out", rel(STS_STREAMS), "--report-out", rel(STS_STREAMS_REPORT)]


def score_command(args: argparse.Namespace, heldout_path: Path, candidates: Path, out: Path, markdown: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/broad_private_generalization_score_v1.py",
        "--heldout",
        rel(heldout_path),
        "--candidates",
        rel(candidates),
        "--control-candidates",
        rel(CONTROL_CANDIDATES),
        "--timeout-seconds",
        str(max(1, int(args.score_timeout_seconds))),
        "--min-heldout-rows",
        str(max(1, int(args.min_rows))),
        "--out",
        rel(out),
        "--markdown-out",
        rel(markdown),
    ]


def fanout_env(*, enabled: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
    if not enabled:
        env["THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY"] = "1"
    return env


def candidate_inventory(candidates: list[dict[str, Any]], composition: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates)
    return {
        "candidate_rows": len(candidates),
        "composition_candidate_rows": len(composition),
        "composition_token_rows": sum(composition_token_candidate(row) for row in candidates),
        "diagnostic_adapter_rows": sum(diagnostic_adapter_candidate(row) for row in candidates),
        "prototype_rows": sum(true(row.get("broad_private_train_prototype_stage")) for row in candidates),
        "top_modes": dict(modes.most_common(20)),
    }


def pass_inventory_summary(score: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        index.setdefault((str(row.get("task_id") or ""), str(row.get("candidate_generation_mode") or "")), []).append(row)
    out = {"passed_result_count": 0, "composition_token_pass_count": 0, "diagnostic_adapter_pass_count": 0, "prototype_pass_count": 0}
    for result in score.get("results") if isinstance(score.get("results"), list) else []:
        if not true(result.get("passed")):
            continue
        out["passed_result_count"] += 1
        rows = index.get((str(result.get("task_id") or ""), str(result.get("pass_candidate_mode") or "")), [])
        if any(composition_token_candidate(row) for row in rows):
            out["composition_token_pass_count"] += 1
        if any(diagnostic_adapter_candidate(row) for row in rows):
            out["diagnostic_adapter_pass_count"] += 1
        if any(true(row.get("broad_private_train_prototype_stage")) for row in rows):
            out["prototype_pass_count"] += 1
    return out


def composition_token_candidate(row: dict[str, Any]) -> bool:
    mode = str(row.get("candidate_generation_mode") or "").lower()
    return "novel_composition_v1" in mode and learned_token_candidate(row)


def learned_token_candidate(row: dict[str, Any]) -> bool:
    mode = str(row.get("candidate_generation_mode") or "").lower()
    return (
        true(row.get("token_level_code_generation_learned"))
        and true(row.get("candidate_syntax_lint_passed"))
        and row.get("deterministic_guardrail_passed") is not False
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and not true(row.get("broad_private_train_prototype_stage"))
        and not true(row.get("broad_private_generalization_semantic_adapter_stage"))
        and not true(row.get("private_residual_v3_semantic_adapter_stage"))
        and not true(row.get("contract_transduced_stage"))
        and not true(row.get("same_seed_non_sts_comparator"))
        and "prototype" not in mode
        and "semantic_adapter" not in mode
    )


def diagnostic_adapter_candidate(row: dict[str, Any]) -> bool:
    return true(row.get("broad_private_generalization_semantic_adapter_stage")) or true(row.get("private_residual_v3_semantic_adapter_stage"))


def commands_succeeded(commands: list[dict[str, Any]], names: list[str]) -> bool:
    by_name = {row.get("name"): row for row in commands}
    return bool(names) and all(by_name.get(name, {}).get("returncode") == 0 for name in names)


def command_evidence(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": row.get("name"), "returncode": row.get("returncode"), "elapsed_seconds": row.get("elapsed_seconds")} for row in commands]


def run_command(name: str, command: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    actual_env = os.environ.copy()
    if env:
        actual_env.update(env)
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, env=actual_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"name": name, "command": command, "returncode": completed.returncode, "elapsed_seconds": round(time.time() - started, 3), "stdout_tail": completed.stdout[-1600:], "stderr_tail": completed.stderr[-2400:]}


def ensure_private_sidecars() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for path in [EMPTY_PUBLIC, PUBLIC_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, EMPTY_STS]:
        path.write_text("", encoding="utf-8")


def checkpoint_default(args: argparse.Namespace) -> Path:
    if str(args.checkpoint_in or "").strip():
        return resolve(args.checkpoint_in)
    trained = REPORTS / "student_code_lm_checkpoint_broad_private_generalization_ladder_v1.json"
    if trained.exists():
        return trained
    preferred = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
    if preferred.exists():
        return preferred
    candidates = sorted(REPORTS.glob("student_code_lm_checkpoint*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else preferred


def release_binary() -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return ROOT / "target" / "release" / name


def artifacts(heldout_path: Path) -> dict[str, str]:
    return {
        "heldout": rel(heldout_path),
        "sts_streams": rel(STS_STREAMS),
        "private_candidates": rel(PRIVATE_CANDIDATES),
        "composition_candidates": rel(COMPOSITION_CANDIDATES),
        "control_candidates": rel(CONTROL_CANDIDATES),
        "score": rel(SCORE),
        "composition_score": rel(COMPOSITION_SCORE),
        "fanout_report": rel(FANOUT_REPORT),
        "control_fanout_report": rel(CONTROL_FANOUT_REPORT),
    }


def next_actions(trigger_state: str, failed: list[dict[str, Any]], pass_rate: float, composition_pass_rate: float, floor: float) -> list[str]:
    if trigger_state == "GREEN":
        return ["Novel-composition private transfer cleared; integrate this gate into the generalization governor and keep public calibration locked."]
    if failed:
        if failed[0]["gate"] == "execute_requested":
            return ["Run with --execute to generate private composition fanout and score evidence."]
        if pass_rate < floor or composition_pass_rate < floor:
            return ["Improve reusable composition token decoding, then rerun this private composition gate."]
        return [f"Repair first failed gate: {failed[0]['gate']}."]
    return ["Refresh composition candidates and rerun this gate."]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "green"}
    return bool(value)


def numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else -1


def file_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size == 0


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


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
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Novel Composition Gate v1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Full pass rate: `{summary.get('pass_rate')}`",
        f"- Composition-only pass rate: `{summary.get('composition_only_pass_rate')}`",
        f"- Composition token rows: `{summary.get('composition_token_rows')}`",
        f"- Composition token passes: `{summary.get('composition_token_pass_count')}`",
        f"- Diagnostic adapter passes: `{summary.get('diagnostic_adapter_pass_count')}`",
        f"- Prototype passes: `{summary.get('prototype_pass_count')}`",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    if not blockers:
        lines.append("- None.")
    else:
        for row in blockers:
            lines.append(f"- `{row.get('gate')}` evidence `{row.get('evidence')}`")
    lines.append("")
    lines.append("## Next Actions")
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
