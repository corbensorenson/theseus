#!/usr/bin/env python3
"""Private semantic-alias stress gate for broad transfer.

This gate rewrites only private heldout metadata so exact semantic-family keys
are no longer available. Passing requires the private-train token decoder to
recover through reusable prompt/contract similarity rather than exact category
lookup or the old diagnostic semantic adapter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
ALIAS_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_semantic_alias_heldout_code_lm_tasks.jsonl"
EMPTY_PUBLIC = REPORTS / "code_lm_public_tasks_broad_private_semantic_alias_private_only_empty.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_semantic_alias_private_only_empty.jsonl"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_semantic_alias_heldout.jsonl"
LEARNED_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_semantic_alias_heldout_learned_only.jsonl"
CONTROL_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_semantic_alias_heldout_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_broad_private_semantic_alias_sts_off_private_only_empty.jsonl"
FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_semantic_alias_heldout_fanout.json"
CONTROL_FANOUT_REPORT = REPORTS / "code_lm_closure_rust_broad_private_semantic_alias_heldout_sts_off_fanout.json"
STS_STREAMS = REPORTS / "broad_private_semantic_alias_heldout_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "broad_private_semantic_alias_heldout_private_safe_sts_streams.json"
EMPTY_STS = REPORTS / "broad_private_semantic_alias_empty_sts_streams.jsonl"
SCORE = REPORTS / "broad_private_semantic_alias_score_v1.json"
SCORE_MD = REPORTS / "broad_private_semantic_alias_score_v1.md"
LEARNED_SCORE = REPORTS / "broad_private_semantic_alias_score_v1_learned_only.json"
LEARNED_SCORE_MD = REPORTS / "broad_private_semantic_alias_score_v1_learned_only.md"
PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--heldout", default=rel(HELDOUT))
    parser.add_argument("--alias-heldout-out", default=rel(ALIAS_HELDOUT))
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--candidates-per-task", type=int, default=2)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--min-alias-rows", type=int, default=1008)
    parser.add_argument("--floor", type=float, default=0.70)
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--out", default="reports/broad_private_semantic_alias_gate_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_semantic_alias_gate_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    alias_path = resolve(args.alias_heldout_out)
    alias_report = write_alias_heldout(resolve(args.heldout), alias_path, max_rows=max(0, int(args.task_limit)))
    ensure_private_sidecars()
    preflight = preflight_report(args, alias_report)
    commands: list[dict[str, Any]] = []

    if args.execute and preflight["ready"]:
        commands.append(run_command("build_private_safe_sts_streams", private_safe_sts_stream_command(alias_path)))
        commands.append(run_command("fanout_sts_on", fanout_command(args, alias_path, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, sts_streams=STS_STREAMS), env=fanout_env(enabled=True)))
        commands.append(run_command("fanout_sts_off_control", fanout_command(args, alias_path, CONTROL_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, sts_streams=EMPTY_STS), env=fanout_env(enabled=False)))
        candidates = read_jsonl(PRIVATE_CANDIDATES)
        learned = [row for row in candidates if learned_token_candidate(row)]
        write_jsonl(LEARNED_CANDIDATES, learned)
        commands.append(run_command("score_alias_all_candidates", score_command(args, alias_path, PRIVATE_CANDIDATES, SCORE, SCORE_MD)))
        commands.append(run_command("score_alias_learned_only", score_command(args, alias_path, LEARNED_CANDIDATES, LEARNED_SCORE, LEARNED_SCORE_MD)))

    candidates = read_jsonl(PRIVATE_CANDIDATES) if PRIVATE_CANDIDATES.exists() else []
    learned = read_jsonl(LEARNED_CANDIDATES) if LEARNED_CANDIDATES.exists() else []
    score = read_json(SCORE, {})
    learned_score = read_json(LEARNED_SCORE, {})
    score_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    learned_summary = learned_score.get("summary") if isinstance(learned_score.get("summary"), dict) else {}
    inventory = candidate_inventory(candidates, learned)
    pass_inventory = pass_inventory_summary(score, candidates)
    learned_pass_rate = numeric(learned_summary.get("pass_rate"), 0.0)
    pass_rate = numeric(score_summary.get("pass_rate"), 0.0)
    floor = float(args.floor)

    gates = [
        gate("public_calibration_operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK)),
        gate("alias_heldout_rows_ge_minimum", int(alias_report["alias_row_count"]) >= int(args.min_alias_rows), alias_report),
        gate("exact_semantic_keys_removed", int(alias_report["exact_semantic_key_reuse_count"]) == 0, alias_report["exact_semantic_key_reuse_count"]),
        gate("preflight_ready", preflight["ready"], preflight),
        gate("execute_requested", bool(args.execute), bool(args.execute)),
        gate("fanout_commands_succeeded", commands_succeeded(commands, ["build_private_safe_sts_streams", "fanout_sts_on", "fanout_sts_off_control"]), command_evidence(commands)),
        gate("score_commands_succeeded", commands_succeeded(commands, ["score_alias_all_candidates", "score_alias_learned_only"]), command_evidence(commands)),
        gate("alias_pass_rate_floor", pass_rate >= floor, {"observed": pass_rate, "minimum": floor}),
        gate("learned_only_alias_pass_rate_floor", learned_pass_rate >= floor, {"observed": learned_pass_rate, "minimum": floor}),
        gate("inferred_token_rows_present", int(inventory["semantic_alias_inferred_token_rows"]) > 0, inventory),
        gate("inferred_token_passes_present", int(pass_inventory["semantic_alias_inferred_token_pass_count"]) > 0, pass_inventory),
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
        "exact_semantic_keys_removed",
        "preflight_ready",
        "public_candidate_manifests_empty",
        "public_data_not_used",
        "external_inference_zero",
    }
    failed = [row for row in gates if not row["passed"]]
    hard_failed = [row for row in failed if row["gate"] in hard]
    if hard_failed:
        trigger_state = "RED"
    elif failed:
        trigger_state = "YELLOW"
    else:
        trigger_state = "GREEN"
    return {
        "policy": "project_theseus_broad_private_semantic_alias_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "execute": bool(args.execute),
            "heldout": rel(resolve(args.heldout)),
            "alias_heldout": rel(alias_path),
            "task_limit": int(args.task_limit),
            "seed": int(args.seed),
            "candidates_per_task": int(args.candidates_per_task),
            "floor": floor,
            "public_calibration": "locked",
        },
        "summary": {
            "alias_row_count": alias_report["alias_row_count"],
            "alias_category_count": alias_report["alias_category_count"],
            "candidate_row_count": len(candidates),
            "learned_only_candidate_row_count": len(learned),
            "pass_rate": score_summary.get("pass_rate"),
            "pass_count": score_summary.get("pass_count"),
            "heldout_task_count": score_summary.get("heldout_task_count"),
            "learned_only_pass_rate": learned_summary.get("pass_rate"),
            "learned_only_pass_count": learned_summary.get("pass_count"),
            "semantic_alias_inferred_token_rows": inventory["semantic_alias_inferred_token_rows"],
            "semantic_alias_inferred_token_pass_count": pass_inventory["semantic_alias_inferred_token_pass_count"],
            "diagnostic_adapter_pass_count": pass_inventory["diagnostic_adapter_pass_count"],
            "prototype_pass_count": pass_inventory["prototype_pass_count"],
            "hard_failed_gate_count": len(hard_failed),
            "failed_gate_count": len(failed),
            "elapsed_seconds": round(time.time() - started, 3),
            "score_semantics": "private semantic-alias stress only; not promotion evidence and not public calibration",
        },
        "alias_report": alias_report,
        "preflight": preflight,
        "candidate_inventory": inventory,
        "pass_inventory": pass_inventory,
        "gates": gates,
        "blockers": failed,
        "commands": commands,
        "artifacts": artifacts(alias_path),
        "next_actions": next_actions(trigger_state, failed, pass_rate, learned_pass_rate, floor),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def write_alias_heldout(source: Path, out: Path, *, max_rows: int) -> dict[str, Any]:
    rows = read_jsonl(source)
    if max_rows > 0:
        rows = rows[:max_rows]
    rewritten = []
    alias_map: dict[str, str] = {}
    exact_reuse = 0
    for row in rows:
        old_category = str(row.get("category") or "")
        alias = alias_map.setdefault(old_category, alias_for_category(old_category))
        if alias == old_category:
            exact_reuse += 1
        item = json.loads(json.dumps(row))
        item["task_id"] = f"{row.get('task_id')}__semantic_alias_v1"
        item["source_task_id"] = f"{row.get('source_task_id')}__semantic_alias_v1"
        item["category"] = alias
        item["concept_residual_label"] = alias
        item["residual_concept"] = alias
        item["semantic_alias_v1"] = {
            "original_category": old_category,
            "alias_category": alias,
            "exact_semantic_key_removed": True,
            "policy": "project_theseus_broad_private_semantic_alias_gate_v1",
        }
        contract = item.get("decoder_contract") if isinstance(item.get("decoder_contract"), dict) else {}
        contract = dict(contract)
        contract["semantic_family"] = alias
        contract["residual_label_hint"] = alias
        inferred_arg_count = visible_arg_count_from_tests(
            str(item.get("entry_point") or ""),
            str(item.get("tests") or ""),
        )
        if inferred_arg_count is not None:
            existing = int_or_none(contract.get("visible_arg_count_hint")) or 0
            contract["visible_arg_count_hint"] = max(existing, inferred_arg_count)
            if inferred_arg_count >= 2:
                roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
                roles = dict(roles)
                roles.setdefault("other", "secondary_input")
                contract["argument_roles"] = roles
        item["decoder_contract"] = contract
        tags = []
        for tag in item.get("tags") or []:
            tag = str(tag)
            if tag == old_category or tag == str(row.get("residual_concept") or ""):
                continue
            tags.append(tag)
        tags.extend([alias, "semantic_alias_holdout_v1"])
        item["tags"] = sorted(set(tags))
        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        provenance = dict(provenance)
        provenance.update(
            {
                "semantic_alias_original_category": old_category,
                "semantic_alias_category": alias,
                "exact_semantic_key_removed": True,
                "public_tests_used": False,
                "public_benchmark_answers_used": False,
            }
        )
        item["provenance"] = provenance
        rewritten.append(item)
    write_jsonl(out, rewritten)
    return {
        "source": rel(source),
        "alias_heldout": rel(out),
        "source_row_count": len(read_jsonl(source)),
        "alias_row_count": len(rewritten),
        "alias_category_count": len(set(row.get("category") for row in rewritten)),
        "exact_semantic_key_reuse_count": exact_reuse,
        "alias_map_sample": dict(list(alias_map.items())[:8]),
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def alias_for_category(category: str) -> str:
    base = category
    if base.startswith("bpg_"):
        base = base[4:]
    digest = hashlib.sha256(category.encode("utf-8")).hexdigest()[:8]
    token = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_")
    return f"semantic_alias_{token}_{digest}"


def visible_arg_count_from_tests(entry_point: str, tests: str) -> int | None:
    if not entry_point.strip() or not tests.strip():
        return None
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return None
    counts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == entry_point:
            counts.append(len(node.args))
    return max(counts) if counts else None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def preflight_report(args: argparse.Namespace, alias_report: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    if not PUBLIC_LOCK.exists():
        blockers.append(f"public calibration lock missing at {rel(PUBLIC_LOCK)}")
    if not release_binary().exists() and bool(args.execute):
        blockers.append(f"release binary missing at {rel(release_binary())}; run cargo build --release -p symliquid-cli")
    checkpoint = checkpoint_default(args)
    if bool(args.execute) and not checkpoint.exists():
        blockers.append(f"checkpoint missing at {rel(checkpoint)}")
    if int(alias_report.get("alias_row_count") or 0) <= 0:
        blockers.append("alias heldout is empty")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "public_lock_active": PUBLIC_LOCK.exists(),
        "release_binary": {"path": rel(release_binary()), "exists": release_binary().exists()},
        "checkpoint": {"path": rel(checkpoint), "exists": checkpoint.exists()},
    }


def fanout_command(args: argparse.Namespace, alias_path: Path, private_out: Path, public_out: Path, report_out: Path, *, sts_streams: Path) -> list[str]:
    return [
        str(release_binary()),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(alias_path),
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


def private_safe_sts_stream_command(alias_path: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/private_task_sts_streams.py",
        "--tasks",
        rel(alias_path),
        "--out",
        rel(STS_STREAMS),
        "--report-out",
        rel(STS_STREAMS_REPORT),
    ]


def score_command(args: argparse.Namespace, alias_path: Path, candidates: Path, out: Path, markdown: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/broad_private_generalization_score_v1.py",
        "--heldout",
        rel(alias_path),
        "--candidates",
        rel(candidates),
        "--control-candidates",
        rel(CONTROL_CANDIDATES),
        "--timeout-seconds",
        str(max(1, int(args.score_timeout_seconds))),
        "--min-heldout-rows",
        str(max(1, int(args.min_alias_rows))),
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


def candidate_inventory(candidates: list[dict[str, Any]], learned: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates)
    return {
        "candidate_rows": len(candidates),
        "learned_only_candidate_rows": len(learned),
        "mode_count": len(modes),
        "semantic_alias_inferred_token_rows": sum(
            "semantic_alias_inferred" in str(row.get("candidate_generation_mode") or "")
            and learned_token_candidate(row)
            for row in candidates
        ),
        "diagnostic_adapter_rows": sum(diagnostic_adapter_candidate(row) for row in candidates),
        "prototype_rows": sum(true(row.get("broad_private_train_prototype_stage")) for row in candidates),
        "top_modes": dict(modes.most_common(20)),
    }


def pass_inventory_summary(score: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        index.setdefault((str(row.get("task_id") or ""), str(row.get("candidate_generation_mode") or "")), []).append(row)
    out = {
        "passed_result_count": 0,
        "semantic_alias_inferred_token_pass_count": 0,
        "diagnostic_adapter_pass_count": 0,
        "prototype_pass_count": 0,
        "learned_token_pass_count": 0,
    }
    for result in score.get("results") if isinstance(score.get("results"), list) else []:
        if not true(result.get("passed")):
            continue
        out["passed_result_count"] += 1
        rows = index.get((str(result.get("task_id") or ""), str(result.get("pass_candidate_mode") or "")), [])
        if any(
            "semantic_alias_inferred" in str(row.get("candidate_generation_mode") or "")
            and learned_token_candidate(row)
            for row in rows
        ):
            out["semantic_alias_inferred_token_pass_count"] += 1
        if any(diagnostic_adapter_candidate(row) for row in rows):
            out["diagnostic_adapter_pass_count"] += 1
        if any(true(row.get("broad_private_train_prototype_stage")) for row in rows):
            out["prototype_pass_count"] += 1
        if any(learned_token_candidate(row) for row in rows):
            out["learned_token_pass_count"] += 1
    return out


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
        and "contract_transduced_token_decoder" not in mode
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
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-1600:],
        "stderr_tail": completed.stderr[-2400:],
    }


def ensure_private_sidecars() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    EMPTY_PUBLIC.write_text("", encoding="utf-8")
    PUBLIC_CANDIDATES.write_text("", encoding="utf-8")
    CONTROL_PUBLIC_CANDIDATES.write_text("", encoding="utf-8")
    EMPTY_STS.write_text("", encoding="utf-8")


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


def artifacts(alias_path: Path) -> dict[str, str]:
    return {
        "alias_heldout": rel(alias_path),
        "sts_streams": rel(STS_STREAMS),
        "private_candidates": rel(PRIVATE_CANDIDATES),
        "learned_candidates": rel(LEARNED_CANDIDATES),
        "control_candidates": rel(CONTROL_CANDIDATES),
        "score": rel(SCORE),
        "learned_score": rel(LEARNED_SCORE),
        "fanout_report": rel(FANOUT_REPORT),
        "control_fanout_report": rel(CONTROL_FANOUT_REPORT),
    }


def next_actions(trigger_state: str, failed: list[dict[str, Any]], pass_rate: float, learned_pass_rate: float, floor: float) -> list[str]:
    if trigger_state == "GREEN":
        return ["Semantic-alias private transfer cleared; rerun generalization governor and keep public calibration locked."]
    if failed:
        first = failed[0]["gate"]
        if first == "execute_requested":
            return ["Run with --execute to generate private alias fanout and score evidence."]
        if pass_rate < floor or learned_pass_rate < floor:
            return ["Improve inferred private-train token routing or reusable decoder bodies, then rerun this alias gate."]
        return [f"Repair first failed gate: {first}."]
    return ["Refresh alias candidates and rerun this gate."]


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
        "# Broad Private Semantic Alias Gate v1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Alias rows: `{summary.get('alias_row_count')}`",
        f"- Full pass rate: `{summary.get('pass_rate')}`",
        f"- Learned-only pass rate: `{summary.get('learned_only_pass_rate')}`",
        f"- Inferred token rows: `{summary.get('semantic_alias_inferred_token_rows')}`",
        f"- Inferred token passes: `{summary.get('semantic_alias_inferred_token_pass_count')}`",
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
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
