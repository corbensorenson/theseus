"""Private execution-shape decoder ablation gate.

This gate compares decoder families on the same private held-out
execution-shaped tasks. It is deliberately private-first: public benchmark
data is not read for training or scoring, and public cards remain calibration
surfaces only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from broad_transfer_residual_decoder_ablation import private_semantic_tests_for_category  # noqa: E402


def training_data_root() -> Path:
    configured = os.environ.get("THESEUS_TRAINING_DATA_ROOT", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/training_data")
    return ROOT / "data" / "training_data"


def training_data_path(*parts: str) -> Path:
    return training_data_root().joinpath(*parts)


def release_binary() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return ROOT / "target" / "release" / f"symliquid-cli{suffix}"


PRIVATE_SOURCE = training_data_path(
    "high_transfer", "private_train", "execution_shaped_programs_residual_code_lm_tasks.jsonl"
)
DEFAULT_CURRICULUM = ROOT / "data/private_code_curriculum/execution_shape_private_ablation_seed14.jsonl"
DEFAULT_PUBLIC_VISIBLE = ROOT / "reports/execution_shape_private_ablation_visible_manifest.jsonl"
DEFAULT_CANDIDATES = ROOT / "reports/execution_shape_private_ablation_candidates.jsonl"
DEFAULT_PUBLIC_CANDIDATES = ROOT / "reports/execution_shape_private_ablation_public_candidates.jsonl"
DEFAULT_CHECKPOINT = ROOT / "reports/execution_shape_private_ablation_checkpoint.json"
DEFAULT_RUST_REPORT = ROOT / "reports/execution_shape_private_ablation_rust.json"
DEFAULT_OUT = ROOT / "reports/execution_shape_private_ablation.json"
DEFAULT_MD = ROOT / "reports/execution_shape_private_ablation.md"
DECODER_SOURCES = [
    ROOT / "crates/symliquid-cli/src/code_lm_closure.rs",
    *(ROOT / "crates/symliquid-cli/src/code_lm_closure").glob("part_*.rs"),
]
DECODER_FINGERPRINT_MARKERS = (
    "semantic_decoder_v2",
    "execution_shape_skeleton",
    "edge_exec_repair",
    "typed_edge_exec_receiver",
    "decoder_contract",
    "contract_guided_skeleton",
    "local_adapter_edge_skeleton",
    "sts_causal_skeleton",
    "candidate_floor",
    "body_token_allowed",
    "syntax_constrained_body",
    "invalid_inline_block_header_body",
    "callable_keyword_argument",
    "archive_context_manager",
    "invalid_overcomposed_generated_line",
)
PUBLIC_GATE_MIN_STUDENT_TOKEN_PASS_RATE = 0.70
DIAGNOSTIC_MIN_SKELETON_PASS_RATE = 0.70

TEMPLATE_MODE_TOKENS = (
    "causal_contract_skeleton_decoder",
    "contract_guided_skeleton_decoder",
    "execution_shape_skeleton_decoder",
    "local_adapter_edge_skeleton_decoder",
    "sts_causal_skeleton_decoder",
    "semantic_plan_v2",
    "edge_exec_repair",
    "private_body_prototype",
    "seeded_body_ngram_token_decoder",
    "sparse_state_sequence_seeded_decoder",
    "native_sts_stream_expression",
    "frequency_baseline",
)

LEARNED_TOKEN_MODE_TOKENS = (
    "contract_guided_token_decoder",
    "full_body_token_beam",
    "greedy_body_token_decoder",
    "private_body_ngram_token_decoder",
    "sparse_state_sequence_decoder",
    "symliquid_recurrent_state_decoder",
)

FAMILIES = {
    "learned_token_decoder_v1": lambda row: learned_token_candidate_row(row),
    "semantic_plan_v2": lambda mode: "semantic_plan_v2" in mode and "edge_exec_repair" not in mode,
    "edge_exec_repair_v1": lambda mode: "edge_exec_repair" in mode,
    "execution_shape_skeleton_decoder_private_v1": lambda mode: "execution_shape_skeleton_decoder" in mode,
}


def template_like_mode(mode: str) -> bool:
    lowered = str(mode or "").lower()
    return any(token in lowered for token in TEMPLATE_MODE_TOKENS)


def learned_token_mode(mode: str) -> bool:
    lowered = str(mode or "").lower()
    return any(token in lowered for token in LEARNED_TOKEN_MODE_TOKENS) and not template_like_mode(lowered)


def learned_token_candidate_row(row: dict[str, Any]) -> bool:
    """Prefer Rust's row-level provenance over brittle mode-name inference."""
    mode = str(row.get("candidate_generation_mode") or "")
    if "token_level_code_generation_learned" in row:
        return (
            bool(row.get("token_level_code_generation_learned"))
            and not template_like_mode(mode)
            and not bool(row.get("same_seed_non_sts_comparator"))
            and not bool(row.get("template_like_candidate"))
            and not bool(row.get("contract_transduced_stage"))
            and not bool(row.get("expression_memory_fallback"))
            and not bool(row.get("sts_candidate_expression_used"))
        )
    return learned_token_mode(mode)


def run_process_tree(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process tree kill did not complete within 5s"
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout or "",
            stderr=((stderr or "") + f"\nTimed out after {timeout_seconds}s; killed process tree.").strip(),
        )


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-source", default=str(PRIVATE_SOURCE))
    parser.add_argument("--curriculum-out", default=str(DEFAULT_CURRICULUM.relative_to(ROOT)))
    parser.add_argument("--public-manifest-out", default=str(DEFAULT_PUBLIC_VISIBLE.relative_to(ROOT)))
    parser.add_argument("--candidate-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--public-candidate-out", default=str(DEFAULT_PUBLIC_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--rust-report-out", default=str(DEFAULT_RUST_REPORT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--train-rows", type=int, default=320)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hv-dim", type=int, default=512)
    parser.add_argument("--max-vocab", type=int, default=640)
    parser.add_argument("--candidates-per-task", type=int, default=12)
    parser.add_argument("--max-work-steps", type=int, default=450000)
    parser.add_argument("--rust-timeout-seconds", type=int, default=7200)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument(
        "--category-filter",
        action="append",
        default=[],
        help="Private execution-shape category to include. May be repeated for targeted gates.",
    )
    parser.add_argument(
        "--allow-diagnostic-templates",
        action="store_true",
        help="Explicitly emit template/skeleton candidates for diagnostics only. Student gates remain token-only.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    source_rows = read_jsonl(resolve(args.private_source))
    curriculum_rows = build_curriculum(
        source_rows,
        seed=args.seed,
        max_train=max(24, args.train_rows),
        max_eval=max(8, args.eval_rows),
        category_filters=set(args.category_filter or []),
    )
    public_visible_rows = build_public_visible_manifest()
    write_jsonl(resolve(args.curriculum_out), curriculum_rows)
    write_jsonl(resolve(args.public_manifest_out), public_visible_rows)

    rust_result: dict[str, Any] = {"skipped": bool(args.skip_rust)}
    if not args.skip_rust:
        rust_result = run_rust(args)

    candidates = read_jsonl(resolve(args.candidate_out))
    eval_rows = [row for row in curriculum_rows if row.get("split") == "eval"]
    ablation = evaluate_families(eval_rows, candidates, seed=args.seed)
    verifier_summary = decoder_contract_verifier_summary(candidates)
    gates = build_gates(
        curriculum_rows,
        public_visible_rows,
        candidates,
        ablation,
        rust_result,
        category_filters=set(args.category_filter or []),
    )
    trigger = "GREEN" if all(row["passed"] for row in gates if row.get("severity") == "hard") else "YELLOW"
    report = {
        "policy": "project_theseus_execution_shape_private_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "purpose": "Compare learned token decoders against diagnostic planner/skeleton families on the same private held-out execution-shaped tasks. Only learned token decoder success may unlock public calibration.",
        "inputs": {
            "private_source": rel(resolve(args.private_source)),
            "seed": args.seed,
            "category_filter": sorted(args.category_filter or []),
            "public_data_rule": "not_used_for_private_ablation_public_benchmarks_remain_calibration_only",
        },
        "outputs": {
            "curriculum": rel(resolve(args.curriculum_out)),
            "public_visible_manifest": rel(resolve(args.public_manifest_out)),
            "private_candidates": rel(resolve(args.candidate_out)),
            "rust_report": rel(resolve(args.rust_report_out)),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            **ablation["summary"],
            **verifier_summary,
            "diagnostic_templates_enabled": bool(args.allow_diagnostic_templates),
            "source_private_rows": len(source_rows),
            "curriculum_rows": len(curriculum_rows),
            "train_rows": sum(1 for row in curriculum_rows if row.get("split") == "train"),
            "eval_rows": len(eval_rows),
            "candidate_rows": len(candidates),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_policy": "hash_lines_matching_decoder_v2_execution_shape_markers",
        },
        "category_distribution": dict(Counter(str(row.get("category") or "") for row in eval_rows)),
        "family_results": ablation["family_results"],
        "decoder_contract_verifier_v1": verifier_summary,
        "dominant_private_residuals": ablation["dominant_private_residuals"],
        "sample_residuals": ablation["sample_residuals"],
        "rust_result": rust_result,
        "gates": gates,
        "next_actions": next_actions(ablation),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if trigger in {"GREEN", "YELLOW"} else 2


def build_curriculum(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_train: int,
    max_eval: int,
    category_filters: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    execution_rows: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category") or "")
        if (
            not category.startswith("private_exec_")
            or row.get("public_benchmark") is not False
            or (category_filters and category not in category_filters)
        ):
            continue
        item = normalize_ablation_row(row)
        materialize_private_semantic_tests(item)
        if str(item.get("tests") or "").strip():
            execution_rows.append(item)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in execution_rows:
        by_category[str(row.get("category") or "")].append(row)
    for items in by_category.values():
        rng.shuffle(items)
    categories = sorted(by_category)
    eval_rows: list[dict[str, Any]] = []
    per_category_target = max(1, max_eval // max(1, len(categories)))
    for category in categories:
        take = min(per_category_target, max(1, len(by_category[category]) // 5), len(by_category[category]))
        eval_rows.extend(by_category[category][:take])
    leftovers = [row for category in categories for row in by_category[category][per_category_target:]]
    rng.shuffle(leftovers)
    while len(eval_rows) < max_eval and leftovers:
        eval_rows.append(leftovers.pop())
    eval_ids = {str(row["task_id"]) for row in eval_rows[:max_eval]}
    train_pool = [row for row in execution_rows if str(row["task_id"]) not in eval_ids]
    rng.shuffle(train_pool)
    train_rows = train_pool[:max_train]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(train_rows):
        item = dict(row)
        item["split"] = "train"
        item["task_id"] = f"ablation_train_{item['task_id']}"
        item["benchmark_evidence_level"] = "private_execution_shape_ablation_train_only"
        item["ablation_role"] = "train"
        out.append(item)
    for idx, row in enumerate(eval_rows[:max_eval]):
        item = dict(row)
        item["split"] = "eval"
        item["task_id"] = f"ablation_eval_{item['task_id']}"
        item["benchmark_evidence_level"] = "private_execution_shape_ablation_eval_only"
        item["ablation_role"] = "heldout_eval"
        item["private_eval_solution_used_for_generation"] = False
        out.append(item)
    return out


def decoder_relevant_source_fingerprint() -> str:
    """Fingerprint decoder-relevant code without staling on unrelated Rust edits."""

    relevant_chunks: list[str] = []
    for source in sorted(DECODER_SOURCES):
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        relevant = "\n".join(
            line
            for line in text.splitlines()
            if any(marker in line for marker in DECODER_FINGERPRINT_MARKERS)
        )
        if relevant:
            relevant_chunks.append(f"{source.relative_to(ROOT).as_posix()}\n{relevant}")
    if not relevant_chunks:
        return ""
    relevant = "\n\n".join(relevant_chunks)
    return hashlib.sha256(relevant.encode("utf-8")).hexdigest()[:16]


def normalize_ablation_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["decoder_contract"] = execution_shape_contract(item)
    item["public_benchmark"] = False
    item["public_benchmark_solutions_included"] = False
    item["public_tests_included"] = False
    item["license_spdx"] = item.get("license_spdx") or "CC0-1.0"
    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    item["provenance"] = {
        **provenance,
        "ablation_policy": "project_theseus_execution_shape_private_ablation_v1",
        "public_benchmark_answers_used": False,
        "public_tests_used": False,
    }
    return item


def materialize_private_semantic_tests(row: dict[str, Any]) -> None:
    if str(row.get("tests") or "").strip():
        row["private_semantic_tests_materialized"] = False
        return
    entry_point = str(row.get("entry_point") or "").strip()
    category = str(row.get("category") or "").strip()
    tests = private_semantic_tests_for_category(category, entry_point)
    if not tests:
        row["private_semantic_tests_materialized"] = False
        row["private_semantic_tests_missing_reason"] = f"no_private_template_for_category:{category or 'unknown'}"
        return
    row["tests"] = tests
    row["private_semantic_tests_materialized"] = True
    row["private_semantic_test_policy"] = "project_theseus_private_contract_semantic_tests_v1"
    row["private_semantic_test_source"] = "private_visible_prompt_category_and_contract_only"
    row["public_tests_used"] = False
    row["public_solutions_used"] = False


def execution_shape_contract(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or "")
    table = {
        "private_exec_archive_config_zip": ("bool", ["branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "archive"]),
        "private_exec_csv_command_outputs": ("list", ["loop", "branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "csv", "system_api"]),
        "private_exec_log_backup_tar": ("str", ["loop", "branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "archive"]),
        "private_exec_zip_flat_directory": ("unknown", ["loop", "branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "archive"]),
        "private_exec_csv_split_shuffle": ("list", ["loop", "branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "csv"]),
        "private_exec_system_info_dict": ("dict", ["branch", "locals", "execution_shaped_program", "edge_conditions", "system_api"]),
        "private_exec_json_extract_field": ("unknown", ["branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "structured_parsing"]),
        "private_exec_urlencode_payload": ("str", ["branch", "locals", "execution_shaped_program", "edge_conditions", "structured_parsing"]),
    }
    return_shape, required = table.get(category, ("unknown", ["execution_shaped_program", "edge_conditions"]))
    return {
        "policy": "project_theseus_decoder_contract_v1",
        "category": category,
        "type_family": "execution_shaped_program",
        "return_shape": return_shape,
        "required_constructs": required,
        "visible_arg_count_hint": 2
        if category
        not in {
            "private_exec_system_info_dict",
            "private_exec_zip_flat_directory",
            "private_exec_urlencode_payload",
            "private_exec_csv_split_shuffle",
        }
        else 0
        if category == "private_exec_system_info_dict"
        else 1,
        "full_body_required": True,
        "guardrail_only": True,
        "public_solutions_used": False,
        "public_tests_used": False,
        "score_semantics": "private held-out execution-shape decoder ablation contract",
    }


def build_public_visible_manifest() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "private_ablation_visible_execution_shape_probe",
            "source_task_id": "private_ablation_visible_0000",
            "card_id": "private_execution_shape_ablation_visible",
            "source_id": "local_private_ablation_visible",
            "split": "public_calibration",
            "category": "private_exec_json_extract_field",
            "prompt": "Visible-only local probe: read structured data and return a field with edge handling.",
            "entry_point": "private_ablation_visible_probe",
            "solution_expr": "",
            "solution_body": "",
            "tags": ["private_ablation_visible_only", "execution_shaped_programs"],
            "benchmark_evidence_level": "private_ablation_visible_prompt_only",
            "public_benchmark": False,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "decoder_contract": {
                "policy": "project_theseus_decoder_contract_v1",
                "category": "private_exec_json_extract_field",
                "type_family": "execution_shaped_program",
                "return_shape": "unknown",
                "required_constructs": ["branch", "locals", "execution_shaped_program", "edge_conditions", "file_path", "structured_parsing"],
                "public_solutions_used": False,
                "public_tests_used": False,
            },
        }
    ]


def run_rust(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    override = os.environ.get("THESEUS_SYMLIQUID_CLI", "").strip()
    exe = Path(override) if override else release_binary()
    if exe.exists():
        prefix = [str(exe)]
    else:
        prefix = ["cargo", "run", "--release", "-p", "symliquid-cli", "--"]
    env = os.environ.copy()
    env["THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES"] = "1"
    if bool(getattr(args, "allow_diagnostic_templates", False)):
        env["THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES"] = "1"
    command = [
        *prefix,
        "train-code-lm-closure",
        "--private-curriculum",
        rel(resolve(args.curriculum_out)),
        "--public-task-manifest",
        rel(resolve(args.public_manifest_out)),
        "--seed",
        str(args.seed),
        "--hv-dim",
        str(args.hv_dim),
        "--max-vocab",
        str(args.max_vocab),
        "--epochs",
        str(args.epochs),
        "--lr",
        "0.08",
        "--candidates-per-task",
        str(args.candidates_per_task),
        "--max-work-steps",
        str(args.max_work_steps),
        "--checkpoint-out",
        rel(resolve(args.checkpoint_out)),
        "--private-candidate-out",
        rel(resolve(args.candidate_out)),
        "--public-candidate-out",
        rel(resolve(args.public_candidate_out)),
        "--report-out",
        rel(resolve(args.rust_report_out)),
    ]
    result = run_process_tree(
        command,
        cwd=ROOT,
        env=env,
        timeout_seconds=args.rust_timeout_seconds if args.rust_timeout_seconds > 0 else None,
    )
    payload = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "command": command,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
    }
    if result.returncode == 124:
        payload["error"] = f"rust_timeout_after_{args.rust_timeout_seconds}s_process_tree_killed"
        write_json(
            resolve(args.rust_report_out),
            {
                "policy": "project_theseus_code_lm_closure_rust_v1",
                "created_utc": now(),
                "trigger_state": "YELLOW",
                "run_status": "timed_out_process_tree_killed",
                "runtime_ms": payload["runtime_ms"],
                "private_candidate_manifest": rel(resolve(args.candidate_out)),
                "public_candidate_manifest": rel(resolve(args.public_candidate_out)),
                "summary": {
                    "timeout_seconds": args.rust_timeout_seconds,
                    "process_tree_killed": True,
                    "score_semantics": "timeout_marker_only_not_learning_evidence",
                },
                "stderr_tail": payload["stderr_tail"],
                "external_inference_calls": 0,
            },
        )
    return payload


def evaluate_families(eval_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if str(row.get("phase") or "") != "private_eval":
            continue
        by_task[str(row.get("task_id") or "")].append(row)
    family_results: dict[str, Any] = {}
    sample_residuals: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_exec_shape_ablation_", dir=str(runtime_tmp_dir())) as tmp:
        tmp_root = Path(tmp)
        for family, predicate in FAMILIES.items():
            passed = 0
            no_candidate = 0
            return_shape_failures = 0
            algorithm_choice = 0
            over_rejection = 0
            residual_counts: Counter[str] = Counter()
            per_category_pass: Counter[str] = Counter()
            per_category_total: Counter[str] = Counter()
            for task in eval_rows:
                task_candidates = [
                    row
                    for row in by_task.get(str(task["task_id"]), [])
                    if (
                        predicate(row)
                        if family == "learned_token_decoder_v1"
                        else predicate(str(row.get("candidate_generation_mode") or ""))
                    )
                ]
                residual_candidates = [row for row in by_task.get(str(task["task_id"]), []) if str(row.get("candidate_generation_mode") or "") == "student_decoder_no_admissible_candidate_residual"]
                if not task_candidates:
                    no_candidate += 1
                    over_rejection += int(bool(residual_candidates))
                    residual_counts["no_admissible_candidate"] += 1
                    add_sample(sample_residuals, task, family, "no_admissible_candidate", "")
                    per_category_total[str(task.get("category") or "")] += 1
                    continue
                category = str(task.get("category") or "")
                per_category_total[category] += 1
                shape_bad = sum(1 for row in task_candidates if not static_return_shape_ok(task, str(row.get("code") or "")))
                result = run_any_candidate(tmp_root, task, task_candidates)
                if result["passed"]:
                    passed += 1
                    per_category_pass[category] += 1
                    continue
                failure = classify_failure(task, result.get("stderr", ""), result.get("stdout", ""))
                residual_counts[failure] += 1
                if shape_bad >= len(task_candidates):
                    return_shape_failures += 1
                elif failure == "return_shape_failure":
                    return_shape_failures += 1
                if failure in {"wrong_semantic_action", "edge_behavior", "algorithm_choice"}:
                    algorithm_choice += 1
                add_sample(sample_residuals, task, family, failure, str(result.get("stderr") or ""))
            total = len(eval_rows)
            family_results[family] = {
                "eval_task_count": total,
                "passed": passed,
                "private_pass_rate": ratio(passed, total),
                "algorithm_choice_residual_count": algorithm_choice,
                "return_shape_failure_count": return_shape_failures,
                "no_admissible_candidate_count": no_candidate,
                "no_admissible_candidate_rate": ratio(no_candidate, total),
                "over_rejection_by_guardrails_count": over_rejection,
                "residual_counts": dict(residual_counts),
                "per_category_pass_rate": {
                    category: ratio(per_category_pass[category], count)
                    for category, count in sorted(per_category_total.items())
                },
            }
    semantic_alg = int(family_results.get("semantic_plan_v2", {}).get("algorithm_choice_residual_count") or 0)
    semantic_shape = int(family_results.get("semantic_plan_v2", {}).get("return_shape_failure_count") or 0)
    for family, result in family_results.items():
        result["algorithm_choice_residual_shrinkage_vs_semantic_plan_v2"] = semantic_alg - int(result.get("algorithm_choice_residual_count") or 0)
        result["return_shape_failure_shrinkage_vs_semantic_plan_v2"] = semantic_shape - int(result.get("return_shape_failure_count") or 0)
    skeleton = family_results.get("execution_shape_skeleton_decoder_private_v1", {})
    semantic = family_results.get("semantic_plan_v2", {})
    learned = family_results.get("learned_token_decoder_v1", {})
    diagnostic_template_candidate_count = sum(
        1 for row in candidates if template_like_mode(str(row.get("candidate_generation_mode") or ""))
    )
    skeleton_zero_categories = [
        category
        for category, rate in sorted((skeleton.get("per_category_pass_rate") or {}).items())
        if float(rate or 0.0) <= 0.0
    ]
    learned_zero_categories = [
        category
        for category, rate in sorted((learned.get("per_category_pass_rate") or {}).items())
        if float(rate or 0.0) <= 0.0
    ]
    skeleton_no_admissible = int(skeleton.get("no_admissible_candidate_count") or 0)
    skeleton_pass_rate = float(skeleton.get("private_pass_rate") or 0.0)
    learned_no_admissible = int(learned.get("no_admissible_candidate_count") or 0)
    learned_pass_rate = float(learned.get("private_pass_rate") or 0.0)
    diagnostic_template_gate_ready = (
        diagnostic_template_candidate_count > 0
        and skeleton_pass_rate >= DIAGNOSTIC_MIN_SKELETON_PASS_RATE
        and skeleton_no_admissible == 0
        and not skeleton_zero_categories
    )
    learned_token_public_gate_ready = (
        learned_pass_rate >= PUBLIC_GATE_MIN_STUDENT_TOKEN_PASS_RATE
        and learned_no_admissible == 0
        and not learned_zero_categories
    )
    dominant = Counter()
    for result in family_results.values():
        dominant.update(result.get("residual_counts") or {})
    learned_dominant = Counter(learned.get("residual_counts") or {})
    operational_residual = learned_dominant.most_common(1)[0][0] if learned_dominant else ""
    diagnostic_residual = dominant.most_common(1)[0][0] if dominant else ""
    return {
        "summary": {
            "families_compared": sorted(family_results),
            "private_eval_task_count": len(eval_rows),
            "learned_token_decoder_pass_rate": learned.get("private_pass_rate", 0.0),
            "learned_token_decoder_no_admissible_candidate_count": learned_no_admissible,
            "learned_token_decoder_no_admissible_candidate_rate": learned.get("no_admissible_candidate_rate", 0.0),
            "learned_token_decoder_zero_pass_categories": learned_zero_categories,
            "learned_token_public_gate_min_pass_rate": PUBLIC_GATE_MIN_STUDENT_TOKEN_PASS_RATE,
            "learned_token_public_gate_ready": learned_token_public_gate_ready,
            "semantic_plan_v2_pass_rate": semantic.get("private_pass_rate", 0.0),
            "edge_exec_repair_v1_pass_rate": family_results.get("edge_exec_repair_v1", {}).get("private_pass_rate", 0.0),
            "execution_shape_skeleton_pass_rate": skeleton.get("private_pass_rate", 0.0),
            "skeleton_competitive_with_semantic": float(skeleton.get("private_pass_rate") or 0.0) >= float(semantic.get("private_pass_rate") or 0.0),
            "skeleton_diagnostic_gate_min_pass_rate": DIAGNOSTIC_MIN_SKELETON_PASS_RATE,
            "skeleton_no_admissible_candidate_count": skeleton_no_admissible,
            "skeleton_no_admissible_candidate_rate": skeleton.get("no_admissible_candidate_rate", 0.0),
            "skeleton_zero_pass_categories": skeleton_zero_categories,
            "diagnostic_template_gate_ready": diagnostic_template_gate_ready,
            "diagnostic_template_candidate_count": diagnostic_template_candidate_count,
            "private_ablation_public_gate_ready": learned_token_public_gate_ready,
            "public_gate_basis": "learned_token_decoder_v1_only_no_templates_no_skeletons",
            "skeleton_algorithm_choice_shrinkage": skeleton.get("algorithm_choice_residual_shrinkage_vs_semantic_plan_v2", 0),
            "skeleton_return_shape_shrinkage": skeleton.get("return_shape_failure_shrinkage_vs_semantic_plan_v2", 0),
            "dominant_residual": operational_residual,
            "dominant_learned_token_residual": operational_residual,
            "dominant_diagnostic_residual": diagnostic_residual,
        },
        "family_results": family_results,
        "dominant_private_residuals": dict(dominant.most_common(12)),
        "sample_residuals": sample_residuals[:24],
    }


def decoder_contract_verifier_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    passed = 0
    failed = 0
    eligible = 0
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidates:
        mode = str(row.get("candidate_generation_mode") or "unknown")
        if "decoder_contract_verifier_v1_passed" not in row:
            continue
        eligible += 1
        if bool(row.get("decoder_contract_verifier_v1_passed")):
            passed += 1
            by_mode[mode]["passed"] += 1
        else:
            failed += 1
            by_mode[mode]["failed"] += 1
            for reason in row.get("decoder_contract_verifier_v1_reasons") or []:
                reasons[str(reason)] += 1
    return {
        "decoder_contract_verifier_v1_candidate_count": eligible,
        "decoder_contract_verifier_v1_pass_count": passed,
        "decoder_contract_verifier_v1_fail_count": failed,
        "decoder_contract_verifier_v1_fail_reasons": dict(reasons.most_common(12)),
        "decoder_contract_verifier_v1_by_mode": {
            mode: dict(counts)
            for mode, counts in sorted(by_mode.items())
        },
    }


def add_sample(samples: list[dict[str, Any]], task: dict[str, Any], family: str, failure: str, stderr: str) -> None:
    if len(samples) >= 24:
        return
    samples.append(
        {
            "task_id": task.get("task_id"),
            "category": task.get("category"),
            "family": family,
            "failure": failure,
            "concept_residual_label": task.get("concept_residual_label"),
            "stderr_tail": stderr[-500:],
        }
    )


def run_any_candidate(root: Path, task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_prelude = "import math\nimport itertools\nimport functools\nimport collections\n\n"
    timeout_seconds = float(os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", "12"))
    root = root.resolve()
    last = {"passed": False, "stderr": "missing candidates", "stdout": "", "returncode": None}
    for idx, candidate in enumerate(candidates, start=1):
        path = root / f"{safe_name(task['task_id'])}_{idx}.py"
        path.write_text(runtime_prelude + str(candidate.get("code") or "") + "\n" + str(task.get("tests") or ""), encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, str(path.resolve())], cwd=root, text=True, capture_output=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            last = {"passed": False, "stderr": f"candidate_timeout_after_{timeout_seconds:g}s", "stdout": exc.stdout or "", "returncode": "timeout"}
            continue
        last = {"passed": result.returncode == 0, "stderr": result.stderr, "stdout": result.stdout, "returncode": result.returncode}
        if last["passed"]:
            return last
    return last


def static_return_shape_ok(task: dict[str, Any], code: str) -> bool:
    shape = str(((task.get("decoder_contract") or {}).get("return_shape") if isinstance(task.get("decoder_contract"), dict) else "") or "unknown")
    if shape == "unknown":
        return True
    compact = "".join(ch for ch in code.lower() if not ch.isspace())
    checks = {
        "list": ["return[]", "returnout", "returnpaths", "return["],
        "dict": ["return{", "returnresult", "returnpayload"],
        "tuple": ["return(", "returndigest,salt"],
        "str": ["return''", "returnarchive", "returnzip", "returnpath", "returnencoded", ".decode("],
        "bool": ["returntrue", "returnfalse", "is true", "is false"],
        "number": ["return0", "returntotal", "returncount", "returnlen("],
    }
    needles = checks.get(shape, [])
    return not needles or any(needle.replace(" ", "") in compact for needle in needles)


def classify_failure(task: dict[str, Any], stderr: str, stdout: str) -> str:
    text = f"{stderr}\n{stdout}".lower()
    if "syntaxerror" in text or "indentationerror" in text:
        return "syntax"
    if "typeerror" in text or "attributeerror" in text or "nameerror" in text:
        return "return_shape_failure" if not static_return_shape_ok(task, "") else "type_or_interface"
    if "filenotfounderror" in text:
        return "edge_behavior"
    if "assertionerror" in text:
        return "wrong_semantic_action"
    if "timeout" in text:
        return "timeout"
    return "runtime"


def build_gates(
    curriculum_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ablation: dict[str, Any],
    rust_result: dict[str, Any],
    *,
    category_filters: set[str] | None = None,
) -> list[dict[str, Any]]:
    train = [row for row in curriculum_rows if row.get("split") == "train"]
    evals = [row for row in curriculum_rows if row.get("split") == "eval"]
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates)
    skeleton = ablation["family_results"].get("execution_shape_skeleton_decoder_private_v1", {})
    learned = ablation["family_results"].get("learned_token_decoder_v1", {})
    zero_categories = ablation["summary"].get("skeleton_zero_pass_categories") or []
    learned_zero_categories = ablation["summary"].get("learned_token_decoder_zero_pass_categories") or []
    no_admissible = int(skeleton.get("no_admissible_candidate_count") or 0)
    learned_no_admissible = int(learned.get("no_admissible_candidate_count") or 0)
    skeleton_pass_rate = float(skeleton.get("private_pass_rate") or 0.0)
    learned_pass_rate = float(learned.get("private_pass_rate") or 0.0)
    diagnostic_template_candidate_count = int(ablation["summary"].get("diagnostic_template_candidate_count") or 0)
    diagnostic_templates_present = diagnostic_template_candidate_count > 0
    category_counts = Counter(str(row.get("category") or "") for row in evals)
    if category_filters:
        category_coverage_ok = bool(evals) and all(category_counts.get(category, 0) > 0 for category in category_filters)
    else:
        category_coverage_ok = len(set(category_counts)) >= 6
    return [
        gate("private_train_eval_split_present", bool(train) and bool(evals), {"train": len(train), "eval": len(evals)}, severity="hard"),
        gate("private_heldout_categories_cover_execution_shapes", category_coverage_ok, dict(category_counts), severity="hard"),
        gate("public_data_not_used", all(row.get("public_benchmark") is False for row in curriculum_rows + public_rows), "private/local generated rows only", severity="hard"),
        gate("rust_decoder_candidates_emitted", bool(candidates) and (rust_result.get("ok") or rust_result.get("skipped")), {"candidate_rows": len(candidates), "rust_ok": rust_result.get("ok")}, severity="hard"),
        gate("same_candidates_compared_across_families", all(name in ablation["family_results"] for name in FAMILIES), list(ablation["family_results"]), severity="hard"),
        gate("student_ablation_template_candidates_absent", diagnostic_template_candidate_count == 0, {"diagnostic_template_candidate_count": diagnostic_template_candidate_count}, severity="hard"),
        gate("mode_families_visible", any(learned_token_candidate_row(row) for row in candidates), dict(modes.most_common(12)), severity="hard"),
        gate("skeleton_competitive_with_semantic_diagnostic_only", (not diagnostic_templates_present) or bool(ablation["summary"].get("skeleton_competitive_with_semantic")), ablation["summary"], severity="soft"),
        gate(
            "skeleton_diagnostic_gate_min_pass_rate",
            (not diagnostic_templates_present) or skeleton_pass_rate >= DIAGNOSTIC_MIN_SKELETON_PASS_RATE,
            {"private_pass_rate": skeleton_pass_rate, "minimum": DIAGNOSTIC_MIN_SKELETON_PASS_RATE, "diagnostic_templates_present": diagnostic_templates_present},
            severity="soft",
        ),
        gate("skeleton_emits_candidate_for_every_private_eval_diagnostic_only", (not diagnostic_templates_present) or no_admissible == 0, skeleton, severity="soft"),
        gate("skeleton_covers_every_execution_shape_category_diagnostic_only", (not diagnostic_templates_present) or not zero_categories, {"zero_pass_categories": zero_categories, "diagnostic_templates_present": diagnostic_templates_present}, severity="soft"),
        gate(
            "learned_token_public_gate_min_pass_rate",
            learned_pass_rate >= PUBLIC_GATE_MIN_STUDENT_TOKEN_PASS_RATE,
            {"private_pass_rate": learned_pass_rate, "minimum": PUBLIC_GATE_MIN_STUDENT_TOKEN_PASS_RATE},
            severity="hard",
        ),
        gate("learned_token_emits_candidate_for_every_private_eval", learned_no_admissible == 0, learned, severity="hard"),
        gate("learned_token_covers_every_execution_shape_category", not learned_zero_categories, {"zero_pass_categories": learned_zero_categories}, severity="hard"),
        gate("public_calibration_gate_uses_no_templates", bool(ablation["summary"].get("private_ablation_public_gate_ready")) and bool(ablation["summary"].get("learned_token_public_gate_ready")), ablation["summary"], severity="hard"),
    ]


def next_actions(ablation: dict[str, Any]) -> list[str]:
    skeleton = ablation["family_results"].get("execution_shape_skeleton_decoder_private_v1", {})
    learned = ablation["family_results"].get("learned_token_decoder_v1", {})
    dominant = str(ablation["summary"].get("dominant_residual") or "")
    diagnostic_templates_present = int(ablation["summary"].get("diagnostic_template_candidate_count") or 0) > 0
    actions = []
    if not bool(ablation["summary"].get("learned_token_public_gate_ready")):
        actions.append("do not run public calibration: learned token decoder has not cleared the private no-template gate")
    if int(learned.get("no_admissible_candidate_count") or 0) > 0:
        actions.append("improve learned full-body token decoder coverage instead of adding skeleton templates")
    learned_zero = ablation["summary"].get("learned_token_decoder_zero_pass_categories") or []
    if learned_zero:
        actions.append("add private token-prediction pressure for learned decoder zero-pass categories: " + ", ".join(str(item) for item in learned_zero[:6]))
    if diagnostic_templates_present and int(skeleton.get("no_admissible_candidate_count") or 0) > 0:
        actions.append("diagnostic only: patch execution_shape_skeleton_decoder to define obligations, not to score capability")
    if diagnostic_templates_present and int(skeleton.get("return_shape_failure_count") or 0) > 0:
        actions.append("tighten return builder selection for list/dict/str/bool execution-shaped contracts")
    if dominant in {"wrong_semantic_action", "edge_behavior"}:
        actions.append("patch library/action plan and branch-loop skeletons for private execution-shaped residuals")
    zero_categories = ablation["summary"].get("skeleton_zero_pass_categories") or []
    if diagnostic_templates_present and zero_categories:
        actions.append("add category-specific skeleton coverage for " + ", ".join(str(item) for item in zero_categories[:6]))
    if bool(ablation["summary"].get("learned_token_public_gate_ready")):
        actions.append("learned token private gate is clear; run the broader decoder_v2/private ablation gate before any single public calibration")
    else:
        actions.append("rerun this private ablation after learned-token coverage changes; public calibration unlocks only after learned_token_public_gate_ready is true")
    return actions


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Execution Shape Private Ablation",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private eval tasks: {summary.get('private_eval_task_count')}",
        f"- Learned Token Decoder: {summary.get('learned_token_decoder_pass_rate')}",
        f"- Learned token public gate ready: {summary.get('learned_token_public_gate_ready')}",
        f"- Semantic Plan V2: {summary.get('semantic_plan_v2_pass_rate')}",
        f"- Edge Exec Repair V1: {summary.get('edge_exec_repair_v1_pass_rate')}",
        f"- Execution Shape Skeleton diagnostic: {summary.get('execution_shape_skeleton_pass_rate')}",
        f"- Public calibration gate ready: {summary.get('private_ablation_public_gate_ready')}",
        f"- Public gate basis: {summary.get('public_gate_basis')}",
        f"- Diagnostic template candidates emitted: {summary.get('diagnostic_template_candidate_count')}",
        f"- Diagnostic templates enabled: {summary.get('diagnostic_templates_enabled')}",
        f"- Learned token no-admissible candidates: {summary.get('learned_token_decoder_no_admissible_candidate_count')}",
        f"- Learned token zero-pass categories: {summary.get('learned_token_decoder_zero_pass_categories')}",
        f"- Skeleton no-admissible candidates: {summary.get('skeleton_no_admissible_candidate_count')}",
        f"- Skeleton zero-pass categories: {summary.get('skeleton_zero_pass_categories')}",
        f"- Dominant residual: {summary.get('dominant_residual')}",
        "",
        "## Family Metrics",
    ]
    for family, result in report.get("family_results", {}).items():
        lines.extend(
            [
                "",
                f"### {family}",
                f"- pass_rate: {result.get('private_pass_rate')}",
                f"- algorithm_choice_residual_count: {result.get('algorithm_choice_residual_count')}",
                f"- return_shape_failure_count: {result.get('return_shape_failure_count')}",
                f"- no_admissible_candidate_count: {result.get('no_admissible_candidate_count')}",
                f"- over_rejection_by_guardrails_count: {result.get('over_rejection_by_guardrails_count')}",
            ]
        )
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark} {row.get('gate')} ({row.get('severity')})")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def runtime_tmp_dir() -> Path:
    configured = os.environ.get("THESEUS_RUNTIME_TMP", "").strip()
    if configured:
        root = Path(configured)
    elif sys.platform.startswith("win"):
        root = Path("D:/ProjectTheseus/tmp")
    else:
        root = ROOT / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
