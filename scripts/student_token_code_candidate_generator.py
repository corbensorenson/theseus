"""Emit real token-level learned student code candidates for code graduation.

This is intentionally only an orchestration wrapper. It exports public task
prompts without tests or reference implementations, then calls the Rust
SymLiquid CLI token generator. The candidate bodies come from the learned Rust
checkpoint, not Python templates or benchmark-specific rules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_graduation as real_code  # noqa: E402
from public_code_case_manifest import filter_tasks_for_card, load_case_manifest, manifest_pool_size  # noqa: E402


DEFAULT_CARDS = "source_evalplus,source_human_eval,source_mbpp"
DEFAULT_TRAINING_SOURCES = "data/training_sources/broad_capability_curriculum_v1_training_sources.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default=DEFAULT_CARDS)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--max-cases-per-card", type=int, default=8)
    parser.add_argument(
        "--case-manifest",
        default="",
        help="Optional public calibration selector manifest. Only task IDs are consumed.",
    )
    parser.add_argument("--training-sources", default=DEFAULT_TRAINING_SOURCES)
    parser.add_argument(
        "--allow-project-code-only",
        action="store_true",
        help=(
            "Permit a prompt-only diagnostic run that learns only from local project code. "
            "Do not use this for capability-transfer readiness claims."
        ),
    )
    parser.add_argument("--project-code-roots", default="scripts,crates")
    parser.add_argument("--max-training-rows-per-source", type=int, default=1200)
    parser.add_argument("--max-project-files", type=int, default=160)
    parser.add_argument("--max-candidates-per-task", type=int, default=8)
    parser.add_argument("--task-manifest-out", default="reports/student_token_code_tasks.jsonl")
    parser.add_argument("--checkpoint-out", default="reports/student_token_code_checkpoint.json")
    parser.add_argument("--out", default="reports/student_code_candidates.jsonl")
    parser.add_argument("--report-out", default="reports/student_token_code_generator.json")
    args = parser.parse_args()

    started = time.perf_counter()
    requested_cards = [card.strip() for card in args.cards.split(",") if card.strip()]
    cards = real_code.expand_requested_cards(requested_cards)
    task_rows = export_visible_tasks(
        cards,
        seed=args.seed,
        max_cases=max(1, args.max_cases_per_card),
        case_manifest=args.case_manifest,
    )
    write_jsonl(resolve(args.task_manifest_out), task_rows)

    removed_stale_artifacts = remove_stale_generation_artifacts(
        [
            resolve(args.out),
            resolve(args.checkpoint_out),
            resolve(args.report_out),
            resolve(str(Path(args.report_out).with_name("student_token_code_generator_wrapper.json"))),
        ]
    )
    command = rust_generator_command(args)
    timeout_seconds = candidate_generation_timeout_seconds(
        task_count=len(task_rows),
        max_candidates=max(1, int(args.max_candidates_per_task)),
    )
    generation_error = ""
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        returncode = result.returncode
        stdout_tail = result.stdout[-1600:]
        stderr_tail = result.stderr[-1600:]
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout_tail = (exc.stdout or "")[-1600:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1600:] if isinstance(exc.stderr, str) else ""
        generation_error = f"rust token generator timed out after {timeout_seconds}s"
    except OSError as exc:
        returncode = 127
        stdout_tail = ""
        stderr_tail = ""
        generation_error = str(exc)[-1600:]
    rust_report = read_json(resolve(args.report_out), {})
    ready_training_sources = int(get_path(rust_report, ["summary", "ready_training_sources"], 0))
    training_rows_used = int(get_path(rust_report, ["summary", "training_rows_used"], 0))
    project_code_only_allowed = bool(args.allow_project_code_only)
    gates = [
        gate("visible_task_manifest_exported", len(task_rows) > 0, f"tasks={len(task_rows)}"),
        gate("task_tests_omitted", all("tests" not in row for row in task_rows), "tests not exported to Rust generator"),
        gate("canonical_solutions_omitted", all("canonical_solution" not in row for row in task_rows), "reference implementations not exported"),
        gate("rust_token_generator_completed", returncode == 0, f"returncode={returncode} error={generation_error}"),
        gate(
            "curated_private_training_sources_loaded",
            ready_training_sources > 0 or project_code_only_allowed,
            {
                "ready_training_sources": ready_training_sources,
                "allow_project_code_only": project_code_only_allowed,
            },
        ),
        gate(
            "curated_private_training_rows_used",
            training_rows_used > 0 or project_code_only_allowed,
            {
                "training_rows_used": training_rows_used,
                "allow_project_code_only": project_code_only_allowed,
            },
        ),
        gate(
            "token_level_code_generation_learned",
            bool(get_path(rust_report, ["summary", "token_level_code_generation_learned"], False)),
            get_path(rust_report, ["summary", "candidate_generation_mode"], ""),
        ),
        gate(
            "full_body_token_candidates_emitted",
            int(get_path(rust_report, ["summary", "full_body_token_candidate_count"], 0)) > 0,
            get_path(rust_report, ["summary", "full_body_token_candidate_count"], 0),
        ),
        gate(
            "grammar_masked_learned_candidates_emitted",
            int(get_path(rust_report, ["summary", "grammar_masked_learned_token_candidate_count"], 0)) > 0,
            get_path(rust_report, ["summary", "grammar_masked_learned_token_candidate_count"], 0),
        ),
        gate(
            "benchmark_promotion_eligible_candidates_emitted",
            int(get_path(rust_report, ["summary", "benchmark_promotion_eligible_candidate_count"], 0)) > 0,
            get_path(rust_report, ["summary", "benchmark_promotion_eligible_candidate_count"], 0),
        ),
        gate(
            "no_expression_memory_fallback_candidates",
            int(get_path(rust_report, ["summary", "expression_memory_fallback_count"], 0)) == 0,
            get_path(rust_report, ["summary", "expression_memory_fallback_count"], 0),
        ),
        gate("external_inference_zero", True, "local Rust/SymLiquid token generator only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    wrapper_report = {
        "policy": "project_theseus_student_token_code_generator_wrapper_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "requested_cards": requested_cards,
        "cards": cards,
        "seed": args.seed,
        "command": command,
        "removed_stale_artifacts": removed_stale_artifacts,
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "generation_error": generation_error,
        "task_manifest": rel(resolve(args.task_manifest_out)),
        "case_manifest": rel(resolve(args.case_manifest)) if args.case_manifest else "",
        "candidate_manifest": rel(resolve(args.out)),
        "checkpoint": rel(resolve(args.checkpoint_out)),
        "rust_report": rel(resolve(args.report_out)),
        "summary": {
            "task_count": len(task_rows),
            "candidate_count": get_path(rust_report, ["summary", "candidate_count"], 0),
            "checkpoint_id": get_path(rust_report, ["summary", "checkpoint_id"], ""),
            "candidate_generation_mode": get_path(rust_report, ["summary", "candidate_generation_mode"], ""),
            "token_level_code_generation_learned": get_path(rust_report, ["summary", "token_level_code_generation_learned"], False),
            "compositional_token_candidate_count": get_path(rust_report, ["summary", "compositional_token_candidate_count"], 0),
            "full_body_token_candidate_count": get_path(rust_report, ["summary", "full_body_token_candidate_count"], 0),
            "grammar_masked_learned_token_candidate_count": get_path(rust_report, ["summary", "grammar_masked_learned_token_candidate_count"], 0),
            "benchmark_promotion_eligible_candidate_count": get_path(rust_report, ["summary", "benchmark_promotion_eligible_candidate_count"], 0),
            "expression_memory_fallback_count": get_path(rust_report, ["summary", "expression_memory_fallback_count"], 0),
            "deterministic_guardrail_failed_candidate_count": get_path(rust_report, ["summary", "deterministic_guardrail_failed_candidate_count"], 0),
            "template_like_candidate_count": get_path(rust_report, ["summary", "template_like_candidate_count"], 0),
            "loop_closure_candidate_count": get_path(rust_report, ["summary", "loop_closure_candidate_count"], 0),
            "ready_training_sources": ready_training_sources,
            "training_rows_used": training_rows_used,
            "project_code_files_seen": get_path(rust_report, ["summary", "project_code_files_seen"], 0),
            "allow_project_code_only": project_code_only_allowed,
            "candidate_generation_timeout_seconds": timeout_seconds,
            "removed_stale_artifact_count": len(removed_stale_artifacts),
            "public_tests_visible_to_generator": False,
            "canonical_solution_seen_by_solver": False,
            "external_inference_calls": 0,
        },
        "gates": gates,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    wrapper_path = resolve(str(Path(args.report_out).with_name("student_token_code_generator_wrapper.json")))
    write_json(wrapper_path, wrapper_report)
    print(json.dumps(wrapper_report, indent=2))
    return 0 if returncode == 0 and trigger_state in {"GREEN", "YELLOW"} else 1


def export_visible_tasks(
    cards: list[str],
    *,
    seed: int,
    max_cases: int,
    case_manifest: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_by_card = load_case_manifest(case_manifest)
    for card_id in cards:
        card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
        source_id = str(card.get("source_id") or card_id.replace("source_", ""))
        source_path = real_code.resolve_source_path(card)
        manifest_rows = manifest_by_card.get(card_id, [])
        load_limit = manifest_pool_size(max_cases, {card_id: manifest_rows}) if manifest_rows else max_cases
        tasks, evidence_level, _semantics = real_code.load_cases(
            card_id,
            source_id,
            source_path,
            seed,
            load_limit,
        )
        if manifest_rows:
            tasks, _missing = filter_tasks_for_card(tasks, manifest_rows)
        for task in tasks:
            rows.append(
                {
                    "task_id": str(task.get("task_id") or ""),
                    "source_task_id": str(task.get("source_task_id") or ""),
                    "card_id": card_id,
                    "source_id": source_id,
                    "case_type": str(task.get("case_type") or ""),
                    "prompt": str(task.get("prompt") or ""),
                    "entry_point": str(task.get("entry_point") or ""),
                    "tags": [str(tag) for tag in task.get("tags", [])] if isinstance(task.get("tags"), list) else [],
                    "benchmark_evidence_level": evidence_level,
                    "case_manifest_selected": bool(manifest_rows),
                    "visible_task_only": True,
                    "tests_exported": False,
                    "canonical_solution_exported": False,
                }
            )
    return rows


def rust_generator_command(args: argparse.Namespace) -> list[str]:
    exe = native_symliquid_cli()
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "code_token_generator.rs",
    ]
    module_dir = ROOT / "crates" / "symliquid-cli" / "src" / "code_token_generator"
    if module_dir.exists():
        source_files.extend(sorted(module_dir.glob("*.rs")))
    exe_fresh = exe.exists() and all(
        exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists()
    )
    if exe_fresh:
        prefix = [str(exe)]
    else:
        prefix = ["cargo", "run", "-p", "symliquid-cli", "--"]
    return [
        *prefix,
        "train-code-token-generator",
        "--task-manifest",
        rel(resolve(args.task_manifest_out)),
        "--training-sources",
        rel(resolve(args.training_sources)),
        "--project-code-roots",
        args.project_code_roots,
        "--seed",
        str(int(args.seed)),
        "--max-training-rows-per-source",
        str(max(1, int(args.max_training_rows_per_source))),
        "--max-project-files",
        str(max(1, int(args.max_project_files))),
        "--max-candidates-per-task",
        str(max(1, int(args.max_candidates_per_task))),
        "--checkpoint-out",
        rel(resolve(args.checkpoint_out)),
        "--out",
        rel(resolve(args.out)),
        "--report-out",
        rel(resolve(args.report_out)),
    ]


def native_symliquid_cli() -> Path:
    native = ROOT / "target" / "release" / "symliquid-cli"
    windows = ROOT / "target" / "release" / "symliquid-cli.exe"
    return native if native.exists() else windows


def candidate_generation_timeout_seconds(*, task_count: int, max_candidates: int) -> int:
    work_units = max(1, int(task_count)) * max(1, int(max_candidates))
    return max(240, min(1800, 60 + work_units))


def remove_stale_generation_artifacts(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                removed.append(rel(path))
        except OSError:
            continue
    return removed


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
