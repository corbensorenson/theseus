#!/usr/bin/env python3
"""Build receipt-bound, task-complete training units for the neural seed.

Raw language-model text remains useful pretraining data, but it does not count as
product-task coverage here. A counted unit must bind visible input, a complete
target, source/license lineage, a source-disjoint split, contamination evidence,
and either a governed blind-English rubric or an independently replayed verifier.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import html.parser
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import training_data_lineage_audit
from neural_seed_functional_verifiers import CHROME, _render_chrome


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "task_complete_training_units.json"
ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "odc-by",
    "public-domain",
    "project-internal",
}
TARGET_EXTENSIONS = {".html", ".htm", ".css"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--prepare-javascript", action="store_true")
    parser.add_argument(
        "--prepare-python-projects",
        action="store_true",
        help="Materialize lockfile-bound Python verifier environments before offline replay.",
    )
    parser.add_argument(
        "--max-executable-units-per-source",
        type=int,
        default=0,
        help="Bound verifier execution for a canary; zero verifies every executable unit.",
    )
    parser.add_argument("--force-verification", action="store_true")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    report = build_report(config, args)
    outputs = config["outputs"]
    write_json(resolve(outputs["report"]), report)
    write_json(resolve(outputs["manifest"]), report["public_manifest"])
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "contract_state": report["contract_state"],
        "coverage_state": report["coverage_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "coverage_gaps": report["coverage_gaps"],
    }, indent=2, sort_keys=True))
    return 0 if report["contract_state"] == "GREEN" else 2


def build_report(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    validate_config(config)
    contamination = training_data_lineage_audit.build_public_contamination_index(
        max_texts=int(config["contamination"]["maximum_public_texts"])
    )
    cache_path = resolve(config["outputs"]["verification_cache"])
    cache = {} if args.force_verification else load_cache(cache_path)
    ledger_path = resolve(config["outputs"]["unit_ledger"])
    prior_units = (
        {}
        if args.force_verification
        else load_prior_unit_cache(
            ledger_path,
            resolve(config["outputs"]["report"]),
            contamination_digest=contamination["digest"],
        )
    )
    cache_updates: dict[str, dict[str, Any]] = {}
    units: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    hard_gaps: list[dict[str, Any]] = []

    for source in config["sources"]:
        if not source.get("enabled", True):
            continue
        try:
            adapter = str(source["adapter"])
            if adapter == "governed_conversation_assistant_turns":
                source_units, source_summary = conversation_units(
                    config, source, contamination, prior_units=prior_units
                )
            elif adapter == "exercism_practice_exercises":
                source_units, source_summary = exercism_units(
                    config,
                    source,
                    contamination,
                    cache=cache,
                    cache_updates=cache_updates,
                    inventory_only=bool(args.inventory_only),
                    prepare_javascript=bool(args.prepare_javascript),
                    max_verify=max(0, int(args.max_executable_units_per_source)),
                )
            elif adapter == "mdn_start_finished_assessments":
                source_units, source_summary = mdn_units(
                    config,
                    source,
                    contamination,
                    cache=cache,
                    cache_updates=cache_updates,
                    inventory_only=bool(args.inventory_only),
                    max_verify=max(0, int(args.max_executable_units_per_source)),
                )
            elif adapter == "python_test_killed_function_holes":
                source_units, source_summary = python_function_hole_units(
                    config,
                    source,
                    contamination,
                    cache=cache,
                    cache_updates=cache_updates,
                    cache_path=cache_path,
                    prepare_python_projects=bool(args.prepare_python_projects),
                    inventory_only=bool(args.inventory_only),
                    max_verify=max(0, int(args.max_executable_units_per_source)),
                )
            else:
                raise ValueError(f"unknown task-complete adapter: {adapter}")
            units.extend(source_units)
            source_summaries.append(source_summary)
        except Exception as exc:  # noqa: BLE001
            hard_gaps.append({
                "kind": "source_adapter_failed",
                "source_id": source.get("id"),
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            })

    duplicate_ids = [key for key, count in Counter(row["unit_id"] for row in units).items() if count > 1]
    if duplicate_ids:
        hard_gaps.append({"kind": "duplicate_unit_identity", "count": len(duplicate_ids), "examples": duplicate_ids[:10]})
    split_leaks = split_leakage(units)
    if split_leaks:
        hard_gaps.append({"kind": "source_task_split_leakage", "count": len(split_leaks), "examples": split_leaks[:10]})
    admitted = [row for row in units if row["decision"] == "admit"]
    if any(row["contamination"]["quarantine"] for row in admitted):
        hard_gaps.append({"kind": "contaminated_unit_admitted"})
    if any(row["public_benchmark_training_rows"] or row["external_inference_calls"] or row["fallback_return_count"] for row in units):
        hard_gaps.append({"kind": "hard_boundary_counter_nonzero"})

    ledger_receipt = write_unit_ledger(ledger_path, units)
    if cache_updates:
        cache.update(cache_updates)
        write_cache(cache_path, cache)
    elif not cache_path.exists():
        write_cache(cache_path, cache)

    coverage, coverage_gaps = coverage_summary(config, units)
    contract_state = "RED" if hard_gaps else "GREEN"
    coverage_state = "GREEN" if not coverage_gaps and contract_state == "GREEN" else "YELLOW"
    trigger_state = "RED" if hard_gaps else coverage_state
    decision_counts = Counter(row["decision"] for row in units)
    verification_counts = Counter(row["verification"]["state"] for row in units)
    strength_counts = Counter(row["verification"]["strength"] for row in units if row["decision"] == "admit")
    split_counts = Counter(row["split"] for row in units if row["decision"] == "admit")
    arm_counts = Counter(row["arm_id"] for row in units if row["decision"] == "admit")
    summary = {
        "source_count": len(source_summaries),
        "unit_count": len(units),
        "admitted_unit_count": len(admitted),
        "decision_counts": dict(sorted(decision_counts.items())),
        "verification_state_counts": dict(sorted(verification_counts.items())),
        "verification_strength_counts": dict(sorted(strength_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "arm_counts": dict(sorted(arm_counts.items())),
        "task_complete_unique_target_positions": sum(int(row["target_positions"]) for row in admitted),
        "public_exact_overlap_unit_count": sum(bool(row["contamination"]["exact_overlap"]) for row in units),
        "public_semantic_overlap_unit_count": sum(bool(row["contamination"]["semantic_overlap"]) for row in units),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "fallback_return_count": 0,
        "raw_user_text_admitted": False,
        "contract_hard_gap_count": len(hard_gaps),
        "coverage_gap_count": len(coverage_gaps),
        "verification_cache_entry_count": len(cache),
        "verification_cache_new_or_replaced_count": len(cache_updates),
        "inventory_only": bool(args.inventory_only),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    config_path = resolve(getattr(args, "config", DEFAULT_CONFIG))
    public_manifest = {
        "policy": "project_theseus_task_complete_training_unit_manifest_v1",
        "created_utc": now(),
        "source_config": rel(config_path),
        "source_config_sha256": file_sha256(config_path),
        "unit_ledger": rel(ledger_path),
        "unit_ledger_sha256": ledger_receipt["sha256"],
        "unit_ledger_count": ledger_receipt["count"],
        "verification_cache": rel(cache_path),
        "verification_cache_sha256": file_sha256(cache_path),
        "coverage": coverage,
        "summary": summary,
        "source_summaries": source_summaries,
        "boundaries": config["boundaries"],
        "raw_unit_payloads_in_manifest": False,
    }
    return {
        "policy": "project_theseus_task_complete_training_units_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "contract_state": contract_state,
        "coverage_state": coverage_state,
        "summary": summary,
        "coverage": coverage,
        "coverage_gaps": coverage_gaps,
        "source_summaries": source_summaries,
        "ledger_receipt": ledger_receipt,
        "hard_gaps": hard_gaps,
        "boundaries": config["boundaries"],
        "score_semantics": (
            "A counted unit binds a complete target to a governed rubric or replayed verifier. "
            "This is data-readiness evidence, not model capability, benchmark performance, or route authority."
        ),
        "non_claims": [
            "Raw source files and parser-only rows do not count toward product-task coverage.",
            "An executable target-pass/starter-fail receipt proves the data unit is coherent, not that a model can solve it.",
            "Blind-rubric binding and source acceptance do not prove every English target is optimal.",
            "Heuristic semantic decontamination reduces known overlap risk but cannot prove universal non-overlap.",
            "No public benchmark payload is emitted into the training-unit ledger.",
        ],
        "public_manifest": public_manifest,
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_task_complete_training_units_v1":
        raise ValueError("unexpected task-complete config policy")
    boundaries = config.get("boundaries") or {}
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count", "teacher_calls"):
        if int(boundaries.get(key) or 0) != 0:
            raise ValueError(f"hard boundary must be zero: {key}")
    split = config["split_policy"]
    total = sum(int(split[key]) for key in ("train_basis_points", "development_basis_points", "confirmation_basis_points"))
    if total != 10000:
        raise ValueError("split basis points must total 10000")
    if set(config["coverage_floors_for_50m_scale_proposal"]) != {
        "english", "python", "javascript_typescript", "html_css", "rust"
    }:
        raise ValueError("coverage floors must name all five seed arms")


def conversation_units(
    config: dict[str, Any],
    source: dict[str, Any],
    contamination: dict[str, Any],
    *,
    prior_units: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for path in sorted(ROOT.glob(source["receipt_glob"])):
        for row in iter_jsonl(path):
            receipt_id = str(row.get("receipt_id") or "")
            if receipt_id:
                receipts[receipt_id] = row
    units: list[dict[str, Any]] = []
    row_count = 0
    reused_contamination = 0
    for path in sorted(ROOT.glob(source["row_glob"])):
        for row in iter_jsonl(path):
            row_count += 1
            target_message = row.get("target_message") if isinstance(row.get("target_message"), dict) else None
            messages = row.get("prompt_messages") if isinstance(row.get("prompt_messages"), list) else []
            if not target_message:
                continue
            visible_messages = [
                {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
                for message in messages if isinstance(message, dict)
            ]
            target = str(target_message.get("content") or "")
            visible = canonical_json({"messages": visible_messages})
            source_task_id = str(row.get("task_id") or stable_id(source["id"], row_count, visible, target))
            receipt = receipts.get(str(row.get("data_admission_receipt_id") or ""), {})
            provenance_class = str(receipt.get("provenance_class") or "unknown")
            identity = stable_id("task-complete-unit", {
                "source_id": source["id"],
                "source_task_id": source_task_id,
                "arm_id": "english",
                "visible_context_sha256": sha256_text(visible),
                "target_sha256": sha256_text(target),
            })
            prior = prior_units.get(identity) or {}
            prior_contamination = (
                prior.get("contamination")
                if prior.get("visible_context_sha256") == sha256_text(visible)
                and prior.get("target_sha256") == sha256_text(target)
                and (prior.get("provenance") or {}).get("source_receipt_sha256")
                == (stable_hash(receipt) if receipt else None)
                else None
            )
            if prior_contamination:
                reused_contamination += 1
            unit = base_unit(
                config,
                source=source,
                source_task_id=source_task_id,
                arm_id="english",
                task_family=english_family(visible_messages),
                visible_context=visible,
                target=target,
                license_spdx=str(row.get("license_spdx") or receipt.get("license_spdx") or ""),
                provenance={
                    "dataset": row.get("source_id"),
                    "data_admission_receipt_id": row.get("data_admission_receipt_id"),
                    "provenance_class": provenance_class,
                    "source_receipt_sha256": stable_hash(receipt) if receipt else None,
                    "static_model_derived": provenance_class == "external_teacher_generated",
                    "live_teacher_call": False,
                },
                contamination=contamination,
                precomputed_contamination=prior_contamination,
            )
            receipt_ok = bool(receipt) and receipt.get("decision") == "admit"
            multi_turn = len(visible_messages) > 1
            verifier = {
                "kind": "governed_conversation_target_binding_v1",
                "strength": "governed_conversation_target_bound",
                "state": "passed" if receipt_ok and target.strip() else "failed",
                "source_receipt_bound": receipt_ok,
                "target_role": target_message.get("role"),
                "multi_turn": multi_turn,
                "utility_evaluation_required_separately": True,
            }
            finish_unit(unit, verifier, config)
            units.append(unit)
    return units, summarize_source(source, units, extra={
        "input_row_count": row_count,
        "source_receipt_count": len(receipts),
        "reused_content_bound_contamination_count": reused_contamination,
    })


def exercism_units(
    config: dict[str, Any],
    source: dict[str, Any],
    contamination: dict[str, Any],
    *,
    cache: dict[str, dict[str, Any]],
    cache_updates: dict[str, dict[str, Any]],
    inventory_only: bool,
    prepare_javascript: bool,
    max_verify: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_root = ensure_source_root(source)
    language = str(source["language"])
    if language == "javascript_typescript" and prepare_javascript:
        prepare_javascript_source(source_root)
    exercises_root = source_root / "exercises" / "practice"
    units: list[dict[str, Any]] = []
    verified_attempts = 0
    skipped_incompatible = 0
    for meta_path in sorted(exercises_root.glob("*/.meta/config.json")):
        exercise_dir = meta_path.parents[1]
        meta = read_json(meta_path)
        file_spec = meta.get("files") if isinstance(meta.get("files"), dict) else {}
        solution_paths = [str(value) for value in file_spec.get("solution", [])]
        test_paths = [str(value) for value in file_spec.get("test", [])]
        example_paths = [str(value) for value in file_spec.get("example", [])]
        if not solution_paths or not test_paths or not example_paths:
            skipped_incompatible += 1
            continue
        target_files = map_example_to_solution(exercise_dir, solution_paths, example_paths)
        if not target_files:
            skipped_incompatible += 1
            continue
        starter_files = read_relative_files(exercise_dir, solution_paths, allow_missing=True)
        test_files = read_relative_files(exercise_dir, test_paths)
        if not test_files:
            skipped_incompatible += 1
            continue
        instructions = read_optional(exercise_dir / ".docs" / "instructions.md")
        if not instructions:
            instructions = str(meta.get("blurb") or exercise_dir.name.replace("-", " "))
        visible = canonical_json({
            "instructions": instructions,
            "starter_files": starter_files,
            "requested_solution_files": solution_paths,
        })
        target = canonical_json({"files": target_files})
        unit = base_unit(
            config,
            source=source,
            source_task_id=f"{source['repo']}:{exercise_dir.name}",
            arm_id=language,
            task_family="exercise_implementation_and_repair",
            visible_context=visible,
            target=target,
            license_spdx=str(source["license_spdx"]),
            provenance={
                "repo": source["repo"],
                "revision": source["revision"],
                "exercise": exercise_dir.name,
                "archive_sha256": source["archive_sha256"],
                "test_file_hashes": {path: sha256_text(text) for path, text in test_files.items()},
                "static_open_corpus": True,
                "live_teacher_call": False,
            },
            contamination=contamination,
        )
        verification_digest = stable_hash({
            "unit_id": unit["unit_id"],
            "target_files": target_files,
            "starter_files": starter_files,
            "test_files": test_files,
            "toolchain": config["toolchains"][language],
        })
        cached = cache.get(unit["unit_id"])
        should_verify = not inventory_only and (max_verify == 0 or verified_attempts < max_verify)
        if cached and cached.get("verification_digest") == verification_digest:
            verifier = cached["verification"]
        elif should_verify:
            verified_attempts += 1
            verifier = verify_exercism(
                source_root,
                exercise_dir,
                language,
                target_files,
                test_paths,
                config["toolchains"][language],
            )
            cache_updates[unit["unit_id"]] = {
                "unit_id": unit["unit_id"],
                "verification_digest": verification_digest,
                "verification": verifier,
            }
        else:
            verifier = {
                "kind": f"{language}_task_complete_verifier_v1",
                "strength": "executable_target_pass_starter_fail",
                "state": "not_run",
                "reason": "inventory_only_or_bounded_verification",
            }
        finish_unit(unit, verifier, config)
        units.append(unit)
    return units, summarize_source(source, units, extra={
        "source_root": rel(source_root),
        "incompatible_exercise_count": skipped_incompatible,
        "new_verification_attempt_count": verified_attempts,
    })


def mdn_units(
    config: dict[str, Any],
    source: dict[str, Any],
    contamination: dict[str, Any],
    *,
    cache: dict[str, dict[str, Any]],
    cache_updates: dict[str, dict[str, Any]],
    inventory_only: bool,
    max_verify: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_root = ensure_source_root(source)
    pairs = mdn_pairs(source_root)
    units: list[dict[str, Any]] = []
    verified_attempts = 0
    for starter_dir, finished_dir, guide_path in pairs:
        target_files = read_tree_files(finished_dir, TARGET_EXTENSIONS)
        starter_files = read_tree_files(starter_dir, TARGET_EXTENSIONS)
        if not target_files or not starter_files:
            continue
        guide = read_optional(guide_path)
        visible = canonical_json({
            "instructions_and_marking_guide": guide,
            "starter_files": starter_files,
        })
        target = canonical_json({"files": target_files})
        source_task_id = f"{source['repo']}:{finished_dir.relative_to(source_root).as_posix()}"
        unit = base_unit(
            config,
            source=source,
            source_task_id=source_task_id,
            arm_id="html_css",
            task_family="html_css_assessment_completion",
            visible_context=visible,
            target=target,
            license_spdx=str(source["license_spdx"]),
            provenance={
                "repo": source["repo"],
                "revision": source["revision"],
                "starter_path": starter_dir.relative_to(source_root).as_posix(),
                "finished_path": finished_dir.relative_to(source_root).as_posix(),
                "marking_guide_sha256": sha256_text(guide),
                "archive_sha256": source["archive_sha256"],
                "static_open_corpus": True,
                "live_teacher_call": False,
            },
            contamination=contamination,
        )
        verification_digest = stable_hash({
            "unit_id": unit["unit_id"],
            "target": target_files,
            "starter": starter_files,
            "toolchain": config["toolchains"]["html_css"],
        })
        cached = cache.get(unit["unit_id"])
        should_verify = not inventory_only and (max_verify == 0 or verified_attempts < max_verify)
        if cached and cached.get("verification_digest") == verification_digest:
            verifier = cached["verification"]
        elif should_verify:
            verified_attempts += 1
            verifier = verify_html_css(starter_files, target_files, config["toolchains"]["html_css"])
            cache_updates[unit["unit_id"]] = {
                "unit_id": unit["unit_id"],
                "verification_digest": verification_digest,
                "verification": verifier,
            }
        else:
            verifier = {
                "kind": "html_css_task_complete_verifier_v1",
                "strength": "dom_a11y_layout_render_delta",
                "state": "not_run",
                "reason": "inventory_only_or_bounded_verification",
            }
        finish_unit(unit, verifier, config)
        units.append(unit)
    return units, summarize_source(source, units, extra={
        "source_root": rel(source_root),
        "matched_assessment_pair_count": len(pairs),
        "new_verification_attempt_count": verified_attempts,
    })


def python_function_hole_units(
    config: dict[str, Any],
    source: dict[str, Any],
    contamination: dict[str, Any],
    *,
    cache: dict[str, dict[str, Any]],
    cache_updates: dict[str, dict[str, Any]],
    cache_path: Path,
    prepare_python_projects: bool,
    inventory_only: bool,
    max_verify: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build independently test-killed implementation-hole units from a pinned repo."""

    source_root = ensure_source_root(source)
    locked_environment = ensure_locked_python_environment(
        source_root, source, prepare=prepare_python_projects
    )
    toolchain = resolve_python_project_toolchain(
        source, source_root=source_root, locked_environment=locked_environment
    )
    holes = discover_python_function_holes(source_root, source)
    rows: list[tuple[dict[str, Any], dict[str, Any], str, dict[str, Any] | None]] = []
    for hole in holes:
        visible = canonical_json({
            "instruction": "Implement the missing Python function body without changing its public contract.",
            "repository": source["repo"],
            "path": hole["path"],
            "qualified_name": hole["qualified_name"],
            "source_with_hole": hole["visible_source"],
        })
        target = canonical_json({
            "path": hole["path"],
            "start_line": hole["start_line"],
            "end_line": hole["end_line"],
            "replacement": hole["target_body"],
        })
        unit = base_unit(
            config,
            source=source,
            # Every hole in one file stays in one split; adjacent implementations
            # cannot leak across train/development/confirmation.
            source_task_id=f"{source['repo']}:{hole['path']}",
            arm_id="python",
            task_family="repository_function_implementation_hole",
            visible_context=visible,
            target=target,
            license_spdx=str(source["license_spdx"]),
            provenance={
                "repo": source["repo"],
                "revision": source.get("revision"),
                "archive_sha256": source["archive_sha256"],
                "path": hole["path"],
                "qualified_name": hole["qualified_name"],
                "target_file_sha256": hole["file_sha256"],
                "test_suite_paths": list(source["test_paths"]),
                "static_open_corpus": True,
                "live_teacher_call": False,
            },
            contamination=contamination,
        )
        verification_digest = stable_hash({
            "unit_id": unit["unit_id"],
            "archive_sha256": source["archive_sha256"],
            "target_body": hole["target_body"],
            "starter_body": hole["starter_body"],
            "toolchain": toolchain,
        })
        cached = cache.get(unit["unit_id"])
        rows.append((unit, hole, verification_digest, cached))

    verified_attempts = 0
    baseline_run: dict[str, Any] | None = None
    baseline_receipt: dict[str, Any] | None = None
    work_context: tempfile.TemporaryDirectory[str] | None = None
    workdir: Path | None = None
    checkpoint_interval = max(1, int(source.get("verification_checkpoint_interval") or 25))
    checkpoint_write_count = 0
    try:
        for unit, hole, verification_digest, cached in rows:
            if cached and cached.get("verification_digest") == verification_digest:
                verifier = cached["verification"]
            elif inventory_only or (max_verify and verified_attempts >= max_verify):
                verifier = {
                    "kind": "python_repository_test_killed_function_hole_v1",
                    "strength": "executable_target_pass_starter_fail",
                    "state": "not_run",
                    "reason": "inventory_only_or_bounded_verification",
                }
            else:
                if workdir is None:
                    work_context = tempfile.TemporaryDirectory(
                        prefix="th-py-unit-", dir="/tmp" if platform.system() == "Darwin" else None
                    )
                    workdir = Path(work_context.name).resolve()
                    shutil.copytree(
                        source_root,
                        workdir,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(
                            ".git", ".pytest_cache", ".venv", ".uv-cache", "__pycache__",
                            "node_modules", "target"
                        ),
                    )
                    baseline_run = run_python_project_tests(workdir, toolchain)
                    baseline_receipt = public_run_receipt(baseline_run)
                verified_attempts += 1
                verifier = verify_python_function_hole(
                    workdir,
                    hole,
                    toolchain,
                    baseline_run=baseline_run or {},
                    baseline_receipt=baseline_receipt or {},
                )
                cache_row = {
                    "unit_id": unit["unit_id"],
                    "verification_digest": verification_digest,
                    "verification": verifier,
                }
                cache[unit["unit_id"]] = cache_row
                cache_updates[unit["unit_id"]] = cache_row
                if verified_attempts % checkpoint_interval == 0:
                    write_cache(cache_path, cache)
                    checkpoint_write_count += 1
            finish_unit(unit, verifier, config)
        if verified_attempts and verified_attempts % checkpoint_interval:
            write_cache(cache_path, cache)
            checkpoint_write_count += 1
    finally:
        if work_context is not None:
            work_context.cleanup()

    cached_target_results = [
        bool(cached["verification"].get("target_passed"))
        for _unit, _hole, verification_digest, cached in rows
        if cached
        and cached.get("verification_digest") == verification_digest
        and isinstance(cached.get("verification"), dict)
        and cached["verification"].get("target_passed") is not None
    ]
    if baseline_run is not None:
        baseline_target_passed: bool | None = bool(baseline_run.get("ok"))
        baseline_target_evidence = "executed"
    elif cached_target_results:
        baseline_target_passed = all(cached_target_results)
        baseline_target_evidence = "content_bound_verification_cache"
    else:
        baseline_target_passed = None
        baseline_target_evidence = "not_run"

    units = [row[0] for row in rows]
    return units, summarize_source(source, units, extra={
        "source_root": rel(source_root),
        "function_hole_candidate_count": len(holes),
        "new_verification_attempt_count": verified_attempts,
        "verification_checkpoint_interval": checkpoint_interval,
        "verification_checkpoint_write_count": checkpoint_write_count,
        "baseline_target_passed": baseline_target_passed,
        "baseline_target_evidence": baseline_target_evidence,
        "locked_environment": locked_environment,
        "toolchain": toolchain,
    })


def discover_python_function_holes(
    source_root: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    minimum_body = int(source.get("minimum_body_bytes") or 8)
    maximum_body = int(source.get("maximum_body_bytes") or 16384)
    context_lines = int(source.get("context_lines") or 40)
    paths: set[Path] = set()
    for pattern in source["source_globs"]:
        paths.update(path for path in source_root.glob(str(pattern)) if path.is_file())
    holes: list[dict[str, Any]] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        lines = text.splitlines(keepends=True)
        relative = path.relative_to(source_root).as_posix()
        for qualified_name, node in top_level_python_functions(tree):
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    body = body[1:]
            if not body or body[0].lineno <= node.lineno:
                continue
            first, last = body[0], body[-1]
            if last.end_lineno is None:
                continue
            start_index = int(first.lineno) - 1
            end_index = int(last.end_lineno)
            target_body = "".join(lines[start_index:end_index])
            body_bytes = len(target_body.encode("utf-8"))
            if not minimum_body <= body_bytes <= maximum_body:
                continue
            indentation_match = re.match(r"[ \t]*", lines[start_index])
            indentation = indentation_match.group(0) if indentation_match else "    "
            newline = "\r\n" if lines[start_index].endswith("\r\n") else "\n"
            starter_body = (
                f'{indentation}raise NotImplementedError("Theseus task-complete implementation hole")'
                f"{newline}"
            )
            starter_file = "".join(lines[:start_index]) + starter_body + "".join(lines[end_index:])
            try:
                compile(starter_file, relative, "exec")
            except SyntaxError:
                continue
            before = "".join(lines[max(0, start_index - context_lines):start_index])
            after = "".join(lines[end_index:min(len(lines), end_index + context_lines)])
            holes.append({
                "path": relative,
                "qualified_name": qualified_name,
                "start_line": start_index + 1,
                "end_line": end_index,
                "target_body": target_body,
                "starter_body": starter_body,
                "visible_source": f"{before}<THESEUS_IMPLEMENTATION_HOLE>\n{after}",
                "file_sha256": sha256_text(text),
            })
    return holes


def top_level_python_functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    rows: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def visit(statements: list[ast.stmt], prefix: str = "") -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append((f"{prefix}{node.name}", node))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(tree.body)
    return rows


def ensure_locked_python_environment(
    source_root: Path, source: dict[str, Any], *, prepare: bool
) -> dict[str, Any] | None:
    policy = source.get("locked_python_environment")
    if not isinstance(policy, dict):
        return None
    if policy.get("manager") != "uv":
        raise ValueError(f"unsupported Python environment manager for {source['id']}")
    lock_path = (source_root / str(policy["lock_file"])).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"Python dependency lock missing: {lock_path}")
    lock_sha256 = file_sha256(lock_path)
    if lock_sha256 != str(policy["lock_sha256"]):
        raise ValueError(f"Python dependency lock identity mismatch: {source['id']}")

    base_python, base_version = select_python_executable(
        source.get("base_python_executable_candidates")
        or source.get("python_executable_candidates")
        or [],
        minimum_version=str(source.get("minimum_python_version") or "3.10"),
        source_root=source_root,
    )
    uv_path = resolve(str(policy["uv_executable"]))
    uv_bootstrapped_with_network = False
    if not uv_path.is_file() and prepare:
        bootstrap_root = uv_path.parents[1]
        expected_path = bootstrap_root / "bin" / "uv"
        if expected_path != uv_path:
            raise ValueError("configured uv executable must use a local venv bin/uv layout")
        bootstrap_root.parent.mkdir(parents=True, exist_ok=True)
        venv_run = subprocess.run(
            [str(base_python), "-m", "venv", str(bootstrap_root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
        if venv_run.returncode:
            raise RuntimeError(f"uv bootstrap venv failed: {venv_run.stderr[-1000:]}")
        install_run = subprocess.run(
            [
                str(bootstrap_root / "bin" / "python"), "-m", "pip", "install",
                "--disable-pip-version-check", f"uv=={policy['uv_version']}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            check=False,
        )
        if install_run.returncode:
            raise RuntimeError(f"uv bootstrap install failed: {install_run.stderr[-1000:]}")
        uv_bootstrapped_with_network = True
    if not uv_path.is_file():
        raise FileNotFoundError(
            f"uv verifier manager missing: {uv_path}; rerun with --prepare-python-projects"
        )
    version_probe = subprocess.run(
        [str(uv_path), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    uv_version_output = version_probe.stdout.strip()
    uv_release = uv_version_output.split()[1] if len(uv_version_output.split()) >= 2 else ""
    uv_version = f"uv {uv_release}" if uv_release else uv_version_output
    if version_probe.returncode or uv_release != str(policy["uv_version"]):
        raise RuntimeError(
            f"uv version mismatch for {source['id']}: expected uv {policy['uv_version']}, "
            f"observed {uv_version_output or version_probe.stderr.strip()}"
        )

    environment_dir = (source_root / str(policy.get("environment_dir") or ".venv")).resolve()
    uv_cache = (source_root / str(policy.get("cache_dir") or ".uv-cache")).resolve()
    groups = [str(value) for value in policy.get("groups") or []]
    sync_command = [
        str(uv_path), "sync", "--locked", "--no-default-groups",
        *[item for group in groups for item in ("--group", group)],
        "--python", str(base_python), "--no-python-downloads",
    ]
    sync_environment = {
        "UV_PROJECT_ENVIRONMENT": str(environment_dir),
        "UV_CACHE_DIR": str(uv_cache),
    }
    prepared_with_network = False
    if prepare:
        completed = subprocess.run(
            sync_command,
            cwd=source_root,
            env={**os.environ, **sync_environment},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(policy.get("prepare_timeout_seconds") or 600),
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"locked Python environment preparation failed: {completed.stderr[-1000:]}")
        prepared_with_network = True

    offline_replay = run_sandboxed(
        [*sync_command, "--offline"],
        source_root,
        int(policy.get("offline_replay_timeout_seconds") or 300),
        extra_env=sync_environment,
    )
    if not offline_replay.get("ok"):
        raise RuntimeError(
            "locked Python environment is not offline-replayable; run with "
            f"--prepare-python-projects: {offline_replay.get('stderr') or offline_replay.get('fault')}"
        )
    environment_python = environment_dir / "bin" / "python"
    if not environment_python.is_file():
        raise FileNotFoundError(f"locked Python interpreter missing: {environment_python}")
    freeze = run_sandboxed(
        [str(uv_path), "pip", "freeze", "--python", str(environment_python)],
        source_root,
        60,
        extra_env=sync_environment,
    )
    if not freeze.get("ok"):
        raise RuntimeError(f"locked Python environment inventory failed: {freeze.get('stderr')}")
    normalized_freeze = "\n".join(
        line.replace(str(source_root), "{SOURCE_ROOT}")
        for line in str(freeze.get("stdout") or "").splitlines()
        if line.strip()
    )
    identity = {
        "manager": "uv",
        "uv_version": uv_version,
        "uv_executable_sha256": file_sha256(uv_path),
        "lock_file": str(policy["lock_file"]),
        "lock_sha256": lock_sha256,
        "groups": groups,
        "base_python_version": base_version,
        "installed_package_count": len(normalized_freeze.splitlines()),
        "installed_packages_sha256": sha256_text(normalized_freeze),
        "offline_replay_valid": True,
        "network_during_verification": "denied",
    }
    return {
        **identity,
        "identity_sha256": stable_hash(identity),
        "prepared_with_network_this_run": prepared_with_network,
        "uv_bootstrapped_with_network_this_run": uv_bootstrapped_with_network,
        "offline_replay": public_run_receipt(offline_replay),
    }


def select_python_executable(
    candidates: Iterable[Any], *, minimum_version: str, source_root: Path
) -> tuple[Path, str]:
    minimum = tuple(int(value) for value in minimum_version.split("."))
    for name in candidates:
        expanded = str(name).replace("{source_root}", str(source_root))
        candidate = shutil.which(expanded)
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "import platform; print(platform.python_version())"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            continue
        observed = tuple(int(value) for value in probe.stdout.strip().split(".")[:2])
        if observed >= minimum:
            # Preserve a virtual-environment launcher path. Resolving its symlink
            # would execute the base interpreter without the venv's sys.prefix.
            return Path(candidate).absolute(), probe.stdout.strip()
    raise RuntimeError(f"no Python interpreter satisfies {minimum_version}")


def resolve_python_project_toolchain(
    source: dict[str, Any], *, source_root: Path, locked_environment: dict[str, Any] | None
) -> dict[str, Any]:
    selected, version = select_python_executable(
        source["python_executable_candidates"],
        minimum_version=str(source.get("minimum_python_version") or "3.10"),
        source_root=source_root,
    )
    command = [
        str(selected) if value == "{python}" else str(value)
        for value in source["test_command"]
    ]
    toolchain = {
        "python_executable": str(selected),
        "python_executable_sha256": file_sha256(selected),
        "python_version": version,
        "command": command,
        "environment": dict(source.get("environment") or {}),
        "network_during_verification": "denied",
        "timeout_seconds": int(source["timeout_seconds"]),
    }
    if locked_environment:
        toolchain["locked_environment_identity"] = {
            key: value for key, value in locked_environment.items()
            if key not in {
                "prepared_with_network_this_run",
                "uv_bootstrapped_with_network_this_run",
                "offline_replay",
            }
        }
    return toolchain


def run_python_project_tests(workdir: Path, toolchain: dict[str, Any]) -> dict[str, Any]:
    environment = {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        **{str(key): str(value) for key, value in toolchain["environment"].items()},
    }
    return run_sandboxed(
        list(toolchain["command"]),
        workdir,
        int(toolchain["timeout_seconds"]),
        extra_env=environment,
    )


def verify_python_function_hole(
    workdir: Path,
    hole: dict[str, Any],
    toolchain: dict[str, Any],
    *,
    baseline_run: dict[str, Any],
    baseline_receipt: dict[str, Any],
) -> dict[str, Any]:
    path = workdir / hole["path"]
    target_text = path.read_text(encoding="utf-8")
    lines = target_text.splitlines(keepends=True)
    start_index = int(hole["start_line"]) - 1
    end_index = int(hole["end_line"])
    observed_body = "".join(lines[start_index:end_index])
    if observed_body != hole["target_body"]:
        return {
            "kind": "python_repository_test_killed_function_hole_v1",
            "strength": "executable_target_pass_starter_fail",
            "state": "failed",
            "reason": "target_span_identity_changed",
            "target_passed": bool(baseline_run.get("ok")),
            "starter_failed": False,
            "target_run": baseline_receipt,
            "toolchain": toolchain,
        }
    starter_text = "".join(lines[:start_index]) + hole["starter_body"] + "".join(lines[end_index:])
    try:
        path.write_text(starter_text, encoding="utf-8")
        starter_run = run_python_project_tests(workdir, toolchain)
    finally:
        path.write_text(target_text, encoding="utf-8")
    starter_failed = bool(
        not starter_run.get("ok")
        and starter_run.get("returncode") is not None
        and not starter_run.get("fault")
    )
    passed = bool(baseline_run.get("ok") and starter_failed)
    return {
        "kind": "python_repository_test_killed_function_hole_v1",
        "strength": "executable_target_pass_starter_fail",
        "state": "passed" if passed else "failed",
        "target_passed": bool(baseline_run.get("ok")),
        "starter_failed": starter_failed,
        "target_run": baseline_receipt,
        "starter_run": public_run_receipt(starter_run),
        "toolchain": toolchain,
    }


def base_unit(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    source_task_id: str,
    arm_id: str,
    task_family: str,
    visible_context: str,
    target: str,
    license_spdx: str,
    provenance: dict[str, Any],
    contamination: dict[str, Any],
    precomputed_contamination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split = split_for(config, source_task_id)
    normalized_license = normalize_license(license_spdx)
    if precomputed_contamination:
        exact = bool(precomputed_contamination.get("exact_overlap"))
        semantic_count = int(precomputed_contamination.get("semantic_match_count") or 0)
        semantic_max = float(precomputed_contamination.get("semantic_max_jaccard") or 0.0)
    else:
        exact = any(
            training_data_lineage_audit.text_sha256(training_data_lineage_audit.normalize_text(text))
            in contamination["exact_hashes"]
            for text in (visible_context, target)
        )
        semantic_count, semantic_max = training_data_lineage_audit.semantic_overlap(
            [visible_context, target],
            contamination,
            threshold=float(config["contamination"]["semantic_jaccard_threshold"]),
        )
    quarantine = bool(
        exact
        or semantic_count >= int(config["contamination"]["semantic_match_count_for_quarantine"])
        or semantic_max >= float(config["contamination"]["single_semantic_match_max_for_quarantine"])
    )
    identity_payload = {
        "source_id": source["id"],
        "source_task_id": source_task_id,
        "arm_id": arm_id,
        "visible_context_sha256": sha256_text(visible_context),
        "target_sha256": sha256_text(target),
    }
    return {
        "policy": config["unit_abi"],
        "unit_id": stable_id("task-complete-unit", identity_payload),
        "source_id": source["id"],
        "source_task_id": source_task_id,
        "arm_id": arm_id,
        "task_family": task_family,
        "split": split,
        "visible_context": visible_context,
        "visible_context_sha256": identity_payload["visible_context_sha256"],
        "target": target,
        "target_sha256": identity_payload["target_sha256"],
        "visible_context_positions": len(visible_context.encode("utf-8")),
        "target_positions": len(target.encode("utf-8")),
        "license_spdx": normalized_license,
        "provenance": provenance,
        "contamination": {
            "public_index_digest": contamination["digest"],
            "exact_overlap": exact,
            "semantic_overlap": bool(semantic_count),
            "semantic_match_count": semantic_count,
            "semantic_max_jaccard": round(float(semantic_max), 6),
            "quarantine": quarantine,
        },
        "public_benchmark_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def finish_unit(unit: dict[str, Any], verifier: dict[str, Any], config: dict[str, Any]) -> None:
    quality = config["quality_policy"]
    reasons: list[str] = []
    if unit["license_spdx"] not in ALLOWED_LICENSES:
        reasons.append("license_not_allowlisted")
    if unit["visible_context_positions"] < int(quality["minimum_visible_context_bytes"]):
        reasons.append("visible_context_too_small")
    if unit["target_positions"] < int(quality["minimum_target_bytes"]):
        reasons.append("target_too_small")
    if unit["visible_context_positions"] + unit["target_positions"] > int(quality["maximum_unit_bytes"]):
        reasons.append("unit_too_large")
    if unit["contamination"]["quarantine"]:
        reasons.append("public_contamination_overlap")
    if verifier.get("state") != "passed":
        reasons.append(f"verifier_{verifier.get('state', 'missing')}")
    unit["verification"] = verifier
    unit["decision_reasons"] = reasons
    unit["decision"] = "quarantine" if unit["contamination"]["quarantine"] else ("admit" if not reasons else "reject")
    unit["task_complete_verified"] = unit["decision"] == "admit"
    unit["receipt_id"] = stable_id("task-complete-receipt", unit["unit_id"], unit["decision"], verifier)


def verify_exercism(
    source_root: Path,
    exercise_dir: Path,
    language: str,
    target_files: dict[str, str],
    test_paths: list[str],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    parent = source_root if language == "javascript_typescript" else None
    with tempfile.TemporaryDirectory(prefix="theseus-task-unit-", dir=parent) as raw:
        workdir = Path(raw).resolve()
        shutil.copytree(exercise_dir, workdir, dirs_exist_ok=True)
        enable_all_tests(workdir, test_paths, language)
        starter_run = run_exercise(workdir, source_root, language, test_paths, toolchain)
        for relative, text in target_files.items():
            path = workdir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        target_run = run_exercise(workdir, source_root, language, test_paths, toolchain)
    passed = bool(target_run.get("ok") and not starter_run.get("ok"))
    return {
        "kind": f"{language}_task_complete_verifier_v1",
        "strength": "executable_target_pass_starter_fail",
        "state": "passed" if passed else "failed",
        "target_passed": bool(target_run.get("ok")),
        "starter_failed": not bool(starter_run.get("ok")),
        "target_run": public_run_receipt(target_run),
        "starter_run": public_run_receipt(starter_run),
        "toolchain": toolchain,
    }


def run_exercise(
    workdir: Path,
    source_root: Path,
    language: str,
    test_paths: list[str],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    timeout = int(toolchain["timeout_seconds"])
    if language == "python":
        return run_sandboxed(
            [sys.executable, "-I", "-m", "unittest", "discover", "-p", "*_test.py"],
            workdir,
            timeout,
        )
    if language == "javascript_typescript":
        jest = source_root / "node_modules" / ".bin" / "jest"
        if not jest.exists():
            return {"ok": False, "fault": "javascript_dependencies_not_prepared"}
        return run_sandboxed(
            [str(jest), *test_paths, "--runInBand", "--no-cache"],
            workdir,
            timeout,
            extra_env={"BABEL_ENV": "test", "NODE_ENV": "test"},
        )
    if language == "rust":
        return run_sandboxed(
            [shutil.which("cargo") or "cargo", "test", "--offline", "--quiet"],
            workdir,
            timeout,
            extra_env={"CARGO_TARGET_DIR": str(workdir / "target")},
        )
    return {"ok": False, "fault": "unsupported_exercism_language"}


def verify_html_css(
    starter_files: dict[str, str], target_files: dict[str, str], toolchain: dict[str, Any]
) -> dict[str, Any]:
    starter_signature = web_signature(starter_files)
    target_signature = web_signature(target_files)
    delta = signature_delta(starter_signature, target_signature)
    target_renders = render_web_bundle(target_files, int(toolchain["timeout_seconds"]))
    starter_renders = render_web_bundle(starter_files, int(toolchain["timeout_seconds"]))
    render_delta = bool(
        target_renders.get("ok")
        and starter_renders.get("ok")
        and target_renders.get("screenshot_sha256") != starter_renders.get("screenshot_sha256")
    )
    passed = bool(delta["target_specific_assertion_count"] > 0 and render_delta)
    return {
        "kind": "html_css_task_complete_verifier_v1",
        "strength": "dom_a11y_layout_render_delta",
        "state": "passed" if passed else "failed",
        "target_static_signature": target_signature,
        "starter_static_signature": starter_signature,
        "target_specific_assertions": delta,
        "target_render": target_renders,
        "starter_render": starter_renders,
        "target_rendered": bool(target_renders.get("ok")),
        "starter_rendered": bool(starter_renders.get("ok")),
        "render_delta": render_delta,
        "toolchain": toolchain,
    }


class DOMSignature(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: Counter[str] = Counter()
        self.attrs: Counter[str] = Counter()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags[tag] += 1
        for key, value in attrs:
            if key in {"role", "aria-label", "aria-labelledby", "alt", "for", "scope"}:
                self.attrs[f"{tag}:{key}:{value or ''}"] += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def web_signature(files: dict[str, str]) -> dict[str, Any]:
    parser = DOMSignature()
    css_properties: Counter[str] = Counter()
    media_queries = 0
    for path, text in files.items():
        if Path(path).suffix.lower() in {".html", ".htm"}:
            try:
                parser.feed(text)
            except Exception:  # noqa: BLE001
                pass
            inline = re.findall(r"<style[^>]*>(.*?)</style>", text, flags=re.IGNORECASE | re.DOTALL)
            for value in inline:
                css_properties.update(re.findall(r"(?:^|[;{])\s*([a-z-]+)\s*:", value, flags=re.IGNORECASE))
                media_queries += len(re.findall(r"@media\b", value, flags=re.IGNORECASE))
        elif Path(path).suffix.lower() == ".css":
            css_properties.update(re.findall(r"(?:^|[;{])\s*([a-z-]+)\s*:", text, flags=re.IGNORECASE))
            media_queries += len(re.findall(r"@media\b", text, flags=re.IGNORECASE))
    return {
        "tags": dict(sorted(parser.tags.items())),
        "accessibility_attrs": dict(sorted(parser.attrs.items())),
        "css_properties": dict(sorted(css_properties.items())),
        "media_query_count": media_queries,
    }


def signature_delta(starter: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    assertions: list[str] = []
    for section in ("tags", "accessibility_attrs", "css_properties"):
        for key, count in target[section].items():
            if int(count) > int(starter[section].get(key, 0)):
                assertions.append(f"{section}:{key}>={count}")
    if int(target["media_query_count"]) > int(starter["media_query_count"]):
        assertions.append(f"media_query_count>={target['media_query_count']}")
    return {
        "target_specific_assertion_count": len(assertions),
        "assertions": assertions[:200],
        "starter_fails_at_least_one": bool(assertions),
    }


def render_web_bundle(files: dict[str, str], timeout: int) -> dict[str, Any]:
    html_paths = sorted(path for path in files if Path(path).suffix.lower() in {".html", ".htm"})
    if not html_paths or not CHROME.exists():
        return {"ok": False, "fault": "html_or_chrome_unavailable"}
    with tempfile.TemporaryDirectory(prefix="theseus-web-unit-") as raw:
        workdir = Path(raw).resolve()
        for relative, text in files.items():
            path = workdir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        entry = workdir / ("index.html" if "index.html" in files else html_paths[0])
        renders = []
        hashes = []
        for width, height, label in ((800, 600, "wide"), (375, 667, "narrow")):
            screenshot = workdir / f"render-{label}.png"
            run = _render_chrome(
                [
                    str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--disable-background-networking", "--disable-crash-reporter",
                    "--disable-crashpad", "--disable-breakpad", "--no-first-run",
                    "--no-default-browser-check", "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=1000", f"--user-data-dir={workdir / ('chrome-' + label)}",
                    f"--window-size={width},{height}", f"--screenshot={screenshot}", entry.as_uri(),
                ],
                workdir,
                screenshot,
                timeout,
            )
            digest = file_sha256(screenshot) if screenshot.exists() else None
            hashes.append(digest)
            renders.append({
                "label": label,
                "ok": bool(run.get("ok")),
                "duration_ms": run.get("duration_ms"),
                "screenshot_sha256": digest,
                "screenshot_bytes": screenshot.stat().st_size if screenshot.exists() else 0,
                "fault": run.get("fault"),
            })
        return {
            "ok": all(row["ok"] for row in renders),
            "renders": renders,
            "screenshot_sha256": hashes,
        }


def run_sandboxed(
    command: list[str], workdir: Path, timeout: int, *, extra_env: dict[str, str] | None = None
) -> dict[str, Any]:
    workdir = workdir.resolve()
    profile = "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f'(deny file-write* (require-not (subpath "{workdir}")))',
        '(allow file-write* (literal "/dev/null"))',
    ])
    env = {
        key: value for key, value in os.environ.items()
        if key in {"HOME", "PATH", "TMPDIR", "NO_COLOR", "CARGO_HOME", "RUSTUP_HOME"}
    }
    env["TMPDIR"] = str(workdir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env or {})
    started = time.monotonic()
    launcher: list[str]
    if platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").is_file():
        launcher = ["/usr/bin/sandbox-exec", "-p", profile]
    elif platform.system() == "Linux" and shutil.which("bwrap"):
        launcher = [
            shutil.which("bwrap") or "bwrap",
            "--unshare-net", "--die-with-parent", "--new-session",
            "--ro-bind", "/", "/", "--bind", str(workdir), str(workdir),
            "--chdir", str(workdir), "--tmpfs", "/tmp",
        ]
    else:
        return {
            "ok": False,
            "fault": "fail_closed_sandbox_unavailable",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
    try:
        completed = subprocess.run(
            [*launcher, *command],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "fault": "timeout",
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }


def public_run_receipt(run: dict[str, Any]) -> dict[str, Any]:
    stdout = str(run.get("stdout") or "")
    stderr = str(run.get("stderr") or "")
    return {
        "ok": bool(run.get("ok")),
        "returncode": run.get("returncode"),
        "fault": run.get("fault"),
        "duration_ms": run.get("duration_ms"),
        "stdout_sha256": sha256_text(stdout),
        "stderr_sha256": sha256_text(stderr),
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def prepare_javascript_source(source_root: Path) -> None:
    node_modules = source_root / "node_modules"
    if node_modules.exists():
        return
    completed = subprocess.run(
        ["corepack", "pnpm", "install", "--frozen-lockfile"],
        cwd=source_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"pnpm install failed: {completed.stderr[-1000:]}")


def ensure_source_root(source: dict[str, Any]) -> Path:
    archive = resolve(source["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"source archive missing: {archive}")
    if file_sha256(archive) != source["archive_sha256"]:
        raise ValueError(f"source archive identity mismatch: {source['id']}")
    extracted_base = ROOT / "runtime" / "task_complete_sources" / "extracted"
    target = extracted_base / archive.name.removesuffix(".tar.gz")
    marker_payload = {
        "source_id": source["id"],
        "revision": source.get("revision"),
        "archive_sha256": source["archive_sha256"],
    }
    marker = target / ".theseus-source.json"
    if target.exists() and marker.is_file():
        observed = read_json(marker)
        if all(observed.get(key) == value for key, value in marker_payload.items()):
            return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, mode="r:gz") as handle:
        root_name = safe_extract_source_archive(handle, target)
    marker.write_text(canonical_json({**marker_payload, "archive_root": root_name}), encoding="utf-8")
    return target


def safe_extract_source_archive(handle: tarfile.TarFile, target: Path) -> str:
    members = handle.getmembers()
    if not members:
        raise ValueError("empty source archive")
    roots = {member.name.split("/", 1)[0] for member in members if member.name}
    if len(roots) != 1:
        raise ValueError("source archive must have one top-level directory")
    root_name = next(iter(roots))
    target_root = target.resolve()
    selected: list[tarfile.TarInfo] = []
    for member in members:
        parts = member.name.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise ValueError(f"unsupported archive member type: {member.name}")
        destination = (target / parts[1]).resolve()
        if destination != target_root and target_root not in destination.parents:
            raise ValueError(f"unsafe archive member: {member.name}")
        copied = copy.copy(member)
        copied.name = parts[1]
        selected.append(copied)
    handle.extractall(target, members=selected, filter="data")
    return root_name


def map_example_to_solution(
    exercise_dir: Path, solution_paths: list[str], example_paths: list[str]
) -> dict[str, str]:
    examples = read_relative_files(exercise_dir, example_paths)
    if len(examples) == len(solution_paths):
        return {
            solution: examples[example]
            for solution, example in zip(solution_paths, example_paths)
        }
    if len(examples) == 1 and len(solution_paths) == 1:
        return {solution_paths[0]: next(iter(examples.values()))}
    targets = read_relative_files(exercise_dir, solution_paths, allow_missing=True)
    mapped = 0
    for example_path, text in examples.items():
        example_name = Path(example_path).name.lower()
        candidate: str | None = None
        if "cargo-example" in example_name:
            candidate = next(
                (path for path in solution_paths if Path(path).name.lower() == "cargo.toml"),
                None,
            )
        elif example_name.startswith("example"):
            suffix = Path(example_path).suffix.lower()
            same_suffix = [path for path in solution_paths if Path(path).suffix.lower() == suffix]
            if len(same_suffix) == 1:
                candidate = same_suffix[0]
        if candidate is None:
            normalized = example_name.replace("example", "").replace("-", "").replace("_", "")
            matches = [
                path
                for path in solution_paths
                if Path(path).name.lower().replace("-", "").replace("_", "") == normalized
            ]
            if len(matches) == 1:
                candidate = matches[0]
        if candidate is not None:
            targets[candidate] = text
            mapped += 1
    if mapped != len(examples) or not any(targets.get(path) for path in solution_paths):
        return {}
    return targets


def enable_all_tests(workdir: Path, test_paths: list[str], language: str) -> None:
    for relative in test_paths:
        path = workdir / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if language == "javascript_typescript":
            text = re.sub(r"\bxtest\s*\(", "test(", text)
            text = re.sub(r"\bxdescribe\s*\(", "describe(", text)
            text = text.replace("test.skip(", "test(").replace("describe.skip(", "describe(")
        elif language == "rust":
            text = re.sub(r"#\s*\[\s*ignore(?:\s*=\s*[^\]]+)?\s*\]\s*", "", text)
        path.write_text(text, encoding="utf-8")


def mdn_pairs(root: Path) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for guide in sorted(root.rglob("marking-guide.md")):
        finished = guide.parent
        candidates: list[Path] = []
        path_text = finished.as_posix()
        replacements = (
            ("assessment-finished", "assessment-start"),
            ("-finished", "-start"),
            ("/finished", "/start"),
        )
        for old, new in replacements:
            if old in path_text:
                candidates.append(Path(path_text.replace(old, new)))
        candidates.extend(sorted(finished.parent.glob("*start*")))
        starter = next((path for path in candidates if path.is_dir()), None)
        if starter and starter != finished:
            pairs.append((starter, finished, guide))
    unique: dict[str, tuple[Path, Path, Path]] = {}
    for row in pairs:
        unique[row[1].as_posix()] = row
    return list(unique.values())


def coverage_summary(
    config: dict[str, Any], units: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    admitted = [row for row in units if row["decision"] == "admit"]
    floors = config["coverage_floors_for_50m_scale_proposal"]
    coverage: dict[str, Any] = {}
    gaps: list[dict[str, Any]] = []
    for arm_id, floor in floors.items():
        rows = [row for row in admitted if row["arm_id"] == arm_id]
        required_strength = floor.get("required_verification_strength")
        strength_rows = [row for row in rows if not required_strength or row["verification"]["strength"] == required_strength]
        positions = sum(int(row["target_positions"]) for row in strength_rows)
        observed: dict[str, Any] = {
            "verified_units": len(strength_rows),
            "target_positions": positions,
            "required_verification_strength": required_strength,
            "minimum_verified_units": int(floor["minimum_verified_units"]),
            "minimum_target_positions": int(floor["minimum_target_positions"]),
        }
        checks = {
            "unit_floor": len(strength_rows) >= int(floor["minimum_verified_units"]),
            "position_floor": positions >= int(floor["minimum_target_positions"]),
        }
        if arm_id == "english":
            human = sum(row["provenance"].get("provenance_class") == "human_contributed" for row in rows)
            multi = sum(bool(row["verification"].get("multi_turn")) for row in rows)
            human_share = human / len(rows) if rows else 0.0
            multi_share = multi / len(rows) if rows else 0.0
            observed.update(human_contributed_share=round(human_share, 6), multi_turn_share=round(multi_share, 6))
            checks.update(
                human_share=human_share >= float(floor["minimum_human_contributed_share"]),
                multi_turn_share=multi_share >= float(floor["minimum_multi_turn_share"]),
            )
        observed["checks"] = checks
        observed["ready"] = all(checks.values())
        coverage[arm_id] = observed
        if not observed["ready"]:
            gaps.append({"arm_id": arm_id, "failed_checks": [key for key, value in checks.items() if not value], "observed": observed})
    return coverage, gaps


def split_for(config: dict[str, Any], source_task_id: str) -> str:
    value = int(hashlib.sha256(f"{config['seed']}:{source_task_id}".encode()).hexdigest()[:16], 16) % 10000
    split = config["split_policy"]
    train_end = int(split["train_basis_points"])
    dev_end = train_end + int(split["development_basis_points"])
    if value < train_end:
        return "train"
    if value < dev_end:
        return "development"
    return "confirmation"


def split_leakage(units: list[dict[str, Any]]) -> list[str]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in units:
        by_source[row["source_task_id"]].add(row["split"])
    return [source_id for source_id, splits in by_source.items() if len(splits) > 1]


def english_family(messages: list[dict[str, str]]) -> str:
    text = " ".join(message["content"] for message in messages).lower()
    if any(token in text for token in ("edit", "correct", "rewrite", "revise")):
        return "correction_following"
    if len(messages) > 1:
        return "multi_turn_continuity"
    if any(token in text for token in ("clarify", "ambiguous", "which do you mean")):
        return "ambiguity_clarification"
    if any(token in text for token in ("code", "python", "javascript", "rust", "html", "css")):
        return "technical_instruction"
    return "general_instruction_following"


def summarize_source(source: dict[str, Any], units: list[dict[str, Any]], *, extra: dict[str, Any]) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in units)
    verification = Counter(row["verification"]["state"] for row in units)
    return {
        "source_id": source["id"],
        "adapter": source["adapter"],
        "repo": source.get("repo"),
        "revision": source.get("revision"),
        "license_spdx": source.get("license_spdx"),
        "unit_count": len(units),
        "decision_counts": dict(sorted(decisions.items())),
        "verification_state_counts": dict(sorted(verification.items())),
        "target_positions": sum(int(row["target_positions"]) for row in units if row["decision"] == "admit"),
        **extra,
    }


def write_unit_ledger(path: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in units:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)
    replay_count = 0
    replay_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            replay_count += 1
            replay_ids.add(row["unit_id"])
    return {
        "path": rel(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "count": replay_count,
        "unique_identity_count": len(replay_ids),
        "replay_valid": replay_count == len(units) == len(replay_ids),
    }


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("unit_id"):
            rows[str(row["unit_id"])] = row
    return rows


def load_prior_unit_cache(
    ledger_path: Path, report_path: Path, *, contamination_digest: str
) -> dict[str, dict[str, Any]]:
    """Reuse only hash-bound row audits against the same public index."""

    if not ledger_path.is_file() or not report_path.is_file():
        return {}
    prior_report = read_json(report_path)
    receipt = prior_report.get("ledger_receipt")
    if not isinstance(receipt, dict):
        return {}
    if (
        receipt.get("replay_valid") is not True
        or str(receipt.get("sha256") or "") != file_sha256(ledger_path)
        or int(receipt.get("count") or 0) != int(receipt.get("unique_identity_count") or -1)
    ):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(ledger_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            contamination = row.get("contamination") if isinstance(row.get("contamination"), dict) else {}
            if (
                row.get("unit_id")
                and contamination.get("public_index_digest") == contamination_digest
                and int(row.get("public_benchmark_training_rows") or 0) == 0
                and int(row.get("external_inference_calls") or 0) == 0
                and int(row.get("fallback_return_count") or 0) == 0
            ):
                rows[str(row["unit_id"])] = row
    return rows


def write_cache(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(canonical_json(rows[key]) + "\n")
    os.replace(temporary, path)


def read_relative_files(base: Path, paths: Iterable[str], *, allow_missing: bool = False) -> dict[str, str]:
    rows: dict[str, str] = {}
    for relative in paths:
        path = base / relative
        if not path.is_file():
            if allow_missing:
                rows[relative] = ""
                continue
            return {}
        rows[relative] = path.read_text(encoding="utf-8", errors="replace")
    return rows


def read_tree_files(base: Path, extensions: set[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions and path.stat().st_size <= 262144:
            rows[path.relative_to(base).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:24]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_license(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
