#!/usr/bin/env python3
"""Join the exact P4 production path before any candidate model call.

This is an evaluator-only mechanics and addressability gate. It renders the
frozen prompts and replays frozen transport oracles through the exact parser,
lowerer, disposable apply sandbox, and visible verifier used by the campaign.
It never opens candidate inference or the hidden evaluator.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_exact_prompt_token_count as token_count
import theseus_p4_cognitive_compilation as p4
import theseus_p4s_cognitive_compilation as p4s
import theseus_p4v2r2_cognitive_compilation as causal
import theseus_p4v2r2r4_cognitive_compilation as release
import theseus_semantic_ir_v2r2 as ir_v2r2


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4v2r2r3_production_conformance_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_p4v2r2r3_production_conformance.json"
INTERRUPTED_LEASE = ROOT / "reports" / "theseus_p4v2r2r4_campaign_leases" / "8babf90cc4da4dbd9f9185282766c3f2.json"
INTERRUPTED_LEASE_SHA256 = "459a7f9eac01ce544574f36c044a98f91caf28b3d468aefdea7d6a9ba5eae7e2"


def typed_edit(actions: list[dict[str, Any]]) -> str:
    lines = [p2a.ACTION_HEADER]
    for action in actions:
        lines.extend(
            [
                f"REPLACE {action['path']} {action['start_line']} {action['end_line']}",
                "<<<",
                str(action.get("replacement") or ""),
                ">>>",
            ]
        )
    lines.append("END")
    return "\n".join(lines)


def plan_edit(actions: list[dict[str, Any]]) -> str:
    return "THESEUS_PLAN_V1\nPLAN\nReplay the evaluator-only oracle.\nTARGET\n" + typed_edit(actions)


def delimiter_variants(oracle: str) -> list[str]:
    variants: list[str] = []
    for style in ("whitespace", "bracket", "quoted_bracket"):
        lines: list[str] = []
        for line in oracle.splitlines():
            if line.startswith(("ALL_OBLIGATIONS ", "OBLIGATIONS ")):
                field, value = line.split(" ", 1)
                ids = value.split(",")
                if style == "whitespace":
                    value = " ".join(ids)
                elif style == "bracket":
                    value = "[" + ", ".join(ids) + "]"
                else:
                    value = "[" + ", ".join(repr(item) for item in ids) + "]"
                line = f"{field} {value}"
            lines.append(line)
        variants.append("\n".join(lines))
    return variants


def count_exact_prompts(prompts: dict[str, str], worker: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=ROOT / "runtime" / "control", delete=False
    ) as handle:
        path = Path(handle.name)
        json.dump({"prompts": prompts}, handle, sort_keys=True)
        handle.write("\n")
    try:
        return token_count.count_prompts(worker, path)
    finally:
        path.unlink(missing_ok=True)


def run_conformance(*, include_tokenizer: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    faults: list[str] = []
    pool = p2a.read_json(POOL)
    overlay = p2a.read_json(INSTRUMENT)
    projected = release.projected_instrument(overlay)
    base_local = p2a.read_json(
        p2a.resolve(str(projected.get("base_local_instrument") or ""))
    )
    protocol = p2a.mapping(base_local.get("candidate_protocol"))
    worker = p2a.resolve(
        str(p2a.mapping(base_local.get("runtime_binding")).get("worker_config") or "")
    )
    prompts: dict[str, str] = {}
    task_receipts: list[dict[str, Any]] = []
    operation_counts = {"REPLACE": 0, "INSERT_BEFORE": 0, "INSERT_AFTER": 0}
    delimiter_cases = 0
    malformed_cases = 0
    interrupted_lease = (
        p2a.read_json(INTERRUPTED_LEASE) if INTERRUPTED_LEASE.is_file() else {}
    )
    preexisting_runtime_receipts = sorted(
        (ROOT / "runtime" / "p2a").glob("*p4v2r2r4_attempt1*.json")
    )
    preexisting_call_starts = sorted(release.CALL_START_DIRECTORY.glob("*.json"))
    preexisting_runs = sorted(
        (ROOT / "reports").glob("theseus_p4v2r2r3_attempt1_*_run.json")
    )
    preexisting_evaluations = sorted(
        (ROOT / "reports").glob("theseus_p4v2r2r3_attempt1_*_evaluation.json")
    )
    if (
        p2a.sha256_file(INTERRUPTED_LEASE) != INTERRUPTED_LEASE_SHA256
        or interrupted_lease.get("state") != "STOPPED_RETAIN_EVIDENCE"
        or interrupted_lease.get("scientific_status") != "NO_DURABLE_OBSERVATION"
        or int(interrupted_lease.get("durable_runtime_receipts") or 0) != 0
        or interrupted_lease.get("durable_campaign_progress_written") is not False
        or preexisting_runtime_receipts
        or preexisting_call_starts
        or preexisting_runs
        or preexisting_evaluations
    ):
        faults.append("interrupted_launch_reuse_custody_invalid")
    original_symbol_table = p4.semantic_symbol_table
    original_verifier = p2a.run_visible_verifier
    try:
        p4.semantic_symbol_table = p4s.semantic_scope_symbol_table
        p2a.run_visible_verifier = release.run_complete_visible_verifier
        for row in p2a.dicts(pool.get("tasks")):
            stem = str(row.get("stem") or "")
            task_path = ROOT / str(row.get("task") or "")
            oracle_path = ROOT / str(row.get("treatment_transport_oracle_ir") or "")
            if p2a.sha256_file(task_path) != str(row.get("task_sha256") or ""):
                faults.append(f"task_binding_invalid:{stem}")
            if p2a.sha256_file(oracle_path) != str(
                row.get("treatment_transport_oracle_ir_sha256") or ""
            ):
                faults.append(f"oracle_binding_invalid:{stem}")
            task = p2a.read_json(task_path)
            oracle = oracle_path.read_text(encoding="utf-8")
            with tempfile.TemporaryDirectory(prefix=f"theseus-conformance-{stem}-") as tmp:
                root = Path(tmp) / "source"
                p2a.extract_source_archive(
                    p2a.resolve(str(task.get("source_archive") or "")),
                    root,
                    str(task.get("source_archive_root") or ""),
                )
                baseline = p2a.inventory(root)
                common = p4.render_common_context(root, task)
                rendered = {
                    arm: causal.render_arm_prompt(arm, task, common, protocol)
                    for arm in p4.ARMS
                }
                if "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>" not in rendered[p4.SEMANTIC]:
                    faults.append(f"semantic_prompt_operation_grammar_incomplete:{stem}")
                if p2a.ACTION_HEADER not in rendered[p4.DIRECT]:
                    faults.append(f"direct_prompt_grammar_missing:{stem}")
                if "THESEUS_PLAN_V1" not in rendered[p4.PLAN] or p2a.ACTION_HEADER not in rendered[p4.PLAN]:
                    faults.append(f"plan_prompt_grammar_missing:{stem}")

                semantic = causal.parse_arm_output(
                    p4.SEMANTIC, oracle, task, root, protocol
                )
                if semantic.get("faults") or not p2a.dicts(semantic.get("actions")):
                    faults.append(f"semantic_oracle_parse_or_lower_red:{stem}")
                for unit in p2a.dicts(semantic.get("units")):
                    operation = str(unit.get("operation") or "")
                    if operation in operation_counts:
                        operation_counts[operation] += 1
                semantic_verification = p4.verify_provisional(
                    root, baseline, task, semantic
                )
                if (
                    semantic_verification.get("apply_faults")
                    or p2a.mapping(semantic_verification.get("visible_verifier")).get("passed") is not True
                ):
                    faults.append(f"semantic_oracle_apply_or_visible_red:{stem}")

                direct_text = typed_edit(p2a.dicts(semantic.get("actions")))
                plan_text = plan_edit(p2a.dicts(semantic.get("actions")))
                arm_artifacts = {
                    p4.DIRECT: direct_text,
                    p4.PLAN: plan_text,
                    p4.SEMANTIC: oracle,
                }
                arm_candidates: dict[str, dict[str, Any]] = {}
                for arm, artifact in arm_artifacts.items():
                    candidate = causal.parse_arm_output(
                        arm, artifact, task, root, protocol
                    )
                    arm_candidates[arm] = candidate
                    verification = p4.verify_provisional(
                        root, baseline, task, candidate
                    )
                    if (
                        candidate.get("faults")
                        or verification.get("apply_faults")
                        or p2a.mapping(verification.get("visible_verifier")).get("passed") is not True
                    ):
                        faults.append(f"production_arm_oracle_replay_red:{stem}:{arm}")
                    repair_prompt = release.render_full_final_prompt(
                        rendered[arm],
                        artifact,
                        candidate,
                        verification,
                        {
                            str(item.get("id") or "")
                            for item in p2a.dicts(task.get("obligations"))
                        },
                    )
                    if artifact not in repair_prompt:
                        faults.append(f"repair_artifact_not_complete:{stem}:{arm}")
                    prompts[f"{stem}:{arm}:first"] = rendered[arm]
                    prompts[f"{stem}:{arm}:repair"] = repair_prompt

                canonical_actions_hash = p2a.stable_hash(semantic.get("actions"))
                for variant in delimiter_variants(oracle):
                    delimiter_cases += 1
                    parsed = causal.parse_arm_output(
                        p4.SEMANTIC, variant, task, root, protocol
                    )
                    if parsed.get("faults") or p2a.stable_hash(parsed.get("actions")) != canonical_actions_hash:
                        faults.append(f"delimiter_round_trip_red:{stem}:{delimiter_cases}")

                malformed = [
                    oracle.rsplit("\nEND", 1)[0],
                    oracle.replace("OP REPLACE", "OP EXECUTE", 1),
                    oracle.replace("SOURCE ", "SOURCE f", 1),
                ]
                for bad in malformed:
                    malformed_cases += 1
                    parsed = causal.parse_arm_output(
                        p4.SEMANTIC, bad, task, root, protocol
                    )
                    if not parsed.get("faults") or parsed.get("actions"):
                        faults.append(f"malformed_surface_accepted:{stem}:{malformed_cases}")

                task_receipts.append(
                    {
                        "stem": stem,
                        "first_prompts": 3,
                        "repair_prompts": 3,
                        "semantic_oracle_parse_lower_apply_visible": not any(
                            fault.endswith(f":{stem}") for fault in faults
                        ),
                        "direct_plan_semantic_oracle_controls": 3,
                        "hidden_evaluator_calls": 0,
                        "candidate_or_control_calls": 0,
                    }
                )
    finally:
        p4.semantic_symbol_table = original_symbol_table
        p2a.run_visible_verifier = original_verifier

    # INSERT_AFTER is parser-reachable but not answer-bearing on this frozen surface.
    # Exercise its exact parser/lowerer using one frozen oracle, without scoring it.
    first_row = p2a.dicts(pool.get("tasks"))[0]
    first_task = p2a.read_json(ROOT / str(first_row.get("task") or ""))
    first_oracle = (ROOT / str(first_row.get("treatment_transport_oracle_ir") or "")).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="theseus-conformance-insert-after-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(first_task.get("source_archive") or "")),
            root,
            str(first_task.get("source_archive_root") or ""),
        )
        original_symbol_table = p4.semantic_symbol_table
        try:
            p4.semantic_symbol_table = p4s.semantic_scope_symbol_table
            insert_after = first_oracle.replace("OP REPLACE", "OP INSERT_AFTER", 1)
            parsed = causal.parse_arm_output(
                p4.SEMANTIC, insert_after, first_task, root, protocol
            )
        finally:
            p4.semantic_symbol_table = original_symbol_table
        if parsed.get("faults") or not parsed.get("actions"):
            faults.append("insert_after_parser_lowerer_mechanics_red")
        else:
            operation_counts["INSERT_AFTER"] += 1

    prompt_report: dict[str, Any]
    if include_tokenizer:
        prompt_report = count_exact_prompts(prompts, worker)
        if prompt_report.get("trigger_state") != "GREEN":
            faults.extend(
                f"prompt_addressability:{fault}"
                for fault in p2a.strings(prompt_report.get("faults"))
            )
    else:
        prompt_report = {
            "trigger_state": "NOT_RUN_IN_UNIT_TEST",
            "prompt_count": len(prompts),
            "project_selected_quality_token_cap": None,
        }
    if any(count < 1 for count in operation_counts.values()):
        faults.append("reachable_operation_coverage_incomplete")
    if len(task_receipts) != 10 or len(prompts) != 60:
        faults.append("frozen_surface_coverage_incomplete")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "bindings": {
            "pool": {"path": p2a.rel(POOL), "sha256": p2a.sha256_file(POOL)},
            "instrument": {"path": p2a.rel(INSTRUMENT), "sha256": p2a.sha256_file(INSTRUMENT)},
            "causal_runner": {"path": p2a.rel(Path(causal.__file__)), "sha256": p2a.sha256_file(Path(causal.__file__))},
            "release_runner": {"path": p2a.rel(Path(release.__file__)), "sha256": p2a.sha256_file(Path(release.__file__))},
            "semantic_parser": {"path": p2a.rel(Path(ir_v2r2.__file__)), "sha256": p2a.sha256_file(Path(ir_v2r2.__file__))},
            "exact_prompt_token_counter": {"path": p2a.rel(Path(token_count.__file__)), "sha256": p2a.sha256_file(Path(token_count.__file__))},
        },
        "coverage": {
            "frozen_tasks": len(task_receipts),
            "production_first_prompts": len(task_receipts) * 3,
            "production_repair_prompts": len(task_receipts) * 3,
            "oracle_parse_lower_apply_visible_passes_required": len(task_receipts) * 3,
            "operation_mechanics": operation_counts,
            "delimiter_round_trips": delimiter_cases,
            "malformed_rejections": malformed_cases,
        },
        "prompt_addressability": prompt_report,
        "interrupted_launch_reuse_custody": {
            "lease": {
                "path": p2a.rel(INTERRUPTED_LEASE),
                "sha256": p2a.sha256_file(INTERRUPTED_LEASE),
                "state": interrupted_lease.get("state"),
                "scientific_status": interrupted_lease.get("scientific_status"),
                "model_call_opened": interrupted_lease.get("model_call_opened"),
            },
            "durable_runtime_receipts_before_gate": len(preexisting_runtime_receipts),
            "durable_call_start_receipts_before_gate": len(preexisting_call_starts),
            "candidate_run_reports_before_gate": len(preexisting_runs),
            "blind_evaluations_before_gate": len(preexisting_evaluations),
            "observed_candidate_or_control_outputs_available_to_release": 0,
            "instrument_changes": [
                "advertise all parser-reachable semantic operations",
                "restore exact direct/plan grammar in the bound local instrument",
                "correct batched exact-token sequence counting",
                "add durable pre-inference call-start custody",
            ],
            "instrument_change_source": "joined static production-path audit and frozen evaluator-only oracle replay",
            "candidate_or_control_output_influenced_instrument": False,
            "maximum_inference": "No durable or author-observable candidate/control output from the interrupted launch was available to influence this release. The in-memory call-open state remains UNKNOWN and is not rewritten as zero calls.",
        },
        "tasks": task_receipts,
        "candidate_or_control_calls": 0,
        "hidden_evaluator_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": "Production-path mechanics, transport conformance, frozen-oracle visible replay, and pre-model physical prompt addressability only; no model, mechanism, D1, D2, or book-support inference.",
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = run_conformance()
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "faults": report["faults"],
                "coverage": report["coverage"],
                "minimum_context_residual_tokens": p2a.mapping(
                    report.get("prompt_addressability")
                ).get("minimum_context_residual_tokens"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
