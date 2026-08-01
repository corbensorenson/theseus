#!/usr/bin/env python3
"""Materialize and qualify the fresh P4-v2r2 recovery successor pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2_task_pool as predecessor  # noqa: E402
import theseus_p4v2r2r1_source_registry as source_registry  # noqa: E402


REGISTRY = ROOT / "configs" / "theseus_p4v2r2r1_task_sources.json"
FETCH = ROOT / "reports" / "theseus_p4v2r2r1_source_fetch.json"
PREDECESSOR_POOL = ROOT / "configs" / "theseus_p4v2r2_task_pool.json"
POOL = ROOT / "configs" / "theseus_p4v2r2r1_task_pool.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r1_online"
HIDDEN = FIXTURES / "theseus_p4v2r2r1_hidden_test.py"
VISIBLE = FIXTURES / "theseus_p4v2r2r1_visible_test.py"
CORRECTIONS = ROOT / "configs" / "theseus_p4v2r2r1_materialization_corrections.json"
EXPECTED_REGISTRY_SHA256 = "92cd51cfe0fba5ad794527573aa6cd3d570394d7d01fabbb6a55ef2bb0e25f56"
EXPECTED_FETCH_SHA256 = "28ad8a9fc9d943f58bda5ced177451d25c95909b3304bcc1edfb0e1134a095e9"
EXPECTED_HIDDEN_SHA256 = "453cc81eb65547799634108105b4a37a7a415ac2c35d80f6524539eb0d2f5603"
EXPECTED_VISIBLE_SHA256 = "9ee63edb08d5c3a89bd3879f87b6a476bbb612287f0a80ae3c75de835eb67662"
EXPECTED_CORRECTIONS_SHA256 = "46f8b54c88c10171f1a3de182a0e703aaea11d9d0c6a737e49d88e3cdf10589c"


def audit_bindings() -> list[str]:
    faults: list[str] = []
    expected = {
        REGISTRY: EXPECTED_REGISTRY_SHA256,
        FETCH: EXPECTED_FETCH_SHA256,
        HIDDEN: EXPECTED_HIDDEN_SHA256,
        VISIBLE: EXPECTED_VISIBLE_SHA256,
        CORRECTIONS: EXPECTED_CORRECTIONS_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or p2a.sha256_file(path) != digest:
            faults.append(f"binding_invalid:{p2a.rel(path)}")
    if source_registry.audit(REGISTRY).get("trigger_state") != "GREEN":
        faults.append("source_registry_audit_red")
    fetch = p2a.read_json(FETCH) if FETCH.is_file() else {}
    if (
        fetch.get("trigger_state") != "GREEN"
        or p2a.strings(fetch.get("faults"))
        or int(fetch.get("candidate_or_control_calls") or 0) != 0
        or int(fetch.get("parent_target_oracle_or_evaluator_executions") or 0) != 0
    ):
        faults.append("source_fetch_invalid")
    return faults


def materialize(*, run_audits: bool) -> dict[str, Any]:
    faults = audit_bindings()
    registry = p2a.read_json(REGISTRY)
    old_pool = p2a.read_json(PREDECESSOR_POOL)
    carried_stems = p2a.strings(registry.get("carried_candidate_unseen_stems"))
    old_by_stem = {
        str(row.get("stem") or ""): row
        for row in p2a.dicts(old_pool.get("tasks"))
    }
    entries: list[dict[str, Any]] = []
    for index, stem in enumerate(carried_stems, start=1):
        source = p2a.mapping(old_by_stem.get(stem))
        if not source:
            faults.append(f"carried_task_missing:{stem}")
            continue
        entry = dict(source)
        entry["campaign_index"] = index
        entry["recovery_custody"] = "candidate_unseen_carried_from_attempt2_interruption"
        entries.append(entry)

    replacement = dict(p2a.mapping(registry.get("replacement_task")))
    corrections = p2a.read_json(CORRECTIONS)
    replacement["oracle_units"] = list(
        next(
            row.get("replacement")
            for row in p2a.dicts(corrections.get("corrections"))
            if row.get("field") == "oracle_units"
        )
    )
    original_globals = {
        "FIXTURES": predecessor.FIXTURES,
        "HIDDEN_TEST": predecessor.HIDDEN_TEST,
        "VISIBLE_TEST": predecessor.VISIBLE_TEST,
    }
    predecessor.FIXTURES = FIXTURES
    predecessor.HIDDEN_TEST = HIDDEN
    predecessor.VISIBLE_TEST = VISIBLE
    try:
        new_entry, new_faults = predecessor.materialize_task(replacement, registry)
    finally:
        for name, value in original_globals.items():
            setattr(predecessor, name, value)
    faults.extend(new_faults)
    task_path = ROOT / str(new_entry["task"])
    evaluator_path = ROOT / str(new_entry["evaluator"])
    task = p2a.read_json(task_path)
    with tempfile.TemporaryDirectory(prefix="theseus-p4v2r2r1-context-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            ROOT / str(task["source_archive"]), root, str(task["source_archive_root"])
        )
        exact_context_characters = len(p2a.render_visible_context(root, task))
    p2a.mapping(task["candidate_visible_context"])["maximum_total_characters"] = (
        exact_context_characters
    )
    p2a.write_json(task_path, task)
    evaluator = p2a.read_json(evaluator_path)
    evaluator["task_manifest_sha256"] = p2a.sha256_file(task_path)
    evaluator["baseline_failure_markers"] = [
        "P4V2R2R1_FAIL_PYDANTIC_AI_OPENROUTER"
    ]
    p2a.write_json(evaluator_path, evaluator)
    new_entry["task_sha256"] = p2a.sha256_file(task_path)
    new_entry["evaluator_sha256"] = p2a.sha256_file(evaluator_path)
    if "candidate_visible_context_budget_invalid" in faults:
        repaired_faults = predecessor.p4s_pool.audit_task_surface(
            task, ROOT / str(task["source_archive"])
        )
        if "candidate_visible_context_budget_invalid" not in repaired_faults:
            faults.remove("candidate_visible_context_budget_invalid")
    new_entry["exact_candidate_visible_context_characters"] = exact_context_characters
    new_entry["campaign_index"] = 10
    new_entry["recovery_custody"] = "new_prospectively_sealed_source_disjoint_replacement"

    if run_audits and not faults:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "theseus_p4_cognitive_compilation_evaluator.py"),
                "--evaluator",
                new_entry["evaluator"],
                "--audit-only",
                "--out",
                new_entry["evaluator_audit"],
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
        )
        audit = p2a.read_json(ROOT / new_entry["evaluator_audit"])
        if result.returncode != 0 or audit.get("trigger_state") != "GREEN":
            faults.append(f"replacement_evaluator_audit_red:{audit.get('faults')}")
        dependency = predecessor.audit_dependency_corruption(new_entry)
        oracle_replay = predecessor.audit_v2r2_oracle(new_entry)
        if dependency.get("rejected") is not True:
            faults.append("replacement_dependency_corruption_not_rejected")
        if oracle_replay.get("trigger_state") != "GREEN":
            faults.append("replacement_v2r2_oracle_replay_red")
        new_entry.update(
            {
                "evaluator_audit_sha256": p2a.sha256_file(ROOT / new_entry["evaluator_audit"]),
                "evaluator_audit_trigger_state": audit.get("trigger_state"),
                "baseline_parent_failed": p2a.mapping(audit.get("baseline_verification")).get("hidden_passed") is False,
                "upstream_target_passed": p2a.mapping(audit.get("target_verification")).get("hidden_passed") is True,
                "compiler_oracle_v1_passed": p2a.mapping(audit.get("compiler_oracle_verification")).get("hidden_passed") is True,
                "four_base_corruptions_rejected": all(
                    p2a.mapping(audit.get("corruption_intervention_rejections")).values()
                ),
                "dependency_corruption": dependency,
                "v2r2_oracle_replay": oracle_replay,
            }
        )
    else:
        new_entry.update(
            {
                "evaluator_audit_sha256": "",
                "evaluator_audit_trigger_state": "NOT_RUN",
                "baseline_parent_failed": False,
                "upstream_target_passed": False,
                "compiler_oracle_v1_passed": False,
                "four_base_corruptions_rejected": False,
                "dependency_corruption": {"rejected": False, "faults": ["not_run"]},
                "v2r2_oracle_replay": {"trigger_state": "NOT_RUN", "faults": ["not_run"]},
            }
        )
    entries.append(new_entry)

    green = sum(row.get("evaluator_audit_trigger_state") == "GREEN" for row in entries)
    oracle_green = sum(
        p2a.mapping(row.get("v2r2_oracle_replay")).get("trigger_state") == "GREEN"
        for row in entries
    )
    dependency_green = sum(
        p2a.mapping(row.get("dependency_corruption")).get("rejected") is True
        for row in entries
    )
    if run_audits and (green, oracle_green, dependency_green) != (10, 10, 10):
        faults.append("not_all_ten_mechanics_floors_green")
    state = (
        "SEALED_BEFORE_SUCCESSOR_CANDIDATE_GENERATION"
        if run_audits and not faults and len(entries) == 10
        else "INVALID_NOT_SEALED"
    )
    pool = {
        "policy": "project_theseus_p4v2r2r1_recovery_task_pool_v1",
        "state": state,
        "partition": "p4v2r2r1_cognitive_compilation_decision_development",
        "sealed_utc": p2a.now(),
        "candidate_generation_opened": False,
        "source_registry": p2a.rel(REGISTRY),
        "source_registry_sha256": p2a.sha256_file(REGISTRY),
        "source_fetch_report": p2a.rel(FETCH),
        "source_fetch_report_sha256": p2a.sha256_file(FETCH),
        "predecessor_pool": p2a.rel(PREDECESSOR_POOL),
        "predecessor_pool_sha256": p2a.sha256_file(PREDECESSOR_POOL),
        "materialization_corrections": p2a.rel(CORRECTIONS),
        "materialization_corrections_sha256": p2a.sha256_file(CORRECTIONS),
        "task_count": len(entries),
        "distinct_repositories": len({str(row.get("repository") or "").lower() for row in entries}),
        "green_evaluator_audits": green,
        "v2r2_oracle_replays_green": oracle_green,
        "dependency_corruptions_rejected": dependency_green,
        "tasks": entries,
        "faults": sorted(set(faults)),
        "generation_budget": {
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_boundary": "model_declared_context_window_minus_exact_prompt_tokens",
            "boundary_hit_invalidates_observation": True,
            "boundary_hit_counts_as_model_or_mechanism_failure": False,
        },
        "counters": {
            "successor_local_model_calls": 0,
            "successor_hosted_model_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": "A GREEN recovery seal establishes only ten mechanics-qualified development tasks with nine candidate-unseen custody proofs and one source-disjoint replacement. It is not candidate, mechanism-survivor, D1, D2, serving, training, or ASI Stack support evidence.",
    }
    p2a.write_json(POOL, pool)
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize(run_audits=not args.materialize_only)
    print(json.dumps({key: report[key] for key in ("state", "task_count", "green_evaluator_audits", "v2r2_oracle_replays_green", "dependency_corruptions_rejected", "faults")}, indent=2, sort_keys=True))
    return 0 if report["state"] == "SEALED_BEFORE_SUCCESSOR_CANDIDATE_GENERATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
