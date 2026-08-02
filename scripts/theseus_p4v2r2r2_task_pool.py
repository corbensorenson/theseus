#!/usr/bin/env python3
"""Materialize and mechanics-qualify the all-new P4-v2r2-r2 task pool."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_task_pool as base_pool  # noqa: E402
import theseus_p4s_task_pool as p4s_pool  # noqa: E402
import theseus_p4v2r2_task_pool as predecessor  # noqa: E402
import theseus_p4v2r2r3_cognitive_compilation as runner  # noqa: E402
import theseus_p4v2r2r2_revision_repair as revision_repair  # noqa: E402
import theseus_p4v2r2r2_source_registry as source_registry  # noqa: E402
import theseus_semantic_ir_v2r2 as ir_v2r2  # noqa: E402


REGISTRY = ROOT / "configs" / "theseus_p4v2r2r2_task_sources.json"
SOURCE_FETCH = ROOT / "reports" / "theseus_p4v2r2r2_source_fetch.json"
REVISION_CORRECTIONS = ROOT / "configs" / "theseus_p4v2r2r2_revision_corrections.json"
REVISION_FETCH = ROOT / "reports" / "theseus_p4v2r2r2_revision_repair_fetch.json"
CONTRACTS = ROOT / "configs" / "theseus_p4v2r2r2_task_contracts.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r2_cognitive_compilation_instrument.json"
INSTRUMENT_AUDIT = ROOT / "reports" / "theseus_p4v2r2r3_instrument_audit.json"
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r2_online"
CORE = FIXTURES / "theseus_p4v2r2r2_evaluator_core.py"
VISIBLE = FIXTURES / "theseus_p4v2r2r2_visible_test.py"
HIDDEN = FIXTURES / "theseus_p4v2r2r2_hidden_test.py"


def correction_maps() -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    revision = p2a.read_json(REVISION_CORRECTIONS)
    revisions = {
        str(row["stem"]): row for row in p2a.dicts(revision.get("corrections"))
    }
    materialization = p2a.read_json(
        ROOT / "configs" / "theseus_p4v2r2r2_materialization_corrections.json"
    )
    licenses = {
        str(row["stem"]): p2a.strings(row.get("materialized_value"))
        for row in p2a.dicts(materialization.get("corrections"))
        if row.get("field") == "license_paths"
    }
    return revisions, licenses


def effective_sources() -> list[dict[str, Any]]:
    registry = p2a.read_json(REGISTRY)
    contracts = p2a.read_json(CONTRACTS)
    contract_by_stem = {
        str(row["stem"]): row for row in p2a.dicts(contracts.get("tasks"))
    }
    revisions, licenses = correction_maps()
    values: list[dict[str, Any]] = []
    for original in p2a.dicts(registry.get("tasks")):
        source = dict(original)
        source.update(contract_by_stem[str(source["stem"])])
        source["registered_parent_revision"] = original["parent_revision"]
        source["registered_target_revision"] = original["target_revision"]
        source["registered_source_root"] = original["source_root"]
        source["registered_target_root"] = original["target_root"]
        correction = revisions.get(str(source["stem"]))
        if correction:
            source["parent_revision"] = correction["corrected_parent_revision"]
            source["target_revision"] = correction["corrected_target_revision"]
            source["source_root"] = correction["corrected_parent_root"]
            source["target_root"] = correction["corrected_target_root"]
            source["revision_correction_applied"] = True
        else:
            source["revision_correction_applied"] = False
        if str(source["stem"]) in licenses:
            source["registered_license_paths"] = original["license_paths"]
            source["license_paths"] = licenses[str(source["stem"])]
        values.append(source)
    return values


def artifact_paths(source: dict[str, Any], label: str) -> dict[str, Path]:
    stem = str(source["stem"])
    corrected = bool(source.get("revision_correction_applied"))
    infix = "_revision_corrected" if corrected else ""
    return {
        "projected": FIXTURES / f"{stem}_{label}{infix}.tar.gz",
        "upstream": FIXTURES / f"{stem}_{label}{infix}_upstream.tar.gz",
        "sanitization": ROOT
        / "reports"
        / f"theseus_{stem}_{label}{infix}_archive_sanitization.json",
    }


def audit_artifact(source: dict[str, Any], label: str) -> list[str]:
    paths = artifact_paths(source, label)
    faults: list[str] = []
    if any(not path.is_file() for path in paths.values()):
        return [f"artifact_missing:{source['stem']}:{label}"]
    report = p2a.read_json(paths["sanitization"])
    root = str(source["source_root" if label == "parent" else "target_root"])
    if report.get("trigger_state") != "GREEN" or p2a.strings(report.get("faults")):
        faults.append(f"sanitization_red:{source['stem']}:{label}")
    if report.get("source_archive_root") != root:
        faults.append(f"sanitization_root_mismatch:{source['stem']}:{label}")
    if p2a.mapping(report.get("input")).get("sha256") != p2a.sha256_file(paths["upstream"]):
        faults.append(f"upstream_digest_mismatch:{source['stem']}:{label}")
    if p2a.mapping(report.get("projected_output")).get("sha256") != p2a.sha256_file(paths["projected"]):
        faults.append(f"projection_digest_mismatch:{source['stem']}:{label}")
    faults.extend(
        f"archive:{source['stem']}:{value}"
        for value in base_pool.audit_archive(paths["projected"], root, source, label)
    )
    return faults


def audit_inputs() -> list[str]:
    faults: list[str] = []
    registry = p2a.read_json(REGISTRY)
    contracts = p2a.read_json(CONTRACTS)
    if source_registry.audit(REGISTRY).get("trigger_state") != "GREEN":
        faults.append("source_registry_red")
    if p2a.read_json(SOURCE_FETCH).get("trigger_state") != "GREEN":
        faults.append("source_fetch_red")
    if revision_repair.audit_corrections().get("trigger_state") != "GREEN":
        faults.append("revision_corrections_red")
    if p2a.read_json(REVISION_FETCH).get("trigger_state") != "GREEN":
        faults.append("revision_fetch_red")
    instrument_audit = runner.audit_instrument(INSTRUMENT)
    p2a.write_json(INSTRUMENT_AUDIT, instrument_audit)
    if instrument_audit.get("trigger_state") != "GREEN":
        faults.append("instrument_red")
    if contracts.get("policy") != "project_theseus_p4v2r2r2_post_fetch_task_contracts_v1":
        faults.append("contracts_policy_invalid")
    if contracts.get("state") != "SEALED_AFTER_SOURCE_AND_EVALUATOR_ADEQUACY_BEFORE_CANDIDATE_OR_CONTROL_GENERATION":
        faults.append("contracts_state_invalid")
    if (
        contracts.get("source_registry_sha256") != p2a.sha256_file(REGISTRY)
        or int(contracts.get("candidate_or_control_calls_before_seal") or 0) != 0
    ):
        faults.append("contracts_custody_invalid")
    source_stems = [str(row["stem"]) for row in p2a.dicts(registry.get("tasks"))]
    contract_rows = p2a.dicts(contracts.get("tasks"))
    if [str(row.get("stem")) for row in contract_rows] != source_stems:
        faults.append("contract_membership_mismatch")
    for row in contract_rows:
        stem = str(row.get("stem") or "")
        obligations = p2a.dicts(row.get("obligations"))
        obligation_ids = [str(item.get("id") or "") for item in obligations]
        if not obligations or len(set(obligation_ids)) != len(obligation_ids):
            faults.append(f"obligations_invalid:{stem}")
        for dependency in p2a.dicts(row.get("obligation_dependencies")):
            if dependency.get("before") not in obligation_ids or dependency.get("after") not in obligation_ids:
                faults.append(f"dependency_invalid:{stem}")
        units = p2a.dicts(row.get("oracle_units"))
        if not units or len(units) > 8:
            faults.append(f"oracle_unit_count_invalid:{stem}")
        if not p2a.dicts(row.get("searches")) or not p2a.dicts(row.get("reads")):
            faults.append(f"visible_context_invalid:{stem}")
        if len(p2a.dicts(row.get("visible_markers"))) != 1:
            faults.append(f"visible_marker_invalid:{stem}")
    for path in (CORE, VISIBLE, HIDDEN):
        if not path.is_file():
            faults.append(f"evaluator_surface_missing:{p2a.rel(path)}")
    for source in effective_sources():
        faults.extend(audit_artifact(source, "parent"))
        faults.extend(audit_artifact(source, "target"))
    return sorted(set(faults))


def build_oracle(
    source: dict[str, Any],
    task: dict[str, Any],
    parent: Path,
    target: Path,
    *,
    transport: str,
) -> str:
    text = predecessor.build_oracle(
        source, task, parent, target, transport=transport
    )
    for unit in p2a.dicts(source.get("oracle_units")):
        replacement = str(unit.get("oracle_replacement") or "")
        if not replacement:
            continue
        unit_id = re.escape(str(unit["id"]))
        pattern = re.compile(
            rf"(UNIT {unit_id}(?: [^\n]*)?\n(?:(?!UNIT ).)*?<<<\n)(.*?)(\n>>>)",
            flags=re.DOTALL,
        )
        text, count = pattern.subn(
            lambda match: match.group(1) + replacement + match.group(3),
            text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"oracle replacement unit not found: {unit['id']}")
    with tempfile.TemporaryDirectory(prefix="theseus-p4v2r2r2-oracle-validate-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(parent, root, str(source["source_root"]))
        parsed = (
            p4.parse_semantic_ir(text, task, root)
            if transport == "v1"
            else ir_v2r2.parse(text, task, root)
        )
        if parsed.get("faults"):
            raise ValueError(
                "custom oracle parse faults: "
                + ",".join(p2a.strings(parsed.get("faults")))
            )
    return text


def materialize_task(source: dict[str, Any], registry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    index = int(source["campaign_index"])
    stem = str(source["stem"])
    suffix = stem.removeprefix("p4v2r2r2_")
    parent_paths = artifact_paths(source, "parent")
    target_paths = artifact_paths(source, "target")
    parent = parent_paths["projected"]
    target = target_paths["projected"]
    task_path = ROOT / "configs" / f"theseus_p4v2r2r2_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p4v2r2r2_evaluator_{suffix}.json"
    oracle_v1_path = FIXTURES / f"{stem}_oracle_v1.semantic_ir"
    oracle_v2_path = FIXTURES / f"{stem}_oracle_v2r2.semantic_ir"
    audit_path = ROOT / "reports" / f"theseus_p4v2r2r2_{suffix}_evaluator_audit.json"
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "campaign_index": index,
        "opaque_task_id": f"p4v2r2r2-cognitive-compilation-{index:02d}",
        "partition": "p4v2r2r2_cognitive_compilation_decision_development",
        "family": "bounded_python_correctness_repair",
        "natural_request": source["natural_request"],
        "source_archive": p2a.rel(parent),
        "source_archive_sha256": p2a.sha256_file(parent),
        "source_archive_root": source["source_root"],
        "source_provenance": {
            "repository": source["repository"],
            "url": f"https://github.com/{source['repository']}",
            "revision": source["parent_revision"],
            "registered_revision": source["registered_parent_revision"],
            "revision_correction_applied": source["revision_correction_applied"],
            "license_spdx": source["license_spdx"],
            "license_paths": source["license_paths"],
            "upstream_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
            "upstream_request_title": source["pull_request_title"],
            "upstream_archive": p2a.rel(parent_paths["upstream"]),
            "upstream_archive_sha256": p2a.sha256_file(parent_paths["upstream"]),
            "archive_sanitization_report": p2a.rel(parent_paths["sanitization"]),
            "archive_sanitization_report_sha256": p2a.sha256_file(parent_paths["sanitization"]),
        },
        "contamination_screen": {
            "public_benchmark": False,
            "previous_theseus_surface": False,
            "source_disjoint_from_all_prior_theseus_development_sources": True,
            "task_selected_before_any_candidate_or_control": True,
            "later_patch_hidden_test_or_oracle_candidate_visible": False,
            "development_task_eligible_for_training_D1_or_D2": False,
        },
        "obligations": source["obligations"],
        "obligation_dependencies": source["obligation_dependencies"],
        "allowed_effect_paths": source["allowed_effect_paths"],
        "candidate_visible_context": {
            "searches": source["searches"],
            "reads": source["reads"],
            "maximum_total_characters": 1_000_000,
        },
        "visible_verifier": {
            "command": base_pool.python312_exec_command(VISIBLE, source["case"]),
            "timeout_seconds": 60,
            "answer_specific": True,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": source["visible_markers"],
        "semantic_ir_contract": {
            "version": "theseus_semantic_ir_v2r2_labeled",
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 80,
            "maximum_units": 8,
            "source_target_obligation_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One P4-v2r2-r2 development observation only; no D1, D2, serving, training, or ASI Stack support claim.",
    }
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = p4s_pool.exact_lowerer_inventory_count(task, parent)
    with tempfile.TemporaryDirectory(prefix="theseus-p4v2r2r2-context-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(parent, root, str(source["source_root"]))
        exact_context = len(p2a.render_visible_context(root, task))
    task["candidate_visible_context"]["maximum_total_characters"] = exact_context
    p2a.write_json(task_path, task)
    faults.extend(p4s_pool.audit_task_surface(task, parent))
    try:
        oracle_v1_path.write_text(
            build_oracle(source, task, parent, target, transport="v1"),
            encoding="utf-8",
        )
        oracle_v2_path.write_text(
            build_oracle(source, task, parent, target, transport="v2r2"),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - oracle construction fails closed.
        faults.append(f"oracle_materialization_fault:{stem}:{type(exc).__name__}:{exc}")
    evaluator = {
        "policy": "project_theseus_p4_cognitive_compilation_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": [f"P4V2R2R2_HIDDEN_FAIL_{str(source['case']).upper()}"],
        "hidden_test_files": [
            {"source": p2a.rel(CORE), "sha256": p2a.sha256_file(CORE), "destination": CORE.name},
            {"source": p2a.rel(HIDDEN), "sha256": p2a.sha256_file(HIDDEN), "destination": HIDDEN.name},
        ],
        "hidden_verifier": {
            "command": base_pool.python312_exec_command(Path(HIDDEN.name), source["case"]),
            "timeout_seconds": 60,
            "network": "forbidden",
        },
        "target_archive": p2a.rel(target),
        "target_archive_sha256": p2a.sha256_file(target),
        "target_archive_root": source["target_root"],
        "target_provenance": {
            "revision": source["target_revision"],
            "registered_revision": source["registered_target_revision"],
            "revision_correction_applied": source["revision_correction_applied"],
            "upstream_archive": p2a.rel(target_paths["upstream"]),
            "upstream_archive_sha256": p2a.sha256_file(target_paths["upstream"]),
            "archive_sanitization_report": p2a.rel(target_paths["sanitization"]),
            "archive_sanitization_report_sha256": p2a.sha256_file(target_paths["sanitization"]),
        },
        "target_must_pass": True,
        "oracle_ir_file": p2a.rel(oracle_v1_path),
        "oracle_ir_sha256": p2a.sha256_file(oracle_v1_path) if oracle_v1_path.is_file() else "",
        "treatment_transport_oracle_ir_file": p2a.rel(oracle_v2_path),
        "treatment_transport_oracle_ir_sha256": p2a.sha256_file(oracle_v2_path) if oracle_v2_path.is_file() else "",
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "route_label_passed_to_scoring": False,
            "later_patch_candidate_visible": False,
            "hidden_test_candidate_visible": False,
            "oracle_candidate_visible": False,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "maximum_inference": "A GREEN audit establishes evaluator reachability and sensitivity for one sealed development task only.",
    }
    p2a.write_json(evaluator_path, evaluator)
    return (
        {
            "campaign_index": index,
            "stem": stem,
            "repository": source["repository"],
            "parent_revision": source["parent_revision"],
            "target_revision": source["target_revision"],
            "revision_correction_applied": source["revision_correction_applied"],
            "task": p2a.rel(task_path),
            "task_sha256": p2a.sha256_file(task_path),
            "evaluator": p2a.rel(evaluator_path),
            "evaluator_sha256": p2a.sha256_file(evaluator_path),
            "oracle_ir": p2a.rel(oracle_v1_path),
            "oracle_ir_sha256": p2a.sha256_file(oracle_v1_path) if oracle_v1_path.is_file() else "",
            "treatment_transport_oracle_ir": p2a.rel(oracle_v2_path),
            "treatment_transport_oracle_ir_sha256": p2a.sha256_file(oracle_v2_path) if oracle_v2_path.is_file() else "",
            "evaluator_audit": p2a.rel(audit_path),
            "exact_candidate_visible_context_characters": exact_context,
        },
        faults,
    )


def materialize(*, run_audits: bool) -> dict[str, Any]:
    faults = audit_inputs()
    registry = p2a.read_json(REGISTRY)
    entries: list[dict[str, Any]] = []
    for source in effective_sources():
        entry, entry_faults = materialize_task(source, registry)
        entries.append(entry)
        faults.extend(entry_faults)
    if run_audits and not faults:
        for entry in entries:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "theseus_p4_cognitive_compilation_evaluator.py"),
                    "--evaluator",
                    entry["evaluator"],
                    "--audit-only",
                    "--out",
                    entry["evaluator_audit"],
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=240,
                check=False,
            )
            audit = p2a.read_json(ROOT / entry["evaluator_audit"])
            dependency = predecessor.audit_dependency_corruption(entry)
            replay = predecessor.audit_v2r2_oracle(entry)
            if result.returncode != 0 or audit.get("trigger_state") != "GREEN":
                faults.append(f"evaluator_audit_red:{entry['stem']}:{audit.get('faults')}")
            if dependency.get("rejected") is not True:
                faults.append(f"dependency_corruption_not_rejected:{entry['stem']}")
            if replay.get("trigger_state") != "GREEN":
                faults.append(f"v2r2_oracle_replay_red:{entry['stem']}:{replay.get('faults')}")
            entry.update(
                {
                    "evaluator_audit_sha256": p2a.sha256_file(ROOT / entry["evaluator_audit"]),
                    "evaluator_audit_trigger_state": audit.get("trigger_state"),
                    "baseline_parent_failed": p2a.mapping(audit.get("baseline_verification")).get("hidden_passed") is False,
                    "upstream_target_passed": p2a.mapping(audit.get("target_verification")).get("hidden_passed") is True,
                    "compiler_oracle_v1_passed": p2a.mapping(audit.get("compiler_oracle_verification")).get("hidden_passed") is True,
                    "four_base_corruptions_rejected": all(p2a.mapping(audit.get("corruption_intervention_rejections")).values()),
                    "dependency_corruption": dependency,
                    "v2r2_oracle_replay": replay,
                }
            )
    else:
        for entry in entries:
            entry.update(
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
    green = sum(row.get("evaluator_audit_trigger_state") == "GREEN" for row in entries)
    replay_green = sum(p2a.mapping(row.get("v2r2_oracle_replay")).get("trigger_state") == "GREEN" for row in entries)
    dependency_green = sum(p2a.mapping(row.get("dependency_corruption")).get("rejected") is True for row in entries)
    if run_audits and (green, replay_green, dependency_green) != (10, 10, 10):
        faults.append("not_all_ten_mechanics_floors_green")
    state = "SEALED_BEFORE_CANDIDATE_GENERATION" if run_audits and not faults else "INVALID_NOT_SEALED"
    pool = {
        "policy": "project_theseus_p4v2r2r2_cognitive_compilation_task_pool_v1",
        "state": state,
        "partition": "p4v2r2r2_cognitive_compilation_decision_development",
        "sealed_utc": p2a.now(),
        "candidate_generation_opened": False,
        "source_registry": {"path": p2a.rel(REGISTRY), "sha256": p2a.sha256_file(REGISTRY)},
        "source_fetch": {"path": p2a.rel(SOURCE_FETCH), "sha256": p2a.sha256_file(SOURCE_FETCH)},
        "revision_corrections": {"path": p2a.rel(REVISION_CORRECTIONS), "sha256": p2a.sha256_file(REVISION_CORRECTIONS)},
        "revision_repair_fetch": {"path": p2a.rel(REVISION_FETCH), "sha256": p2a.sha256_file(REVISION_FETCH)},
        "task_contracts": {"path": p2a.rel(CONTRACTS), "sha256": p2a.sha256_file(CONTRACTS)},
        "instrument": {"path": p2a.rel(INSTRUMENT), "sha256": p2a.sha256_file(INSTRUMENT)},
        "evaluator_surface": {
            "core": {"path": p2a.rel(CORE), "sha256": p2a.sha256_file(CORE)},
            "visible": {"path": p2a.rel(VISIBLE), "sha256": p2a.sha256_file(VISIBLE)},
            "hidden": {"path": p2a.rel(HIDDEN), "sha256": p2a.sha256_file(HIDDEN)},
        },
        "task_count": len(entries),
        "distinct_repositories": len({str(row["repository"]).lower() for row in entries}),
        "green_evaluator_audits": green,
        "v2r2_oracle_replays_green": replay_green,
        "dependency_corruptions_rejected": dependency_green,
        "tasks": entries,
        "faults": sorted(set(faults)),
        "generation_boundary": {
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_context_boundary": "pinned_model_context_window_minus_exact_prompt_tokens",
            "boundary_hit_invalidates_observation": True,
        },
        "counters": {
            "local_model_calls": 0,
            "hosted_model_calls": 0,
            "teacher_calls": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "training_rows_written": 0,
        },
        "maximum_inference": "A GREEN seal establishes ten fresh licensed parent-fail/target-pass tasks, evaluator-only Semantic-IR v1/v2r2 replay, and corruption sensitivity before candidate generation. It is not model, mechanism-survivor, D1, D2, training, serving, or ASI Stack support evidence.",
    }
    p2a.write_json(POOL, pool)
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize(run_audits=not args.materialize_only)
    print(json.dumps({key: report[key] for key in ("state", "task_count", "green_evaluator_audits", "v2r2_oracle_replays_green", "dependency_corruptions_rejected", "faults")}, indent=2, sort_keys=True))
    return 0 if report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
