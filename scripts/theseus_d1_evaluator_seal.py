#!/usr/bin/env python3
"""Prospectively qualify and seal machine-verifiable D1 tasks before model calls.

The source frame is ordered without archive or candidate outcomes.  This owner
then applies a predeclared construct-validity filter: exactly one Python callable
must change, target tests are split deterministically into visible and hidden
sets, parent must fail both sets, and target plus an exact callable transplant
must pass both.  Every rejected identity remains in the report.  Candidate and
control generation remains forbidden until a complete final cohort is sealed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_d1_pre_model_evaluator_seal_v1"
POOL_POLICY = "project_theseus_d1_sealed_task_pool_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_evaluator_seal.json"


@dataclass(frozen=True)
class CallableSource:
    qualified_name: str
    kind: str
    signature: str
    start_line: int
    end_line: int
    source: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = preflight(config, config_path=config_path)
    if args.execute and report.get("execution_authorized") is True:
        report = execute(config, config_path=config_path)
    write_json(resolve(args.out or str(config["qualification_report"])), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    registry_override: dict[str, Any] | None = None,
    materialization_override: dict[str, Any] | None = None,
    sandbox_override: dict[str, Any] | None = None,
    lease_exists_override: bool | None = None,
) -> dict[str, Any]:
    faults = validate_config(config)
    existing = audit_existing_pool(config)
    if existing.get("passed") is True:
        return {
            "policy": POLICY,
            "created_utc": now(),
            "trigger_state": "GREEN",
            "activation_state": "D1_PRE_MODEL_EVALUATOR_COHORT_ALREADY_SEALED",
            "terminal": True,
            "execution_authorized": False,
            "candidate_or_control_calls_authorized": False,
            "faults": [],
            "config": identity(config_path),
            "existing_pool": existing,
            "qualified_task_count": existing.get("task_count"),
            "required_task_count": existing.get("task_count"),
            "pre_model_rejection_count": 0,
            "parent_target_oracle_executions": 0,
            "candidate_or_control_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "project_selected_quality_token_cap": None,
            "maximum_inference": str(config.get("maximum_inference") or ""),
        }
    instrument_path = resolve(str(config.get("instrument") or ""))
    registry_path = resolve(str(config.get("source_registry") or ""))
    materialization_path = resolve(str(config.get("materialization_report") or ""))
    sandbox_path = resolve(str(config.get("sandbox_qualification") or ""))
    instrument = read_optional(instrument_path)
    registry = registry_override if registry_override is not None else read_optional(registry_path)
    materialization = (
        materialization_override
        if materialization_override is not None
        else read_optional(materialization_path)
    )
    sandbox = sandbox_override if sandbox_override is not None else read_optional(sandbox_path)
    if instrument.get("policy") != (
        "project_theseus_d1_fresh_source_disjoint_qualification_instrument_v1"
    ):
        faults.append("instrument_invalid")
    source_ready = audit_source_inputs(registry, materialization)
    sandbox_ready = (
        sandbox.get("policy") == "project_theseus_d1_untrusted_evaluator_sandbox_v1"
        and sandbox.get("trigger_state") == "GREEN"
        and sandbox.get("untrusted_execution_authorized") is True
        and not strings(sandbox.get("faults"))
    )
    lease_path = resolve(str(config.get("active_lease") or ""))
    lease_exists = lease_path.exists() if lease_exists_override is None else lease_exists_override
    execution_authorized = not faults and source_ready["passed"] and sandbox_ready and not lease_exists
    state = "CONTRACT_INVALID" if faults else "WAITING_FOR_GREEN_D1_SOURCE_MATERIALIZATION"
    if not faults and source_ready["passed"] and not sandbox_ready:
        state = "WAITING_FOR_GREEN_D1_EVALUATOR_SANDBOX"
    if execution_authorized:
        state = "READY_FOR_PRE_MODEL_EVALUATOR_QUALIFICATION"
    if lease_exists and not faults and source_ready["passed"] and sandbox_ready:
        state = "WAITING_FOR_EXCLUSIVE_EVALUATOR_SEAL_LEASE"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if faults else "GREEN" if execution_authorized else "PAUSED",
        "activation_state": state,
        "terminal": False,
        "execution_authorized": execution_authorized,
        "candidate_or_control_calls_authorized": False,
        "faults": sorted(set(faults)),
        "config": identity(config_path),
        "instrument": input_identity(instrument_path, instrument),
        "source_registry": input_identity(registry_path, registry, registry_override),
        "source_materialization": input_identity(
            materialization_path, materialization, materialization_override
        ),
        "source_input_audit": source_ready,
        "sandbox_qualification": input_identity(sandbox_path, sandbox, sandbox_override),
        "sandbox_ready": sandbox_ready,
        "qualified_task_count": 0,
        "required_task_count": required_task_count(instrument),
        "pre_model_rejection_count": 0,
        "parent_target_oracle_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def execute(config: dict[str, Any], *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    if before.get("execution_authorized") is not True:
        return before
    lease_path = resolve(str(config["active_lease"]))
    lease_id = uuid.uuid4().hex
    lease = {"policy": POLICY, "lease_id": lease_id, "state": "RUNNING", "created_utc": now()}
    try:
        write_json_exclusive(lease_path, lease)
    except FileExistsError:
        raced = dict(before)
        raced["trigger_state"] = "PAUSED"
        raced["execution_authorized"] = False
        raced["activation_state"] = "LEASE_ACQUISITION_RACE"
        return raced
    try:
        registry = read_json(resolve(str(config["source_registry"])))
        materialization = read_json(resolve(str(config["materialization_report"])))
        report = qualify_registry(config, registry, materialization)
        if report.get("final_pool_ready") is True:
            persisted = persist_final_pool(config, mapping(report["final_pool"]))
            report["final_pool"] = persisted
            write_json_exclusive(resolve(str(config["final_task_pool"])), persisted)
            report["final_pool_written"] = identity(resolve(str(config["final_task_pool"])))
        else:
            report["final_pool_written"] = {}
    finally:
        archive_dir = resolve(str(config["lease_archive_directory"]))
        archive_dir.mkdir(parents=True, exist_ok=True)
        lease["state"] = "COMPLETED"
        lease["completed_utc"] = now()
        write_json(lease_path, lease)
        os.replace(lease_path, archive_dir / f"{lease_id}.json")
    return report


def qualify_registry(
    config: dict[str, Any],
    registry: dict[str, Any],
    materialization: dict[str, Any],
    *,
    runner: Callable[[Path, list[str], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_audit = audit_source_inputs(registry, materialization)
    faults = list(source_audit["faults"])
    instrument = read_json(resolve(str(config["instrument"])))
    needed = required_task_count(instrument)
    materialized_by_index = {
        int(row.get("campaign_index") or 0): row
        for row in dictionaries(materialization.get("tasks"))
    }
    qualified: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    execute_runner = runner or run_pytest_sandboxed
    for task in dictionaries(registry.get("tasks")):
        index = int(task.get("campaign_index") or 0)
        materialized = materialized_by_index.get(index, {})
        row = qualify_task(config, task, materialized, runner=execute_runner)
        rows.append(row)
        if row.get("qualified") is True and len(qualified) < needed:
            qualified.append(row)
    pool_ready = not faults and len(qualified) == needed
    pool = build_pool(config, registry, qualified, instrument) if pool_ready else {}
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "activation_state": (
            "D1_PRE_MODEL_EVALUATOR_COHORT_SEALED"
            if pool_ready
            else "D1_EVALUATOR_QUALIFIED_COHORT_UNDERFILLED_APPEND_METADATA_INTERVALS"
        ),
        "terminal": pool_ready,
        "execution_authorized": False,
        "candidate_or_control_calls_authorized": False,
        "faults": sorted(set(faults)),
        "required_task_count": needed,
        "qualified_task_count": len(qualified),
        "pre_model_rejection_count": len(rows) - sum(row.get("qualified") is True for row in rows),
        "qualification_rows": rows,
        "final_pool_ready": pool_ready,
        "final_pool": pool,
        "all_pre_model_rejections_retained": True,
        "rejections_are_model_or_mechanism_negative_evidence": False,
        "parent_target_oracle_executions": sum(int(row.get("execution_count") or 0) for row in rows),
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def qualify_task(
    config: dict[str, Any],
    task: dict[str, Any],
    materialized: dict[str, Any],
    *,
    runner: Callable[[Path, list[str], dict[str, Any]], dict[str, Any]],
    prompt_counter: Callable[[dict[str, str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    faults: list[str] = []
    artifacts = {str(row.get("label") or ""): row for row in dictionaries(materialized.get("artifacts"))}
    parent_artifact = artifacts.get("parent", {})
    target_artifact = artifacts.get("target", {})
    parent_archive = resolve(str(parent_artifact.get("normalized") or ""))
    target_archive = resolve(str(target_artifact.get("normalized") or ""))
    if not parent_archive.is_file() or sha256_file(parent_archive) != parent_artifact.get("normalized_sha256"):
        faults.append("parent_archive_invalid")
    if not target_archive.is_file() or sha256_file(target_archive) != target_artifact.get("normalized_sha256"):
        faults.append("target_archive_invalid")
    changed_source_paths = source_paths(task)
    if len(changed_source_paths) != 1:
        faults.append("changed_source_file_count_not_one")
    test_paths = changed_test_paths(task)
    if not test_paths:
        faults.append("changed_test_paths_missing")
    change: dict[str, Any] = {}
    split: dict[str, list[str]] = {"visible": [], "hidden": []}
    receipts: dict[str, Any] = {}
    if not faults:
        source_path = changed_source_paths[0]
        parent_text = archive_text(parent_archive, str(parent_artifact.get("source_archive_root") or ""), source_path)
        target_text = archive_text(target_archive, str(target_artifact.get("source_archive_root") or ""), source_path)
        change, change_faults = changed_callable(parent_text, target_text)
        faults.extend(change_faults)
        nodeids: list[str] = []
        for path in test_paths:
            try:
                test_text = archive_text(target_archive, str(target_artifact.get("source_archive_root") or ""), path)
            except (KeyError, OSError, tarfile.TarError):
                continue
            nodeids.extend(pytest_nodeids(path, test_text))
        split = split_test_nodeids(
            str(task.get("selection_digest") or stable_hash(task)), sorted(set(nodeids))
        )
        if not split["visible"] or not split["hidden"]:
            faults.append("visible_hidden_test_split_inadequate")
    execution_count = 0
    if not faults:
        sandbox_config = read_json(resolve(str(config["sandbox_config"])))
        with extracted(parent_archive, str(parent_artifact.get("source_archive_root") or "")) as parent_root, extracted(
            target_archive, str(target_artifact.get("source_archive_root") or "")
        ) as target_root:
            receipts["parent_visible"] = runner(parent_root, split["visible"], sandbox_config)
            receipts["parent_hidden"] = runner(parent_root, split["hidden"], sandbox_config)
            receipts["target_visible"] = runner(target_root, split["visible"], sandbox_config)
            receipts["target_hidden"] = runner(target_root, split["hidden"], sandbox_config)
            execution_count += 4
            transplant_root = Path(tempfile.mkdtemp(prefix="theseus-d1-transplant-", dir="/private/tmp"))
            try:
                extract_archive(parent_archive, transplant_root, str(parent_artifact.get("source_archive_root") or ""))
                apply_callable_transplant(transplant_root / changed_source_paths[0], change)
                receipts["transplant_visible"] = runner(transplant_root, split["visible"], sandbox_config)
                receipts["transplant_hidden"] = runner(transplant_root, split["hidden"], sandbox_config)
                execution_count += 2
            finally:
                import shutil

                shutil.rmtree(transplant_root, ignore_errors=True)
        faults.extend(qualification_faults(receipts))
    qualified = not faults
    task_manifest = (
        build_task_manifest(task, parent_artifact, change, split) if qualified else {}
    )
    evaluator_manifest = (
        build_evaluator_manifest(task, target_artifact, change, split)
        if qualified
        else {}
    )
    prompt_addressability: dict[str, Any] = {}
    if qualified:
        with extracted(
            parent_archive,
            str(parent_artifact.get("source_archive_root") or ""),
        ) as parent_root:
            prompts = render_initial_prompts(config, parent_root, task_manifest)
        prompt_addressability = (
            prompt_counter(prompts)
            if prompt_counter is not None
            else count_initial_prompts_exact(config, prompts)
        )
        if (
            prompt_addressability.get("trigger_state") != "GREEN"
            or strings(prompt_addressability.get("faults"))
        ):
            faults.append("initial_prompt_not_physically_addressable")
            qualified = False
            task_manifest = {}
            evaluator_manifest = {}
        else:
            task_manifest["candidate_visible_context"][
                "initial_prompt_addressability"
            ] = prompt_addressability
    return {
        "campaign_index": task.get("campaign_index"),
        "repository": task.get("repository"),
        "selection_digest": task.get("selection_digest"),
        "qualified": qualified,
        "faults": sorted(set(faults)),
        "changed_callable": change,
        "test_split": split,
        "run_receipts": receipts,
        "execution_count": execution_count,
        "task_manifest": task_manifest,
        "evaluator_manifest": evaluator_manifest,
        "initial_prompt_addressability": prompt_addressability,
        "candidate_or_control_calls": 0,
    }


def changed_callable(parent_text: str, target_text: str) -> tuple[dict[str, Any], list[str]]:
    try:
        parent_tree = ast.parse(parent_text)
        target_tree = ast.parse(target_text)
    except SyntaxError:
        return {}, ["source_AST_parse_failed"]
    parent = callable_inventory(parent_tree, parent_text)
    target = callable_inventory(target_tree, target_text)
    common = sorted(set(parent).intersection(target))
    changed = [name for name in common if ast_fingerprint(parent[name][0]) != ast_fingerprint(target[name][0])]
    if set(parent) != set(target):
        return {}, ["callable_inventory_changed"]
    if len(changed) != 1:
        return {}, [f"changed_callable_count_not_one:{len(changed)}"]
    name = changed[0]
    parent_node, parent_source = parent[name]
    target_node, target_source = target[name]
    if callable_signature(parent_node) != callable_signature(target_node):
        return {}, ["changed_callable_signature_changed"]
    parent_masked = copy.deepcopy(parent_tree)
    target_masked = copy.deepcopy(target_tree)
    replace_callable_body(parent_masked, name)
    replace_callable_body(target_masked, name)
    if ast_fingerprint(parent_masked) != ast_fingerprint(target_masked):
        return {}, ["AST_outside_changed_callable_differs"]
    return {
        "qualified_name": name,
        "kind": type(parent_node).__name__,
        "signature": callable_signature(parent_node),
        "path_start_line": int(parent_node.lineno),
        "path_end_line": int(parent_node.end_lineno or parent_node.lineno),
        "parent_source": parent_source,
        "parent_source_sha256": sha256_text(parent_source),
        "parent_AST_node_count": sum(1 for _ in ast.walk(parent_node)),
        "target_source": target_source,
        "target_source_sha256": sha256_text(target_source),
    }, []


def callable_inventory(tree: ast.AST, text: str) -> dict[str, tuple[ast.AST, str]]:
    found: dict[str, tuple[ast.AST, str]] = {}

    def walk(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*prefix, node.name))
                found[name] = (node, ast.get_source_segment(text, node) or "")
            if isinstance(node, ast.ClassDef):
                walk(node.body, (*prefix, node.name))

    walk(getattr(tree, "body", []), ())
    return found


def replace_callable_body(tree: ast.AST, qualified_name: str) -> None:
    target: ast.AST | None = None

    def walk(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        nonlocal target
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ".".join((*prefix, node.name)) == qualified_name:
                    target = node
            if isinstance(node, ast.ClassDef):
                walk(node.body, (*prefix, node.name))

    walk(getattr(tree, "body", []), ())
    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        target.body = [ast.Pass()]


def callable_signature(node: ast.AST) -> str:
    clone = copy.deepcopy(node)
    if isinstance(clone, (ast.FunctionDef, ast.AsyncFunctionDef)):
        clone.body = [ast.Pass()]
        return ast.unparse(clone).split("\n", 1)[0].removesuffix(":")
    return ""


def ast_fingerprint(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def pytest_nodeids(path: str, text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    rows: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            rows.append(f"{path}::{node.name}")
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    rows.append(f"{path}::{node.name}::{child.name}")
    return rows


def split_test_nodeids(identity_value: str, nodeids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(
        nodeids,
        key=lambda nodeid: hashlib.sha256(f"{identity_value}||{nodeid}".encode()).hexdigest(),
    )
    return {"visible": ordered[::2], "hidden": ordered[1::2]}


def qualification_faults(receipts: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    for key, receipt in receipts.items():
        if receipt.get("boundary_hit") is True:
            faults.append(f"sandbox_boundary_hit:{key}")
    for key in ("parent_visible", "parent_hidden"):
        if receipts.get(key, {}).get("passed") is not False:
            faults.append(f"{key}_did_not_fail")
    for key in ("target_visible", "target_hidden", "transplant_visible", "transplant_hidden"):
        if receipts.get(key, {}).get("passed") is not True:
            faults.append(f"{key}_did_not_pass")
    return faults


def run_pytest_sandboxed(root: Path, nodeids: list[str], sandbox_config: dict[str, Any]) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import theseus_d1_evaluator_sandbox as sandbox

    command = [str(sandbox_config["python"]), "-m", "pytest", "-q", *nodeids]
    receipt = sandbox.run_sandboxed(command, workdir=root, config=sandbox_config)
    receipt["passed"] = receipt.get("returncode") == 0 and receipt.get("boundary_hit") is False
    receipt["pytest_nodeids_sha256"] = stable_hash(nodeids)
    return receipt


def apply_callable_transplant(path: Path, change: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = int(change["path_start_line"])
    end = int(change["path_end_line"])
    replacement = str(change["target_source"]).splitlines()
    if replacement:
        prefix = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        replacement[0] = prefix + replacement[0]
    path.write_text("\n".join([*lines[: start - 1], *replacement, *lines[end:]]) + "\n", encoding="utf-8")


def build_task_manifest(
    task: dict[str, Any], parent: dict[str, Any], change: dict[str, Any], split: dict[str, list[str]]
) -> dict[str, Any]:
    request = "\n\n".join(
        value.strip()
        for value in (str(task.get("pull_request_title") or ""), str(task.get("pull_request_body") or ""))
        if value.strip()
    )
    path = source_paths(task)[0]
    return {
        "policy": "project_theseus_p4_cognitive_compilation_task_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": "d1-" + str(task.get("selection_digest") or "")[:24],
        "campaign_index": task.get("campaign_index"),
        "partition": "d1_fresh_source_disjoint_qualification",
        "family": "bounded_python_correctness_repair",
        "natural_request": request,
        "allowed_effect_paths": [path],
        "obligations": [
            {"id": "O1", "kind": "require", "text": request},
            {"id": "O2", "kind": "preserve", "text": "Preserve behavior not changed by the requested repair."},
            {"id": "O3", "kind": "non_goal", "text": f"Do not modify files outside {path}."},
        ],
        "obligation_dependencies": [{"before": "O1", "after": "O2"}],
        "candidate_visible_context": {
            "reads": [{"path": path, "start_line": 1, "end_line": 10**9}],
            "searches": [{"literal": change["qualified_name"].split(".")[-1], "paths": [path]}],
            "physical_context_addressability_checked_before_final_pool_seal": True,
            "complete_repair_prompt_addressability_recomputed_at_generation": True,
            "project_selected_character_or_token_cap": None,
        },
        "source_archive": parent.get("normalized"),
        "source_archive_sha256": parent.get("normalized_sha256"),
        "source_archive_root": parent.get("source_archive_root"),
        "visible_verifier": {
            "command": ["python3", "D1_SANDBOXED_PYTEST", *split["visible"]],
            "candidate_prompt_visibility": False,
            "answer_specific": True,
            "project_selected_quality_token_cap": None,
        },
        "visible_feedback_map": [
            {"marker": "FAILED", "obligation_ids": ["O1", "O2"]}
        ],
        "semantic_ir_contract": {
            "version": "theseus_semantic_ir_v2r2_labeled",
            "maximum_symbol_nodes": change["parent_AST_node_count"],
            "maximum_units": 1,
            "source_target_obligation_loss_and_dependency_identity_required": True,
            "project_selected_quality_token_cap": None,
        },
        "source_provenance": {
            "repository": task.get("repository"),
            "url": task.get("repository_url"),
            "license_spdx": task.get("license_spdx"),
            "pull_request": task.get("pull_request"),
            "pull_request_url": task.get("pull_request_url"),
            "revision": task.get("parent_revision"),
            "target_revision": task.get("target_revision"),
            "merged_utc": task.get("merged_utc"),
        },
        "contamination_screen": {
            "task_selected_before_any_candidate_or_control": True,
            "source_disjoint_from_all_prior_theseus_sources": True,
            "target_test_or_source_candidate_visible": False,
            "eligible_for_training": False,
        },
        "maximum_inference": "One fresh D1 observation for the exact frozen Python callable-repair implementation only.",
    }


def build_evaluator_manifest(
    task: dict[str, Any], target: dict[str, Any], change: dict[str, Any], split: dict[str, list[str]]
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_d1_route_blind_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "target_archive": target.get("normalized"),
        "target_archive_sha256": target.get("normalized_sha256"),
        "target_archive_root": target.get("source_archive_root"),
        "hidden_pytest_nodeids": split["hidden"],
        "test_overlay_paths": changed_test_paths(task),
        "visible_pytest_nodeids_sha256": stable_hash(split["visible"]),
        "oracle_callable_source": change.get("target_source"),
        "oracle_callable_source_sha256": change.get("target_source_sha256"),
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "target_source_candidate_visible": False,
            "test_source_candidate_visible": False,
            "route_label_passed_to_scoring": False,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "maximum_inference": "Evaluator construct-validity for one sealed D1 task only.",
    }


def build_pool(
    config: dict[str, Any], registry: dict[str, Any], qualified: list[dict[str, Any]], instrument: dict[str, Any]
) -> dict[str, Any]:
    tasks = []
    for index, row in enumerate(qualified, 1):
        tasks.append(
            {
                "campaign_index": index,
                "source_campaign_index": row.get("campaign_index"),
                "repository": row.get("repository"),
                "selection_digest": row.get("selection_digest"),
                "task_manifest": row.get("task_manifest"),
                "evaluator_manifest": row.get("evaluator_manifest"),
            }
        )
    return {
        "policy": POOL_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_OR_CONTROL_GENERATION",
        "campaign_id": instrument.get("campaign_id"),
        "claim_id": instrument.get("claim_id"),
        "task_count": len(tasks),
        "distinct_repository_count": len({str(row.get("repository") or "").lower() for row in tasks}),
        "tasks": tasks,
        "source_registry_sha256": stable_hash(registry),
        "candidate_or_control_calls": 0,
        "post_candidate_task_replacement_allowed": False,
        "automatic_book_support_promotion": False,
        "project_selected_quality_token_cap": None,
    }


def persist_final_pool(config: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    destination = resolve(str(config["sealed_task_root"]))
    if destination.exists():
        raise FileExistsError(f"sealed_task_root_already_exists:{relative(destination)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent)
    )
    persisted = copy.deepcopy(pool)
    try:
        for row in dictionaries(persisted.get("tasks")):
            index = int(row.get("campaign_index") or 0)
            digest = str(row.get("selection_digest") or "")[:16]
            task_name = f"d1_{index:02d}_{digest}_task.json"
            evaluator_name = f"d1_{index:02d}_{digest}_evaluator.json"
            task_path = temporary / task_name
            evaluator_path = temporary / evaluator_name
            task_manifest = mapping(row.pop("task_manifest", {}))
            evaluator_manifest = mapping(row.pop("evaluator_manifest", {}))
            write_json_exclusive(task_path, task_manifest)
            evaluator_manifest["task_manifest"] = str(destination / task_name)
            evaluator_manifest["task_manifest_sha256"] = sha256_file(task_path)
            write_json_exclusive(evaluator_path, evaluator_manifest)
            row["task"] = str(destination / task_name)
            row["task_sha256"] = sha256_file(task_path)
            row["evaluator"] = str(destination / evaluator_name)
            row["evaluator_sha256"] = sha256_file(evaluator_path)
        os.replace(temporary, destination)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return persisted


def audit_existing_pool(config: dict[str, Any]) -> dict[str, Any]:
    pool_path = resolve(str(config.get("final_task_pool") or ""))
    if not pool_path.is_file():
        return {"present": False, "passed": False, "faults": []}
    pool = read_json(pool_path)
    faults: list[str] = []
    if pool.get("policy") != POOL_POLICY or pool.get("state") != (
        "SEALED_BEFORE_CANDIDATE_OR_CONTROL_GENERATION"
    ):
        faults.append("existing_pool_policy_or_state_invalid")
    tasks = dictionaries(pool.get("tasks"))
    declared = int(pool.get("task_count") or 0)
    if declared != 44 or len(tasks) != declared:
        faults.append("existing_pool_task_count_invalid")
    if len({str(row.get("repository") or "").lower() for row in tasks}) != declared:
        faults.append("existing_pool_repository_count_invalid")
    for row in tasks:
        for path_key, digest_key in (
            ("task", "task_sha256"),
            ("evaluator", "evaluator_sha256"),
        ):
            path = resolve(str(row.get(path_key) or ""))
            if not path.is_file() or sha256_file(path) != row.get(digest_key):
                faults.append(f"existing_pool_binding_invalid:{path_key}")
    if int(pool.get("candidate_or_control_calls") or 0) != 0:
        faults.append("existing_pool_candidate_calls_nonzero")
    if pool.get("post_candidate_task_replacement_allowed") is not False:
        faults.append("existing_pool_replacement_allowed")
    if pool.get("project_selected_quality_token_cap") is not None:
        faults.append("existing_pool_quality_token_cap_present")
    return {
        "present": True,
        "passed": not faults,
        "faults": sorted(set(faults)),
        "path": relative(pool_path),
        "sha256": sha256_file(pool_path),
        "task_count": declared,
    }


def audit_source_inputs(registry: dict[str, Any], materialization: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    if not registry:
        faults.append("source_registry_missing")
    if not materialization:
        faults.append("source_materialization_missing")
    if registry and registry.get("policy") != "project_theseus_d1_online_source_registry_v1":
        faults.append("source_registry_policy_invalid")
    if materialization and (
        materialization.get("policy") != "project_theseus_d1_source_materialization_v1"
        or materialization.get("trigger_state") != "GREEN"
        or strings(materialization.get("faults"))
    ):
        faults.append("source_materialization_not_green")
    if registry and materialization:
        tasks = dictionaries(registry.get("tasks"))
        materialized = dictionaries(materialization.get("tasks"))
        if len(tasks) != len(materialized):
            faults.append("source_materialization_task_count_mismatch")
    return {"passed": not faults, "faults": sorted(set(faults))}


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVELY_BOUND_BEFORE_D1_ARCHIVE_OR_EVALUATOR_EXECUTION":
        faults.append("state_invalid")
    shape = mapping(config.get("task_shape"))
    if shape.get("candidate_visible_test_source") is not False or shape.get("candidate_visible_target_source") is not False:
        faults.append("answer_identifying_surface_candidate_visible")
    cohort = mapping(config.get("cohort_policy"))
    if cohort.get("candidate_or_control_calls_before_final_pool_seal") is not False:
        faults.append("candidate_calls_before_pool_seal_allowed")
    if cohort.get("all_pre_model_rejections_retained") is not True:
        faults.append("pre_model_rejections_not_retained")
    if cohort.get("post_candidate_task_replacement_allowed") is not False:
        faults.append("post_candidate_replacement_allowed")
    execution = mapping(config.get("execution"))
    if execution.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    addressability = mapping(config.get("prompt_addressability"))
    if addressability.get("project_selected_quality_token_cap") is not None:
        faults.append("prompt_addressability_quality_cap_present")
    if addressability.get("non_addressable_prompt_is_model_or_mechanism_failure") is not False:
        faults.append("prompt_addressability_negative_evidence_invalid")
    if set(strings(addressability.get("required_learned_arms"))) != {
        "direct_target_generation",
        "natural_language_plan_control",
        "typed_semantic_ir_treatment",
    }:
        faults.append("prompt_addressability_arm_set_invalid")
    for path_key, hash_key in (
        ("python", "python_sha256"),
        ("counter", "counter_sha256"),
        ("worker_config", "worker_config_sha256"),
        ("causal_instrument", "causal_instrument_sha256"),
    ):
        owner = resolve(str(addressability.get(path_key) or ""))
        if not owner.is_file() or sha256_file(owner) != str(
            addressability.get(hash_key) or ""
        ):
            faults.append(f"prompt_addressability_binding_invalid:{path_key}")
    authority = mapping(config.get("authority"))
    if authority.get("user_or_operator_approval_required") is not False:
        faults.append("user_gate_present")
    for key in (
        "candidate_or_control_calls_authorized", "external_inference_authorized",
        "teacher_calls_authorized", "training_rows_authorized", "serving_authorized",
        "D2_authorized", "book_support_promotion_authorized",
    ):
        if authority.get(key) is not False:
            faults.append(f"cross_stage_authority_present:{key}")
    return faults


def render_initial_prompts(
    config: dict[str, Any], root: Path, task: dict[str, Any]
) -> dict[str, str]:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import theseus_assistant_p2a as p2a
    import theseus_p4_cognitive_compilation as p4
    import theseus_p4s_cognitive_compilation as p4s
    import theseus_p4v2r2_cognitive_compilation as causal

    addressability = mapping(config.get("prompt_addressability"))
    overlay = read_json(resolve(str(addressability.get("causal_instrument") or "")))
    base = read_json(resolve(str(overlay.get("base_instrument") or "")))
    local = read_json(resolve(str(base.get("base_local_instrument") or "")))
    protocol = mapping(local.get("candidate_protocol"))
    original_symbol_table = p4.semantic_symbol_table
    try:
        p4.semantic_symbol_table = p4s.semantic_scope_symbol_table
        common = p4.render_common_context(root, task)
    finally:
        p4.semantic_symbol_table = original_symbol_table
    return {
        arm: causal.render_arm_prompt(arm, task, common, protocol)
        for arm in p2a.strings(addressability.get("required_learned_arms"))
    }


def count_initial_prompts_exact(
    config: dict[str, Any], prompts: dict[str, str]
) -> dict[str, Any]:
    addressability = mapping(config.get("prompt_addressability"))
    python = resolve(str(addressability.get("python") or ""))
    counter = resolve(str(addressability.get("counter") or ""))
    worker = resolve(str(addressability.get("worker_config") or ""))
    with tempfile.TemporaryDirectory(
        prefix="theseus-d1-prompt-addressability-", dir="/private/tmp"
    ) as temporary:
        input_path = Path(temporary) / "prompts.json"
        output_path = Path(temporary) / "receipt.json"
        write_json(input_path, {"prompts": prompts})
        completed = subprocess.run(
            [
                str(python),
                str(counter),
                "--worker-config",
                str(worker),
                "--prompts",
                str(input_path),
                "--out",
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if not output_path.is_file():
            return {
                "policy": "project_theseus_exact_local_prompt_addressability_v1",
                "trigger_state": "RED",
                "faults": [
                    f"prompt_counter_failed:{completed.returncode}",
                ],
                "candidate_or_control_calls": 0,
                "project_selected_quality_token_cap": None,
            }
        report = read_json(output_path)
        if completed.returncode != 0 and report.get("trigger_state") == "GREEN":
            report["trigger_state"] = "RED"
            report["faults"] = sorted(
                set(strings(report.get("faults")) + ["prompt_counter_returncode_nonzero"])
            )
        return report


def source_paths(task: dict[str, Any]) -> list[str]:
    return sorted(
        str(row.get("filename") or "")
        for row in dictionaries(task.get("changed_files"))
        if str(row.get("status") or "") == "modified"
        and str(row.get("filename") or "").endswith(".py")
        and not is_test_path(str(row.get("filename") or ""))
    )


def changed_test_paths(task: dict[str, Any]) -> list[str]:
    return sorted(
        str(row.get("filename") or "")
        for row in dictionaries(task.get("changed_files"))
        if str(row.get("status") or "") != "removed"
        and str(row.get("filename") or "").endswith(".py")
        and is_test_path(str(row.get("filename") or ""))
    )


def is_test_path(value: str) -> bool:
    return any(part.lower().startswith("test") or part.lower() in {"tests", "spec", "specs"} for part in PurePosixPath(value).parts)


class extracted:
    def __init__(self, archive: Path, root_name: str):
        self.archive = archive
        self.root_name = root_name
        self.temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self.temporary = tempfile.TemporaryDirectory(prefix="theseus-d1-evaluator-", dir="/private/tmp")
        root = Path(self.temporary.name)
        extract_archive(self.archive, root, self.root_name)
        return root

    def __exit__(self, *_: Any) -> None:
        if self.temporary:
            self.temporary.cleanup()


def extract_archive(archive: Path, destination: Path, root_name: str) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != root_name or ".." in path.parts or path.is_absolute():
                raise ValueError("archive_member_outside_declared_root")
            if not member.isfile() and not member.isdir():
                raise ValueError("archive_non_regular_member")
            relative_path = Path(*path.parts[1:])
            target = destination / relative_path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("archive_member_unreadable")
            target.write_bytes(source.read())


def archive_text(archive: Path, root_name: str, relative_path: str) -> str:
    name = f"{root_name}/{relative_path}"
    with tarfile.open(archive, "r:gz") as handle:
        source = handle.extractfile(name)
        if source is None:
            raise KeyError(name)
        return source.read().decode("utf-8")


def required_task_count(instrument: dict[str, Any]) -> int:
    return int(mapping(instrument.get("power_design")).get("design_derived_cohort_size") or 0)


def input_identity(path: Path, value: dict[str, Any], override: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "path": relative(path),
        "present": bool(value),
        "sha256": stable_hash(value) if override is not None and value else sha256_file(path),
    }


def identity(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def read_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if isinstance(row, str)] if isinstance(value, list) else []


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state", "activation_state", "execution_authorized",
            "qualified_task_count", "required_task_count", "pre_model_rejection_count",
            "candidate_or_control_calls", "project_selected_quality_token_cap", "faults",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
