#!/usr/bin/env python3
"""Prospectively frozen P2 real-work pair on the canonical assistant runtime."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_assistant_route_integrity as route_integrity
import theseus_assistant_runtime as assistant_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "theseus_assistant_p2_canary.json"
POLICY = "project_theseus_p2_real_work_canary_run_v1"
ARMS = (route_integrity.DIRECT_MODE, route_integrity.INTEGRATED_MODE)
FORBIDDEN_TASK_FIELDS = {
    "answer",
    "expected",
    "hidden_tests",
    "solution",
    "source_task_id",
    "tests",
}


class CanaryFault(ValueError):
    """A prospective P2 contract or execution fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=rel(DEFAULT_MANIFEST))
    parser.add_argument("--repair-amendment", default="")
    parser.add_argument("--out", default="reports/theseus_assistant_p2_canary.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    repair_path = resolve(args.repair_amendment) if args.repair_amendment else None
    report = audit_manifest(manifest_path) if args.audit_only and repair_path is None else (
        audit_repair_amendment(manifest_path, repair_path)
        if args.audit_only and repair_path is not None
        else run_canary(manifest_path, repair_path)
    )
    write_json(resolve(args.out), report)
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = read_json(manifest_path)
    faults: list[str] = []
    if manifest.get("policy") != "project_theseus_p2_real_work_canary_v1":
        faults.append("manifest_policy_invalid")
    if manifest.get("state") != "PROSPECTIVELY_BOUND_BEFORE_CANDIDATE_GENERATION":
        faults.append("manifest_not_prospectively_bound")
    if tuple(strings(manifest.get("arm_order"))) != ARMS:
        faults.append("arm_order_invalid")
    task = mapping(manifest.get("task"))
    if FORBIDDEN_TASK_FIELDS.intersection(task):
        faults.append("answer_identifying_task_field_present")
    natural_request = str(task.get("natural_request") or "")
    if not natural_request.strip():
        faults.append("natural_request_missing")
    parent = mapping(manifest.get("parent_source"))
    overlay = resolve(str(parent.get("overlay_patch") or ""))
    if sha256_file(overlay) != str(parent.get("overlay_patch_sha256") or ""):
        faults.append("parent_overlay_digest_mismatch")
    commit = str(parent.get("commit") or "")
    if git("rev-parse", f"{commit}^{{commit}}") != commit:
        faults.append("parent_commit_unavailable")
    instrument = mapping(manifest.get("frozen_instrument"))
    runtime_config = read_json(resolve(str(instrument.get("runtime_config") or "")))
    local = mapping(runtime_config.get("local_inference"))
    model_contract = route_integrity.load_model_contract(
        str(local.get("worker_config") or ""),
        str(local.get("runtime_preflight") or ""),
        maximum_tokens=int(local.get("product_maximum_tokens") or 0),
        required_repo_id=str(local.get("required_repo_id") or ""),
        required_revision=str(local.get("required_revision") or ""),
        required_snapshot_manifest_sha256=str(local.get("required_snapshot_manifest_sha256") or ""),
    )
    identity = mapping(model_contract.get("identity"))
    required_identity = {
        "repo_id": instrument.get("model_repo_id"),
        "revision": instrument.get("model_revision"),
        "identity_sha256": instrument.get("model_identity_sha256"),
        "snapshot_manifest_sha256": instrument.get("snapshot_manifest_sha256"),
        "decoder_sha256": instrument.get("decoder_sha256"),
    }
    if model_contract.get("ready") is not True:
        faults.append("frozen_model_contract_not_ready")
    if any(identity.get(key) != value for key, value in required_identity.items()):
        faults.append("frozen_model_identity_mismatch")
    if int(local.get("product_maximum_tokens") or 0) != int(instrument.get("maximum_generation_tokens") or 0):
        faults.append("generation_budget_mismatch")
    context_receipt: dict[str, Any] = {}
    if not faults:
        try:
            with tempfile.TemporaryDirectory(prefix="theseus-p2-audit-") as tmp:
                snapshot = Path(tmp) / "parent"
                create_parent_snapshot(commit, overlay, snapshot)
                context = visible_source_context(snapshot, task)
                context_receipt = {
                    "visible_source_sha256": sha256_text(context),
                    "visible_source_characters": len(context),
                    "visible_source_paths": [str(row.get("path") or "") for row in dicts(task.get("visible_source"))],
                }
        except (OSError, subprocess.SubprocessError, tarfile.TarError, CanaryFault) as exc:
            faults.append(f"parent_snapshot_invalid:{type(exc).__name__}")
    return {
        "policy": "project_theseus_p2_real_work_canary_alignment_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "manifest_sha256": sha256_file(manifest_path),
        "parent_commit": commit,
        "parent_overlay_sha256": sha256_file(overlay),
        "natural_request_sha256": sha256_text(natural_request),
        "model_identity": identity,
        "context_receipt": context_receipt,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def audit_repair_amendment(manifest_path: Path, amendment_path: Path) -> dict[str, Any]:
    amendment = read_json(amendment_path)
    faults: list[str] = []
    if amendment.get("policy") != "project_theseus_p2_loop_efficiency_repair_v1":
        faults.append("repair_policy_invalid")
    if amendment.get("state") != "PROSPECTIVELY_BOUND_BEFORE_REPAIR_GENERATION":
        faults.append("repair_not_prospectively_bound")
    if resolve(str(amendment.get("base_manifest") or "")) != manifest_path:
        faults.append("repair_base_manifest_path_mismatch")
    if sha256_file(manifest_path) != str(amendment.get("base_manifest_sha256") or ""):
        faults.append("repair_base_manifest_digest_mismatch")
    prior_run = resolve(str(amendment.get("prior_run") or ""))
    prior_evaluation = resolve(str(amendment.get("prior_evaluation") or ""))
    if sha256_file(prior_run) != str(amendment.get("prior_run_sha256") or ""):
        faults.append("repair_prior_run_digest_mismatch")
    if sha256_file(prior_evaluation) != str(amendment.get("prior_evaluation_sha256") or ""):
        faults.append("repair_prior_evaluation_digest_mismatch")
    evaluation = read_json(prior_evaluation)
    if evaluation.get("disposition") != "P2_LOOP_EFFICIENCY_REPAIR_AUTHORIZED":
        faults.append("repair_not_authorized_by_independent_evaluator")
    changes = mapping(amendment.get("frozen_changes"))
    required_true = (
        "same_task",
        "same_parent",
        "same_visible_source",
        "same_arm_order",
        "same_model_repo_revision_snapshot",
        "same_decoder",
        "same_effect_sandbox",
        "same_independent_evaluator",
        "cumulative_product_budgets",
    )
    if any(changes.get(key) is not True for key in required_true):
        faults.append("repair_matched_boundary_not_frozen")
    if changes.get("candidate_output_encoding") != "raw_git_unified_diff":
        faults.append("repair_output_encoding_invalid")
    if int(changes.get("same_maximum_generation_tokens") or 0) != 512:
        faults.append("repair_generation_budget_changed")
    if not str(changes.get("prompt_suffix") or "").strip():
        faults.append("repair_prompt_suffix_missing")
    return {
        "policy": "project_theseus_p2_loop_efficiency_repair_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "base_manifest_sha256": sha256_file(manifest_path),
        "repair_amendment_sha256": sha256_file(amendment_path),
        "prior_run_sha256": sha256_file(prior_run),
        "prior_evaluation_sha256": sha256_file(prior_evaluation),
        "counters": zero_counters(),
    }


def run_canary(manifest_path: Path, repair_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = read_json(manifest_path)
    alignment = audit_manifest(manifest_path)
    if alignment.get("trigger_state") != "GREEN":
        return invalid_report(manifest_path, alignment, "alignment_audit_red")
    repair: dict[str, Any] = {}
    repair_audit: dict[str, Any] = {}
    prior_report: dict[str, Any] = {}
    if repair_path is not None:
        repair = read_json(repair_path)
        repair_audit = audit_repair_amendment(manifest_path, repair_path)
        if repair_audit.get("trigger_state") != "GREEN":
            return invalid_report(manifest_path, repair_audit, "repair_amendment_audit_red")
        prior_report = read_json(resolve(str(repair.get("prior_run") or "")))
    task = mapping(manifest.get("task"))
    parent = mapping(manifest.get("parent_source"))
    overlay = resolve(str(parent.get("overlay_patch") or ""))
    with tempfile.TemporaryDirectory(prefix="theseus-p2-visible-") as tmp:
        snapshot = Path(tmp) / "parent"
        create_parent_snapshot(str(parent.get("commit") or ""), overlay, snapshot)
        source_context = visible_source_context(snapshot, task)
    candidate_prompt = render_candidate_prompt(str(task.get("natural_request") or ""), source_context)
    repair_changes = mapping(repair.get("frozen_changes"))
    if repair_changes:
        candidate_prompt += "\n\n[prospectively_frozen_repair_amendment]\n" + str(repair_changes.get("prompt_suffix") or "")
    output_encoding = str(repair_changes.get("candidate_output_encoding") or "json_candidate_envelope")
    skip_dogfood = repair_changes.get("skip_dogfood") is not False
    run_suffix = "r1" if repair else "r0"
    attempts: list[dict[str, Any]] = []
    for arm in strings(manifest.get("arm_order")):
        attempts.append(run_arm(arm, candidate_prompt, task, manifest, output_encoding, skip_dogfood, run_suffix))
    direct = next((row for row in attempts if row.get("arm_id") == route_integrity.DIRECT_MODE), {})
    integrated = next((row for row in attempts if row.get("arm_id") == route_integrity.INTEGRATED_MODE), {})
    pair = route_integrity.compare_matched_pair(
        mapping(mapping(direct.get("runtime_report")).get("route_integrity")),
        mapping(mapping(integrated.get("runtime_report")).get("route_integrity")),
    )
    budgets = product_budget_receipt(
        attempts,
        mapping(manifest.get("product_budgets")),
        started,
        prior_report=prior_report,
    )
    faults = []
    if pair.get("ready") is not True:
        faults.append("matched_route_pair_invalid")
    if budgets.get("ready") is not True:
        faults.append("product_budget_exceeded")
    if any(row.get("runtime_trigger_state") == "RED" for row in attempts):
        faults.append("runtime_arm_red")
    sealed = sum(bool(mapping(row.get("candidate")).get("candidate_seal")) for row in attempts)
    trigger = "RED" if faults else ("GREEN" if sealed == len(attempts) else "YELLOW")
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger,
        "faults": faults,
        "scope": "one reusable L0 product canary; no D1, D2, subsystem, model, or general utility claim",
        "experiment_id": repair.get("experiment_id") or manifest.get("experiment_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "repair_amendment_sha256": sha256_file(repair_path) if repair_path else "",
        "repair_amendment_audit": repair_audit,
        "alignment_audit": alignment,
        "task": {
            "opaque_task_id": task.get("opaque_task_id"),
            "task_pair_id": task.get("task_pair_id"),
            "family": task.get("family"),
            "natural_request_sha256": sha256_text(str(task.get("natural_request") or "")),
            "candidate_prompt_sha256": sha256_text(candidate_prompt),
            "variant_results": attempts,
        },
        "matched_pair": pair,
        "product_budget": budgets,
        "denominators": {
            "tasks": 1,
            "arms": len(attempts),
            "model_calls": sum(int(mapping(row.get("resource_metrics")).get("model_calls") or 0) for row in attempts),
            "cumulative_model_calls": sum(
                int(row.get("model_calls") or 0)
                for row in mapping(budgets.get("arms")).values()
                if isinstance(row, dict)
            ),
            "sealed_candidates": sealed,
            "malformed_or_abstained_candidates": len(attempts) - sealed,
        },
        "ablation_boundary": manifest.get("ablation_boundary"),
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def run_arm(
    arm: str,
    prompt: str,
    task: dict[str, Any],
    manifest: dict[str, Any],
    output_encoding: str,
    skip_dogfood: bool,
    run_suffix: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise CanaryFault("unsupported_arm")
    session = f"p2_001_{run_suffix}_{arm}"
    out = ROOT / "runtime" / "p2_real_work_canary" / f"{session}_runtime.json"
    markdown = out.with_suffix(".md")
    events = ROOT / "runtime" / "p2_real_work_canary" / f"{session}_events.jsonl"
    viea = ROOT / "runtime" / "p2_real_work_canary" / f"{session}_viea.jsonl"
    argv = [
        "theseus_assistant_runtime.py",
        "--execution-mode", arm,
        "--intent", "code",
        "--session-id", session,
        "--prompt", prompt,
        "--out", rel(out),
        "--markdown-out", rel(markdown),
        "--events-out", rel(events),
        "--viea-trace-out", rel(viea),
        "--skip-context-refresh",
    ]
    if skip_dogfood:
        argv.append("--skip-dogfood")
    before = time.perf_counter()
    old_argv = sys.argv
    capture = io.StringIO()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            returncode = assistant_runtime.main()
    finally:
        sys.argv = old_argv
    elapsed_ms = round((time.perf_counter() - before) * 1000.0, 3)
    runtime_report = read_json(out)
    assistant_text = str(runtime_report.get("assistant_text") or "")
    parsed, parse_faults = parse_candidate_output(
        assistant_text,
        strings(task.get("allowed_effect_paths")),
        mapping(manifest.get("candidate_output_schema")),
        output_encoding=output_encoding,
    )
    candidate = seal_candidate(parsed, task, arm, prompt) if parsed else {}
    backend_path = resolve(str(mapping(runtime_report.get("generation_backend")).get("out") or ""))
    backend = read_json(backend_path)
    metrics = mapping(backend.get("metrics"))
    return {
        "arm_id": arm,
        "candidate_output_sha256": sha256_text(assistant_text),
        "candidate_output_encoding": output_encoding,
        "candidate": candidate,
        "parse_faults": parse_faults,
        "runtime_trigger_state": runtime_report.get("trigger_state"),
        "runtime_returncode": returncode,
        "runtime_report_sha256": sha256_file(out),
        "runtime_report_path": rel(out),
        "runtime_report": {
            "trigger_state": runtime_report.get("trigger_state"),
            "inputs": runtime_report.get("inputs"),
            "summary": runtime_report.get("summary"),
            "generation_request_binding": runtime_report.get("generation_request_binding"),
            "route_integrity": runtime_report.get("route_integrity"),
        },
        "resource_metrics": {
            "model_calls": int(metrics.get("local_model_inference_calls") or 0),
            "prompt_tokens": metrics.get("prompt_tokens"),
            "generated_tokens": metrics.get("generated_tokens"),
            "generation_wall_ms": metrics.get("generation_wall_ms"),
            "arm_wall_ms": elapsed_ms,
            "mlx_peak_memory_gib": metrics.get("mlx_peak_memory_gib"),
            "tool_calls": 0,
            "verification_count": 1,
        },
    }


def parse_candidate_output(
    text: str,
    allowed_paths: list[str],
    schema: dict[str, Any],
    *,
    output_encoding: str = "json_candidate_envelope",
) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    raw = str(text or "").strip()
    if output_encoding == "raw_git_unified_diff":
        fenced_diff = re.search(r"```(?:diff)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
        patch = (fenced_diff.group(1) if fenced_diff else raw).strip()
        proposed = patch_paths(patch)
        if not (patch.startswith("diff --git ") or patch.startswith("--- a/")):
            faults.append("candidate_git_diff_header_missing")
        if not proposed:
            faults.append("candidate_patch_paths_missing")
        if not set(proposed).issubset(set(allowed_paths)):
            faults.append("candidate_path_outside_authority")
        if len(proposed) > int(schema.get("maximum_proposed_paths") or 0):
            faults.append("candidate_too_many_paths")
        if len(patch.encode("utf-8")) > int(schema.get("maximum_patch_bytes") or 0):
            faults.append("candidate_patch_too_large")
        return (
            {
                "patch_unified_diff": patch + "\n",
                "proposed_paths": proposed,
                "verification_commands": [],
                "abstained": False,
            }
            if not faults
            else {},
            faults,
        )
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    candidate_text = fenced.group(1) if fenced else raw
    if not candidate_text.startswith("{"):
        start, end = candidate_text.find("{"), candidate_text.rfind("}")
        candidate_text = candidate_text[start : end + 1] if start >= 0 and end > start else ""
    try:
        value = json.loads(candidate_text)
    except (json.JSONDecodeError, TypeError):
        return {}, ["candidate_json_invalid"]
    if not isinstance(value, dict):
        return {}, ["candidate_not_object"]
    required = set(strings(schema.get("required_fields")))
    if not required.issubset(value):
        faults.append("candidate_required_fields_missing")
    if value.get("abstained") is True:
        faults.append("candidate_abstained")
    patch = str(value.get("patch_unified_diff") or "")
    if not patch.strip():
        faults.append("candidate_patch_missing")
    if len(patch.encode("utf-8")) > int(schema.get("maximum_patch_bytes") or 0):
        faults.append("candidate_patch_too_large")
    proposed = strings(value.get("proposed_paths"))
    if sorted(proposed) != sorted(set(proposed)) or not proposed:
        faults.append("candidate_proposed_paths_invalid")
    if not set(proposed).issubset(set(allowed_paths)):
        faults.append("candidate_path_outside_authority")
    if len(proposed) > int(schema.get("maximum_proposed_paths") or 0):
        faults.append("candidate_too_many_paths")
    return (value if not faults else {}), faults


def seal_candidate(candidate: dict[str, Any], task: dict[str, Any], arm: str, prompt: str) -> dict[str, Any]:
    output = {
        "patch_unified_diff": str(candidate.get("patch_unified_diff") or ""),
        "proposed_paths": strings(candidate.get("proposed_paths")),
        "verification_commands": strings(candidate.get("verification_commands")),
        "abstained": bool(candidate.get("abstained")),
    }
    seal = {
        "policy": "project_theseus_p2_candidate_seal_v1",
        "candidate_output_sha256": stable_hash(output),
        "natural_request_sha256": sha256_text(str(task.get("natural_request") or "")),
        "candidate_prompt_sha256": sha256_text(prompt),
        "parent_source_binding": "commit_plus_frozen_overlay",
    }
    return {
        "candidate_output": output,
        "candidate_seal": seal,
        "worker_id": "frozen_tmax_canonical_assistant",
        "arm_id_attached_after_generation": arm,
        "candidate_authored_success_flags_trusted": False,
    }


def product_budget_receipt(
    attempts: list[dict[str, Any]],
    budget: dict[str, Any],
    pair_started: float,
    *,
    prior_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prior_attempts = {
        str(row.get("arm_id") or ""): mapping(row.get("resource_metrics"))
        for row in dicts(mapping(prior_report).get("task", {}).get("variant_results"))
    }
    arm_checks: dict[str, Any] = {}
    for row in attempts:
        metrics = mapping(row.get("resource_metrics"))
        prior = prior_attempts.get(str(row.get("arm_id") or ""), {})
        calls = int(metrics.get("model_calls") or 0) + int(prior.get("model_calls") or 0)
        wall = float(metrics.get("arm_wall_ms") or 0.0) + float(prior.get("arm_wall_ms") or 0.0)
        arm_checks[str(row.get("arm_id") or "")] = {
            "model_calls": calls,
            "arm_wall_ms": wall,
            "model_calls_within_budget": 0 <= calls <= int(budget.get("maximum_model_calls_per_arm") or 0),
            "wall_within_budget": 0.0 <= wall <= float(budget.get("maximum_arm_wall_ms") or 0.0),
        }
    prior_pair_wall = float(mapping(mapping(prior_report).get("product_budget")).get("pair_wall_ms") or 0.0)
    pair_wall = round(prior_pair_wall + (time.perf_counter() - pair_started) * 1000.0, 3)
    ready = bool(arm_checks) and all(
        row["model_calls_within_budget"] and row["wall_within_budget"] for row in arm_checks.values()
    ) and pair_wall <= float(budget.get("maximum_pair_wall_ms") or 0.0)
    return {
        "policy": "project_theseus_p2_product_budget_v1",
        "ready": ready,
        "arms": arm_checks,
        "pair_wall_ms": pair_wall,
        "pair_wall_within_budget": pair_wall <= float(budget.get("maximum_pair_wall_ms") or 0.0),
        "limits": budget,
        "prior_pair_wall_ms": prior_pair_wall,
        "cumulative": bool(prior_report),
    }


def patch_paths(patch: str) -> list[str]:
    return sorted(
        {
            match.group(1).strip()
            for match in re.finditer(r"^\+\+\+ (?:b/)?(.+)$", patch, flags=re.MULTILINE)
            if match.group(1).strip() != "/dev/null"
        }
    )


def render_candidate_prompt(natural_request: str, source_context: str) -> str:
    return f"{natural_request}\n\n[visible_parent_source]\n{source_context}"


def visible_source_context(snapshot: Path, task: dict[str, Any]) -> str:
    blocks: list[str] = []
    for row in dicts(task.get("visible_source")):
        relative = str(row.get("path") or "")
        path = snapshot / relative
        if not path.is_file():
            raise CanaryFault(f"visible_source_missing:{relative}")
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(1, int(row.get("start_line") or 1))
        end = min(len(lines), int(row.get("end_line") or len(lines)))
        blocks.append(f"--- {relative}:{start}-{end} ---\n" + "\n".join(lines[start - 1 : end]))
    if not blocks:
        raise CanaryFault("visible_source_empty")
    return "\n\n".join(blocks)


def create_parent_snapshot(commit: str, overlay: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination.parent / "parent.tar"
    with archive.open("wb") as handle:
        result = subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise CanaryFault("parent_archive_failed")
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise CanaryFault("unsafe_parent_archive_member")
        tar.extractall(destination, filter="data")
    applied = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(overlay)],
        cwd=destination,
        text=True,
        capture_output=True,
    )
    if applied.returncode != 0:
        raise CanaryFault("parent_overlay_apply_failed")


def invalid_report(manifest_path: Path, alignment: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED",
        "faults": [reason],
        "manifest_sha256": sha256_file(manifest_path),
        "alignment_audit": alignment,
        "counters": zero_counters(),
    }


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "denominators": report.get("denominators"),
        "product_budget_ready": mapping(report.get("product_budget")).get("ready"),
        "matched_pair_ready": mapping(report.get("matched_pair")).get("ready"),
    }


def zero_counters() -> dict[str, int]:
    return {
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "user_facing_effects": 0,
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if isinstance(row, str) and row] if isinstance(value, list) else []


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
