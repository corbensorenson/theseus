"""Resource-aware Hive scheduler for Project Theseus.

The scheduler does not invent arbitrary work. It reads the local node report,
peer reports, autonomy policy, and resource governor output, then emits a
placement plan for safe registered task kinds. With --execute it may submit only
registered low/medium-risk Hive tasks to authorized peers.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402
import compute_market  # noqa: E402
import hive_security  # noqa: E402
import hive_node_registry  # noqa: E402
import theseus_runtime  # noqa: E402
import viea_spine_records  # noqa: E402

DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = ROOT / "reports" / "hive_scheduler.json"
METAL_ROUTE_POLICY = ROOT / "configs" / "macos_metal_route_policy.json"
METAL_CANARY_POLICY = ROOT / "configs" / "macos_metal_scheduler_canary_policy.json"
METAL_DRY_RUN_REPORT = ROOT / "reports" / "macos_metal_scheduler_dry_run.json"
METAL_DRY_RUN_TRAIN_REPORT = ROOT / "reports" / "macos_metal_scheduler_dry_run_train_report.json"
METAL_DRY_RUN_ARTIFACT = ROOT / "reports" / "macos_metal_scheduler_dry_run_readout_artifact.json"
METAL_PARITY_LADDER = ROOT / "reports" / "macos_metal_parity_ladder.json"
METAL_MLX_PARITY_AUDIT = ROOT / "reports" / "macos_mlx_parity_audit.json"
METAL_CANARY_REPORT = ROOT / "reports" / "macos_metal_scheduler_canary.json"
METAL_CANARY_TRAIN_REPORT = ROOT / "reports" / "macos_metal_scheduler_canary_train_report.json"
METAL_CANARY_ARTIFACT = ROOT / "reports" / "macos_metal_scheduler_canary_readout_artifact.json"
METAL_TOKEN_ROUTE_POLICY = ROOT / "configs" / "macos_metal_token_superposition_route_policy.json"
METAL_TOKEN_CANARY_POLICY = ROOT / "configs" / "macos_metal_token_superposition_scheduler_canary_policy.json"
METAL_TOKEN_LADDER = ROOT / "reports" / "macos_metal_token_superposition_ladder.json"
METAL_TOKEN_REPORT = ROOT / "reports" / "token_superposition_metal_training.json"
METAL_TOKEN_ARTIFACT = ROOT / "reports" / "macos_metal_token_superposition_readout_artifact.json"
METAL_TOKEN_CANARY_REPORT = ROOT / "reports" / "macos_metal_token_superposition_scheduler_canary.json"
METAL_TOKEN_CANARY_TRAIN_REPORT = ROOT / "reports" / "macos_metal_token_superposition_scheduler_canary_train_report.json"
METAL_TOKEN_CANARY_ARTIFACT = ROOT / "reports" / "macos_metal_token_superposition_scheduler_canary_readout_artifact.json"
WORKER_CHUNK_TASKS = {
    "cuda_eval_chunk",
    "cuda_training_chunk",
    "cuda_rollout_chunk",
    "mlx_eval_chunk",
    "mlx_training_chunk",
    "mlx_rollout_chunk",
}
ROUTE_VALIDATOR_VIEW_GROUPS = (
    "governance_records",
    "failure_boundaries",
    "authority_records",
    "resource_route_records",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe-peers", action="store_true")
    parser.add_argument("--worker-chunks", action="store_true")
    parser.add_argument("--sync-artifacts", action="store_true")
    parser.add_argument(
        "--macos-metal-dry-run",
        action="store_true",
        help="Plan, and with --execute run, the guarded local train-rollout-metal scheduler dry-run.",
    )
    parser.add_argument(
        "--macos-metal-canary",
        action="store_true",
        help="Plan, and with --execute run, the reviewed local train-rollout-metal scheduler canary; production routing remains disabled.",
    )
    parser.add_argument(
        "--macos-metal-token-superposition-canary",
        action="store_true",
        help="Plan, and with --execute run, the reviewed local train-token-superposition-metal scheduler canary; production routing remains disabled.",
    )
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    ensure_local_reports(policy)
    registry = hive_node_registry.build_registry(policy)
    write_json(ROOT / "reports" / "hive_node_registry.json", registry)
    nodes = registry.get("nodes") if isinstance(registry.get("nodes"), list) else []
    local_nodes = [node for node in nodes if node.get("is_local")]
    status = local_nodes[0] if local_nodes else read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    resource = read_json(ROOT / "reports" / "resource_governor.json", {})
    candidate = read_json(ROOT / "reports" / "candidate_promotion_gate.json", {})
    license_status = license_manager.status_report(write_report=True)
    peers = [node for node in nodes if not node.get("is_local")]
    route_validator_receipt = hive_route_validator_receipt()
    placements = attach_viea_route_records(policy, build_placements(policy, nodes, resource, candidate), route_validator_receipt)
    execution = []
    if args.execute and args.probe_peers:
        for placement in placements:
            if placement.get("task_kind") == "resource_probe" and placement.get("target") == "remote":
                execution.append(submit_remote_task(policy, placement, {"reason": "scheduler_probe"}, route_validator_receipt=route_validator_receipt))
    if args.execute and args.worker_chunks:
        for placement in placements:
            if placement.get("task_kind") in WORKER_CHUNK_TASKS:
                payload = placement.get("payload") if isinstance(placement.get("payload"), dict) else {}
                execution.append(submit_remote_task(policy, placement, {**payload, "reason": "scheduler_worker_chunk"}, route_validator_receipt=route_validator_receipt))
    artifact_sync = {}
    if args.sync_artifacts or (args.execute and args.worker_chunks):
        artifact_sync = run_artifact_sync(policy)
    macos_metal_dry_run = {}
    if args.macos_metal_dry_run:
        macos_metal_dry_run = run_macos_metal_scheduler_dry_run(
            policy,
            local_nodes[0] if local_nodes else status,
            execute=bool(args.execute),
        )
    macos_metal_canary = {}
    if args.macos_metal_canary:
        macos_metal_canary = run_macos_metal_scheduler_canary(
            policy,
            local_nodes[0] if local_nodes else status,
            execute=bool(args.execute),
        )
    macos_metal_token_superposition_canary = {}
    if args.macos_metal_token_superposition_canary:
        macos_metal_token_superposition_canary = run_macos_metal_token_superposition_scheduler_canary(
            policy,
            local_nodes[0] if local_nodes else status,
            execute=bool(args.execute),
        )
    execution_receipt_smoke = build_viea_execution_receipt_smoke(policy, placements, route_validator_receipt)
    report = {
        "policy": "project_theseus_hive_scheduler_v0",
        "created_utc": now(),
        "enabled": bool(policy.get("enabled", True)),
        "local_node_id": status.get("node_id"),
        "node_count": len(nodes),
        "peer_count": len(peers),
        "summary": summarize(nodes, placements),
        "viea_spine": summarize_viea_route_records(placements, execution, execution_receipt_smoke, route_validator_receipt),
        "route_validator_receipt": route_validator_receipt,
        "node_registry": {
            "report": "reports/hive_node_registry.json",
            "summary": registry.get("summary", {}),
            "created_utc": registry.get("created_utc"),
        },
        "placements": placements,
        "jobs": summarize_jobs(placements),
        "execution": execution,
        "viea_execution_receipt_smoke": execution_receipt_smoke,
        "artifact_sync": artifact_sync,
        "macos_metal_scheduler_dry_run": macos_metal_dry_run,
        "macos_metal_scheduler_canary": macos_metal_canary,
        "macos_metal_token_superposition_scheduler_canary": macos_metal_token_superposition_canary,
        "routing_policy": policy.get("resource_routing", {}),
        "license": compact_license(license_status),
        "safety": {
            "remote_task_submission": "secret_required" if get_path(policy, ["security", "requires_shared_secret_for_remote_tasks"], True) else "open_on_trusted_network",
            "shared_secret_configured": bool(shared_secret(policy)),
            "arbitrary_shell_allowed": False,
        },
    }
    write_json(ROOT / args.out, report)
    jobs_out = ROOT / str(get_path(policy, ["jobs", "scheduler_report_path"], "reports/hive_jobs.json"))
    write_json(jobs_out, {"policy": "project_theseus_hive_job_plan_v0", "created_utc": now(), **summarize_jobs(placements)})
    print(json.dumps(report, indent=2))
    return 0


def ensure_local_reports(policy: dict[str, Any]) -> None:
    status_path = ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json"))
    if not status_path.exists():
        subprocess.run(
            [sys.executable, "scripts/hive_node.py", "--policy", str(DEFAULT_POLICY.relative_to(ROOT)), "probe"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            env=theseus_runtime.runtime_env(),
        )


def build_placements(
    policy: dict[str, Any],
    nodes: list[dict[str, Any]],
    resource: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    trusted = [node for node in nodes if get_path(node, ["trust", "trusted"], True) is not False]
    light_nodes = [node for node in trusted if node.get("light_task_allowed", True)]
    training_nodes = [node for node in light_nodes if node.get("training_allowed", True) and not node.get("training_blockers")]
    ranked = sorted(light_nodes or nodes, key=lambda node: node_score(node), reverse=True)
    ranked_training = sorted(training_nodes, key=lambda node: node_score(node), reverse=True)
    best_training = best_for_capability(ranked_training, ["nvidia_cuda", "apple_mlx", "cpu_worker"])
    best_cuda = best_for_capability(ranked_training, ["nvidia_cuda"]) if any(node_has_capability(node, "nvidia_cuda") for node in ranked_training) else {}
    best_mlx = best_for_capability(ranked_training, ["mlx_apple", "mlx_cuda", "apple_mlx"]) if any(node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]) for node in ranked_training) else {}
    best_inference = best_for_capability(ranked, ["apple_mlx", "mlx_apple", "nvidia_cuda", "cpu_worker"])
    local = nodes[0] if nodes else {}
    can_run_profile = get_path(resource, ["decision", "can_run_requested_profile"], True) is not False
    chunks_license = license_manager.check_feature("distributed_worker_chunks", write_report=True)
    chunks_allowed = bool(chunks_license.get("allowed"))
    placements = [
        placement("resource_probe", local, "local", "Keep local resource and capability reports current."),
        placement("capability_refresh", local, "local", "Capability matrix is cheap and should stay local on every app node."),
        placement("compute_market_status", local, "local", "Keep work-credit gas accounting and local wallet status current."),
        placement("readiness_check", local, "local", "Launch readiness remains coordinator-owned."),
    ]
    if best_training and best_training.get("node_id") != local.get("node_id"):
        placements.append(
            placement(
                "training_smoke",
                best_training,
                "remote",
                "A stronger peer is a better target for smoke training/eval work.",
            )
        )
    elif best_training:
        placements.append(
            placement(
                "training_smoke",
                best_training,
                "local",
                "Local node is currently the best available training node.",
            )
        )
    if chunks_allowed:
        placements.extend(build_worker_job_placements(policy, ranked_training, local, can_run_profile))
    if best_inference and best_inference.get("node_id") != local.get("node_id"):
        placements.append(
            placement(
                "checkpoint_chat",
                best_inference,
                "remote",
                "Weak clients can route checkpoint/live chat to the best authorized peer.",
                payload={"checkpoint_id": "live", "prompt": "Summarize current Project Theseus status."},
            )
        )
    else:
        placements.append(
            placement(
                "checkpoint_chat",
                local,
                "local",
                "Local checkpoint chat remains available without external inference.",
                payload={"checkpoint_id": "live", "prompt": "Summarize current Project Theseus status."},
            )
        )
    if not bool(candidate.get("promote")):
        placements.append(
            placement(
                "readiness_check",
                local,
                "local",
                "Candidate promotion is blocked; keep readiness, residuals, and frontier pressure visible.",
            )
        )
    if not chunks_allowed:
        placements.append(
            placement(
                "readiness_check",
                local,
                "local",
                "Distributed worker chunks are license-gated until this install is registered or licensed.",
                payload={"license_required": True, "feature": "distributed_worker_chunks"},
            )
        )
    if get_path(resource, ["decision", "can_run_requested_profile"], True) is False:
        placements.append(
            placement(
                "resource_probe",
                local,
                "local",
                "Resource governor is throttled; do not offload heavier work until local envelope is clear.",
            )
        )
    return placements


def hive_route_validator_receipt() -> dict[str, Any]:
    return viea_spine_records.materialized_view_consumer_receipt(
        "hive_scheduler_route_validator",
        required_groups=list(ROUTE_VALIDATOR_VIEW_GROUPS),
    )


def attach_viea_route_records(policy: dict[str, Any], placements: list[dict[str, Any]], route_validator_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    secret_configured = bool(shared_secret(policy))
    out = []
    for index, row in enumerate(placements):
        placement_row = dict(row)
        placement_row["placement_id"] = placement_row.get("placement_id") or viea_spine_records.stable_id(
            "hive_placement",
            index,
            placement_row.get("task_kind"),
            placement_row.get("target"),
            placement_row.get("node_id"),
            placement_row.get("payload"),
            placement_row.get("compute_market"),
        )
        placement_row["viea_route_records"] = build_viea_route_records(
            policy,
            placement_row,
            placement_index=index,
            secret_configured=secret_configured,
            route_validator_receipt=route_validator_receipt,
        )
        out.append(placement_row)
    return out


def build_viea_route_records(
    policy: dict[str, Any],
    placement_row: dict[str, Any],
    *,
    placement_index: int,
    secret_configured: bool,
    route_validator_receipt: dict[str, Any],
) -> dict[str, Any]:
    task_kind = str(placement_row.get("task_kind") or "unknown_task")
    target = str(placement_row.get("target") or "unknown_target")
    node_id = str(placement_row.get("node_id") or "unknown_node")
    payload = placement_row.get("payload") if isinstance(placement_row.get("payload"), dict) else {}
    network = placement_row.get("network") if isinstance(placement_row.get("network"), dict) else {}
    compute = placement_row.get("compute_market") if isinstance(placement_row.get("compute_market"), dict) else {}
    route_id = viea_spine_records.stable_id(
        "hive_route",
        placement_row.get("placement_id"),
        task_kind,
        target,
        node_id,
        payload.get("job_id"),
        compute.get("quote_id"),
    )
    is_remote = target == "remote"
    authorized = not is_remote or secret_configured
    output_artifacts = payload.get("output_artifacts") if isinstance(payload.get("output_artifacts"), list) else []
    backend_requirements = payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else []
    failure_state = "planned_authorized" if authorized else "blocked_missing_shared_secret"
    authority_scope = payload.get("allowed_task_scope") if isinstance(payload.get("allowed_task_scope"), list) else [task_kind]
    common = {
        "route_id": route_id,
        "placement_id": placement_row.get("placement_id"),
        "task_kind": task_kind,
        "target": target,
        "node_id": node_id,
        "node_name": placement_row.get("node_name"),
        "job_id": payload.get("job_id"),
        "job_family": payload.get("job_family"),
        "arm_id": payload.get("arm_id"),
        "backend_requirements": backend_requirements,
        "route_phase": "planned",
        "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
        "route_validator_ready": bool(route_validator_receipt.get("ready")),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    authority_transition = {
        **common,
        "record_type": "authority_transition",
        "record_id": viea_spine_records.stable_id("hive_authority_transition", route_id),
        "state": failure_state,
        "status": "SUPPORTED" if authorized else "BLOCKED",
        "support_state": "SUPPORTED" if authorized else "UNSUPPORTED",
        "authority_scope": authority_scope,
        "requires_shared_secret": bool(is_remote),
        "shared_secret_configured": secret_configured,
        "remote_arbitrary_shell_allowed": False,
    }
    authority_use_receipt = {
        **common,
        "record_type": "authority_use_receipt",
        "record_id": viea_spine_records.stable_id("hive_authority_use", route_id),
        "state": "planned_not_executed",
        "status": "READY" if authorized else "BLOCKED",
        "support_state": "SUPPORTED" if authorized else "UNSUPPORTED",
        "verifier_state": "planning_gate",
        "authority_receipt_ref": authority_transition["record_id"],
        "allowed_task_scope": authority_scope,
    }
    runtime_adapter_invocation = {
        **common,
        "record_type": "runtime_adapter_invocation",
        "record_id": viea_spine_records.stable_id("hive_runtime_adapter", route_id),
        "adapter_id": "hive.local_task_adapter" if target == "local" else "hive.remote_task_api_adapter",
        "state": "planned",
        "status": "READY" if authorized else "BLOCKED",
        "bounded_task_kind": task_kind,
        "remote_endpoint_present": bool(placement_row.get("api_url")) if is_remote else True,
    }
    resource_budget = {
        **common,
        "record_type": "resource_budget",
        "record_id": viea_spine_records.stable_id("hive_resource_budget", route_id),
        "gas_estimate_micro_twc": int(compute.get("gas_estimate_micro_twc") or 0),
        "provider_payout_micro_twc": int(compute.get("provider_payout_micro_twc") or 0),
        "estimated_latency_ms": int(network.get("estimated_latency_ms") or 0),
        "network_class": network.get("class"),
        "task_fit": network.get("task_fit"),
        "accounting_only": compute.get("accounting_only"),
    }
    costed_route = {
        **common,
        "record_type": "costed_route",
        "record_id": viea_spine_records.stable_id("hive_costed_route", route_id),
        "quote_id": compute.get("quote_id"),
        "currency_symbol": compute.get("currency_symbol"),
        "gas_estimate_micro_twc": int(compute.get("gas_estimate_micro_twc") or 0),
        "node_score": placement_row.get("node_score"),
        "network_class": network.get("class"),
    }
    generation_mode = {
        **common,
        "record_type": "generation_mode",
        "record_id": viea_spine_records.stable_id("hive_generation_mode", route_id),
        "state": "scheduler_routing",
        "status": "TOOL_ORCHESTRATION_ONLY",
        "learned_generation_claim_allowed": False,
        "tool_or_router_claim": True,
        "non_claim": "Hive scheduler placement is orchestration evidence and cannot support learned-generation promotion claims.",
    }
    failure_boundary = {
        **common,
        "record_type": "failure_boundary",
        "record_id": viea_spine_records.stable_id("hive_failure_boundary", route_id),
        "failure_id": viea_spine_records.stable_id("hive_route_failure", route_id),
        "state": failure_state,
        "status": "READY" if authorized else "BLOCKED",
        "terminal": not authorized,
        "structured_non_solved": not authorized,
        "fallback_return_used": False,
        "blocked_reason": "missing_shared_secret" if is_remote and not secret_configured else "",
    }
    artifact_graph = {
        **common,
        "record_type": "artifact_graph_record",
        "record_id": viea_spine_records.stable_id("hive_artifact_graph", route_id),
        "artifact_kind": "hive_route_plan",
        "output_artifacts": output_artifacts,
        "evidence_ref": "reports/hive_scheduler.json",
        "content_hash": viea_spine_records.stable_hash(
            {
                "task_kind": task_kind,
                "target": target,
                "node_id": node_id,
                "payload": payload,
                "compute_market": compute,
                "placement_index": placement_index,
            }
        ),
    }
    evidence_transition = {
        **common,
        "record_type": "evidence_transition_record",
        "record_id": viea_spine_records.stable_id("hive_evidence_transition", route_id),
        "state": "planned_route_to_scheduler_report",
        "status": "SUPPORTED" if authorized else "BLOCKED",
        "support_state": "SUPPORTED" if authorized else "UNSUPPORTED",
        "evidence_ref": "reports/hive_scheduler.json",
        "artifact_ref": artifact_graph["record_id"],
    }
    claim_record = {
        **common,
        "record_type": "claim_record",
        "record_id": viea_spine_records.stable_id("hive_route_claim", route_id),
        "claim_id": viea_spine_records.stable_id("hive_route_claim", route_id),
        "claim_type": "hive_registered_task_route_traceability",
        "state": "planned_route_traceable",
        "status": "SUPPORTED" if authorized else "BLOCKED",
        "support_state": "SUPPORTED" if authorized else "UNSUPPORTED",
        "verifier_state": "route_validator_ready" if route_validator_receipt.get("ready") else "route_validator_not_ready",
        "artifact_ref": artifact_graph["record_id"],
        "evidence_ref": "reports/hive_scheduler.json",
        "non_claim": "Hive route claim proves bounded registered-task route traceability; it is not model capability or learned-generation evidence.",
    }
    return {
        "authority_transition": authority_transition,
        "authority_use_receipt": authority_use_receipt,
        "runtime_adapter_invocation": runtime_adapter_invocation,
        "resource_budget": resource_budget,
        "costed_route": costed_route,
        "generation_mode": generation_mode,
        "failure_boundary": failure_boundary,
        "artifact_graph_record": artifact_graph,
        "claim_record": claim_record,
        "evidence_transition_record": evidence_transition,
    }


def route_id_for_placement(placement_row: dict[str, Any]) -> str:
    records = placement_row.get("viea_route_records") if isinstance(placement_row.get("viea_route_records"), dict) else {}
    for value in records.values():
        if isinstance(value, dict) and value.get("route_id"):
            return str(value.get("route_id"))
    payload = placement_row.get("payload") if isinstance(placement_row.get("payload"), dict) else {}
    compute = placement_row.get("compute_market") if isinstance(placement_row.get("compute_market"), dict) else {}
    return viea_spine_records.stable_id(
        "hive_route",
        placement_row.get("placement_id"),
        placement_row.get("task_kind"),
        placement_row.get("target"),
        placement_row.get("node_id"),
        payload.get("job_id"),
        compute.get("quote_id"),
    )


def build_viea_execution_records(
    policy: dict[str, Any],
    placement_row: dict[str, Any],
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    execution_phase: str,
    route_validator_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_validator = route_validator_receipt if isinstance(route_validator_receipt, dict) else hive_route_validator_receipt()
    task_kind = str(placement_row.get("task_kind") or result.get("kind") or "unknown_task")
    target = str(placement_row.get("target") or "unknown_target")
    node_id = str(placement_row.get("node_id") or "unknown_node")
    network = placement_row.get("network") if isinstance(placement_row.get("network"), dict) else {}
    compute = placement_row.get("compute_market") if isinstance(placement_row.get("compute_market"), dict) else {}
    clean_payload = dict(payload)
    clean_payload.pop("manifest", None)
    route_id = route_id_for_placement(placement_row)
    execution_id = viea_spine_records.stable_id(
        "hive_execution",
        route_id,
        execution_phase,
        result.get("ok"),
        result.get("error"),
        result.get("task_id") or get_path(result, ["task", "task_id"], ""),
    )
    attempted = bool(result.get("attempted", execution_phase not in {"receipt_schema_smoke", "not_executed"}))
    ok = bool(result.get("ok")) and not result.get("error")
    status = "SUPPORTED" if ok else ("READY" if not attempted else "FAILED")
    support_state = "SUPPORTED" if ok or not attempted else "UNSUPPORTED"
    state = "queued" if ok and attempted else ("schema_verified_not_submitted" if not attempted else "submission_failed")
    terminal = bool(attempted and not ok)
    task = result.get("task") if isinstance(result.get("task"), dict) else {}
    job = task.get("job") if isinstance(task.get("job"), dict) else {}
    job_id = clean_payload.get("job_id") or job.get("job_id") or get_path(result, ["task", "job", "job_id"], "")
    job_family = clean_payload.get("job_family") or job.get("family") or clean_payload.get("family")
    arm_id = clean_payload.get("arm_id")
    backend_requirements = clean_payload.get("backend_requirements") if isinstance(clean_payload.get("backend_requirements"), list) else []
    output_artifacts = clean_payload.get("output_artifacts") if isinstance(clean_payload.get("output_artifacts"), list) else []
    authority_scope = clean_payload.get("allowed_task_scope") if isinstance(clean_payload.get("allowed_task_scope"), list) else [task_kind]
    secret_configured = bool(shared_secret(policy))
    is_remote = target == "remote"
    authorized = not is_remote or secret_configured
    common = {
        "route_id": route_id,
        "execution_id": execution_id,
        "placement_id": placement_row.get("placement_id"),
        "task_kind": task_kind,
        "target": target,
        "node_id": node_id,
        "node_name": placement_row.get("node_name"),
        "job_id": job_id,
        "job_family": job_family,
        "arm_id": arm_id,
        "backend_requirements": backend_requirements,
        "route_phase": execution_phase,
        "route_validator_receipt_id": route_validator.get("receipt_id"),
        "route_validator_ready": bool(route_validator.get("ready")),
        "attempted": attempted,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    authority_transition = {
        **common,
        "record_type": "authority_transition",
        "record_id": viea_spine_records.stable_id("hive_execution_authority_transition", execution_id),
        "state": "execution_authorized" if authorized else "blocked_missing_shared_secret",
        "status": "SUPPORTED" if authorized else "BLOCKED",
        "support_state": "SUPPORTED" if authorized else "UNSUPPORTED",
        "authority_scope": authority_scope,
        "requires_shared_secret": bool(is_remote),
        "shared_secret_configured": secret_configured,
        "remote_arbitrary_shell_allowed": False,
    }
    authority_use_receipt = {
        **common,
        "record_type": "authority_use_receipt",
        "record_id": viea_spine_records.stable_id("hive_execution_authority_use", execution_id),
        "state": state,
        "status": status if authorized else "BLOCKED",
        "support_state": support_state if authorized else "UNSUPPORTED",
        "verifier_state": "submission_receipt" if attempted else "schema_smoke",
        "authority_receipt_ref": authority_transition["record_id"],
        "allowed_task_scope": authority_scope,
    }
    runtime_adapter_invocation = {
        **common,
        "record_type": "runtime_adapter_invocation",
        "record_id": viea_spine_records.stable_id("hive_execution_runtime_adapter", execution_id),
        "adapter_id": "hive.local_task_adapter" if target == "local" else "hive.remote_task_api_adapter",
        "state": "submitted" if attempted else "schema_smoke",
        "status": status if authorized else "BLOCKED",
        "bounded_task_kind": task_kind,
        "remote_endpoint_present": bool(placement_row.get("api_url")) if is_remote else True,
    }
    resource_budget = {
        **common,
        "record_type": "resource_budget",
        "record_id": viea_spine_records.stable_id("hive_execution_resource_budget", execution_id),
        "gas_estimate_micro_twc": int(compute.get("gas_estimate_micro_twc") or 0),
        "provider_payout_micro_twc": int(compute.get("provider_payout_micro_twc") or 0),
        "estimated_latency_ms": int(network.get("estimated_latency_ms") or 0),
        "network_class": network.get("class"),
        "task_fit": network.get("task_fit"),
        "accounting_only": compute.get("accounting_only"),
    }
    costed_route = {
        **common,
        "record_type": "costed_route",
        "record_id": viea_spine_records.stable_id("hive_execution_costed_route", execution_id),
        "quote_id": compute.get("quote_id"),
        "currency_symbol": compute.get("currency_symbol"),
        "gas_estimate_micro_twc": int(compute.get("gas_estimate_micro_twc") or 0),
        "node_score": placement_row.get("node_score"),
        "network_class": network.get("class"),
    }
    generation_mode = {
        **common,
        "record_type": "generation_mode",
        "record_id": viea_spine_records.stable_id("hive_execution_generation_mode", execution_id),
        "state": "scheduler_execution_receipt",
        "status": "TOOL_ORCHESTRATION_ONLY",
        "learned_generation_claim_allowed": False,
        "tool_or_router_claim": True,
        "non_claim": "Hive execution receipts prove bounded task submission traceability, not learned-generation capability.",
    }
    failure_boundary = {
        **common,
        "record_type": "failure_boundary",
        "record_id": viea_spine_records.stable_id("hive_execution_failure_boundary", execution_id),
        "failure_id": viea_spine_records.stable_id("hive_execution_failure", execution_id),
        "state": state if authorized else "blocked_missing_shared_secret",
        "status": status if authorized else "BLOCKED",
        "terminal": terminal or not authorized,
        "structured_non_solved": terminal or not authorized,
        "fallback_return_used": False,
        "blocked_reason": "missing_shared_secret" if is_remote and not secret_configured else str(result.get("error") or ""),
    }
    artifact_graph = {
        **common,
        "record_type": "artifact_graph_record",
        "record_id": viea_spine_records.stable_id("hive_execution_artifact_graph", execution_id),
        "artifact_kind": "hive_task_submission_receipt",
        "output_artifacts": output_artifacts,
        "evidence_ref": "reports/hive_scheduler.json",
        "content_hash": viea_spine_records.stable_hash(
            {
                "task_kind": task_kind,
                "target": target,
                "node_id": node_id,
                "placement_id": placement_row.get("placement_id"),
                "execution_phase": execution_phase,
                "attempted": attempted,
                "ok": ok,
                "error": result.get("error"),
                "payload": clean_payload,
            }
        ),
    }
    evidence_transition = {
        **common,
        "record_type": "evidence_transition_record",
        "record_id": viea_spine_records.stable_id("hive_execution_evidence_transition", execution_id),
        "state": "task_submission_to_scheduler_report" if attempted else "receipt_schema_to_scheduler_report",
        "status": status if authorized else "BLOCKED",
        "support_state": support_state if authorized else "UNSUPPORTED",
        "evidence_ref": "reports/hive_scheduler.json",
        "artifact_ref": artifact_graph["record_id"],
    }
    claim_record = {
        **common,
        "record_type": "claim_record",
        "record_id": viea_spine_records.stable_id("hive_execution_claim", execution_id),
        "claim_id": viea_spine_records.stable_id("hive_execution_claim", execution_id),
        "claim_type": "hive_registered_task_submission_traceability",
        "state": "task_submission_traceable" if attempted else "receipt_schema_traceable",
        "status": status if authorized else "BLOCKED",
        "support_state": support_state if authorized else "UNSUPPORTED",
        "verifier_state": "submission_receipt" if attempted else "schema_smoke",
        "artifact_ref": artifact_graph["record_id"],
        "evidence_ref": "reports/hive_scheduler.json",
        "non_claim": "Hive execution claim proves bounded registered-task submission traceability; it is not model capability or learned-generation evidence.",
    }
    return {
        "authority_transition": authority_transition,
        "authority_use_receipt": authority_use_receipt,
        "runtime_adapter_invocation": runtime_adapter_invocation,
        "resource_budget": resource_budget,
        "costed_route": costed_route,
        "generation_mode": generation_mode,
        "failure_boundary": failure_boundary,
        "artifact_graph_record": artifact_graph,
        "claim_record": claim_record,
        "evidence_transition_record": evidence_transition,
    }


def attach_viea_execution_records(
    policy: dict[str, Any],
    placement: dict[str, Any],
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    execution_phase: str,
    route_validator_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(result)
    out.setdefault("placement_id", placement.get("placement_id"))
    out.setdefault("task_kind", placement.get("task_kind"))
    out.setdefault("target", placement.get("target"))
    out.setdefault("node_id", placement.get("node_id"))
    out["viea_execution_records"] = build_viea_execution_records(
        policy,
        placement,
        payload,
        out,
        execution_phase=execution_phase,
        route_validator_receipt=route_validator_receipt,
    )
    out.setdefault("public_training_rows_written", 0)
    out.setdefault("external_inference_calls", 0)
    out.setdefault("fallback_return_count", 0)
    return out


def build_viea_execution_receipt_smoke(policy: dict[str, Any], placements: list[dict[str, Any]], route_validator_receipt: dict[str, Any]) -> dict[str, Any]:
    sample = next((row for row in placements if row.get("task_kind") == "resource_probe"), placements[0] if placements else {})
    if not sample:
        return {
            "policy": "project_theseus_hive_execution_receipt_smoke_v1",
            "ready": False,
            "attempted": False,
            "reason": "no_placement_available",
            "viea_execution_records": {},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    payload = {"reason": "scheduler_execution_receipt_schema_smoke"}
    result = {
        "ok": True,
        "attempted": False,
        "reason": "schema_smoke_no_task_submitted",
        "placement_id": sample.get("placement_id"),
        "task_kind": sample.get("task_kind"),
        "target": sample.get("target"),
        "node_id": sample.get("node_id"),
    }
    records = build_viea_execution_records(
        policy,
        sample,
        payload,
        result,
        execution_phase="receipt_schema_smoke",
        route_validator_receipt=route_validator_receipt,
    )
    return {
        "policy": "project_theseus_hive_execution_receipt_smoke_v1",
        "ready": bool(records),
        "attempted": False,
        "reason": "schema_smoke_no_task_submitted",
        "placement_id": sample.get("placement_id"),
        "task_kind": sample.get("task_kind"),
        "target": sample.get("target"),
        "node_id": sample.get("node_id"),
        "viea_execution_records": records,
        "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
        "route_validator_ready": bool(route_validator_receipt.get("ready")),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def placement(
    task_kind: str,
    node: dict[str, Any],
    target: str,
    reason: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    market_quote = compute_market.quote_task(task_kind, payload or {}, node, write_report=False)
    return {
        "task_kind": task_kind,
        "target": target,
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "node_score": round(node_score(node), 4),
        "network": node_network_profile(node),
        "reason": reason,
        "payload": payload or {},
        "compute_market": {
            "quote_id": market_quote.get("quote_id"),
            "currency_symbol": market_quote.get("currency_symbol"),
            "gas_estimate_micro_twc": market_quote.get("gas_estimate_micro_twc"),
            "provider_payout_micro_twc": market_quote.get("provider_payout_micro_twc"),
            "accounting_only": market_quote.get("accounting_only"),
        },
    }


def build_worker_job_placements(
    policy: dict[str, Any],
    nodes: list[dict[str, Any]],
    local: dict[str, Any],
    can_run_local_profile: bool,
) -> list[dict[str, Any]]:
    jobs_policy = policy.get("jobs") if isinstance(policy.get("jobs"), dict) else {}
    parallel = jobs_policy.get("embarrassingly_parallel") if isinstance(jobs_policy.get("embarrassingly_parallel"), dict) else {}
    max_cuda = int(parallel.get("max_jobs_per_cuda_node", 3))
    max_mlx = int(parallel.get("max_jobs_per_mlx_node", 2))
    out: list[dict[str, Any]] = []
    seed_base = 20260515
    for node_index, node in enumerate(nodes):
        is_local = node.get("node_id") == local.get("node_id")
        if is_local and not can_run_local_profile:
            continue
        target = "local" if is_local else "remote"
        if node_has_capability(node, "nvidia_cuda") and node_has_slot(node, ["cuda"]):
            cuda_jobs = [
                (
                    "cuda_eval_chunk",
                    "eval_shard",
                    "NVIDIA node can run an independent Rust/CUDA eval shard.",
                    {
                        "cases_per_task": 8,
                        "epochs": 1,
                        "samples_per_launch": 512,
                        "hv_dim": 1024,
                    },
                ),
                (
                    "cuda_training_chunk",
                    "readout_chunk",
                    "NVIDIA node can run an independent Rust/CUDA readout-training arm.",
                    {
                        "cases_per_task": 24,
                        "epochs": 3,
                        "samples_per_launch": 512,
                        "hv_dim": 1536,
                    },
                ),
                (
                    "cuda_rollout_chunk",
                    "rollout_chunk",
                    "NVIDIA node can run an independent Rust/CUDA rollout-training arm.",
                    {
                        "cases_per_task": 12,
                        "epochs": 2,
                        "state_epochs": 1,
                        "samples_per_launch": 512,
                        "rollout_batch": 384,
                        "hv_dim": 1536,
                        "seq_len": 32,
                    },
                ),
            ][:max_cuda]
            for job_index, (task_kind, family, reason, params) in enumerate(cuda_jobs):
                chunk_id = f"scheduler_{family}_{safe_node_name(node)}_{job_index}"
                payload = job_payload(
                    task_kind,
                    chunk_id,
                    family=family,
                    arm_id="rust_cuda_systems_arm" if task_kind != "cuda_rollout_chunk" else "rl_control_arm",
                    backend_requirements=["nvidia_cuda", "rust_cuda"],
                    seed=seed_base + node_index * 100 + job_index,
                    params=params,
                )
                payload.update(requester_fields(local, node, task_kind))
                out.append(placement(task_kind, node, target, reason, payload=payload))
        if node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]) and node_has_slot(node, ["mlx_apple", "mlx_cuda"]):
            mlx_jobs = [
                (
                    "mlx_eval_chunk",
                    "mlx_eval_shard",
                    "MLX node can run an independent BabyLM eval shard.",
                    {"train_limit": 128, "eval_limit": 128, "feature_dim": 512, "steps": 1},
                ),
                (
                    "mlx_training_chunk",
                    "mlx_readout_chunk",
                    "MLX node can run an independent BabyLM readout-training arm.",
                    {"train_limit": 512, "eval_limit": 256, "feature_dim": 1024, "steps": 24},
                ),
                (
                    "mlx_rollout_chunk",
                    "mlx_rollout_chunk",
                    "MLX node can run an independent rollout/control probe.",
                    {"cases_per_task": 64, "eval_cases": 64, "epochs": 6, "hv_dim": 1024, "obs_dim": 32, "seq_len": 32},
                ),
            ][:max_mlx]
            for job_index, (task_kind, family, reason, params) in enumerate(mlx_jobs):
                chunk_id = f"scheduler_{family}_{safe_node_name(node)}_{job_index}"
                payload = job_payload(
                    task_kind,
                    chunk_id,
                    family=family,
                    arm_id="apple_mlx_control_arm" if task_kind == "mlx_rollout_chunk" else "apple_mlx_worker_arm",
                    backend_requirements=["mlx_apple_or_mlx_cuda"],
                    seed=seed_base + node_index * 100 + 20 + job_index,
                    params=params,
                )
                payload.update(requester_fields(local, node, task_kind))
                out.append(placement(task_kind, node, target, reason, payload=payload))
    return out


def requester_fields(local: dict[str, Any], target: dict[str, Any], task_kind: str) -> dict[str, Any]:
    return {
        "requester_node_id": str(local.get("node_id") or ""),
        "requester_node_name": str(local.get("node_name") or local.get("hostname") or ""),
        "target_node_id": str(target.get("node_id") or ""),
        "target_node_name": str(target.get("node_name") or target.get("hostname") or ""),
        "allowed_task_scope": [task_kind],
    }


def job_payload(
    task_kind: str,
    chunk_id: str,
    *,
    family: str,
    arm_id: str,
    backend_requirements: list[str],
    seed: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "profile": "smoke",
        "chunk_id": chunk_id,
        "job_id": f"job_{chunk_id}",
        "job_family": family,
        "arm_id": arm_id,
        "backend_requirements": backend_requirements,
        "train_seed": seed,
        "eval_seed": seed + 10000,
        "merge_policy": "append_report_then_gate",
        "priority": 75,
        "lease_seconds": 1800,
        "max_retries": 1,
        "output_artifacts": [{"type": "worker_report", "path": f"reports/hive_chunks/{chunk_id}.json"}],
    }
    payload.update(params)
    if task_kind.startswith("mlx_"):
        payload.pop("train_seed", None)
        payload.pop("eval_seed", None)
    return payload


def safe_node_name(node: dict[str, Any]) -> str:
    raw = str(node.get("node_name") or node.get("node_id") or "node")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)[:48]


def submit_remote_task(
    policy: dict[str, Any],
    placement: dict[str, Any],
    payload: dict[str, Any],
    *,
    route_validator_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_url = str(placement.get("api_url") or "")
    if placement.get("target") == "local":
        api_url = f"http://127.0.0.1:{int(get_path(policy, ['node', 'http_port'], 8791))}"
    if not api_url:
        return attach_viea_execution_records(
            policy,
            placement,
            payload,
            {"ok": False, "error": "missing_api_url"},
            execution_phase="submission_failed",
            route_validator_receipt=route_validator_receipt,
        )
    signed_payload = dict(payload)
    secret = shared_secret(policy)
    if secret and "manifest" not in signed_payload:
        signed_payload["manifest"] = hive_security.build_manifest(
            str(placement.get("task_kind") or ""),
            dict(signed_payload),
            hive_id=str(placement.get("hive_id") or get_path(policy, ["federation", "default_hive_id"], "")),
            join_token=secret,
            scope=[str(placement.get("task_kind") or "")],
        )
    body = json.dumps({"kind": placement.get("task_kind"), "payload": signed_payload}).encode("utf-8")
    req = urlrequest.Request(
        api_url.rstrip("/") + "/api/hive/tasks",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if secret:
        req.add_header("X-Theseus-Hive-Secret", secret)
    try:
        with urlrequest.urlopen(req, timeout=10) as response:  # noqa: S310 - private/local hive endpoint.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        return attach_viea_execution_records(
            policy,
            placement,
            payload,
            {"ok": False, "error": str(exc)},
            execution_phase="submission_failed",
            route_validator_receipt=route_validator_receipt,
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return attach_viea_execution_records(
            policy,
            placement,
            payload,
            {"ok": False, "error": "non_json_response", "body": raw[:500]},
            execution_phase="submission_failed",
            route_validator_receipt=route_validator_receipt,
        )
    result = value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response"}
    return attach_viea_execution_records(
        policy,
        placement,
        payload,
        result,
        execution_phase="submitted" if bool(result.get("ok")) else "submission_failed",
        route_validator_receipt=route_validator_receipt,
    )


def run_artifact_sync(policy: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "scripts/hive_artifact_sync.py", "--out", "reports/hive_artifact_sync.json", "--relay-results"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300, env=theseus_runtime.runtime_env())
    if result.returncode != 0:
        return {"ok": False, "returncode": result.returncode, "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:]}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "stdout_tail": result.stdout[-1000:]}
    return value if isinstance(value, dict) else {"ok": True}


def run_macos_metal_scheduler_dry_run(
    policy: dict[str, Any],
    local: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    route_policy = read_json(METAL_ROUTE_POLICY, {})
    dry_run_bounds = {
        "cases_per_task": 4,
        "epochs": 2,
        "samples_per_launch": 2,
        "rollout_batch": 2,
        "obs_dim": 3,
        "hidden_dim": 4,
        "reservoir_dim": 5,
        "hv_dim": 8,
        "seq_len": 3,
        "output_dim": 4,
        "max_kernel_launches": 64,
    }
    payload = {
        "profile": "guarded_smoke",
        "job_id": "job_macos_metal_train_rollout_scheduler_dry_run",
        "job_family": "macos_metal_train_rollout_dry_run",
        "arm_id": "apple_metal_control_arm",
        "backend_requirements": ["apple_metal"],
        "command": "train-rollout-metal",
        "parity_for": "train-rollout-cuda",
        "route_policy": str(METAL_ROUTE_POLICY.relative_to(ROOT)),
        "production_scheduler_routing_enabled": False,
        "merge_policy": "append_report_only_no_promotion",
        "priority": 10,
        "lease_seconds": 300,
        "output_artifacts": [
            {"type": "worker_report", "path": str(METAL_DRY_RUN_TRAIN_REPORT.relative_to(ROOT))},
            {"type": "readout_artifact", "path": str(METAL_DRY_RUN_ARTIFACT.relative_to(ROOT))},
        ],
        **dry_run_bounds,
    }
    payload.update(requester_fields(local, local, "train_rollout_metal_dry_run"))
    planned = placement(
        "train_rollout_metal_dry_run",
        local,
        "local",
        "Guarded local-only scheduler dry-run for train-rollout-metal; production routing remains disabled.",
        payload=payload,
    )
    command = [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-rollout-metal",
        "--cases-per-task",
        str(dry_run_bounds["cases_per_task"]),
        "--epochs",
        str(dry_run_bounds["epochs"]),
        "--samples-per-launch",
        str(dry_run_bounds["samples_per_launch"]),
        "--rollout-batch",
        str(dry_run_bounds["rollout_batch"]),
        "--obs-dim",
        str(dry_run_bounds["obs_dim"]),
        "--hidden-dim",
        str(dry_run_bounds["hidden_dim"]),
        "--reservoir-dim",
        str(dry_run_bounds["reservoir_dim"]),
        "--hv-dim",
        str(dry_run_bounds["hv_dim"]),
        "--seq-len",
        str(dry_run_bounds["seq_len"]),
        "--output-dim",
        str(dry_run_bounds["output_dim"]),
        "--model-out",
        str(METAL_DRY_RUN_ARTIFACT.relative_to(ROOT)),
        "--out",
        str(METAL_DRY_RUN_TRAIN_REPORT.relative_to(ROOT)),
    ]
    preflight_checks = {
        "route_policy_present": bool(route_policy),
        "route_policy_command_matches": route_policy.get("command") == "train-rollout-metal",
        "route_policy_backend_matches": route_policy.get("backend") == "apple_metal",
        "bounded_smoke_route_enabled": bool(route_policy.get("bounded_smoke_route_enabled")),
        "production_scheduler_routing_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "dry_run_not_registered_worker_chunk": "train_rollout_metal_dry_run" not in WORKER_CHUNK_TASKS,
        "local_only_target": planned.get("target") == "local",
        "no_remote_task_scope_change": bool(route_policy.get("does_not_change_hive_remote_task_scope")),
        "no_arbitrary_remote_execution": bool(route_policy.get("no_arbitrary_remote_execution")),
        "bounded_kernel_launch_cap_declared": dry_run_bounds["max_kernel_launches"] <= 64,
    }
    execution = {"attempted": False, "reason": "execute_not_requested"}
    if execute and all(preflight_checks.values()):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
            env=theseus_runtime.runtime_env(),
        )
        execution = {
            "attempted": True,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    elif execute:
        execution = {
            "attempted": False,
            "reason": "preflight_failed",
            "failed_checks": [name for name, passed in preflight_checks.items() if not passed],
        }
    train_report = read_json(METAL_DRY_RUN_TRAIN_REPORT, {}) if execute else {}
    artifact = read_json(METAL_DRY_RUN_ARTIFACT, {}) if execute else {}
    result_checks = (
        validate_macos_metal_dry_run_result(train_report, artifact, dry_run_bounds)
        if execute
        else {"not_applicable_until_execute": True}
    )
    report = {
        "ok": all(preflight_checks.values()) and (not execute or (bool(execution.get("ok")) and all(result_checks.values()))),
        "policy": "project_theseus_macos_metal_scheduler_dry_run_v0",
        "created_utc": now(),
        "mode": "execute" if execute else "plan",
        "route_policy": str(METAL_ROUTE_POLICY.relative_to(ROOT)),
        "planned_placement": planned,
        "command": command,
        "execution": execution,
        "train_report": str(METAL_DRY_RUN_TRAIN_REPORT.relative_to(ROOT)),
        "artifact": str(METAL_DRY_RUN_ARTIFACT.relative_to(ROOT)),
        "dry_run_bounds": dry_run_bounds,
        "checks": {
            "preflight": preflight_checks,
            "result": result_checks,
        },
        "guardrails": {
            "local_only": True,
            "remote_task_submitted": False,
            "production_scheduler_routing_enabled": False,
            "model_promotion_allowed": False,
            "train_rollout_parity_claim_allowed": False,
            "public_benchmark_training_used": False,
            "teacher_used": False,
            "external_inference_calls": 0,
            "no_fallback_returns": True,
        },
        "next_action": "keep production routing locked; only consider route enablement after a separate operator-gated scheduler policy promotes this dry-run evidence",
    }
    write_json(METAL_DRY_RUN_REPORT, report)
    return report


def run_macos_metal_scheduler_canary(
    policy: dict[str, Any],
    local: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    route_policy = read_json(METAL_ROUTE_POLICY, {})
    canary_policy = read_json(METAL_CANARY_POLICY, {})
    dry_run = read_json(METAL_DRY_RUN_REPORT, {})
    ladder = read_json(METAL_PARITY_LADDER, {})
    parity_audit = read_json(METAL_MLX_PARITY_AUDIT, {})
    bounds = dict(canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {})
    payload = {
        "profile": "reviewed_local_canary",
        "job_id": "job_macos_metal_train_rollout_scheduler_canary",
        "job_family": "macos_metal_train_rollout_canary",
        "arm_id": "apple_metal_control_arm",
        "backend_requirements": ["apple_metal"],
        "command": "train-rollout-metal",
        "parity_for": "train-rollout-cuda",
        "route_policy": str(METAL_ROUTE_POLICY.relative_to(ROOT)),
        "canary_policy": str(METAL_CANARY_POLICY.relative_to(ROOT)),
        "production_scheduler_routing_enabled": False,
        "merge_policy": "append_report_only_no_promotion",
        "priority": 20,
        "lease_seconds": 600,
        "output_artifacts": [
            {"type": "worker_report", "path": str(METAL_CANARY_TRAIN_REPORT.relative_to(ROOT))},
            {"type": "readout_artifact", "path": str(METAL_CANARY_ARTIFACT.relative_to(ROOT))},
        ],
        **bounds,
    }
    payload.update(requester_fields(local, local, "train_rollout_metal_local_canary"))
    planned = placement(
        "train_rollout_metal_local_canary",
        local,
        "local",
        "Reviewed local-only scheduler canary for train-rollout-metal; production routing remains disabled.",
        payload=payload,
    )
    command = [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-rollout-metal",
        "--cases-per-task",
        str(int(bounds.get("cases_per_task") or 1)),
        "--epochs",
        str(int(bounds.get("epochs") or 1)),
        "--samples-per-launch",
        str(int(bounds.get("samples_per_launch") or 1)),
        "--rollout-batch",
        str(int(bounds.get("rollout_batch") or 1)),
        "--obs-dim",
        str(int(bounds.get("obs_dim") or 1)),
        "--hidden-dim",
        str(int(bounds.get("hidden_dim") or 1)),
        "--reservoir-dim",
        str(int(bounds.get("reservoir_dim") or 1)),
        "--hv-dim",
        str(int(bounds.get("hv_dim") or 8)),
        "--seq-len",
        str(int(bounds.get("seq_len") or 1)),
        "--output-dim",
        str(int(bounds.get("output_dim") or 1)),
        "--tolerance",
        str(float(bounds.get("tolerance") or 0.0001)),
        "--model-out",
        str(METAL_CANARY_ARTIFACT.relative_to(ROOT)),
        "--out",
        str(METAL_CANARY_TRAIN_REPORT.relative_to(ROOT)),
    ]
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    required_reports = canary_policy.get("required_reports") if isinstance(canary_policy.get("required_reports"), list) else []
    preflight_checks = {
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_scheduler_canary_policy_v0",
        "command_matches": canary_policy.get("command") == "train-rollout-metal",
        "task_kind_matches": canary_policy.get("task_kind") == "train_rollout_metal_local_canary",
        "backend_matches": canary_policy.get("backend") == "apple_metal",
        "local_only_policy": canary_policy.get("local_only") is True,
        "planned_placement_local_only": planned.get("target") == "local",
        "canary_not_registered_worker_chunk": "train_rollout_metal_local_canary" not in WORKER_CHUNK_TASKS,
        "route_policy_present": bool(route_policy),
        "route_policy_command_matches": route_policy.get("command") == "train-rollout-metal",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "production_scheduler_routing_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": canary_policy.get("remote_task_submitted") is False,
        "no_remote_task_scope_change": bool(canary_policy.get("does_not_change_hive_remote_task_scope")),
        "no_arbitrary_remote_execution": bool(canary_policy.get("no_arbitrary_remote_execution")),
        "dry_run_ok_required": requires.get("macos_metal_scheduler_dry_run_ok") is True and bool(dry_run.get("ok")),
        "ladder_ok_required": requires.get("macos_metal_parity_ladder_ok") is True and ladder.get("trigger_state") == "GREEN",
        "audit_counts_ladder": requires.get("macos_mlx_parity_audit_counts_ladder") is True
        and int(get_path(parity_audit, ["summary", "native_train_rollout_parity_ladder_count"], 0) or 0) >= 1,
        "required_reports_declared": all(
            path in required_reports
            for path in [
                "reports/macos_metal_scheduler_dry_run.json",
                "reports/macos_metal_parity_ladder.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "bounded_kernel_launch_cap_declared": 0 < int(bounds.get("max_kernel_launches") or 0) <= 256,
        "tolerance_bounded": 0.0 < float(bounds.get("tolerance") or 0.0) <= 0.0005,
        "promotion_locked_by_policy": requires.get("model_promotion_allowed") is False,
        "parity_claim_locked_by_policy": requires.get("train_rollout_parity_claim_allowed") is False
        and requires.get("native_hot_loop_parity_claim_allowed") is False,
        "no_external_inference_by_policy": requires.get("external_inference_calls") == 0,
        "teacher_disabled_by_policy": requires.get("teacher_used") is False,
        "public_training_zero_by_policy": requires.get("public_training_rows") == 0,
        "no_fallback_returns_by_policy": requires.get("no_fallback_returns") is True,
    }
    execution = {"attempted": False, "reason": "execute_not_requested"}
    if execute and all(preflight_checks.values()):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=240,
            env=theseus_runtime.runtime_env(),
        )
        execution = {
            "attempted": True,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    elif execute:
        execution = {
            "attempted": False,
            "reason": "preflight_failed",
            "failed_checks": [name for name, passed in preflight_checks.items() if not passed],
        }
    train_report = read_json(METAL_CANARY_TRAIN_REPORT, {}) if execute else {}
    artifact = read_json(METAL_CANARY_ARTIFACT, {}) if execute else {}
    result_checks = (
        validate_macos_metal_canary_result(train_report, artifact, bounds)
        if execute
        else {"not_applicable_until_execute": True}
    )
    report = {
        "ok": all(preflight_checks.values()) and (not execute or (bool(execution.get("ok")) and all(result_checks.values()))),
        "policy": "project_theseus_macos_metal_scheduler_canary_v0",
        "created_utc": now(),
        "mode": "execute" if execute else "plan",
        "route_policy": str(METAL_ROUTE_POLICY.relative_to(ROOT)),
        "canary_policy": str(METAL_CANARY_POLICY.relative_to(ROOT)),
        "planned_placement": planned,
        "command": command,
        "execution": execution,
        "train_report": str(METAL_CANARY_TRAIN_REPORT.relative_to(ROOT)),
        "artifact": str(METAL_CANARY_ARTIFACT.relative_to(ROOT)),
        "canary_bounds": bounds,
        "checks": {
            "preflight": preflight_checks,
            "result": result_checks,
        },
        "guardrails": {
            "local_only": True,
            "remote_task_submitted": False,
            "production_scheduler_routing_enabled": False,
            "registers_worker_chunk": False,
            "model_promotion_allowed": False,
            "train_rollout_parity_claim_allowed": False,
            "native_hot_loop_parity_claim_allowed": False,
            "public_benchmark_training_used": False,
            "teacher_used": False,
            "external_inference_calls": 0,
            "no_fallback_returns": True,
        },
        "next_action": "keep production routing locked; any route enablement requires a separate operator-reviewed policy change",
    }
    write_json(METAL_CANARY_REPORT, report)
    return report


def run_macos_metal_token_superposition_scheduler_canary(
    policy: dict[str, Any],
    local: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    route_policy = read_json(METAL_TOKEN_ROUTE_POLICY, {})
    canary_policy = read_json(METAL_TOKEN_CANARY_POLICY, {})
    ladder = read_json(METAL_TOKEN_LADDER, {})
    parity_audit = read_json(METAL_MLX_PARITY_AUDIT, {})
    previous_report = read_json(METAL_TOKEN_REPORT, {})
    previous_artifact = read_json(METAL_TOKEN_ARTIFACT, {})
    bounds = dict(canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {})
    inputs = [
        str(item)
        for item in canary_policy.get("inputs", [])
        if isinstance(item, str) and item.strip()
    ]
    payload = {
        "profile": "reviewed_local_token_superposition_canary",
        "job_id": "job_macos_metal_token_superposition_scheduler_canary",
        "job_family": "macos_metal_token_superposition_canary",
        "arm_id": "apple_metal_token_superposition_control_arm",
        "backend_requirements": ["apple_metal"],
        "command": "train-token-superposition-metal",
        "parity_for": "train-token-superposition-cuda",
        "route_policy": str(METAL_TOKEN_ROUTE_POLICY.relative_to(ROOT)),
        "canary_policy": str(METAL_TOKEN_CANARY_POLICY.relative_to(ROOT)),
        "production_scheduler_routing_enabled": False,
        "merge_policy": "append_report_only_no_promotion",
        "priority": 20,
        "lease_seconds": 600,
        "output_artifacts": [
            {"type": "worker_report", "path": str(METAL_TOKEN_CANARY_TRAIN_REPORT.relative_to(ROOT))},
            {"type": "readout_artifact", "path": str(METAL_TOKEN_CANARY_ARTIFACT.relative_to(ROOT))},
        ],
        **bounds,
    }
    payload.update(requester_fields(local, local, "train_token_superposition_metal_local_canary"))
    planned = placement(
        "train_token_superposition_metal_local_canary",
        local,
        "local",
        "Reviewed local-only scheduler canary for train-token-superposition-metal; production routing remains disabled.",
        payload=payload,
    )
    command = [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-token-superposition-metal",
        "--input",
        ",".join(inputs),
        "--train-seed",
        str(int(bounds.get("train_seed") or 1)),
        "--max-language-rows",
        str(int(bounds.get("max_language_rows") or 32)),
        "--max-code-files",
        str(int(bounds.get("max_code_files") or 0)),
        "--max-chars-per-doc",
        str(int(bounds.get("max_chars_per_doc") or 6000)),
        "--max-vocab",
        str(int(bounds.get("max_vocab") or 32)),
        "--hv-dim",
        str(int(bounds.get("hv_dim") or 64)),
        "--train-samples",
        str(int(bounds.get("train_samples") or 64)),
        "--eval-samples",
        str(int(bounds.get("eval_samples") or 32)),
        "--baseline-epochs",
        str(int(bounds.get("baseline_epochs") or 1)),
        "--bag-sizes",
        str(bounds.get("bag_sizes") or "4"),
        "--recovery-ratios",
        str(bounds.get("recovery_ratios") or "0.5"),
        "--samples-per-launch",
        str(int(bounds.get("samples_per_launch") or 8)),
        "--gate-tolerance",
        str(float(bounds.get("gate_tolerance") or 0.002)),
        "--model-out",
        str(METAL_TOKEN_CANARY_ARTIFACT.relative_to(ROOT)),
        "--out",
        str(METAL_TOKEN_CANARY_TRAIN_REPORT.relative_to(ROOT)),
    ]
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    required_reports = canary_policy.get("required_reports") if isinstance(canary_policy.get("required_reports"), list) else []
    input_scope = canary_policy.get("input_scope") if isinstance(canary_policy.get("input_scope"), list) else []
    preflight_checks = {
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy")
        == "project_theseus_macos_metal_token_superposition_scheduler_canary_policy_v0",
        "command_matches": canary_policy.get("command") == "train-token-superposition-metal",
        "task_kind_matches": canary_policy.get("task_kind") == "train_token_superposition_metal_local_canary",
        "backend_matches": canary_policy.get("backend") == "apple_metal",
        "local_only_policy": canary_policy.get("local_only") is True,
        "planned_placement_local_only": planned.get("target") == "local",
        "canary_not_registered_worker_chunk": "train_token_superposition_metal_local_canary" not in WORKER_CHUNK_TASKS,
        "route_policy_present": bool(route_policy),
        "route_policy_command_matches": route_policy.get("command") == "train-token-superposition-metal",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "production_scheduler_routing_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": canary_policy.get("remote_task_submitted") is False,
        "no_remote_task_scope_change": bool(canary_policy.get("does_not_change_hive_remote_task_scope")),
        "no_arbitrary_remote_execution": bool(canary_policy.get("no_arbitrary_remote_execution")),
        "input_scope_private": "data/training_data/high_transfer/private_train" in input_scope
        and all(private_token_input(path) for path in inputs),
        "ladder_ok_required": requires.get("macos_metal_token_superposition_ladder_ok") is True
        and ladder.get("trigger_state") == "GREEN",
        "previous_artifact_ok_required": requires.get("canonical_readout_artifact") is True
        and bool(get_path(previous_report, ["artifact_write", "production_checkpoint_compatible"], False))
        and bool(previous_artifact),
        "audit_counts_token_ladder": requires.get("macos_mlx_parity_audit_counts_token_ladder") is True
        and int(get_path(parity_audit, ["summary", "native_train_token_superposition_ladder_count"], 0) or 0) >= 1,
        "audit_counts_token_artifact": requires.get("macos_mlx_parity_audit_counts_token_artifact") is True
        and int(get_path(parity_audit, ["summary", "native_train_token_superposition_artifact_equivalence_count"], 0) or 0) >= 1,
        "required_reports_declared": all(
            path in required_reports
            for path in [
                "reports/token_superposition_metal_training.json",
                "reports/macos_metal_token_superposition_readout_artifact.json",
                "reports/macos_metal_token_superposition_ladder.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "bounded_kernel_launch_cap_declared": 0 < int(bounds.get("max_kernel_launches") or 0) <= 256,
        "max_code_files_zero": int(bounds.get("max_code_files") or 0) == 0,
        "promotion_locked_by_policy": requires.get("model_promotion_allowed") is False,
        "parity_claim_locked_by_policy": requires.get("train_token_superposition_parity_claim_allowed") is False
        and requires.get("native_hot_loop_parity_claim_allowed") is False,
        "no_external_inference_by_policy": requires.get("external_inference_calls") == 0,
        "teacher_disabled_by_policy": requires.get("teacher_used") is False,
        "public_training_zero_by_policy": requires.get("public_training_rows") == 0,
        "no_fallback_returns_by_policy": requires.get("no_fallback_returns") is True,
        "public_calibration_not_run_by_policy": requires.get("public_calibration_not_run") is True,
    }
    execution = {"attempted": False, "reason": "execute_not_requested"}
    if execute and all(preflight_checks.values()):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=240,
            env=theseus_runtime.runtime_env(),
        )
        execution = {
            "attempted": True,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    elif execute:
        execution = {
            "attempted": False,
            "reason": "preflight_failed",
            "failed_checks": [name for name, passed in preflight_checks.items() if not passed],
        }
    train_report = read_json(METAL_TOKEN_CANARY_TRAIN_REPORT, {}) if execute else {}
    artifact = read_json(METAL_TOKEN_CANARY_ARTIFACT, {}) if execute else {}
    result_checks = (
        validate_macos_metal_token_superposition_canary_result(train_report, artifact, bounds, inputs)
        if execute
        else {"not_applicable_until_execute": True}
    )
    report = {
        "ok": all(preflight_checks.values()) and (not execute or (bool(execution.get("ok")) and all(result_checks.values()))),
        "policy": "project_theseus_macos_metal_token_superposition_scheduler_canary_v0",
        "created_utc": now(),
        "mode": "execute" if execute else "plan",
        "route_policy": str(METAL_TOKEN_ROUTE_POLICY.relative_to(ROOT)),
        "canary_policy": str(METAL_TOKEN_CANARY_POLICY.relative_to(ROOT)),
        "planned_placement": planned,
        "command": command,
        "execution": execution,
        "train_report": str(METAL_TOKEN_CANARY_TRAIN_REPORT.relative_to(ROOT)),
        "artifact": str(METAL_TOKEN_CANARY_ARTIFACT.relative_to(ROOT)),
        "canary_bounds": bounds,
        "inputs": inputs,
        "checks": {
            "preflight": preflight_checks,
            "result": result_checks,
        },
        "guardrails": {
            "local_only": True,
            "remote_task_submitted": False,
            "production_scheduler_routing_enabled": False,
            "registers_worker_chunk": False,
            "model_promotion_allowed": False,
            "train_token_superposition_parity_claim_allowed": False,
            "native_hot_loop_parity_claim_allowed": False,
            "public_benchmark_training_used": False,
            "teacher_used": False,
            "external_inference_calls": 0,
            "no_fallback_returns": True,
            "public_calibration_run": False,
        },
        "next_action": "keep production routing locked; any route enablement requires a separate operator-reviewed policy change",
    }
    write_json(METAL_TOKEN_CANARY_REPORT, report)
    return report


def validate_macos_metal_dry_run_result(
    train_report: dict[str, Any],
    artifact: dict[str, Any],
    bounds: dict[str, Any],
) -> dict[str, bool]:
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    hv_dim = int(get_path(train_report, ["config", "hv_dim"], 0) or 0)
    output_dim = int(get_path(train_report, ["config", "output_dim"], 0) or 0)
    kernel_launches = int(train_report.get("kernel_launches") or 0)
    return {
        "train_report_ok": bool(train_report.get("ok")),
        "command_matches": train_report.get("command") == "train-rollout-metal",
        "backend_matches": train_report.get("backend") == "apple_metal",
        "artifact_write_attempted": bool(get_path(train_report, ["artifact_write", "attempted"], False)),
        "artifact_write_production_compatible": bool(get_path(train_report, ["artifact_write", "production_checkpoint_compatible"], False)),
        "artifact_file_loaded": bool(artifact),
        "artifact_hv_dim_matches_report": artifact.get("hv_dim") == hv_dim == int(bounds["hv_dim"]),
        "artifact_output_dim_matches_report": artifact.get("output_dim") == output_dim == int(bounds["output_dim"]),
        "artifact_label_count_matches": len(labels) == output_dim,
        "artifact_weights_count_matches": len(weights) == hv_dim * output_dim,
        "artifact_bias_count_matches": len(bias) == output_dim,
        "feature_set_matches_report": artifact.get("feature_set") == train_report.get("feature_set"),
        "kernel_launches_bounded": 0 < kernel_launches <= int(bounds["max_kernel_launches"]),
        "work_receipt_accepted": bool(get_path(train_report, ["work_receipt", "accepted"], False)),
        "scheduler_routing_still_disabled": get_path(train_report, ["report_contract", "scheduler_routing_enabled"], True) is False,
        "promotion_still_locked": not bool(train_report.get("model_promotion_allowed")) and not bool(get_path(train_report, ["promotion_decision", "promote_to_training_lane"], False)),
        "parity_claim_still_locked": not bool(train_report.get("parity_claim_allowed")) and not bool(train_report.get("train_rollout_parity_claim_allowed")) and not bool(train_report.get("full_cli_parity_claim_allowed")),
        "external_inference_zero": int(train_report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": not bool(train_report.get("teacher_used")),
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": bool(get_path(train_report, ["guardrails", "no_fallback_returns"], False)),
    }


def validate_macos_metal_canary_result(
    train_report: dict[str, Any],
    artifact: dict[str, Any],
    bounds: dict[str, Any],
) -> dict[str, bool]:
    base = validate_macos_metal_dry_run_result(train_report, artifact, bounds)
    runtime = train_report.get("runtime_profile") if isinstance(train_report.get("runtime_profile"), dict) else {}
    guardrails = train_report.get("guardrails") if isinstance(train_report.get("guardrails"), dict) else {}
    contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    tolerance = float(bounds.get("tolerance") or 0.0)
    kernel_launches = int(train_report.get("kernel_launches") or 0)
    base.update(
        {
            "mode_is_native_metal": runtime.get("backend") == "apple_metal",
            "native_rust_owned": runtime.get("native_rust_owned") is True,
            "python_mlx_bridge_not_used": runtime.get("python_mlx_bridge_used") is False,
            "train_rows_match_bounds": int(train_report.get("train_rows") or 0) == int(bounds.get("cases_per_task") or 0),
            "eval_rows_match_bounds": int(train_report.get("eval_rows") or 0) == int(bounds.get("cases_per_task") or 0),
            "kernel_launches_within_canary_cap": 0 < kernel_launches <= int(bounds.get("max_kernel_launches") or 0),
            "tolerance_declared": abs(float(get_path(train_report, ["args", "tolerance"], 0.0) or 0.0) - tolerance) <= 1.0e-8,
            "tolerance_bounded": 0.0 < tolerance <= 0.0005,
            "report_contract_matches_cli_surface": bool(contract.get("matches_train_rollout_cli_surface")),
            "no_scheduler_route_enabled": contract.get("scheduler_routing_enabled") is False
            and guardrails.get("does_not_route_scheduler_to_metal") is True,
            "symbolic_fallback_false": train_report.get("symbolic_fallback") is False,
        }
    )
    return base


def validate_macos_metal_token_superposition_canary_result(
    train_report: dict[str, Any],
    artifact: dict[str, Any],
    bounds: dict[str, Any],
    expected_inputs: list[str],
) -> dict[str, bool]:
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    dataset = train_report.get("dataset") if isinstance(train_report.get("dataset"), dict) else {}
    args = train_report.get("args") if isinstance(train_report.get("args"), dict) else {}
    contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    artifact_write = train_report.get("artifact_write") if isinstance(train_report.get("artifact_write"), dict) else {}
    hv_dim = int(dataset.get("hv_dim") or get_path(train_report, ["config", "hv_dim"], 0) or 0)
    output_dim = int(dataset.get("vocab_size") or 0)
    kernel_launches = token_superposition_kernel_launches(train_report)
    actual_inputs = [item for item in str(args.get("input") or "").split(",") if item]
    return {
        "train_report_ok": bool(train_report.get("ok")),
        "command_matches": train_report.get("command") == "train-token-superposition-metal",
        "backend_matches": train_report.get("backend") == "apple_metal",
        "implementation_matches": train_report.get("implementation") == "rust_metal_token_superposition_readout_cli",
        "input_paths_match_policy": actual_inputs == expected_inputs,
        "input_paths_private": all(private_token_input(path) for path in actual_inputs),
        "project_code_excluded": args.get("include_project_code") is False
        and int(args.get("max_code_files") or 0) == 0,
        "artifact_write_attempted": bool(artifact_write.get("attempted")),
        "artifact_write_production_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "artifact_file_loaded": bool(artifact),
        "artifact_hv_dim_matches_report": artifact.get("hv_dim") == hv_dim == int(bounds["hv_dim"]),
        "artifact_output_dim_matches_report": artifact.get("output_dim") == output_dim == int(bounds["max_vocab"]),
        "artifact_label_count_matches": len(labels) == output_dim,
        "artifact_weights_count_matches": len(weights) == hv_dim * output_dim,
        "artifact_bias_count_matches": len(bias) == output_dim,
        "feature_set_matches_report": artifact.get("feature_set") == artifact_write.get("feature_set")
        == "metal_token_superposition_readout_private_residual_train_eval",
        "kernel_launches_bounded": 0 < kernel_launches <= int(bounds["max_kernel_launches"]),
        "work_receipt_accepted": bool(get_path(train_report, ["work_receipt", "accepted"], False))
        and get_path(train_report, ["work_receipt", "task_kind"], "") == "train_token_superposition_metal_cli",
        "scheduler_routing_still_disabled": contract.get("scheduler_routing_enabled") is False,
        "python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "promotion_still_locked": not bool(train_report.get("model_promotion_allowed"))
        and not bool(get_path(train_report, ["promotion_decision", "promote_to_training_lane"], False)),
        "parity_claim_still_locked": not bool(train_report.get("train_token_superposition_parity_claim_allowed"))
        and not bool(train_report.get("full_cli_parity_claim_allowed"))
        and artifact_write.get("train_token_superposition_parity_claim_allowed") is False,
        "external_inference_zero": int(train_report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": not bool(train_report.get("teacher_used")),
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0,
        "public_calibration_not_run": bool(get_path(train_report, ["guardrails", "no_public_calibration"], False)),
        "no_fallback_returns": bool(get_path(train_report, ["guardrails", "no_fallback_returns"], False)),
    }


def token_superposition_kernel_launches(report: dict[str, Any]) -> int:
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    variants = report.get("variants") if isinstance(report.get("variants"), list) else []
    return int(baseline.get("kernel_launches") or 0) + sum(
        int(row.get("kernel_launches") or 0)
        for row in variants
        if isinstance(row, dict)
    )


def private_token_input(path: str) -> bool:
    if not path:
        return False
    root = (ROOT / "data" / "training_data" / "high_transfer" / "private_train").resolve()
    candidate = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.exists() and candidate.suffix == ".jsonl"


def summarize(nodes: list[dict[str, Any]], placements: list[dict[str, Any]]) -> dict[str, Any]:
    remote = [row for row in placements if row.get("target") == "remote"]
    worker_chunks = [row for row in placements if row.get("task_kind") in WORKER_CHUNK_TASKS]
    light_nodes = [node for node in nodes if node.get("light_task_allowed", True)]
    training_nodes = [node for node in light_nodes if node.get("training_allowed", True) and not node.get("training_blockers")]
    caps = {}
    network_classes: dict[str, int] = {}
    for node in nodes:
        network = node_network_profile(node)
        network_class = str(network.get("class") or "unknown")
        network_classes[network_class] = network_classes.get(network_class, 0) + 1
        for cap in node.get("capabilities") or []:
            cap_id = cap.get("id")
            if cap_id:
                caps[cap_id] = caps.get(cap_id, 0) + 1
    gas = sum(int(get_path(row, ["compute_market", "gas_estimate_micro_twc"], 0) or 0) for row in placements)
    return {
        "nodes": len(nodes),
        "remote_placements": len(remote),
        "real_worker_chunks": len(worker_chunks),
        "gas_estimate_micro_twc": gas,
        "capabilities": caps,
        "network_classes": network_classes,
        "best_training_node": (best_for_capability(training_nodes, ["nvidia_cuda", "apple_mlx", "cpu_worker"]) or {}).get("node_name"),
        "best_cuda_node": (best_for_capability(training_nodes, ["nvidia_cuda"]) or {}).get("node_name")
        if any(node_has_capability(node, "nvidia_cuda") for node in training_nodes)
        else None,
        "best_mlx_node": (best_for_capability(training_nodes, ["mlx_apple", "mlx_cuda", "apple_mlx"]) or {}).get("node_name")
        if any(node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]) for node in training_nodes)
        else None,
        "best_inference_node": (best_for_capability(light_nodes, ["apple_mlx", "mlx_apple", "nvidia_cuda", "cpu_worker"]) or {}).get("node_name"),
        "training_blocked_nodes": [
            {"node_name": node.get("node_name"), "training_blockers": node.get("training_blockers")}
            for node in nodes
            if node.get("training_blockers")
        ],
    }


def summarize_viea_route_records(
    placements: list[dict[str, Any]],
    execution: list[dict[str, Any]] | None = None,
    execution_receipt_smoke: dict[str, Any] | None = None,
    route_validator_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    route_count = 0
    execution_receipt_count = 0
    route_record_total = 0
    execution_receipt_record_total = 0
    blocked_routes = 0
    missing_record_placements = []

    def count_records(records: dict[str, Any], *, execution_receipt: bool = False) -> int:
        nonlocal execution_receipt_count, blocked_routes
        if execution_receipt and records:
            execution_receipt_count += 1
        record_total = 0
        for record in records.values():
            if not isinstance(record, dict):
                continue
            record_total += 1
            record_type = str(record.get("record_type") or "")
            counts[record_type] = counts.get(record_type, 0) + 1
            if record_type == "failure_boundary" and record.get("status") == "BLOCKED":
                blocked_routes += 1
        return record_total

    for row in placements:
        records = row.get("viea_route_records") if isinstance(row.get("viea_route_records"), dict) else {}
        if records:
            route_count += 1
        else:
            missing_record_placements.append(row.get("placement_id") or row.get("task_kind"))
        route_record_total += count_records(records)
    smoke = execution_receipt_smoke if isinstance(execution_receipt_smoke, dict) else {}
    route_validator = route_validator_receipt if isinstance(route_validator_receipt, dict) else {}
    smoke_records = smoke.get("viea_execution_records") if isinstance(smoke.get("viea_execution_records"), dict) else {}
    execution_receipt_record_total += count_records(smoke_records, execution_receipt=True)
    for row in execution or []:
        records = row.get("viea_execution_records") if isinstance(row.get("viea_execution_records"), dict) else {}
        execution_receipt_record_total += count_records(records, execution_receipt=True)
    return {
        "producer_profile": "hive_scheduler_route_records_v1",
        "route_record_placement_count": route_count,
        "route_record_count": route_record_total,
        "execution_receipt_record_set_count": execution_receipt_count,
        "execution_receipt_record_count": execution_receipt_record_total,
        "execution_receipt_smoke_ready": bool(smoke.get("ready")),
        "route_validator_ready": bool(route_validator.get("ready")),
        "route_validator_receipt_id": route_validator.get("receipt_id"),
        "route_validator_missing_group_count": len(route_validator.get("missing_required_groups") if isinstance(route_validator.get("missing_required_groups"), list) else []),
        "spine_record_count": sum(counts.values()),
        "record_counts": dict(sorted(counts.items())),
        "blocked_route_count": blocked_routes,
        "missing_record_placements": missing_record_placements,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "Hive route records prove scheduler traceability, not learned-generation capability.",
    }


def summarize_jobs(placements: list[dict[str, Any]]) -> dict[str, Any]:
    jobs = []
    by_arm: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    for row in placements:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if not payload.get("job_id"):
            continue
        arm_id = str(payload.get("arm_id") or "unknown_arm")
        backends = payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else []
        by_arm[arm_id] = by_arm.get(arm_id, 0) + 1
        for backend in backends:
            by_backend[str(backend)] = by_backend.get(str(backend), 0) + 1
        jobs.append(
            {
                "job_id": payload.get("job_id"),
                "task_kind": row.get("task_kind"),
                "arm_id": arm_id,
                "job_family": payload.get("job_family"),
                "target": row.get("target"),
                "node_id": row.get("node_id"),
                "node_name": row.get("node_name"),
                "backend_requirements": backends,
                "merge_policy": payload.get("merge_policy"),
                "priority": payload.get("priority"),
                "lease_seconds": payload.get("lease_seconds"),
                "output_artifacts": payload.get("output_artifacts", []),
                "compute_market": row.get("compute_market", {}),
            }
        )
    return {
        "job_count": len(jobs),
        "by_arm": by_arm,
        "by_backend": by_backend,
        "jobs": jobs,
        "distributed_optimizer_sync": False,
        "parallelism_model": "embarrassingly_parallel_artifact_merge",
    }


def best_for_capability(nodes: list[dict[str, Any]], capability_order: list[str]) -> dict[str, Any]:
    for capability in capability_order:
        matching = [
            node
            for node in nodes
            if any(cap.get("id") == capability for cap in node.get("capabilities") or [])
        ]
        if matching:
            return max(matching, key=lambda node: node_score(node))
    return max(nodes, key=lambda node: node_score(node)) if nodes else {}


def node_has_capability(node: dict[str, Any], capability: str) -> bool:
    return any(cap.get("id") == capability for cap in node.get("capabilities") or [])


def node_has_any_capability(node: dict[str, Any], capabilities: list[str]) -> bool:
    wanted = set(capabilities)
    return any(cap.get("id") in wanted for cap in node.get("capabilities") or [])


def node_has_slot(node: dict[str, Any], slot_types: list[str]) -> bool:
    wanted = set(slot_types)
    slots = node.get("slots") if isinstance(node.get("slots"), list) else []
    if slots:
        return any(slot.get("slot_type") in wanted and int(slot.get("capacity") or 0) > 0 for slot in slots if isinstance(slot, dict))
    if "cuda" in wanted and node_has_capability(node, "nvidia_cuda"):
        return True
    if wanted & {"mlx_apple", "mlx_cuda"} and node_has_any_capability(node, ["mlx_apple", "mlx_cuda", "apple_mlx"]):
        return True
    if "cpu" in wanted and node_has_capability(node, "cpu_worker"):
        return True
    return False


def compact_license(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration_complete": status.get("registration_complete"),
        "tier": get_path(status, ["entitlement", "tier"], ""),
        "source": get_path(status, ["entitlement", "source"], ""),
        "paid": get_path(status, ["entitlement", "paid"], False),
        "can_run_worker_chunks": get_path(status, ["feature_summary", "can_run_worker_chunks"], False),
        "next_action": status.get("next_action"),
    }


def node_network_profile(node: dict[str, Any]) -> dict[str, Any]:
    parsed = urlparse(str(node.get("api_url") or ""))
    host = parsed.hostname or ""
    scope = host_scope(host)
    if scope == "loopback":
        network_class = "local"
        latency_ms = 0
        task_fit = "interactive_and_training"
    elif scope in {"private_ip", "private_dns"}:
        network_class = "lan_or_private_tunnel"
        latency_ms = 8
        task_fit = "interactive_and_training"
    elif scope in {"public_ip", "public_dns", "unknown_dns"}:
        network_class = "wan"
        latency_ms = 80
        task_fit = "bounded_async_chunks"
    else:
        network_class = "relay_or_unknown"
        latency_ms = 120
        task_fit = "bounded_async_chunks"
    return {
        "class": network_class,
        "host_scope": scope,
        "estimated_latency_ms": latency_ms,
        "task_fit": task_fit,
    }


def host_scope(host: str) -> str:
    if not host:
        return "unknown"
    lowered = host.lower()
    if lowered in {"localhost", "ip6-localhost"}:
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if lowered.endswith(".local") or lowered.endswith(".lan") or lowered.endswith(".home"):
            return "private_dns"
        if "." in lowered:
            return "public_dns"
        return "unknown_dns"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private or ip.is_link_local:
        return "private_ip"
    return "public_ip"


def node_score(node: dict[str, Any]) -> float:
    score = 0.0
    for cap in node.get("capabilities") or []:
        score += float(cap.get("score") or 0.0)
    nvidia = get_path(node, ["resources", "nvidia"], {})
    if nvidia.get("available"):
        free = [
            float(gpu.get("memory_free_mib") or 0.0)
            for gpu in nvidia.get("gpus") or []
            if isinstance(gpu, dict)
        ]
        score += min(0.4, (max(free) if free else 0.0) / 24000.0)
    memory = get_path(node, ["resources", "memory"], {})
    score += min(0.15, float(memory.get("available_gib") or memory.get("total_gib") or 0.0) / 256.0)
    latency_ms = float(node_network_profile(node).get("estimated_latency_ms") or 0.0)
    score -= min(0.25, latency_ms / 1000.0)
    score -= power_penalty(node)
    return score


def power_penalty(node: dict[str, Any]) -> float:
    power = get_path(node, ["resources", "power"], {})
    if not isinstance(power, dict) or not power:
        return 0.0
    penalty = 0.0
    if power.get("on_ac_power") is False:
        penalty += 0.45
        pct = power.get("battery_percent")
        if isinstance(pct, (int, float)) and float(pct) < 40:
            penalty += 0.55
    return penalty


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "api_url": status.get("api_url"),
        "capabilities": status.get("capabilities") or [],
        "resources": status.get("resources") or {},
        "slots": status.get("slots") or [],
        "runtime_paths": status.get("runtime_paths") or {},
    }


def shared_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    env_value = os.environ.get(env_name, "")
    if env_value:
        return env_value
    join_cfg = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    if isinstance(join_cfg, dict) and join_cfg.get("join_token"):
        return str(join_cfg.get("join_token") or "")
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            token = str(profile.get("join_token") or "")
            if token:
                return token
    return ""


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
