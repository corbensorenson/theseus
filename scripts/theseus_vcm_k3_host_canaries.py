#!/usr/bin/env python3
"""Prospectively sealed, non-scoring physical host canaries for VCM K3."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_local_inference_backend_v2 as backend  # noqa: E402
import theseus_semantic_ir_production_adequacy_backend as host_backend  # noqa: E402
import theseus_vcm_k3_route_preflight as route_owner  # noqa: E402

POLICY = "project_theseus_vcm_k3_non_scoring_host_canaries_v1"
CONFIG_POLICY = "project_theseus_vcm_k3_non_scoring_host_canaries_config_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_k3_host_canaries.json"
NORMAL_TERMINATIONS = {"parser_complete", "model_eos"}


class CanaryMlxChatModel(host_backend.AdequacyLocalMlxChatModel):
    """Use the exact K3 system envelope and stop only at marker, EOS, or safety."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        exact = [dict(row) for row in messages]
        exact[0]["content"] = route_owner.SYSTEM_PROMPT
        return super().generate(exact)


def identity(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path), "bytes": path.stat().st_size}


def read_bound(cfg: dict[str, Any], path_key: str, hash_key: str, faults: list[str]) -> dict[str, Any]:
    path = p2a.resolve(str(cfg.get(path_key) or ""))
    if not path.is_file() or p2a.sha256_file(path) != cfg.get(hash_key):
        faults.append(f"binding_invalid:{path_key}")
        return {}
    try:
        return p2a.read_json(path)
    except Exception as exc:
        faults.append(f"binding_read_failed:{path_key}:{type(exc).__name__}")
        return {}


def validate_source_binding(cfg: dict[str, Any], path_key: str, hash_key: str, faults: list[str]) -> None:
    path = p2a.resolve(str(cfg.get(path_key) or ""))
    if not path.is_file() or p2a.sha256_file(path) != cfg.get(hash_key):
        faults.append(f"binding_invalid:{path_key}")


def materialize_selected_prompts(
    cfg: dict[str, Any], preflight: dict[str, Any], faults: list[str]
) -> list[dict[str, Any]]:
    preflight_cfg = p2a.read_json(p2a.resolve(str(cfg.get("route_preflight_config") or "")))
    store = read_bound(preflight_cfg, "k2_store", "k2_store_sha256", faults)
    k2_report = read_bound(preflight_cfg, "k2_report", "k2_report_sha256", faults)
    visible = {str(row.get("request_id")): row for row in p2a.dicts(k2_report.get("rows"))}
    stores = {str(row.get("request_id")): row for row in p2a.dicts(store.get("rows"))}
    packet_rows = {
        (str(row.get("request_id")), str(row.get("route"))): row
        for row in p2a.dicts(read_bound(cfg, "packet_manifest", "packet_manifest_sha256", faults).get("rows"))
    }
    plan = p2a.mapping(preflight.get("host_canary_plan"))
    order = p2a.strings(cfg.get("route_execution_order"))
    selected: list[dict[str, Any]] = []
    for route in order:
        frozen = p2a.mapping(plan.get(route))
        request_id = str(frozen.get("request_id") or "")
        store_row = p2a.mapping(stores.get(request_id))
        surface = p2a.mapping(p2a.mapping(visible.get(request_id)).get("candidate_surface"))
        request = str(surface.get("natural_language_request") or "")
        selector = p2a.mapping(store_row.get("selector"))
        frontier = p2a.dicts(selector.get("frontier"))
        page_cache = route_owner.load_text_pages(store_row)
        request_terms = p2a.strings(selector.get("request_terms"))
        enriched: list[dict[str, Any]] = []
        for raw in frontier:
            row = dict(raw)
            searchable = f"{row.get('path', '')}\n{page_cache.get(str(row.get('path') or ''), '')}".lower()
            row["matched_request_terms"] = (
                [term for term in request_terms if term in searchable]
                if int(row.get("score") or 0) > 0
                else []
            )
            enriched.append(row)
        vcm = route_owner.minimum_request_cover(enriched)
        ordinary = [row for row in enriched if int(row.get("score") or 0) > 0]
        pages = route_owner.route_pages(route, vcm, ordinary, enriched, request_id)
        context = route_owner.render_context(
            route, pages, lambda row: page_cache[str(row.get("path") or "")], request_id
        )
        prompt = route_owner.render_prompt(request, route, context)
        packet = p2a.mapping(packet_rows.get((request_id, route)))
        if p2a.sha256_text(prompt) != frozen.get("prompt_sha256"):
            faults.append(f"selected_prompt_hash_mismatch:{route}")
        if packet.get("prompt_sha256") != frozen.get("prompt_sha256"):
            faults.append(f"packet_plan_mismatch:{route}")
        if packet.get("physically_addressable") is not True:
            faults.append(f"selected_packet_not_physically_addressable:{route}")
        selected.append(
            {
                "route": route,
                "request_id": request_id,
                "prompt": prompt,
                "prompt_sha256": p2a.sha256_text(prompt),
                "expected_prompt_tokens": int(frozen.get("exact_chat_prompt_tokens") or 0),
                "expected_context_residual_tokens": int(frozen.get("physical_context_residual_tokens") or 0),
                "route_context_digest": str(packet.get("context_information_sha256") or ""),
                "execution_mode": "integrated_local_model" if route == "governed_vcm" else "direct_local_model",
            }
        )
    return selected


def build_plan(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    for path_key, hash_key in (
        ("owner", "owner_sha256"),
        ("audit_owner", "audit_owner_sha256"),
        ("test", "test_sha256"),
        ("route_preflight_config", "route_preflight_config_sha256"),
        ("backend_owner", "backend_owner_sha256"),
        ("host_interlock_owner", "host_interlock_owner_sha256"),
        ("worker_config", "worker_config_sha256"),
        ("runtime_preflight", "runtime_preflight_sha256"),
    ):
        validate_source_binding(cfg, path_key, hash_key, faults)
    preflight = read_bound(cfg, "route_preflight_report", "route_preflight_report_sha256", faults)
    audit = read_bound(cfg, "route_preflight_audit", "route_preflight_audit_sha256", faults)
    if preflight.get("trigger_state") != "GREEN" or audit.get("trigger_state") != "GREEN":
        faults.append("static_route_preflight_not_green")
    if int(audit.get("audited_packet_count") or 0) != 36:
        faults.append("static_route_denominator_invalid")
    authority = p2a.mapping(cfg.get("authority"))
    if authority.get("non_scoring_host_canary_calls_authorized") != 6:
        faults.append("host_canary_authority_invalid")
    if any(int(authority.get(key) or 0) != 0 for key in ("candidate_screen_calls_authorized", "hidden_evaluator_calls_authorized", "external_reference_calls_authorized")):
        faults.append("forbidden_call_authority_open")
    if cfg.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    selected = materialize_selected_prompts(cfg, preflight, faults)
    if len(selected) != 6 or len({row["route"] for row in selected}) != 6:
        faults.append("host_canary_route_denominator_invalid")
    report = {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K3_NONSCORING_HOST_CANARY_PLAN_GREEN_EXECUTION_READY" if not faults else "K3_NONSCORING_HOST_CANARY_PLAN_INVALID",
        "faults": sorted(set(faults)),
        "config": identity(path),
        "planned_call_count": len(selected),
        "route_execution_order": [row["route"] for row in selected],
        "selected_receipts": [{k: v for k, v in row.items() if k != "prompt"} for row in selected],
        "project_selected_quality_token_cap": None,
        "capability_or_mechanism_evidence": False,
        "candidate_screen_calls_authorized": 0,
        "hidden_evaluator_calls": 0,
        "external_reference_calls": 0,
        "local_model_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }
    return report, selected


def source_control_seal_green(cfg: dict[str, Any]) -> bool:
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    required = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(cfg.get("owner")), str(cfg.get("audit_owner")), str(cfg.get("test")), p2a.rel(DEFAULT_CONFIG)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return tracked.returncode == 0 and not tracked.stdout.strip() and required.returncode == 0


def execute(
    path: Path = DEFAULT_CONFIG,
    *,
    runner: Callable[..., dict[str, Any]] = backend.run_backend,
    seal_check: Callable[[dict[str, Any]], bool] = source_control_seal_green,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    model_factory: Callable[[dict[str, Any], Path, int], Any] | None = None,
) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    report, selected = build_plan(path)
    if report["trigger_state"] != "GREEN" or not seal_check(cfg):
        report["trigger_state"] = "RED"
        report["state"] = "K3_NONSCORING_HOST_CANARY_EXECUTION_NOT_AUTHORIZED"
        report["faults"] = sorted(set(p2a.strings(report.get("faults")) + ["prospective_source_control_seal_not_green"]))
        return report
    wall_seconds = float(cfg.get("host_safety_wall_seconds_per_call") or 0)
    cached: dict[str, Any] = {}

    def default_factory(card: dict[str, Any], snapshot: Path, maximum: int) -> Any:
        if "model" not in cached:
            cached["model"] = CanaryMlxChatModel(
                card, snapshot, maximum,
                completion_predicate=lambda text: "CANARY_OK" in text,
                maximum_wall_seconds=wall_seconds,
            )
        return cached["model"]

    factory = model_factory or default_factory
    calls: list[dict[str, Any]] = []
    for ordinal, row in enumerate(selected, start=1):
        raw = runner(
            worker_config_path=p2a.resolve(str(cfg.get("worker_config"))),
            runtime_preflight_path=p2a.resolve(str(cfg.get("runtime_preflight"))),
            execution_mode=row["execution_mode"],
            route_context_digest=row["route_context_digest"],
            session_id=f"vcm_k3_host_canary_{ordinal}",
            prompt=row["prompt"],
            maximum_tokens=0,
            required_repo_id=str(cfg.get("required_repo_id") or ""),
            required_revision=str(cfg.get("required_revision") or ""),
            required_snapshot_manifest_sha256=str(cfg.get("required_snapshot_manifest_sha256") or ""),
            model_factory=factory,
        )
        metrics = p2a.mapping(raw.get("metrics"))
        answer = str(p2a.mapping(raw.get("response")).get("answer") or "")
        termination = str(metrics.get("termination_reason") or "")
        faults: list[str] = []
        if raw.get("trigger_state") != "GREEN":
            faults.append("backend_not_green")
        if int(metrics.get("exact_prompt_tokens") or 0) != row["expected_prompt_tokens"]:
            faults.append("actual_prompt_token_count_mismatch")
        if int(metrics.get("effective_context_residual_tokens") or 0) != row["expected_context_residual_tokens"]:
            faults.append("actual_context_residual_mismatch")
        if termination not in NORMAL_TERMINATIONS:
            faults.append("invalid_host_operability_termination")
        if termination == "parser_complete" and "CANARY_OK" not in answer:
            faults.append("canary_marker_not_observed")
        if metrics.get("host_safety_wall_time_hit") is True or metrics.get("physical_context_boundary_hit") is True:
            faults.append("host_or_physical_safety_interlock_activated")
        receipt = {
            **{k: v for k, v in row.items() if k != "prompt"},
            "call_ordinal": ordinal,
            "trigger_state": "GREEN" if not faults else "RED",
            "faults": sorted(set(faults + p2a.strings(raw.get("faults")))),
            "termination_reason": termination,
            "generated_tokens": metrics.get("generated_tokens"),
            "actual_prompt_tokens": metrics.get("exact_prompt_tokens"),
            "actual_context_residual_tokens": metrics.get("effective_context_residual_tokens"),
            "runtime_ms": metrics.get("runtime_ms"),
            "generation_wall_ms": metrics.get("generation_wall_ms"),
            "load_wall_ms": metrics.get("load_wall_ms"),
            "mlx_peak_memory_gib": metrics.get("mlx_peak_memory_gib"),
            "prompt_tokens_per_second": metrics.get("prompt_tokens_per_second"),
            "generation_tokens_per_second": metrics.get("generation_tokens_per_second"),
            "response_chars": len(answer),
            "response_sha256": p2a.sha256_text(answer),
            "raw_response_stored": False,
            "capability_or_mechanism_evidence": False,
            "hidden_evaluator_calls": 0,
            "external_reference_calls": 0,
        }
        calls.append(receipt)
        report.update({
            "state": "K3_NONSCORING_PHYSICAL_HOST_CANARIES_RUNNING",
            "calls": calls,
            "local_model_calls": len(calls),
            "completed_call_count": len(calls),
        })
        if checkpoint:
            checkpoint(report)
        if receipt["trigger_state"] != "GREEN" and cfg.get("stop_after_first_invalid_observation") is True:
            break
    all_green = len(calls) == 6 and all(row["trigger_state"] == "GREEN" for row in calls)
    report["trigger_state"] = "GREEN" if all_green else "RED"
    report["state"] = "K3_NONSCORING_PHYSICAL_HOST_CANARIES_GREEN" if all_green else "INCONCLUSIVE_EXPERIMENT_HOST_OPERABILITY"
    report["completed_call_count"] = len(calls)
    report["unexecuted_route_count"] = 6 - len(calls)
    report["nine_task_screen_authorized"] = all_green
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    out = p2a.resolve(args.out or str(cfg.get("report") or ""))
    checkpoint = lambda report: p2a.write_json(out, report)
    report = execute(path, checkpoint=checkpoint) if args.execute else build_plan(path)[0]
    p2a.write_json(out, report)
    print(json.dumps({k: report.get(k) for k in ("trigger_state", "state", "faults", "planned_call_count", "completed_call_count", "local_model_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
