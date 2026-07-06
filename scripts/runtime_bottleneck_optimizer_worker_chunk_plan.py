#!/usr/bin/env python3
"""Worker-chunk plan for the current Theseus runtime bottleneck.

The generic Hive scheduler proves CUDA eval/train/rollout chunks. This report
is narrower: it plans the actual Code LM iteration loop that is currently
blocking speed work: fanout, ranker prefilter, staged verifier, and benchmark
regression checks. It is plan/proof only by default and never launches training
or public calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "runtime_bottleneck_optimizer_worker_chunk_plan.json"
DEFAULT_MARKDOWN = REPORTS / "runtime_bottleneck_optimizer_worker_chunk_plan.md"
DEFAULT_LEASE_OUT = REPORTS / "runtime_bottleneck_optimizer_worker_chunk_leases.jsonl"

REQUIRED_STAGES = [
    "code_lm_fanout",
    "candidate_ranker_prefilter",
    "staged_verifier",
    "benchmark_regression",
]

STAGED_VERIFICATION = [
    {
        "stage": "lint_parse",
        "parallel_group": "verification_fast_fail",
        "fail_fast": True,
        "cache_namespace": "candidate_ast_parse_v1",
        "reward_signal": "syntax_valid_candidate",
    },
    {
        "stage": "compile_or_import",
        "parallel_group": "verification_fast_fail",
        "fail_fast": True,
        "cache_namespace": "candidate_import_contract_v1",
        "reward_signal": "interface_load_candidate",
    },
    {
        "stage": "cheap_behavior",
        "parallel_group": "verification_contract_smoke",
        "fail_fast": True,
        "cache_namespace": "candidate_contract_smoke_v1",
        "reward_signal": "contract_admissible_candidate",
    },
    {
        "stage": "sandbox_full_tests",
        "parallel_group": "verification_sandbox_tail",
        "fail_fast": False,
        "cache_namespace": "candidate_private_or_unlocked_calibration_full_v1",
        "reward_signal": "behavioral_success_candidate",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--lease-out", default=str(DEFAULT_LEASE_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    reports = collect_reports()
    plan = build_plan(reports)
    plan["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    out = resolve(args.out)
    lease_out = resolve(args.lease_out)
    write_json(out, plan)
    write_jsonl(lease_out, plan["leases"])
    plan["lease_ledger"] = rel(lease_out)
    write_json(out, plan)
    write_text(resolve(args.markdown_out), markdown_report(plan))
    print(json.dumps(plan, indent=2))
    return 0 if plan.get("plan_ready") else 2


def collect_reports() -> dict[str, Any]:
    return {
        "system_efficiency": read_json(REPORTS / "system_efficiency_audit.json", {}),
        "performance": read_json(REPORTS / "performance_optimizer.json", {}),
        "resource_governor": read_json(REPORTS / "resource_governor.json", {}),
        "decoder_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {}),
        "transfer_proof": read_json(REPORTS / "private_public_transfer_proof.json", {}),
        "train_once": latest_train_once_report(),
        "real_code": read_json(REPORTS / "real_code_benchmark_graduation.json", {}),
    }


def latest_train_once_report() -> dict[str, Any]:
    preferred = REPORTS / "code_lm_train_once_fanout_eligible_receiver_inventory_router_v1.json"
    if preferred.exists():
        return read_json(preferred, {})
    return read_json(REPORTS / "code_lm_train_once_fanout.json", {})


def build_plan(reports: dict[str, Any]) -> dict[str, Any]:
    train_once = reports["train_once"] if isinstance(reports.get("train_once"), dict) else {}
    system_efficiency = reports["system_efficiency"] if isinstance(reports.get("system_efficiency"), dict) else {}
    speed_proof = system_efficiency.get("current_speed_proof") if isinstance(system_efficiency.get("current_speed_proof"), dict) else {}
    resource = reports["resource_governor"] if isinstance(reports.get("resource_governor"), dict) else {}
    decoder_gate = reports["decoder_gate"] if isinstance(reports.get("decoder_gate"), dict) else {}
    transfer_proof = reports["transfer_proof"] if isinstance(reports.get("transfer_proof"), dict) else {}
    real_code = reports["real_code"] if isinstance(reports.get("real_code"), dict) else {}

    paths = train_once.get("paths") if isinstance(train_once.get("paths"), dict) else {}
    summary = train_once.get("summary") if isinstance(train_once.get("summary"), dict) else {}
    timing = summary.get("phase_timing_ms") if isinstance(summary.get("phase_timing_ms"), dict) else {}
    fanout_timing = timing.get("fanout_report") if isinstance(timing.get("fanout_report"), dict) else {}
    speed_targets = speed_targets_from(speed_proof, fanout_timing)
    cached_artifacts = cached_artifacts_from(paths, speed_proof)
    public_boundary_clean = public_boundary_ok(decoder_gate, transfer_proof, train_once, real_code)
    resource_profile = resource_profile_from(resource)
    chunks = build_chunks(cached_artifacts, speed_targets, resource_profile, public_boundary_clean)
    leases = [lease_for_chunk(chunk) for chunk in chunks]
    ready_checks = readiness_checks(chunks, cached_artifacts, speed_proof, public_boundary_clean, decoder_gate, transfer_proof)
    required_coverage = {stage: any(chunk["stage"] == stage for chunk in chunks) for stage in REQUIRED_STAGES}
    plan_ready = all(required_coverage.values()) and all(row["passed"] for row in ready_checks)

    return {
        "policy": "runtime_bottleneck_optimizer_worker_chunk_plan_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if plan_ready else "YELLOW",
        "plan_ready": plan_ready,
        "objective": "staged cached worker plan for Code LM fanout, ranker, verifier, and benchmark/regression evaluation",
        "mode": "plan_and_speed_proof_only_no_training_no_public_calibration",
        "summary": {
            "required_stage_count": len(REQUIRED_STAGES),
            "planned_stage_count": sum(1 for ok in required_coverage.values() if ok),
            "planned_chunk_count": len(chunks),
            "parallel_group_count": len({chunk["parallel_group"] for chunk in chunks}),
            "cache_namespace_count": len({name for chunk in chunks for name in chunk["cache_namespaces"]}),
            "lease_count": len(leases),
            "resource_throttle_handled": resource_profile["resource_throttle_handled"],
            "bounded_smoke_profile": resource_profile["bounded_smoke_profile"],
            "public_tests_or_solutions_used": False,
            "transfer_gates_ready": bool(speed_proof.get("transfer_gates_ready"))
            or bool(transfer_proof.get("ready_for_public_calibration")),
            "current_speed_proof_ready": bool(speed_proof.get("ready")),
            "public_limit8_ms_per_task": speed_proof.get("public_limit8_ms_per_task"),
            "private_scale_ms_per_task": speed_proof.get("private_scale_private_ms_per_task"),
            "benchmark_parallel_verification_enabled": bool(get_path(real_code, ["summary", "parallel_verification_enabled"], False)),
        },
        "required_stage_coverage": required_coverage,
        "ready_checks": ready_checks,
        "resource_envelope": resource_profile,
        "cached_artifacts": cached_artifacts,
        "staged_verification_contract": STAGED_VERIFICATION,
        "worker_chunks": chunks,
        "leases": leases,
        "speed_proof": speed_proof_report(speed_proof, fanout_timing, real_code),
        "safety": {
            "training_launched": False,
            "public_calibration_launched": False,
            "public_benchmark_training": False,
            "public_tests_or_solutions_used": False,
            "public_boundary": "public tasks remain metadata/calibration only; worker chunks consume candidate manifests, timing, and staged-verification cache keys",
            "external_inference_calls": 0,
        },
        "consumers": [
            "performance_optimizer",
            "system_efficiency_audit",
            "asi_wall_breaker_governor",
            "autonomy_watchdog",
            "hive_work_board_executor",
        ],
        "external_inference_calls": 0,
    }


def speed_targets_from(speed_proof: dict[str, Any], fanout_timing: dict[str, Any]) -> dict[str, Any]:
    return {
        "public_limit8_ms_per_task": number(speed_proof.get("public_limit8_ms_per_task")),
        "private_scale_ms_per_task": number(speed_proof.get("private_scale_private_ms_per_task")),
        "public_candidate_generation_ms": number(fanout_timing.get("public_candidate_generation_and_write")),
        "private_candidate_generation_ms": number(fanout_timing.get("private_candidate_generation_and_write")),
        "ranker_public_ms": number(fanout_timing.get("public_ranker_prefilter_verifier_cache_summary")),
        "ranker_private_ms": number(fanout_timing.get("private_ranker_prefilter_verifier_cache_summary")),
    }


def cached_artifacts_from(paths: dict[str, Any], speed_proof: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        ("checkpoint", paths.get("checkpoint")),
        ("private_candidates", paths.get("private_candidates")),
        ("public_candidates", paths.get("public_candidates")),
        ("fanout_report", paths.get("fanout_report")),
        ("closure_report", paths.get("closure_report")),
        ("phase_ledger", paths.get("phase_ledger")),
        ("current_source_smoke_fanout", paths.get("current_source_smoke_fanout_report")),
        ("private_scale_smoke", speed_proof.get("private_scale_smoke_report")),
        ("public_limit8_smoke", speed_proof.get("public_limit8_smoke_report")),
        ("same_seed_comparator_smoke", speed_proof.get("same_seed_comparator_smoke_report")),
    ]
    out = []
    for kind, value in candidates:
        if not value:
            continue
        path = resolve(str(value))
        out.append({
            "kind": kind,
            "path": rel(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "cache_key": cache_key(kind, path),
        })
    return out


def resource_profile_from(resource: dict[str, Any]) -> dict[str, Any]:
    can_run = get_path(resource, ["decision", "can_run_requested_profile"], None)
    recommended = str(get_path(resource, ["decision", "recommended_profile"], "smoke") or "smoke")
    throttle_reasons = get_path(resource, ["decision", "throttle_reasons"], [])
    throttle_reasons = throttle_reasons if isinstance(throttle_reasons, list) else []
    return {
        "can_run_requested_profile": can_run,
        "recommended_profile": recommended,
        "throttle_reasons": [str(row) for row in throttle_reasons],
        "resource_throttle_handled": can_run is not False or recommended in {"smoke", "micro", "diagnostic"},
        "bounded_smoke_profile": recommended if recommended in {"smoke", "micro", "diagnostic"} else "smoke",
        "gpu": get_path(resource, ["current_resources", "gpu"], {}),
    }


def build_chunks(
    cached_artifacts: list[dict[str, Any]],
    speed_targets: dict[str, Any],
    resource_profile: dict[str, Any],
    public_boundary_clean: bool,
) -> list[dict[str, Any]]:
    artifact_keys = {row["kind"]: row for row in cached_artifacts}
    lease_seconds = 900 if resource_profile["bounded_smoke_profile"] == "smoke" else 300
    chunks = [
        chunk(
            "code_lm_fanout_public_metadata",
            "code_lm_fanout",
            "candidate_fanout",
            "candidate_fanout",
            ["checkpoint", "public_candidates", "public_limit8_smoke"],
            artifact_keys,
            lease_seconds,
            speed_targets,
            public_boundary_clean,
            max_runtime_ms=max(2_000, int(number(speed_targets.get("public_candidate_generation_ms")) * 4 or 2_000)),
        ),
        chunk(
            "code_lm_private_scale_fanout_cache",
            "code_lm_fanout",
            "candidate_fanout",
            "candidate_fanout_private_cached",
            ["checkpoint", "private_scale_smoke", "private_candidates"],
            artifact_keys,
            lease_seconds,
            speed_targets,
            public_boundary_clean,
            max_runtime_ms=max(15_000, int(number(speed_targets.get("private_scale_ms_per_task")) * 512)),
        ),
        chunk(
            "candidate_ranker_prefilter_top_slice",
            "candidate_ranker_prefilter",
            "ranker_prefilter",
            "ranker_prefilter",
            ["private_candidates", "public_candidates", "fanout_report"],
            artifact_keys,
            600,
            speed_targets,
            public_boundary_clean,
            max_runtime_ms=max(1_000, int(number(speed_targets.get("ranker_private_ms")) + number(speed_targets.get("ranker_public_ms")) + 1_000)),
        ),
        chunk(
            "staged_verifier_cascade_cache",
            "staged_verifier",
            "verification_fast_fail",
            "verifier_cascade",
            ["public_candidates", "private_candidates", "same_seed_comparator_smoke"],
            artifact_keys,
            900,
            speed_targets,
            public_boundary_clean,
            max_runtime_ms=5_000,
        ),
        chunk(
            "benchmark_regression_fast_parallel",
            "benchmark_regression",
            "benchmark_regression",
            "benchmark_regression",
            ["public_candidates", "closure_report"],
            artifact_keys,
            900,
            speed_targets,
            public_boundary_clean,
            max_runtime_ms=15_000,
        ),
    ]
    return chunks


def chunk(
    chunk_id: str,
    stage: str,
    parallel_group: str,
    worker_kind: str,
    artifact_kinds: list[str],
    artifact_keys: dict[str, dict[str, Any]],
    lease_seconds: int,
    speed_targets: dict[str, Any],
    public_boundary_clean: bool,
    *,
    max_runtime_ms: int,
) -> dict[str, Any]:
    inputs = [artifact_keys[kind] for kind in artifact_kinds if kind in artifact_keys]
    return {
        "chunk_id": chunk_id,
        "stage": stage,
        "worker_kind": worker_kind,
        "backend": "rust_cuda_or_cpu_control",
        "parallel_group": parallel_group,
        "lease_seconds": lease_seconds,
        "max_runtime_ms": max_runtime_ms,
        "status": "ready" if inputs and public_boundary_clean else "blocked",
        "input_artifacts": inputs,
        "output_artifacts": [
            {
                "path": f"reports/runtime_worker_chunks/{chunk_id}.json",
                "type": "speed_proof_or_stage_result",
            }
        ],
        "cache_namespaces": cache_namespaces_for(stage),
        "phase_timing_required": True,
        "public_tests_or_solutions_allowed": False,
        "speed_targets": speed_targets,
        "score_semantics": "runtime_optimizer_worker_chunk_not_capability_promotion",
    }


def cache_namespaces_for(stage: str) -> list[str]:
    if stage == "code_lm_fanout":
        return ["checkpoint_model_load_v1", "task_manifest_contract_v1", "candidate_manifest_sidecar_v1"]
    if stage == "candidate_ranker_prefilter":
        return ["candidate_structural_rank_features_v1", "candidate_quality_score_v1"]
    if stage == "staged_verifier":
        return [str(row["cache_namespace"]) for row in STAGED_VERIFICATION]
    if stage == "benchmark_regression":
        return ["benchmark_loader_manifest_v1", "regression_result_cache_v1"]
    return []


def lease_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    created = datetime.now(timezone.utc)
    lease_id = cache_key(chunk["chunk_id"], Path(chunk["chunk_id"]))[:20]
    return {
        "policy": "runtime_bottleneck_optimizer_worker_chunk_lease_v1",
        "lease_id": f"lease_{lease_id}",
        "chunk_id": chunk["chunk_id"],
        "stage": chunk["stage"],
        "status": "planned_ready" if chunk["status"] == "ready" else "planned_blocked",
        "created_utc": created.isoformat(),
        "expires_utc": (created + timedelta(seconds=int(chunk["lease_seconds"]))).isoformat(),
        "lease_seconds": chunk["lease_seconds"],
        "bounded": True,
        "execute_by_default": False,
        "no_training": True,
        "no_public_calibration": True,
    }


def readiness_checks(
    chunks: list[dict[str, Any]],
    cached_artifacts: list[dict[str, Any]],
    speed_proof: dict[str, Any],
    public_boundary_clean: bool,
    decoder_gate: dict[str, Any],
    transfer_proof: dict[str, Any],
) -> list[dict[str, Any]]:
    existing_cache_count = sum(1 for row in cached_artifacts if row.get("exists"))
    return [
        check("required_stage_coverage", {chunk["stage"] for chunk in chunks} >= set(REQUIRED_STAGES), sorted(REQUIRED_STAGES)),
        check("cached_artifacts_present", existing_cache_count >= 5, {"existing_cache_count": existing_cache_count}),
        check("bounded_leases_present", all(int(chunk.get("lease_seconds") or 0) <= 900 for chunk in chunks), [chunk["lease_seconds"] for chunk in chunks]),
        check("parallel_groups_present", len({chunk["parallel_group"] for chunk in chunks}) >= 4, [chunk["parallel_group"] for chunk in chunks]),
        check("phase_timing_required", all(chunk.get("phase_timing_required") for chunk in chunks), True),
        check("current_speed_proof_ready", bool(speed_proof.get("ready")), speed_proof.get("score_semantics")),
        check("decoder_gate_ready", bool(decoder_gate.get("ready_for_public_calibration")), decoder_gate.get("trigger_state")),
        check("transfer_proof_ready", bool(transfer_proof.get("ready_for_public_calibration")), transfer_proof.get("trigger_state")),
        check("public_boundary_clean", public_boundary_clean, "no public tests/solutions in training or planner chunks"),
    ]


def public_boundary_ok(
    decoder_gate: dict[str, Any],
    transfer_proof: dict[str, Any],
    train_once: dict[str, Any],
    real_code: dict[str, Any],
) -> bool:
    decoder_summary = decoder_gate.get("summary") if isinstance(decoder_gate.get("summary"), dict) else {}
    real_summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    train_summary = train_once.get("summary") if isinstance(train_once.get("summary"), dict) else {}
    public_safety = get_path(train_summary, ["public_manifest_diagnostics", "safety"], {})
    unsafe_rows = int(number(get_path(public_safety, ["unsafe_public_rows"], 0)))
    public_used = bool(get_path(public_safety, ["public_tests_or_solutions_used"], False))
    integrity_ok = bool(real_summary.get("student_candidate_benchmark_integrity_valid", True))
    return (
        bool(decoder_gate.get("ready_for_public_calibration"))
        and bool(transfer_proof.get("ready_for_public_calibration"))
        and not public_used
        and unsafe_rows == 0
        and integrity_ok
    )


def speed_proof_report(speed_proof: dict[str, Any], fanout_timing: dict[str, Any], real_code: dict[str, Any]) -> dict[str, Any]:
    real_summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    return {
        "policy": "runtime_bottleneck_optimizer_speed_proof_v1",
        "source": "system_efficiency_current_speed_proof_plus_train_once_fanout_timing",
        "current_speed_proof_ready": bool(speed_proof.get("ready")),
        "public_limit8_ms_per_task": speed_proof.get("public_limit8_ms_per_task"),
        "private_scale_private_ms_per_task": speed_proof.get("private_scale_private_ms_per_task"),
        "train_once_current_source_public_generation_ms": speed_proof.get("train_once_current_source_public_generation_ms"),
        "train_once_current_source_private_generation_ms": speed_proof.get("train_once_current_source_private_generation_ms"),
        "canonical_public_generation_ms": fanout_timing.get("public_candidate_generation_and_write"),
        "canonical_private_generation_ms": fanout_timing.get("private_candidate_generation_and_write"),
        "benchmark_parallel_verification_enabled": bool(real_summary.get("parallel_verification_enabled")),
        "benchmark_verification_workers": real_summary.get("verification_workers"),
        "verification_stage_count": real_summary.get("verification_stage_count"),
        "score_semantics": "fresh runtime speed/control proof, not capability promotion evidence",
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Runtime Bottleneck Worker-Chunk Plan",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Plan ready: `{report.get('plan_ready')}`",
        f"- Planned chunks: `{summary.get('planned_chunk_count')}`",
        f"- Parallel groups: `{summary.get('parallel_group_count')}`",
        f"- Resource throttle handled: `{summary.get('resource_throttle_handled')}`",
        f"- Public tests or solutions used: `{summary.get('public_tests_or_solutions_used')}`",
        "",
        "## Chunks",
    ]
    for chunk_row in report.get("worker_chunks", []):
        lines.append(
            f"- `{chunk_row.get('status')}` `{chunk_row.get('chunk_id')}` "
            f"stage=`{chunk_row.get('stage')}` group=`{chunk_row.get('parallel_group')}` "
            f"lease=`{chunk_row.get('lease_seconds')}s` max=`{chunk_row.get('max_runtime_ms')}ms`"
        )
    lines.extend(["", "## Checks"])
    for row in report.get("ready_checks", []):
        lines.append(f"- `{row.get('passed')}` `{row.get('id')}`: `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def check(check_id: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "evidence": evidence}


def cache_key(kind: str, path: Path) -> str:
    payload = f"{kind}:{rel(path)}:{path.stat().st_mtime_ns if path.exists() else 0}:{path.stat().st_size if path.exists() else 0}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return num if math.isfinite(num) else 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
