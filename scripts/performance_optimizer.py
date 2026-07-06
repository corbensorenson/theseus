"""Performance optimizer report for SymLiquid and Theseus Hive.

This is deliberately report-first. It does not mutate profiles by default; it
turns CUDA/MLX/resource/Hive evidence into concrete next actions so the
autonomy loop can prefer the fastest safe backend and avoid blind long runs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "performance_policy.json"
DEFAULT_OUT = ROOT / "reports" / "performance_optimizer.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "performance_optimizer.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    reports = collect_reports()
    report = build_report(policy, reports)
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, markdown_report(report))
    print(json.dumps(report, indent=2))
    return 0


def collect_reports() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "resource_governor": read_json(reports / "resource_governor.json", {}),
        "hive_status": read_json(reports / "hive_status.json", {}),
        "hive_scheduler": read_json(reports / "hive_scheduler.json", {}),
        "preflight": read_json(reports / "training_preflight_report.json", {}),
        "cuda_standalone": read_json(reports / "preflight_cuda_standalone_smoke.json", {}),
        "cuda_rollout": read_json(reports / "preflight_cuda_rollout_smoke.json", {}),
        "profile_stress": read_json(reports / "profile_vram_stress.json", {}),
        "worker_chunks": read_jsonl_tail(reports / "hive_worker_chunk_ledger.jsonl", 80),
        "runtime_worker_chunk_plan": read_json(reports / "runtime_bottleneck_optimizer_worker_chunk_plan.json", {}),
        "training_profile": read_json(reports / "training_ratchet_profile_run.json", {}),
        "launch_readiness": read_json(reports / "autonomy_launch_readiness.json", {}),
    }


def build_report(policy: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    resource = reports["resource_governor"]
    hive = reports["hive_status"]
    scheduler = reports["hive_scheduler"]
    chunks = [row for row in reports["worker_chunks"] if isinstance(row, dict)]
    cuda_chunks = [row for row in chunks if str(row.get("backend")) == "rust_cuda"]
    mlx_chunks = [row for row in chunks if str(row.get("backend")) in {"apple_mlx", "mlx_apple", "mlx_cuda"}]
    recent_ok_chunks = [row for row in chunks if row.get("ok")]

    gpu = get_path(resource, ["current_resources", "gpu"], {})
    mlx = get_path(hive, ["resources", "mlx"], {})
    cap_ids = {
        cap.get("id")
        for cap in get_path(hive, ["capabilities"], [])
        if isinstance(cap, dict)
    }
    scheduler_summary = scheduler.get("summary") if isinstance(scheduler.get("summary"), dict) else {}
    freshness = control_plane_freshness(reports)
    runtime_plan = reports["runtime_worker_chunk_plan"] if isinstance(reports.get("runtime_worker_chunk_plan"), dict) else {}
    cuda_metrics = best_metrics([reports["cuda_standalone"], reports["cuda_rollout"], *cuda_chunks])
    mlx_metrics = best_metrics(mlx_chunks)
    bottlenecks = bottleneck_list(policy, reports, cuda_chunks, mlx_chunks, freshness)
    recommendations = recommendation_list(policy, reports, bottlenecks)
    score = performance_score(policy, reports, cuda_chunks, mlx_chunks, bottlenecks)
    trigger_state = "GREEN" if score >= 0.78 and not high_severity(bottlenecks) else ("YELLOW" if score >= 0.48 else "RED")

    return {
        "policy": "project_theseus_performance_optimizer_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "score": round(score, 4),
        "summary": {
            "preferred_training_backend": choose_training_backend(policy, gpu, mlx, cap_ids),
            "preferred_inference_backend": choose_inference_backend(policy, gpu, mlx, cap_ids),
            "cuda_available": bool(gpu.get("available")),
            "mlx_available": bool(mlx.get("available")),
            "scheduler_worker_chunks": int(number(scheduler_summary.get("real_worker_chunks", 0))),
            "recent_worker_chunks": len(chunks),
            "recent_ok_worker_chunks": len(recent_ok_chunks),
            "recent_cuda_chunks": len(cuda_chunks),
            "recent_mlx_chunks": len(mlx_chunks),
            "resource_can_run_profile": get_path(resource, ["decision", "can_run_requested_profile"], None),
            "resource_recommended_profile": get_path(resource, ["decision", "recommended_profile"], None),
            "gpu_free_mib": number(gpu.get("memory_free_mib", 0)),
            "spillover_free_gib": number(get_path(resource, ["current_resources", "spillover", "selected", "free_gib"], 0)),
            "runtime_worker_chunk_plan_ready": runtime_worker_plan_ready(runtime_plan),
            "runtime_worker_chunk_plan_chunks": int(number(get_path(runtime_plan, ["summary", "planned_chunk_count"], 0))),
            "runtime_worker_chunk_plan_stages": int(number(get_path(runtime_plan, ["summary", "planned_stage_count"], 0))),
            "runtime_worker_chunk_plan_resource_throttle_handled": bool(
                get_path(runtime_plan, ["summary", "resource_throttle_handled"], False)
            ),
        },
        "control_plane_freshness": freshness,
        "cuda": {
            "available": bool(gpu.get("available")),
            "gpu": gpu,
            "metrics": cuda_metrics,
            "targets": get_path(policy, ["cuda", "throughput_targets"], {}),
            "rules": get_path(policy, ["cuda", "optimization_rules"], []),
        },
        "mlx": {
            "available": bool(mlx.get("available")),
            "status": mlx,
            "metrics": mlx_metrics,
            "feature_cache": get_path(policy, ["mlx", "feature_cache"], {}),
            "rules": get_path(policy, ["mlx", "optimization_rules"], []),
        },
        "hive": {
            "node_name": hive.get("node_name"),
            "capabilities": sorted(str(cap) for cap in cap_ids if cap),
            "scheduler_summary": scheduler_summary,
            "worker_chunk_backends": summarize_chunk_backends(chunks),
            "runtime_bottleneck_worker_plan": compact_runtime_worker_plan(runtime_plan),
        },
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "efficiency_law": get_path(policy, ["autonomy", "speed_principle"], ""),
        "external_inference_calls": 0,
    }


def choose_training_backend(policy: dict[str, Any], gpu: dict[str, Any], mlx: dict[str, Any], cap_ids: set[Any]) -> str:
    for backend in get_path(policy, ["backend_priority", "training"], ["rust_cuda", "apple_mlx", "rust_cpu"]):
        if backend == "rust_cuda" and (gpu.get("available") or "nvidia_cuda" in cap_ids):
            return backend
        if backend == "apple_mlx" and (mlx.get("available") or {"apple_mlx", "mlx_apple", "mlx_cuda"} & cap_ids):
            return backend
        if backend == "rust_cpu":
            return backend
    return "rust_cpu"


def choose_inference_backend(policy: dict[str, Any], gpu: dict[str, Any], mlx: dict[str, Any], cap_ids: set[Any]) -> str:
    for backend in get_path(policy, ["backend_priority", "inference"], ["apple_mlx", "rust_cuda", "rust_cpu"]):
        if backend == "apple_mlx" and (mlx.get("available") or {"apple_mlx", "mlx_apple", "mlx_cuda"} & cap_ids):
            return backend
        if backend == "rust_cuda" and (gpu.get("available") or "nvidia_cuda" in cap_ids):
            return backend
        if backend == "rust_cpu":
            return backend
        if backend == "best_authorized_hive_peer":
            return backend
    return "rust_cpu"


def best_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "train_examples_per_second",
        "eval_examples_per_second",
        "examples_per_second",
        "train_runtime_ms",
        "eval_runtime_ms",
        "runtime_ms",
        "runtime_ms_child",
        "kernel_launches",
        "cuda_fallback",
        "train_accuracy",
        "eval_accuracy",
        "accuracy",
        "feature_ms",
        "mlx_transfer_ms",
        "mlx_train_ms",
        "mlx_eval_ms",
        "cache_hits",
    ]
    candidates = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        merged = {**metrics}
        for key in metric_keys:
            if key in report:
                merged[key] = report[key]
        child = report.get("child_report") if isinstance(report.get("child_report"), dict) else {}
        child_metrics = child.get("metrics") if isinstance(child.get("metrics"), dict) else {}
        merged.update(child_metrics)
        for key in metric_keys:
            if key in child:
                merged[key] = child[key]
        eps = number(merged.get("train_examples_per_second", merged.get("examples_per_second", 0)))
        if eps or merged:
            candidates.append((eps, merged))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def summarize_chunk_backends(chunks: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in chunks:
        backend = str(row.get("backend") or "unknown")
        out[backend] = out.get(backend, 0) + 1
    return out


def bottleneck_list(
    policy: dict[str, Any],
    reports: dict[str, Any],
    cuda_chunks: list[dict[str, Any]],
    mlx_chunks: list[dict[str, Any]],
    freshness: dict[str, Any],
) -> list[dict[str, Any]]:
    resource = reports["resource_governor"]
    scheduler = reports["hive_scheduler"]
    runtime_plan = reports["runtime_worker_chunk_plan"] if isinstance(reports.get("runtime_worker_chunk_plan"), dict) else {}
    runtime_plan_ready = runtime_worker_plan_ready(runtime_plan)
    gpu = get_path(resource, ["current_resources", "gpu"], {})
    mlx = get_path(reports["hive_status"], ["resources", "mlx"], {})
    bottlenecks: list[dict[str, Any]] = []
    if not gpu.get("available") and not mlx.get("available"):
        bottlenecks.append(issue("RED", "no_accelerator", "No CUDA or MLX accelerator is currently visible."))
    if get_path(resource, ["decision", "can_run_requested_profile"], True) is False and not runtime_worker_plan_handles_throttle(runtime_plan):
        bottlenecks.append(issue("YELLOW", "resource_governor_throttled", "; ".join(get_path(resource, ["decision", "throttle_reasons"], []))))
    if gpu.get("available") and not cuda_chunks:
        bottlenecks.append(issue("YELLOW", "no_recent_cuda_worker_chunks", "Scheduler has not executed a recent bounded CUDA worker chunk."))
    gpu_util = number(gpu.get("utilization_gpu_percent", 0))
    active_jobs = int(number(get_path(resource, ["current_resources", "active_jobs", "training_job_count"], 0)))
    if gpu.get("available") and active_jobs > 0 and gpu_util < 15.0:
        bottlenecks.append(
            issue(
                "YELLOW",
                "accelerator_idle_during_training",
                f"training_jobs={active_jobs} but gpu_utilization={gpu_util:.1f}%; move the active lane into Rust/CUDA kernels or defer it behind CUDA worker chunks.",
            )
        )
    if mlx.get("available") and not mlx_chunks:
        bottlenecks.append(issue("YELLOW", "no_recent_mlx_worker_chunks", "MLX is visible but no recent MLX worker chunk has run."))
    if (
        bool(freshness.get("hive_scheduler_after_resource_governor", True))
        and int(number(get_path(scheduler, ["summary", "real_worker_chunks"], 0))) < 3
        and gpu.get("available")
        and not runtime_plan_ready
    ):
        bottlenecks.append(
            issue(
                "YELLOW",
                "incomplete_worker_chunk_plan",
                "Hive scheduler or runtime bottleneck planner should plan fanout, ranker, verifier, benchmark/regression, and CUDA proof chunks.",
            )
        )
    targets = get_path(policy, ["cuda", "throughput_targets"], {})
    rollout = reports["cuda_rollout"]
    rollout_eps = number(rollout.get("train_examples_per_second", get_path(rollout, ["metrics", "examples_per_second"], 0)))
    if rollout and rollout_eps < float(targets.get("rollout_examples_per_second_min", 10.0)):
        bottlenecks.append(issue("RED", "cuda_rollout_below_target", f"rollout eps={rollout_eps:.3f}"))
    return bottlenecks


def recommendation_list(policy: dict[str, Any], reports: dict[str, Any], bottlenecks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = [
        {
            "priority": "high",
            "action": "Keep Rust/CUDA as the Windows NVIDIA hot path; Python should schedule, audit, and report.",
            "owner": "rust_cuda_systems_arm",
        },
        {
            "priority": "high",
            "action": "Run CUDA eval, CUDA readout training, and CUDA rollout chunks as separate bounded worker proofs before longer profiles.",
            "owner": "hive_scheduler",
        },
        {
            "priority": "medium",
            "action": "On Mac, run mlx_eval_chunk, mlx_training_chunk, and mlx_rollout_chunk; verify MLX timing and feature/cache telemetry.",
            "owner": "apple_mlx_worker",
        },
        {
            "priority": "medium",
            "action": "Treat public benchmarks as scoring pressure and keep repeated environment/task setup compiled into registered tools.",
            "owner": "loop_closure_tool_arm",
        },
    ]
    for bottleneck in bottlenecks:
        if bottleneck.get("id") == "cuda_rollout_below_target":
            recommendations.insert(
                0,
                {
                    "priority": "critical",
                    "action": "Stop long training until CUDA rollout throughput recovers above target.",
                    "owner": "resource_governor",
                },
            )
        if bottleneck.get("id") == "accelerator_idle_during_training":
            recommendations.insert(
                0,
                {
                    "priority": "critical",
                    "action": "Do not spend long wall-clock on CPU-bound training while the NVIDIA GPU is idle; route the next eligible work to CUDA worker chunks or port the hot loop.",
                    "owner": "rust_cuda_systems_arm",
                },
            )
    return recommendations


def performance_score(
    policy: dict[str, Any],
    reports: dict[str, Any],
    cuda_chunks: list[dict[str, Any]],
    mlx_chunks: list[dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
) -> float:
    resource = reports["resource_governor"]
    hive = reports["hive_status"]
    scheduler = reports["hive_scheduler"]
    gpu = get_path(resource, ["current_resources", "gpu"], {})
    mlx = get_path(hive, ["resources", "mlx"], {})
    score = 0.15
    score += 0.2 if gpu.get("available") else 0.0
    score += 0.15 if mlx.get("available") else 0.0
    score += 0.15 if get_path(resource, ["decision", "can_run_requested_profile"], False) else 0.0
    runtime_plan = reports.get("runtime_worker_chunk_plan") if isinstance(reports.get("runtime_worker_chunk_plan"), dict) else {}
    planned_chunks = int(number(get_path(runtime_plan, ["summary", "planned_chunk_count"], 0))) if runtime_worker_plan_ready(runtime_plan) else 0
    scheduler_or_runtime_chunks = max(int(number(get_path(scheduler, ["summary", "real_worker_chunks"], 0))), min(planned_chunks, 5))
    score += min(scheduler_or_runtime_chunks / 3.0, 1.0) * 0.15
    score += min(len([row for row in cuda_chunks + mlx_chunks if row.get("ok")]) / 3.0, 1.0) * 0.15
    score += 0.1 if not any(row.get("severity") == "RED" for row in bottlenecks) else -0.15
    score += min(number(get_path(resource, ["efficiency", "score"], 0.0)), 1.0) * 0.1
    return max(0.0, min(score, 1.0))


def runtime_worker_plan_ready(plan: dict[str, Any]) -> bool:
    if not isinstance(plan, dict) or not plan:
        return False
    if plan.get("policy") != "runtime_bottleneck_optimizer_worker_chunk_plan_v1":
        return False
    coverage = plan.get("required_stage_coverage") if isinstance(plan.get("required_stage_coverage"), dict) else {}
    required = {
        "code_lm_fanout",
        "candidate_ranker_prefilter",
        "staged_verifier",
        "benchmark_regression",
    }
    checks = plan.get("ready_checks") if isinstance(plan.get("ready_checks"), list) else []
    return bool(plan.get("plan_ready")) and required.issubset({key for key, value in coverage.items() if value}) and all(
        bool(row.get("passed")) for row in checks if isinstance(row, dict)
    )


def runtime_worker_plan_handles_throttle(plan: dict[str, Any]) -> bool:
    return runtime_worker_plan_ready(plan) and bool(get_path(plan, ["summary", "resource_throttle_handled"], False))


def compact_runtime_worker_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or not plan:
        return {"present": False, "ready": False}
    summary = plan.get("summary") if isinstance(plan.get("summary"), dict) else {}
    return {
        "present": True,
        "ready": runtime_worker_plan_ready(plan),
        "created_utc": plan.get("created_utc"),
        "planned_chunk_count": summary.get("planned_chunk_count"),
        "planned_stage_count": summary.get("planned_stage_count"),
        "parallel_group_count": summary.get("parallel_group_count"),
        "resource_throttle_handled": summary.get("resource_throttle_handled"),
        "public_tests_or_solutions_used": summary.get("public_tests_or_solutions_used"),
    }


def high_severity(bottlenecks: list[dict[str, Any]]) -> bool:
    return any(row.get("severity") == "RED" for row in bottlenecks)


def issue(severity: str, issue_id: str, detail: str) -> dict[str, Any]:
    return {"severity": severity, "id": issue_id, "detail": detail}


def control_plane_freshness(reports: dict[str, Any]) -> dict[str, Any]:
    resource_time = parse_utc(get_path(reports["resource_governor"], ["created_utc"], ""))
    scheduler_time = parse_utc(get_path(reports["hive_scheduler"], ["created_utc"], ""))
    scheduler_after_resource = (
        resource_time is None
        or scheduler_time is None
        or scheduler_time >= resource_time
    )
    return {
        "resource_governor_created_utc": get_path(reports["resource_governor"], ["created_utc"], ""),
        "hive_scheduler_created_utc": get_path(reports["hive_scheduler"], ["created_utc"], ""),
        "hive_scheduler_after_resource_governor": scheduler_after_resource,
        "interpretation": (
            "Worker-plan bottlenecks are only promotion-relevant when the scheduler report was built after "
            "the current resource-governor decision."
        ),
    }


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Project Theseus Performance Optimizer",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Score: `{report.get('score')}`",
        f"- Preferred training backend: `{summary.get('preferred_training_backend')}`",
        f"- Preferred inference backend: `{summary.get('preferred_inference_backend')}`",
        f"- Worker chunks planned/recent: `{summary.get('scheduler_worker_chunks')}` / `{summary.get('recent_worker_chunks')}`",
        f"- GPU free MiB: `{summary.get('gpu_free_mib')}`",
        f"- Spillover free GiB: `{summary.get('spillover_free_gib')}`",
        "",
        "## Bottlenecks",
    ]
    bottlenecks = report.get("bottlenecks") if isinstance(report.get("bottlenecks"), list) else []
    if bottlenecks:
        for row in bottlenecks:
            lines.append(f"- `{row.get('severity')}` `{row.get('id')}`: {row.get('detail')}")
    else:
        lines.append("- None blocking.")
    lines.extend(["", "## Recommendations"])
    for row in report.get("recommendations", []):
        if isinstance(row, dict):
            lines.append(f"- `{row.get('priority')}` {row.get('action')}")
    return "\n".join(lines) + "\n"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[Any]:
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    out = []
    for row in rows:
        try:
            out.append(json.loads(row))
        except json.JSONDecodeError:
            continue
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any) -> Any:
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
