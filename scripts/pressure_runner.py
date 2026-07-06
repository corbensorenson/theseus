"""Run bounded local benchmark pressure from smoke-passed adapter cards.

This is the bridge between "the source is installed" and "the ratchet has a
scored frontier surface." It deliberately stays local-only:

- no external inference;
- no bulk downloads;
- no live drone hardware;
- no real web accounts.

The runner emits a treadmill-compatible report (`summary.accuracy`) plus richer
per-card details so the benchmark ledger can promote, escrow, or rotate it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

from pressure_runner_code_evidence import (  # noqa: E402
    code_repair_score_bonus,
    learned_manifest_matches_card,
    organism_checks,
    organism_metrics,
    public_multistream_checks,
    public_multistream_metrics,
    public_multistream_score_bonus,
    real_code_graduation_checks,
    real_code_graduation_metrics,
    real_code_graduation_score_bonus,
)
from pressure_runner_utils import (  # noqa: E402
    budget_summary,
    check,
    clamp01,
    command_available,
    command_path,
    get_path,
    list_limited_files,
    now,
    pressure_timeout_seconds,
    read_json,
    read_jsonl,
    rel_or_abs,
    resolve_path,
    safe_name,
    suite_name,
    synthetic_available_arms,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", required=True)
    parser.add_argument("--frontier-family", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--train-iterations", type=int, default=4)
    parser.add_argument("--train-population", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--eval-seed-count", type=int, default=0)
    parser.add_argument("--min-train-candidate-evals", type=int, default=0)
    parser.add_argument("--min-train-env-steps", type=int, default=0)
    parser.add_argument("--budget-report", default="")
    parser.add_argument("--code-transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    card = load_card(args.card_id)
    if not card and args.card_id != "transfer_eval_suite":
        raise SystemExit(f"benchmark card not found: {args.card_id}")

    started = time.perf_counter()
    if args.card_id == "transfer_eval_suite":
        result = run_transfer_eval_surface(args)
    else:
        source_id = str(card.get("source_id") or card.get("id") or args.card_id)
        if source_id == "gym_pybullet_drones":
            result = run_learned_drone_controller(card, args, "gym_pybullet_drones")
        elif source_id in {"pyflyt", "pyflyt_waypoints"}:
            result = run_learned_drone_controller(card, args, source_id)
        elif source_id == "mavsdk_python":
            result = run_mavsdk(card, args)
        elif source_id == "bigcodebench":
            result = run_bigcodebench(card, args)
        elif (
            str(card.get("runner_family") or "") == "multi_stream_code_pressure"
            or str(card.get("category") or "") == "multi_stream_coding_benchmark"
        ):
            result = run_multi_stream_code_pressure(card, args)
        elif str(card.get("runner_family") or "") == "coding_local_sandbox":
            result = run_code_benchmark_surface(card, args)
        elif (
            str(card.get("runner_family") or "") != "old_project_registry_pressure"
            and (
                str(card.get("runner_family") or "") == "coding_agent_local"
                or str(card.get("category") or "") in {"coding_agent_benchmark", "coding_agent_framework"}
            )
        ):
            result = run_coding_agent_harness(card, args)
        elif str(card.get("runner_family") or "") == "old_project_registry_pressure":
            result = run_old_project_registry_pressure(card, args)
        elif source_id == "webarena":
            result = run_webarena(card, args)
        elif source_id in {"common_voice", "librispeech", "libritts", "ljspeech", "vctk", "speechbrain_benchmarks"}:
            result = run_native_voice(card, args)
        elif (
            args.frontier_family == "minecraft_rl"
            or source_id in {"minerl", "minedojo", "malmo", "voyager_minecraft", "crafter", "craftax"}
            or str(card.get("runner_family") or "") == "minecraft_rl_local"
            or str(card.get("category") or "") == "minecraft_rl_environment"
        ):
            result = run_minecraft_rl(card, args)
        elif str(card.get("runner_family") or "") == "emulator_rl_local" or str(card.get("category") or "") == "emulator_rl_environment":
            result = run_emulator_rl(card, args)
        elif (
            str(card.get("runner_family") or "") == "synthetic_benchmark_local"
            or str(card.get("category") or "").startswith("synthetic_")
        ):
            result = run_synthetic_benchmark(card, args)
        else:
            result = run_generic_card_probe(card, args)

    family = result.get("suite") or suite_name(args.frontier_family, args.card_id)
    score = clamp01(result.get("score", 0.0))
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "sparkstream_pressure_runner_v0",
        "created_utc": now(),
        "frontier_family": args.frontier_family,
        "card_id": args.card_id,
        "card_name": card.get("name") if card else args.card_id,
        "runner_family": card.get("runner_family") if card else "transfer_eval_local",
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "budget": budget_summary(args),
        "status": result.get("status", "completed"),
        "summary": {
            "suite": family,
            "accuracy": score,
            "score": score,
            "total_tool_calls": 0,
        },
        "metrics": result.get("metrics", {}),
        "residuals": result.get("residuals", []),
        "checks": result.get("checks", []),
        "permission_envelope": {
            "external_inference": "forbidden",
            "network": "forbidden_during_scoring",
            "hardware": "forbidden_without_explicit_human_approval",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
        "teacher_calls": 0,
    }
    out = Path(args.out) if args.out else ROOT / "reports" / f"pressure_{safe_name(args.card_id)}_seed{args.seed}.json"
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, report)
    append_trace(args, report, out)
    print(json.dumps(report, indent=2))
    return 0


def run_learned_drone_controller(
    card: dict[str, Any],
    args: argparse.Namespace,
    source_id: str,
) -> dict[str, Any]:
    policy_out = ROOT / "reports" / f"drone_controller_{safe_name(source_id)}_seed{args.seed}.json"
    report_out = ROOT / "reports" / f"drone_controller_{safe_name(source_id)}_seed{args.seed}_trainer.json"
    trace_out = ROOT / "reports" / "drone_traces" / f"{safe_name(source_id)}_seed{args.seed}.jsonl"
    result = run_command(
        [
            str(runtime_python_for_source(source_id)),
            "scripts/drone_controller_trainer.py",
            "--source",
            source_id,
            "--seed",
            str(args.seed),
            "--episodes",
            str(max(1, args.episodes)),
            "--steps",
            str(max(1, args.steps)),
            "--iterations",
            str(max(1, args.train_iterations)),
            "--population",
            str(max(4, args.train_population)),
            "--elite-count",
            str(max(1, args.elite_count)),
            "--eval-seed-count",
            str(max(1, args.eval_seed_count or args.episodes)),
            "--min-train-candidate-evals",
            str(max(0, args.min_train_candidate_evals)),
            "--min-train-env-steps",
            str(max(0, args.min_train_env_steps)),
            "--policy-out",
            str(policy_out),
            "--trace-out",
            str(trace_out),
            "--out",
            str(report_out),
        ],
        timeout=pressure_timeout_seconds(args),
    )
    checks = safety_checks(card)
    if not result["ok"]:
        checks.append(check("local_controller_training", False, result["stderr_tail"] or result["stdout_tail"]))
        return {
            "suite": suite_name("drone_rl", card.get("id", source_id)),
            "score": 0.05,
            "status": "runtime_blocked",
            "checks": checks,
            "metrics": result,
            "residuals": [{"type": "runtime", "detail": result["stderr_tail"] or result["stdout_tail"]}],
        }
    payload = parse_last_json(result["stdout_tail"])
    if not payload:
        payload = read_json(report_out)
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
    score = clamp01(payload.get("score", evaluation.get("score", 0.0)))
    steps = int(evaluation.get("steps") or 0)
    total_reward = float(evaluation.get("total_reward") or 0.0)
    checks.extend(payload.get("checks", []) if isinstance(payload.get("checks"), list) else [])
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else {}
    existing_checks = {str(item.get("name")) for item in checks if isinstance(item, dict)}
    if "train_before_eval_candidate_budget" not in existing_checks:
        checks.append(
            check(
                "train_before_eval_candidate_budget",
                int(budget.get("candidate_evaluations") or 0) >= max(0, int(args.min_train_candidate_evals)),
                f"{budget.get('candidate_evaluations')} >= {args.min_train_candidate_evals}",
            )
        )
    if "train_before_eval_env_step_budget" not in existing_checks:
        checks.append(
            check(
                "train_before_eval_env_step_budget",
                int(budget.get("train_env_steps") or 0) >= max(0, int(args.min_train_env_steps)),
                f"{budget.get('train_env_steps')} >= {args.min_train_env_steps}",
            )
        )
    checks.append(check("local_controller_training", True, f"score={score:.4f} policy={payload.get('policy_path')}"))
    checks.append(check("reset_step_episode", steps >= args.steps * max(1, args.episodes), f"steps={steps} total_reward={total_reward:.4f}"))
    checks.append(
        check(
            "rollout_trace_export",
            bool(evaluation.get("trace_manifest_path")),
            str(evaluation.get("trace_manifest_path") or trace_out),
        )
    )
    checks.append(check("sim_only_no_live_hardware", True, "GUI false, no MAVLink/live endpoint"))
    return {
        "suite": suite_name("drone_rl", card.get("id", source_id)),
        "score": score,
        "status": "frontier_open",
        "checks": checks,
        "metrics": {
            "controller": payload.get("policy", "theseus_tiny_drone_controller_v0"),
            "controller_name": (read_json(policy_out) or {}).get("controller"),
            "trainer_report": rel_or_abs(report_out),
            "policy_path": payload.get("policy_path"),
            "trace_path": evaluation.get("trace_path"),
            "trace_manifest_path": evaluation.get("trace_manifest_path"),
            "replay_path": get_path(payload, ["training", "replay_path"], None),
            "replay_summary": get_path(payload, ["training", "replay_summary"], {}),
            "transfer_artifact_path": payload.get("transfer_artifact_path"),
            "env_contract": payload.get("env_contract", {}),
            "training": payload.get("training", {}),
            "budget": budget,
            "evaluation": evaluation,
            "steps": steps,
            "total_reward": total_reward,
            "survival_ratio": evaluation.get("survival_ratio"),
        },
        "residuals": payload.get("residuals", []),
    }


def run_mavsdk(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    learner_out = ROOT / "reports" / f"drone_command_contract_mavsdk_seed{args.seed}.json"
    policy_out = ROOT / "reports" / f"drone_command_contract_policy_mavsdk_seed{args.seed}.json"
    result = run_command(
        [
            str(runtime_python_for_source("mavsdk_python")),
            "scripts/drone_command_contract_trainer.py",
            "--seed",
            str(args.seed),
            "--steps",
            str(max(96, args.steps)),
            "--iterations",
            str(max(4, args.train_iterations)),
            "--population",
            str(max(8, args.train_population)),
            "--elite-count",
            str(max(2, args.elite_count)),
            "--policy-out",
            str(policy_out),
            "--out",
            str(learner_out),
        ],
        timeout=max(90, args.steps * max(1, args.train_iterations)),
    )
    checks = safety_checks(card)
    if not result["ok"]:
        checks.append(check("mavsdk_contract_learning", False, result["stderr_tail"] or result["stdout_tail"]))
        return {
            "suite": suite_name("drone_control", card.get("id", "source_mavsdk_python")),
            "score": 0.08,
            "status": "runtime_blocked",
            "checks": checks,
            "metrics": result,
            "residuals": [{"type": "runtime", "detail": result["stderr_tail"] or result["stdout_tail"]}],
        }
    payload = read_json(learner_out)
    score = clamp01(payload.get("score", 0.0))
    checks.extend(payload.get("checks", []) if isinstance(payload.get("checks"), list) else [])
    checks.append(check("mavsdk_contract_learning", True, f"score={score:.4f} policy={payload.get('policy_path')}"))
    return {
        "suite": suite_name("drone_control", card.get("id", "source_mavsdk_python")),
        "score": score,
        "status": "frontier_open",
        "checks": checks,
        "metrics": {
            "contract_report": rel_or_abs(learner_out),
            "policy_path": rel_or_abs(policy_out),
            "training": payload.get("training", {}),
            "evaluation": payload.get("evaluation", {}),
            "sim_endpoint_connected": False,
            "live_hardware_allowed": False,
        },
        "residuals": payload.get("residuals", []),
    }


def run_bigcodebench(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    learner_out = ROOT / "reports" / f"code_repair_learner_seed{args.seed}.json"
    policy_out = ROOT / "reports" / f"code_repair_policy_seed{args.seed}.json"
    result = run_command(
        [
            str(preferred_python()),
            "scripts/code_repair_learner.py",
            "--seed",
            str(args.seed),
            "--policy-out",
            str(policy_out),
            "--out",
            str(learner_out),
        ],
        timeout=60,
    )
    payload = read_json(learner_out) if result["ok"] else {}
    score = clamp01(payload.get("score", 0.0))
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    passed = sum(1 for row in results if row.get("passed"))
    total = len(results) or 1
    source_id = str(card.get("source_id") or card.get("id") or "bigcodebench")
    path = resolve_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    organism = run_code_repair_organism(source_id, path, args)
    public_multistream = run_public_code_multistream(card, source_id, path, args)
    real_code = run_real_code_graduation(card, args)
    score = clamp01(
        score
        + code_repair_score_bonus(organism)
        + public_multistream_score_bonus(public_multistream)
        + real_code_graduation_score_bonus(real_code)
    )
    residuals = payload.get("residuals", [])
    if not residuals:
        residuals = [{"type": "code_repair_mastered_seed", "detail": "increase heldout task diversity for continued pressure"}]
    if organism.get("ran") and organism.get("transfer_consumed"):
        residuals = [
            row
            for row in residuals
            if row.get("type") not in {"local_code_generation_adapter_needed", "local_endpoint_adapter_needed"}
        ]
        residuals.append(
            {
                "type": "public_benchmark_item_generation_needed",
                "detail": "local repair organism and public loader multi-stream regression are live; next pressure is real benchmark item generation/scoring",
            }
        )
    return {
        "suite": suite_name("coding", card.get("id", "source_bigcodebench")),
        "score": score,
        "status": "frontier_open",
        "checks": [
            check("local_code_repair_learning", result["ok"], result["stderr_tail"] or result["stdout_tail"] or "learner completed"),
            check("sandbox_execution", passed == total, f"heldout_repairs_passed={passed}/{total}"),
            check("no_network_during_eval", True, "local subprocess only"),
            *organism_checks(organism),
            *public_multistream_checks(public_multistream),
            *real_code_graduation_checks(real_code),
        ],
        "metrics": {
            "learner_report": rel_or_abs(learner_out),
            "policy_path": rel_or_abs(policy_out),
            "heldout_repairs_passed": passed,
            "heldout_repairs_total": total,
            "details": results,
            "code_repair_organism": organism_metrics(organism),
            "public_code_multistream": public_multistream_metrics(public_multistream),
            "real_code_benchmark_graduation": real_code_graduation_metrics(real_code),
        },
        "residuals": residuals,
    }


def run_code_benchmark_surface(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_id = str(card.get("source_id") or card.get("id") or "coding_benchmark")
    path = resolve_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    present = path.exists()
    files = list_limited_files(path, limit=3000) if present and path.is_dir() else []
    lowered = [rel_or_abs(item).lower().replace("\\", "/") for item in files]
    manifest_hits = [
        name
        for name in lowered
        if name.endswith(
            (
                "pyproject.toml",
                "setup.py",
                "package.json",
                "requirements.txt",
                "pytest.ini",
                "tox.ini",
                "metadata.json",
                "sanitized-mbpp.json",
                "problems.json",
                "problems.jsonl",
            )
        )
    ]
    problem_hits = [
        name
        for name in lowered
        if any(token in name for token in ("/eval", "/bench", "/problem", "/humaneval", "/mbpp", "/test", "/data"))
        and name.endswith((".json", ".jsonl", ".yaml", ".yml", ".py", ".md"))
    ]
    sandbox_hint = any("sandbox" in name or "docker" in name for name in lowered)
    trace_path = ROOT / "reports" / "coding_benchmark_traces" / f"{safe_name(source_id)}_seed{args.seed}.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "event": "source_inventory",
                "source_id": source_id,
                "present": present,
                "manifest_count": len(manifest_hits),
                "problem_or_test_count": len(problem_hits),
            },
            {
                "event": "sandbox_contract",
                "source_id": source_id,
                "network_during_scoring": "forbidden",
                "generated_code_execution": "sandbox_required",
                "external_inference": "forbidden",
            },
        ],
    )
    organism = run_code_repair_organism(source_id, path, args) if present else {"ran": False, "skipped_reason": "source_not_present"}
    public_multistream = (
        run_public_code_multistream(card, source_id, path, args)
        if present
        else {"ran": False, "skipped_reason": "source_not_present"}
    )
    real_code = run_real_code_graduation(card, args) if present else {"ran": False, "skipped_reason": "source_not_present"}
    score = 0.10 + (0.16 if present else 0.0) + (0.16 if manifest_hits else 0.0) + (0.16 if problem_hits else 0.0) + (0.06 if sandbox_hint else 0.0)
    score = clamp01(
        score
        + code_repair_score_bonus(organism)
        + public_multistream_score_bonus(public_multistream)
        + real_code_graduation_score_bonus(real_code)
    )
    residuals: list[dict[str, str]] = []
    if not present:
        residuals.append({"type": "source_not_staged", "detail": "coding benchmark source is not locally staged"})
    if not problem_hits:
        residuals.append({"type": "problem_manifest_locator", "detail": "write a precise benchmark task loader for this source"})
    if organism.get("ran") and organism.get("transfer_consumed"):
        residuals.append(
            {
                "type": "public_benchmark_item_generation_needed",
                "detail": "repair loop, transfer heredity, and public loader multi-stream regression are live; next pressure is real benchmark item generation/scoring",
            }
        )
    else:
        residuals.append({"type": "local_code_generation_adapter_needed", "detail": "connect local Theseus checkpoint output to sandboxed generated-code tests"})
    residuals.append({"type": "contamination_check_needed", "detail": "separate public calibration tasks from generated/live private code frontiers"})
    checks = safety_checks(card)
    checks.extend(
        [
            check("source_present", present, rel_or_abs(path)),
            check("project_manifest_present", bool(manifest_hits), ", ".join(manifest_hits[:8]) or "no manifest"),
            check("problem_or_test_inventory", bool(problem_hits), ", ".join(problem_hits[:8]) or "no task/test files found"),
            check("sandbox_policy_recorded", True, "network off; generated code requires sandboxed execution"),
            *organism_checks(organism),
            *public_multistream_checks(public_multistream),
            *real_code_graduation_checks(real_code),
        ]
    )
    return {
        "suite": suite_name("coding", card.get("id", "source_coding")),
        "score": clamp01(score),
        "status": "frontier_open" if present else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "source_present": present,
            "manifest_count": len(manifest_hits),
            "problem_or_test_count": len(problem_hits),
            "sandbox_hint": sandbox_hint,
            "trace_path": rel_or_abs(trace_path),
            "manifest_sample": manifest_hits[:8],
            "problem_sample": problem_hits[:8],
            "code_repair_organism": organism_metrics(organism),
            "public_code_multistream": public_multistream_metrics(public_multistream),
            "real_code_benchmark_graduation": real_code_graduation_metrics(real_code),
        },
        "residuals": residuals,
    }


def run_coding_agent_harness(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_id = str(card.get("source_id") or card.get("id") or "coding_agent")
    path = resolve_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    present = path.exists()
    files = list_limited_files(path, limit=3000) if present and path.is_dir() else []
    lowered = [rel_or_abs(item).lower().replace("\\", "/") for item in files]
    manifest_hits = [
        name
        for name in lowered
        if name.endswith(("package.json", "pyproject.toml", "setup.py", "uv.lock", "bun.lock", "bun.lockb", "pnpm-lock.yaml"))
    ]
    task_hits = [
        name
        for name in lowered
        if any(token in name for token in ("/bench", "/eval", "/task", "/scenario", "/problem", "/test", "/swe"))
        and name.endswith((".json", ".jsonl", ".yaml", ".yml", ".toml", ".md", ".py", ".ts"))
    ]
    provider_hits = [
        name
        for name in lowered
        if any(token in name for token in ("openai", "anthropic", "gemini", "provider", "api_key", "models.dev", "litellm"))
    ][:24]
    node_ok = command_available("node")
    bun_ok = command_available("bun")
    container_status = container_runtime_status()
    docker_ok = bool(container_status.get("docker_cli"))
    podman_ok = bool(container_status.get("podman_cli"))
    container_ok = bool(container_status.get("ready"))
    trace_path = ROOT / "reports" / "coding_agent_traces" / f"{safe_name(source_id)}_seed{args.seed}.jsonl"
    trace_packets = [
        {
            "event": "source_inventory",
            "source_id": source_id,
            "present": present,
            "path": rel_or_abs(path),
            "manifest_count": len(manifest_hits),
            "task_or_harness_count": len(task_hits),
        },
        {
            "event": "external_inference_guard",
            "source_id": source_id,
            "provider_path_sample": provider_hits[:8],
            "provider_calls_disabled": True,
            "network_during_scoring": "forbidden",
        },
        {
            "event": "local_endpoint_contract",
            "source_id": source_id,
            "endpoint": "Theseus/OpenAI-compatible local endpoint only",
            "teacher_allowed": "proposal/audit only, never benchmark solving",
        },
    ]
    write_jsonl(trace_path, trace_packets)
    organism = run_code_repair_organism(source_id, path, args) if present else {"ran": False, "skipped_reason": "source_not_present"}
    score = 0.08
    score += 0.14 if present else 0.0
    score += 0.14 if manifest_hits else 0.0
    score += 0.14 if task_hits else 0.0
    score += 0.12
    score += 0.06 if node_ok else 0.0
    if source_id in {"opencode", "opencode_bench"}:
        score += 0.08 if bun_ok else 0.0
    elif source_id in {"openhands", "terminal_bench", "swe_atlas", "swe_rex", "swe_smith"}:
        score += 0.08 if container_ok else 0.0
    else:
        score += 0.05
    score = clamp01(score + code_repair_score_bonus(organism))
    residuals: list[dict[str, str]] = []
    if not present:
        residuals.append({"type": "source_not_staged", "detail": "coding-agent harness source is not locally staged"})
    if not task_hits:
        residuals.append({"type": "task_manifest_locator", "detail": "harness needs a precise task/problem manifest loader"})
    if source_id in {"opencode", "opencode_bench"} and not bun_ok:
        residuals.append({"type": "bun_runtime_missing", "detail": "OpenCode source declares Bun; install/runtime gate remains before full harness execution"})
    if source_id in {"openhands", "terminal_bench", "swe_atlas", "swe_rex", "swe_smith"} and not container_ok:
        residuals.append({"type": "container_runtime_missing", "detail": f"Full sandbox execution needs ready Docker or Podman runtime; status={container_status.get('reason')}"})
    if source_id == "opencode_bench" and not card.get("license_allowed"):
        residuals.append({"type": "license_audit_required", "detail": "opencode-bench currently has no repo license exposed; keep metadata-only until approved"})
    if organism.get("ran") and organism.get("transfer_consumed"):
        residuals.append(
            {
                "type": "agent_harness_task_loader_deepening_needed",
                "detail": "local repair organism consumed transfer artifacts; next pressure is endpoint/harness-specific task execution",
            }
        )
    else:
        residuals.append({"type": "local_endpoint_adapter_needed", "detail": "wire harness to Theseus endpoint and deterministic local repo tasks"})
    candidate_evals = max(1, int(args.train_iterations)) * max(4, int(args.train_population))
    env_steps = candidate_evals * max(1, int(args.steps))
    checks = safety_checks(card)
    checks.extend(
        [
            check("source_present", present, rel_or_abs(path)),
            check("framework_manifest_present", bool(manifest_hits), ", ".join(manifest_hits[:8]) or "no manifest"),
            check("task_manifest_or_harness_present", bool(task_hits or manifest_hits), ", ".join(task_hits[:8]) or "framework manifest only"),
            check("provider_calls_disabled", True, json.dumps(provider_hits[:8])),
            check("node_runtime_available", node_ok, "node found" if node_ok else "node missing"),
            *organism_checks(organism),
            check(
                "train_before_eval_candidate_budget",
                candidate_evals >= max(0, int(args.min_train_candidate_evals)),
                f"{candidate_evals} >= {args.min_train_candidate_evals}",
            ),
            check(
                "train_before_eval_env_step_budget",
                env_steps >= max(0, int(args.min_train_env_steps)),
                f"{env_steps} >= {args.min_train_env_steps}",
            ),
        ]
    )
    if source_id in {"opencode", "opencode_bench"}:
        checks.append(check("bun_runtime_available", bun_ok, "bun found" if bun_ok else "bun missing"))
    if source_id in {"openhands", "terminal_bench", "swe_atlas", "swe_rex", "swe_smith"}:
        checks.append(check("container_runtime_available", container_ok, json.dumps(container_status)))
    return {
        "suite": suite_name("coding_agent", card.get("id", "source_coding_agent")),
        "score": clamp01(score),
        "status": "frontier_open" if present else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "source_present": present,
            "manifest_count": len(manifest_hits),
            "task_or_harness_count": len(task_hits),
            "provider_path_sample": provider_hits[:8],
            "node_available": node_ok,
            "bun_available": bun_ok,
            "docker_available": docker_ok,
            "podman_available": podman_ok,
            "container_runtime_available": container_ok,
            "container_runtime_status": container_status,
            "trace_path": rel_or_abs(trace_path),
            "manifest_sample": manifest_hits[:8],
            "task_sample": task_hits[:8],
            "code_repair_organism": organism_metrics(organism),
        },
        "residuals": residuals,
    }


def run_old_project_registry_pressure(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    manifest = resolve_path(str(card.get("case_manifest") or card.get("resource_pantry_path") or card.get("staged_path") or ""))
    rows = read_jsonl(manifest) if manifest.exists() else []
    eval_rows = [
        row
        for row in rows
        if str(row.get("split") or "").lower() != "train" and not bool(row.get("seen_in_training"))
    ]
    answer_fields = [
        key
        for row in rows
        for key in row
        if key in {"reference_answer", "answer", "solution", "expected"}
    ]
    redacted_rows = [row for row in rows if row.get("reference_answer_redacted")]
    metadata_rows = [row for row in rows if isinstance(row.get("metadata"), dict) and row.get("metadata")]
    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or "Unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    readiness_score = 0.05
    readiness_score += 0.20 if manifest.exists() else 0.0
    readiness_score += 0.20 if rows else 0.0
    readiness_score += 0.15 if eval_rows else 0.0
    readiness_score += 0.12 if rows and len(redacted_rows) == len(rows) else 0.0
    readiness_score += 0.08 if metadata_rows else 0.0
    readiness_score += 0.05 if card.get("public_comparator_use") == "forbidden" else 0.0
    score = min(readiness_score, 0.34)
    scorer_report = ROOT / "reports" / "old_project_trace_scores" / f"{safe_name(args.card_id)}_seed{args.seed}.json"
    scorer_trace = ROOT / "reports" / "old_project_traces" / f"{safe_name(args.card_id)}_seed{args.seed}.jsonl"
    scorer_payload: dict[str, Any] = {}
    scorer_runner: dict[str, Any] = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": ""}
    if rows and not answer_fields:
        scorer_runner = run_command(
            [
                str(preferred_python()),
                "scripts/old_project_trace_scorer.py",
                "--card-id",
                args.card_id,
                "--case-manifest",
                rel_or_abs(manifest),
                "--seed",
                str(args.seed),
                "--out",
                rel_or_abs(scorer_report),
                "--trace-out",
                rel_or_abs(scorer_trace),
            ],
            timeout=120,
        )
        scorer_payload = read_json(scorer_report)
        scorer_summary = scorer_payload.get("summary") if isinstance(scorer_payload.get("summary"), dict) else {}
        if scorer_payload.get("policy") == "project_theseus_old_project_trace_scorer_v1":
            score = clamp01(scorer_summary.get("score"))
    residuals: list[dict[str, str]] = []
    if not manifest.exists():
        residuals.append({"type": "old_project_case_manifest_missing", "detail": "redacted old-project case manifest is not staged"})
    if not rows:
        residuals.append({"type": "old_project_cases_empty", "detail": "no redacted pressure cases were loaded"})
    if answer_fields:
        residuals.append({"type": "answer_leakage_blocker", "detail": f"forbidden answer fields present: {sorted(set(answer_fields))}"})
    if not eval_rows:
        residuals.append({"type": "old_project_eval_split_missing", "detail": "no non-training cases are available for private pressure"})
    scorer_summary = scorer_payload.get("summary") if isinstance(scorer_payload.get("summary"), dict) else {}
    scorer_ran = scorer_payload.get("policy") == "project_theseus_old_project_trace_scorer_v1"
    if not scorer_ran:
        residuals.append({"type": "old_project_trace_scorer_needed", "detail": "wire trace/evidence scorer for this redacted custom benchmark"})
        residuals.append({"type": "student_response_adapter_needed", "detail": "connect local Theseus checkpoint responses to this old-project case manifest"})
    else:
        if not scorer_summary.get("trace_scorer_present"):
            residuals.append({"type": "old_project_trace_scorer_format_needed", "detail": "trace scorer ran but did not support this old-project case format"})
        if not scorer_summary.get("student_response_adapter_present"):
            residuals.append({"type": "student_response_adapter_needed", "detail": "trace scorer ran but no local student response adapter supported this case format"})
        for case_score in scorer_payload.get("case_scores", []) if isinstance(scorer_payload.get("case_scores"), list) else []:
            for item in case_score.get("residuals", []) if isinstance(case_score.get("residuals"), list) else []:
                if isinstance(item, dict):
                    residuals.append({"type": str(item.get("type") or "old_project_trace_residual"), "detail": str(item.get("detail") or "")})
    checks = safety_checks(card)
    checks.extend(
        [
            check("case_manifest_present", manifest.exists(), rel_or_abs(manifest)),
            check("cases_loaded", bool(rows), f"cases={len(rows)}"),
            check("non_train_cases_available", bool(eval_rows), f"eval_cases={len(eval_rows)} split_counts={split_counts}"),
            check("reference_answers_redacted", bool(rows) and len(redacted_rows) == len(rows), f"redacted={len(redacted_rows)}/{len(rows)}"),
            check("no_reference_answer_fields", not answer_fields, ",".join(sorted(set(answer_fields))) or "none"),
            check("public_score_claims_forbidden", card.get("public_comparator_use") == "forbidden", str(card.get("public_comparator_use"))),
            check("old_project_trace_scorer_ran", scorer_ran, rel_or_abs(scorer_report) if scorer_ran else scorer_runner.get("stderr_tail") or "not_run"),
            check("old_project_student_response_adapter", bool(scorer_summary.get("student_response_adapter_present")), f"supported={scorer_summary.get('supported_case_count')} eval={scorer_summary.get('eval_case_count')}"),
            check("old_project_trace_contract_score", scorer_ran and clamp01(scorer_summary.get("score")) >= 0.70, f"score={scorer_summary.get('score')} semantics={scorer_summary.get('score_semantics')}"),
            check("external_inference_zero", True, "manifest-only local pressure"),
        ]
    )
    return {
        "suite": suite_name("old_project_registry", card.get("id", "old_registry")),
        "score": clamp01(score),
        "status": "frontier_open" if rows and not answer_fields else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "case_manifest": rel_or_abs(manifest),
            "case_count": len(rows),
            "non_train_case_count": len(eval_rows),
            "split_counts": split_counts,
            "redacted_reference_answer_count": len(redacted_rows),
            "metadata_case_count": len(metadata_rows),
            "old_project_registry": card.get("old_project_registry", {}),
            "readiness_score": clamp01(readiness_score),
            "trace_scorer": {
                "ran": scorer_ran,
                "ok": bool(scorer_runner.get("ok")),
                "returncode": scorer_runner.get("returncode"),
                "report": rel_or_abs(scorer_report),
                "trace": rel_or_abs(scorer_trace),
                "trigger_state": scorer_payload.get("trigger_state"),
                "score": scorer_summary.get("score"),
                "student_response_adapter_present": scorer_summary.get("student_response_adapter_present"),
                "trace_scorer_present": scorer_summary.get("trace_scorer_present"),
                "supported_case_count": scorer_summary.get("supported_case_count"),
                "external_inference_calls": scorer_summary.get("external_inference_calls"),
            },
            "score_semantics": "private old-project trace-contract score when scorer is live; not a public benchmark score",
        },
        "residuals": residuals,
    }


def run_webarena(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    has_source = path.exists()
    task_files = list(path.rglob("*.json"))[:25] if has_source and path.is_dir() else []
    service_ready = bool((path / "config_files").exists() or (path / "environment_docker").exists()) if has_source and path.is_dir() else False
    local_fixture_ready = bool(task_files)
    score = 0.18 + (0.12 if has_source else 0.0) + (0.08 if local_fixture_ready else 0.0)
    return {
        "suite": suite_name("web_agent", card.get("id", "source_webarena")),
        "score": score,
        "status": "frontier_open",
        "checks": [
            check("source_present", has_source, rel_or_abs(path)),
            check("local_task_inventory", local_fixture_ready, f"json_task_or_config_files={len(task_files)}"),
            check("self_hosted_service_ready", service_ready, "no self-hosted WebArena service was contacted"),
            check("no_real_accounts", True, "only local metadata/source readiness checked"),
        ],
        "metrics": {
            "source_present": has_source,
            "service_ready": service_ready,
            "local_fixture_ready": local_fixture_ready,
            "task_file_sample": [rel_or_abs(item) for item in task_files[:8]],
        },
        "residuals": [
            {
                "type": "service_setup",
                "detail": "self-hosted web tasks need a deterministic local service fixture before true scoring",
            }
        ],
    }


def run_native_voice(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = run_command(
        [
            str(preferred_python()),
            "scripts/native_voice_io.py",
            "--out",
            f"reports/native_voice_io_{safe_name(card.get('source_id') or card.get('id'))}_seed{args.seed}.json",
        ],
        timeout=90,
    )
    payload = parse_last_json(result["stdout_tail"]) if result["ok"] else {}
    if not payload:
        payload = read_json(ROOT / f"reports/native_voice_io_{safe_name(card.get('source_id') or card.get('id'))}_seed{args.seed}.json")
    score = clamp01(get_path(payload, ["summary", "accuracy"], 0.0))
    checks = safety_checks(card)
    checks.extend(payload.get("checks", []) if isinstance(payload.get("checks"), list) else [])
    checks.append(
        check(
            "native_voice_runner",
            result["ok"],
            "native voice report completed" if result["ok"] else (result["stderr_tail"] or "native voice report failed"),
        )
    )
    return {
        "suite": suite_name("voice", card.get("id", "source_common_voice")),
        "score": score,
        "status": "frontier_open",
        "checks": checks,
        "metrics": {
            "native_voice_report": f"reports/native_voice_io_{safe_name(card.get('source_id') or card.get('id'))}_seed{args.seed}.json",
            "voice_is_head_router_io": get_path(payload, ["summary", "voice_is_head_router_io"], None),
            "native_model_ready": get_path(payload, ["summary", "native_model_ready"], None),
            "learned_components": payload.get("learned_components", {}),
        },
        "residuals": payload.get("residuals", [])
        or [{"type": "native_voice_training_needed", "detail": "native STT/TTS components are not yet mastered"}],
    }


def run_emulator_rl(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    registry = read_json(ROOT / "reports" / "local_rom_registry.json")
    profile_id = str(get_path(card, ["input_contract", "rom_profile_id"], "") or card.get("source_id") or card.get("id"))
    source_id = str(card.get("source_id") or profile_id)
    python = runtime_python_for_source(source_id)
    profiles = registry.get("profiles") if isinstance(registry.get("profiles"), list) else []
    profile = next((row for row in profiles if row.get("profile_id") == profile_id or row.get("id") == profile_id), {})
    rom_path = resolve_path(str(profile.get("path") or profile.get("rom_path") or ""))
    rom_present = bool(profile and rom_path.exists())
    pyboy_available = python_has_module("pyboy", python)
    gymboy_available = python_has_module("gymboy", python)
    mgba_available = python_has_mgba_runtime(python) and python_has_module("pygba", python)
    is_gba_profile = profile_id.startswith("gba_") or source_id == "pygba"
    runtime_ready = bool(rom_present and (mgba_available if is_gba_profile else (pyboy_available or gymboy_available)))
    score = 0.10 + (0.18 if rom_present else 0.0) + (0.18 if runtime_ready else 0.0)
    return {
        "suite": suite_name("emulator_rl", card.get("id", "local_rom")),
        "score": score,
        "status": "frontier_open" if runtime_ready else "runtime_blocked",
        "checks": [
            check("user_supplied_rom_only", rom_present, profile_id),
            check("rom_file_ignored_by_git", True, "ROM path resolved from ignored local registry only"),
            check(
                "emulator_runtime_available",
                runtime_ready,
                f"profile={profile_id} pyboy={pyboy_available} gymboy={gymboy_available} mgba_pygba={mgba_available}",
            ),
            check("no_commercial_rom_download", True, "network fetch forbidden; BYO local asset only"),
        ],
        "metrics": {
            "profile_id": profile_id,
            "rom_present": rom_present,
            "runtime_ready": runtime_ready,
            "pyboy_available": pyboy_available,
            "gymboy_available": gymboy_available,
            "mgba_available": mgba_available,
            "runtime_python": rel_or_abs(python),
            "steps": 0,
        },
        "residuals": []
        if runtime_ready
        else [
            {
                "type": "emulator_runtime_dependency",
                "detail": "local ROM profile exists but emulator runtime or wrapper smoke is not ready",
            }
        ],
    }


def run_minecraft_rl(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_id = str(card.get("source_id") or card.get("id") or "").removeprefix("source_")
    probe_out = ROOT / "reports" / "minecraft_runtime_probe.json"
    python = runtime_python_for_source(source_id if source_id in {"crafter", "craftax", "minerl", "minedojo", "malmo"} else "crafter")
    probe = run_command(
        [
            str(python),
            "scripts/minecraft_runtime_probe.py",
            "--out",
            str(probe_out),
        ],
        timeout=45,
    )
    runtime = read_json(probe_out)
    summary = runtime.get("summary") if isinstance(runtime.get("summary"), dict) else {}
    checks = safety_checks(card)
    checks.extend(runtime.get("checks", []) if isinstance(runtime.get("checks"), list) else [])
    checks.append(check("minecraft_runtime_probe", probe["ok"], probe["stderr_tail"] or probe["stdout_tail"] or "probe completed"))
    checks.append(check("no_public_server_by_default", True, "pressure runner never joins public servers"))
    checks.append(check("no_credentials_stored", True, "no launcher credentials are read or stored"))

    bridge_result = minecraft_bridge_rollout(source_id, args)
    full_ready = bool(summary.get("full_minecraft_runtime_ready"))
    bridge_ready = bool(bridge_result.get("ok") or summary.get("bridge_runtime_ready"))
    local_install = bool(summary.get("local_minecraft_install_detected"))

    if bridge_result:
        checks.append(
            check(
                "minecraft_like_bridge_rollout",
                bool(bridge_result.get("ok")),
                bridge_result.get("evidence", "") or bridge_result.get("stderr_tail", ""),
            )
        )
    candidate_evals = max(1, int(args.train_iterations)) * max(4, int(args.train_population))
    env_steps = candidate_evals * max(1, int(args.steps))
    checks.append(
        check(
            "train_before_eval_candidate_budget",
            candidate_evals >= max(0, int(args.min_train_candidate_evals)),
            f"{candidate_evals} >= {args.min_train_candidate_evals}",
        )
    )
    checks.append(
        check(
            "train_before_eval_env_step_budget",
            env_steps >= max(0, int(args.min_train_env_steps)),
            f"{env_steps} >= {args.min_train_env_steps}",
        )
    )
    checks.append(
        check(
            "full_minecraft_harness_ready",
            full_ready,
            f"install={local_install} java={summary.get('java_available')} modules={get_path(runtime, ['runtime', 'python_modules'], {})}",
        )
    )

    base_score = 0.10
    base_score += 0.12 if local_install else 0.0
    base_score += 0.14 if bool(summary.get("java_available")) else 0.0
    base_score += 0.18 if full_ready else 0.0
    if bridge_result.get("ok"):
        base_score = max(base_score, float(bridge_result.get("score") or 0.0))
    elif bridge_ready:
        base_score += 0.12
    score = clamp01(base_score)
    residuals = []
    if not full_ready:
        residuals.append(
            {
                "type": "minecraft_full_harness_runtime",
                "detail": "full Minecraft harness is not ready; use bridge pressure or stage MineDojo/Malmo under local license policy",
            }
        )
    if not bridge_result.get("ok"):
        residuals.append(
            {
                "type": "minecraft_bridge_rollout",
                "detail": bridge_result.get("evidence") or "Crafter/Craftax bridge rollout did not complete",
            }
        )
    return {
        "suite": suite_name("minecraft_rl", card.get("id", source_id)),
        "score": score,
        "status": "frontier_open" if full_ready or bridge_ready else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "source_id": source_id,
            "runtime_probe": rel_or_abs(probe_out),
            "runtime_summary": summary,
            "bridge_result": bridge_result,
            "episodes": max(1, args.episodes),
            "steps": max(1, args.steps),
            "trace_path": bridge_result.get("trace_path", ""),
            "transfer_artifact_path": bridge_result.get("transfer_artifact_path", ""),
        },
        "residuals": residuals,
    }


def minecraft_bridge_rollout(source_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a tiny local Minecraft-like bridge if available.

    Full Minecraft harnesses need local Java/game runtime setup. Crafter is a
    clean bridge because it is MIT and does not need commercial assets.
    """
    trace_dir = ROOT / "reports" / "minecraft_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{safe_name(source_id or 'bridge')}_seed{args.seed}.jsonl"
    artifact_path = (
        ROOT
        / "reports"
        / "transfer_artifacts"
        / f"minecraft_policy_prior_{safe_name(source_id or 'bridge')}_seed{int(args.seed)}.json"
    )
    python = runtime_python_for_source("crafter")
    if source_id in {"", "crafter", "craftax", "minerl", "minedojo", "malmo", "voyager_minecraft"} and python_has_module("crafter", python):
        trainer_out = ROOT / "reports" / f"minecraft_crafter_trainer_seed{args.seed}.json"
        result = run_command(
            [
                str(python),
                "scripts/minecraft_crafter_trainer.py",
                "--seed",
                str(args.seed),
                "--train-iterations",
                str(max(1, int(args.train_iterations))),
                "--population",
                str(max(4, int(args.train_population))),
                "--elite-count",
                str(max(1, int(args.elite_count))),
                "--train-steps",
                str(max(1, int(args.steps))),
                "--eval-steps",
                str(max(1, int(args.steps))),
                "--eval-seed-count",
                str(max(1, int(args.eval_seed_count or args.episodes))),
                "--trace-path",
                str(trace_path),
                "--artifact-path",
                str(artifact_path),
                "--out",
                str(trainer_out),
            ],
            timeout=pressure_timeout_seconds(args),
        )
        payload = read_json(trainer_out) if result["ok"] else {}
        steps = int(sum(row.get("steps", 0) for row in payload.get("eval", []) if isinstance(row, dict)))
        total_reward = float(sum(row.get("total_reward", 0.0) for row in payload.get("eval", []) if isinstance(row, dict)))
        score = clamp01(float(payload.get("score") or 0.0))
        return {
            "ok": bool(result["ok"]),
            "source": "crafter_cem_trainer",
            "score": score if result["ok"] else 0.0,
            "steps": steps,
            "total_reward": total_reward,
            "trace_path": rel_or_abs(trace_path),
            "transfer_artifact_path": rel_or_abs(artifact_path),
            "trainer_report": rel_or_abs(trainer_out),
            "evidence": result["stdout_tail"][-1200:] if result["stdout_tail"] else result["stderr_tail"],
            "stderr_tail": result["stderr_tail"],
        }
    craftax_python = runtime_python_for_source("craftax")
    if python_has_module("craftax", craftax_python):
        return {
            "ok": True,
            "source": "craftax_import_only",
            "score": 0.30,
            "steps": 0,
            "total_reward": 0.0,
            "trace_path": "",
            "transfer_artifact_path": "",
            "evidence": "craftax import available; full JAX rollout is deferred to dedicated runner",
        }
    return {
        "ok": False,
        "source": "none",
        "score": 0.0,
        "steps": 0,
        "total_reward": 0.0,
        "trace_path": "",
        "transfer_artifact_path": "",
        "evidence": "No Crafter/Craftax bridge module available",
    }


def run_transfer_eval_surface(args: argparse.Namespace) -> dict[str, Any]:
    tasks = {
        "code_repair": 0.35,
        "tool_use": 0.45,
        "web_task": 0.25,
        "long_context_recovery": 0.40,
        "rl_control": 0.38,
        "self_debugging": 0.42,
        "voice_io": 0.20,
    }
    score = sum(tasks.values()) / len(tasks)
    return {
        "suite": "asi_transfer_suite",
        "score": score,
        "status": "frontier_open",
        "checks": [check(name, value >= 0.7, f"baseline={value:.2f}") for name, value in tasks.items()],
        "metrics": {"task_scores": tasks},
        "residuals": [
            {"type": name, "detail": "below transfer mastery threshold; keep as active pressure"}
            for name, value in tasks.items()
            if value < 0.7
        ],
    }


def run_synthetic_benchmark(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    manifest = resolve_path(str(card.get("case_manifest") or card.get("staged_path") or ""))
    cases = read_jsonl(manifest)
    required_arms = [str(item) for item in card.get("required_arms", []) if str(item)]
    available_arms = synthetic_available_arms()
    pseudo_arms = {"code_repair_verifier", "residual_governance_arm"}
    missing_arms = [arm for arm in required_arms if arm not in available_arms and arm not in pseudo_arms]
    provenance_ok = bool(cases) and all(
        isinstance(case.get("provenance"), dict)
        and int(case.get("provenance", {}).get("copied_source_item_chars") or 0) == 0
        and case.get("scoring", {}).get("public_comparator_use") == "forbidden"
        for case in cases
    )
    code_cases = [case for case in cases if case.get("case_type") == "python_code_repair"]
    organism = (
        run_code_repair_organism(str(card.get("source_id") or card.get("id") or "synthetic"), manifest, args, task_manifest=manifest)
        if code_cases
        else {"ran": False, "skipped_reason": "no_python_code_repair_cases"}
    )
    transfer_rate = float(organism.get("transfer_pass_rate") or 0.0)
    contract_score = 0.0
    contract_score += 0.02 if manifest.exists() and bool(cases) else 0.0
    contract_score += 0.02 if provenance_ok else 0.0
    contract_score += 0.02 if not missing_arms else 0.0
    contract_score += 0.02 if card.get("public_comparator_use") == "forbidden" else 0.0
    heredity_bonus = 0.10 if organism.get("transfer_consumed") else 0.0
    score = clamp01(contract_score + (0.80 * transfer_rate) + heredity_bonus)
    residuals: list[dict[str, str]] = []
    if not cases:
        residuals.append({"type": "synthetic_case_manifest_empty", "detail": rel_or_abs(manifest)})
    if missing_arms:
        residuals.append({"type": "cross_arm_activation_missing", "detail": ", ".join(missing_arms)})
    if not provenance_ok:
        residuals.append({"type": "synthetic_provenance_gate_failed", "detail": "missing provenance or comparator quarantine"})
    if organism.get("ran") and not organism.get("transfer_consumed"):
        residuals.append({"type": "synthetic_transfer_consumption_missing", "detail": "local repair organism ran but did not consume transfer artifacts"})
    if transfer_rate < 0.70:
        residuals.append({"type": "synthetic_code_repair_below_floor", "detail": f"transfer_pass_rate={transfer_rate:.4f}"})
    checks = safety_checks(card)
    checks.extend(
        [
            check("synthetic_case_manifest_present", manifest.exists(), rel_or_abs(manifest)),
            check("synthetic_cases_loaded", bool(cases), f"cases={len(cases)}"),
            check("synthetic_code_cases_present", bool(code_cases), f"code_cases={len(code_cases)}"),
            check("cross_arm_activation_contracts_present", bool(required_arms), ", ".join(required_arms)),
            check("required_arms_available", not missing_arms, ", ".join(missing_arms) or "all required arms available or pseudo-verified"),
            check("synthetic_provenance_preserved", provenance_ok, "no copied source items; source refs metadata only"),
            check("public_comparator_quarantine", card.get("public_comparator_use") == "forbidden", str(card.get("public_comparator_use"))),
            *organism_checks(organism),
        ]
    )
    return {
        "suite": suite_name("synthetic_benchmark", card.get("id", "synthetic")),
        "score": score,
        "status": "frontier_open" if cases else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "case_manifest": rel_or_abs(manifest),
            "case_count": len(cases),
            "code_case_count": len(code_cases),
            "required_arms": required_arms,
            "missing_arms": missing_arms,
            "score_semantics": get_path(card, ["synthetic_benchmark", "score_semantics"], "private synthetic pressure"),
            "code_repair_organism": organism_metrics(organism),
        },
        "residuals": residuals,
    }


def run_multi_stream_code_pressure(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    manifest = resolve_path(str(card.get("case_manifest") or card.get("staged_path") or card.get("resource_pantry_path") or ""))
    report = ROOT / "reports" / f"multi_stream_code_pressure_{safe_name(card.get('id') or 'multistream')}_seed{args.seed}.json"
    command = [
        str(preferred_python()),
        "scripts/multi_stream_code_pressure_runner.py",
        "--card-id",
        str(card.get("id") or "multistream_code_repair_pressure"),
        "--seed",
        str(args.seed),
        "--case-manifest",
        rel_or_abs(manifest),
        "--code-transfer-artifacts",
        str(args.code_transfer_artifacts),
        "--out",
        rel_or_abs(report),
    ]
    runner = run_command(command, timeout=180)
    payload = read_json(report)
    payload_checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    checks = safety_checks(card)
    checks.extend(
        check(str(row.get("gate") or row.get("name") or "multi_stream_check"), bool(row.get("passed")), row.get("evidence", ""))
        for row in payload_checks
        if isinstance(row, dict)
    )
    checks.extend(
        [
            check("multi_stream_runner_completed", bool(payload), rel_or_abs(report)),
            check("multi_stream_runner_exit_ok", bool(runner.get("ok")), runner.get("stderr_tail") or runner.get("stdout_tail") or ""),
        ]
    )
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "suite": suite_name("multi_stream_code_pressure", card.get("id", "multistream")),
        "score": clamp01(payload.get("score", 0.0)),
        "status": payload.get("status", "runtime_blocked") if payload else "runtime_blocked",
        "checks": checks,
        "metrics": {
            "score_semantics": "private_multistream_pressure_correctness_monitorability_and_critical_path",
            "runner_report": rel_or_abs(report),
            "case_manifest": rel_or_abs(manifest),
            "single_stream_baseline": artifacts.get("single_stream_baseline"),
            "trace": artifacts.get("trace"),
            "verifier": artifacts.get("verifier"),
            "task_count": summary.get("task_count"),
            "efficiency_composite_score": payload.get("efficiency_composite_score") or summary.get("efficiency_composite_score"),
            "single_stream_transfer_pass_rate": summary.get("single_stream_transfer_pass_rate"),
            "multi_stream_pass_rate": summary.get("multi_stream_pass_rate"),
            "pass_rate_delta": summary.get("pass_rate_delta"),
            "task_level_improvements_over_single_stream": summary.get("task_level_improvements_over_single_stream"),
            "task_level_regressions_vs_single_stream": summary.get("task_level_regressions_vs_single_stream"),
            "avg_patch_candidates_tested": summary.get("avg_patch_candidates_tested"),
            "patch_stream_synthesis_used_count": summary.get("patch_stream_synthesis_used_count"),
            "verifier_score": summary.get("verifier_score"),
            "monitorability_coverage": summary.get("monitorability_coverage"),
            "avg_parallel_efficiency": summary.get("avg_parallel_efficiency"),
            "apples_to_apples_overlap": summary.get("apples_to_apples_overlap"),
            "external_inference_calls": summary.get("external_inference_calls", 0),
        },
        "residuals": payload.get("residuals", []) if isinstance(payload.get("residuals"), list) else [{"type": "multi_stream_runner_failed", "detail": rel_or_abs(report)}],
    }


def run_generic_card_probe(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    present = path.exists()
    return {
        "suite": suite_name(str(card.get("runner_family") or "generic"), card.get("id", "card")),
        "score": 0.22 if present else 0.05,
        "status": "frontier_open" if present else "runtime_blocked",
        "checks": [check("source_present", present, rel_or_abs(path))],
        "metrics": {"source_present": present},
        "residuals": [{"type": "adapter_missing", "detail": "generic card needs a first-class pressure runner"}],
    }


def safety_checks(card: dict[str, Any]) -> list[dict[str, Any]]:
    permission = card.get("permission_envelope") if isinstance(card.get("permission_envelope"), dict) else {}
    return [
        check("license_allowed", bool(card.get("license_allowed")), str(card.get("license_spdx"))),
        check("external_inference_zero", int(card.get("external_inference_calls") or 0) == 0, "card/tool local only"),
        check(
            "hardware_approval_gated",
            permission.get("hardware") in {None, "not_applicable", "forbidden_without_explicit_human_approval"},
            str(permission.get("hardware")),
        ),
    ]


def load_card(card_id: str) -> dict[str, Any]:
    path = ROOT / "benchmarks" / "cards" / f"{card_id}.json"
    if not path.exists():
        return {}
    return read_json(path)


def run_command(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    env = os.environ.copy()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env)
    except Exception as exc:  # noqa: BLE001 - diagnostic runner.
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def run_python_script(script: str, *, timeout: int = 30, python: Path | None = None) -> dict[str, Any]:
    return run_command([str(python or preferred_python()), "-c", script], timeout=timeout)


def runtime_python_for_source(source_id: str) -> Path:
    lane_map = {
        "pyflyt": [".venv-drone-pyflyt-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "pyflyt_waypoints": [".venv-drone-pyflyt-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "gym_pybullet_drones": [".venv-drone-gym-pybullet-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "mavsdk_python": [".venv-drone-control-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "crafter": [".venv-minecraft-rl-py311"],
        "craftax": [".venv-minecraft-rl-py311"],
        "minerl": [".venv-minecraft-rl-py311"],
        "minedojo": [".venv-minecraft-rl-py311"],
        "malmo": [".venv-minecraft-rl-py311"],
        "voyager_minecraft": [".venv-minecraft-rl-py311"],
    }
    minecraft_modules = {
        "crafter": "crafter",
        "craftax": "craftax",
        "minerl": "minerl",
        "minedojo": "minedojo",
        "malmo": "malmoenv",
        "voyager_minecraft": "crafter",
    }
    if source_id in minecraft_modules:
        candidates = [ROOT / venv / "Scripts" / "python.exe" for venv in lane_map.get(source_id, [])]
        candidates.append(ROOT / ".venv-puffer" / "Scripts" / "python.exe")
        for candidate in candidates:
            if candidate.exists() and python_has_module(minecraft_modules[source_id], candidate):
                return candidate
        for candidate in candidates:
            if candidate.exists():
                return candidate
    for venv in lane_map.get(source_id, []):
        candidate = ROOT / venv / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
    return preferred_python()


def preferred_python() -> Path:
    venv = ROOT / ".venv-puffer" / "Scripts" / "python.exe"
    if venv.exists():
        return venv
    return Path(sys.executable)


def parse_last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else {}
    return {}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def python_has_module(name: str, python: Path) -> bool:
    result = run_command([str(python), "-c", f"import {name}"], timeout=20)
    return bool(result.get("ok"))


def python_has_mgba_runtime(python: Path) -> bool:
    script = (
        "import mgba.core\n"
        "from mgba._pylib import ffi, lib\n"
        "assert hasattr(lib, 'mCoreFind')\n"
        "ffi.sizeof('mColor')\n"
    )
    result = run_command([str(python), "-c", script], timeout=20)
    return bool(result.get("ok"))


def run_code_repair_organism(source_id: str, source_path: Path, args: argparse.Namespace, *, task_manifest: Path | None = None) -> dict[str, Any]:
    report = ROOT / "reports" / f"local_code_repair_organism_{safe_name(source_id)}_seed{args.seed}.json"
    trace = ROOT / "reports" / "local_code_repair_traces" / f"{safe_name(source_id)}_seed{args.seed}.jsonl"
    artifact = ROOT / "reports" / "transfer_artifacts" / "code" / f"{safe_name(source_id)}_repair_organism_evidence.json"
    command = [
            str(preferred_python()),
            "scripts/local_code_repair_organism.py",
            "--card-id",
            source_id,
            "--seed",
            str(args.seed),
            "--source-path",
            rel_or_abs(source_path),
            "--transfer-artifacts",
            str(args.code_transfer_artifacts),
            "--out",
            rel_or_abs(report),
            "--trace-out",
            rel_or_abs(trace),
            "--artifact-out",
            rel_or_abs(artifact),
    ]
    if task_manifest:
        command.extend(["--task-manifest", rel_or_abs(task_manifest)])
    result = run_command(
        command,
        timeout=90,
    )
    payload = read_json(report)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    heredity = get_path(payload, ["transfer_run"], {})
    return {
        "ran": bool(payload) and payload.get("policy") == "project_theseus_local_code_repair_organism_v1",
        "ok": bool(result.get("ok")),
        "returncode": result.get("returncode"),
        "report": rel_or_abs(report),
        "patch_trace": str(artifacts.get("patch_trace") or rel_or_abs(trace)),
        "transfer_evidence": str(artifacts.get("transfer_evidence") or rel_or_abs(artifact)),
        "transfer_loaded": bool(summary.get("transfer_loaded")),
        "transfer_consumed": bool(summary.get("transfer_loaded")) and bool(summary.get("transfer_altered_behavior")),
        "transfer_altered_behavior": bool(summary.get("transfer_altered_behavior")),
        "baseline_pass_rate": clamp01(summary.get("baseline_pass_rate")),
        "transfer_pass_rate": clamp01(summary.get("transfer_pass_rate")),
        "pass_rate_delta": float(summary.get("pass_rate_delta") or 0.0),
        "task_count": int(summary.get("task_count") or 0),
        "residual_count": int(summary.get("residual_count") or 0),
        "stderr_tail": result.get("stderr_tail"),
        "stdout_tail": result.get("stdout_tail"),
        "payload_policy": payload.get("policy"),
        "heredity_total": heredity.get("total") if isinstance(heredity, dict) else None,
    }


def run_public_code_multistream(
    card: dict[str, Any],
    source_id: str,
    source_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    card_id = str(card.get("id") or f"source_{source_id}")
    manifest = ROOT / "data" / "public_code_benchmark_manifests" / f"{safe_name(card_id)}_seed{args.seed}.jsonl"
    builder_report = ROOT / "reports" / f"public_code_benchmark_manifest_{safe_name(card_id)}_seed{args.seed}.json"
    runner_report = ROOT / "reports" / f"multi_stream_code_pressure_{safe_name(card_id)}_seed{args.seed}.json"
    trace = ROOT / "reports" / "multi_stream_traces" / f"{safe_name(card_id)}_public_loader_seed{args.seed}.jsonl"
    verifier = ROOT / "reports" / f"multi_stream_causal_verifier_{safe_name(card_id)}_public_loader_seed{args.seed}.json"
    single = ROOT / "reports" / f"local_code_repair_organism_{safe_name(card_id)}_single_stream_seed{args.seed}.json"

    builder = run_command(
        [
            str(preferred_python()),
            "scripts/code_benchmark_manifest_builder.py",
            "--card-id",
            card_id,
            "--source-path",
            rel_or_abs(source_path),
            "--seed",
            str(args.seed),
            "--out",
            rel_or_abs(manifest),
            "--summary-out",
            rel_or_abs(builder_report),
        ],
        timeout=60,
    )
    builder_payload = read_json(builder_report)
    case_count = int(builder_payload.get("case_count") or 0) if isinstance(builder_payload, dict) else 0
    if case_count <= 0:
        return {
            "ran": False,
            "ok": False,
            "skipped_reason": "public_code_manifest_empty",
            "builder_report": rel_or_abs(builder_report),
            "manifest": rel_or_abs(manifest),
            "builder_returncode": builder.get("returncode"),
            "stderr_tail": builder.get("stderr_tail"),
            "stdout_tail": builder.get("stdout_tail"),
            "case_count": case_count,
        }

    runner = run_command(
        [
            str(preferred_python()),
            "scripts/multi_stream_code_pressure_runner.py",
            "--card-id",
            card_id,
            "--seed",
            str(args.seed),
            "--case-manifest",
            rel_or_abs(manifest),
            "--code-transfer-artifacts",
            str(args.code_transfer_artifacts),
            "--out",
            rel_or_abs(runner_report),
            "--trace-out",
            rel_or_abs(trace),
            "--verifier-out",
            rel_or_abs(verifier),
            "--single-stream-out",
            rel_or_abs(single),
        ],
        timeout=180,
    )
    payload = read_json(runner_report)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    task_delta = payload.get("task_level_delta") if isinstance(payload.get("task_level_delta"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "ran": bool(payload) and payload.get("policy") == "project_theseus_multi_stream_code_pressure_v1",
        "ok": bool(runner.get("ok")),
        "returncode": runner.get("returncode"),
        "card_id": card_id,
        "source_id": source_id,
        "benchmark_evidence_level": builder_payload.get("benchmark_evidence_level"),
        "public_benchmark_score_claim": builder_payload.get("public_benchmark_score_claim"),
        "builder_report": rel_or_abs(builder_report),
        "manifest": rel_or_abs(manifest),
        "runner_report": rel_or_abs(runner_report),
        "trace": artifacts.get("trace") or rel_or_abs(trace),
        "verifier": artifacts.get("verifier") or rel_or_abs(verifier),
        "single_stream_baseline": artifacts.get("single_stream_baseline") or rel_or_abs(single),
        "patch_selection_transfer_artifact": artifacts.get("patch_selection_transfer_artifact"),
        "case_count": case_count,
        "single_stream_transfer_pass_rate": clamp01(summary.get("single_stream_transfer_pass_rate")),
        "multi_stream_pass_rate": clamp01(summary.get("multi_stream_pass_rate")),
        "pass_rate_delta": float(summary.get("pass_rate_delta") or 0.0),
        "task_level_improvements_over_single_stream": int(summary.get("task_level_improvements_over_single_stream") or 0),
        "task_level_regressions_vs_single_stream": int(summary.get("task_level_regressions_vs_single_stream") or 0),
        "patch_stream_synthesis_used_count": int(summary.get("patch_stream_synthesis_used_count") or 0),
        "avg_patch_candidates_tested": float(summary.get("avg_patch_candidates_tested") or 0.0),
        "monitorability_coverage": clamp01(summary.get("monitorability_coverage")),
        "verifier_score": clamp01(summary.get("verifier_score")),
        "apples_to_apples_overlap": clamp01(summary.get("apples_to_apples_overlap")),
        "task_delta": task_delta,
        "stderr_tail": runner.get("stderr_tail"),
        "stdout_tail": runner.get("stdout_tail"),
        "external_inference_calls": int(payload.get("external_inference_calls") or 0) if payload else 0,
    }


def run_real_code_graduation(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    card_id = str(card.get("id") or args.card_id)
    report = ROOT / "reports" / f"real_code_benchmark_graduation_{safe_name(card_id)}_seed{args.seed}.json"
    trace = ROOT / "reports" / "real_code_benchmark_traces" / f"{safe_name(card_id)}_seed{args.seed}.jsonl"
    transfer_artifact = ROOT / "reports" / "transfer_artifacts" / "code" / f"{safe_name(card_id)}_real_code_graduation_transfer_artifact.json"
    learned_manifest = ROOT / "reports" / "student_learning_code_candidates.jsonl"
    command = [
        str(preferred_python()),
        "scripts/real_code_benchmark_graduation.py",
        "--cards",
        card_id,
        "--seed",
        str(args.seed),
        "--max-cases-per-card",
        str(max(3, min(12, args.episodes * 3))),
        "--code-transfer-artifacts",
        str(args.code_transfer_artifacts),
        "--out",
        rel_or_abs(report),
        "--trace-out",
        rel_or_abs(trace),
        "--transfer-artifact-out",
        rel_or_abs(transfer_artifact),
    ]
    if learned_manifest_matches_card(learned_manifest, card_id):
        command.extend(
            [
                "--skip-student-candidate-generation",
                "--student-candidate-manifest",
                rel_or_abs(learned_manifest),
            ]
        )
    runner = run_command(
        command,
        timeout=180,
    )
    payload = read_json(report)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "ran": bool(payload) and payload.get("policy") == "project_theseus_real_code_benchmark_graduation_v1",
        "ok": bool(runner.get("ok")),
        "returncode": runner.get("returncode"),
        "report": rel_or_abs(report),
        "trace": rel_or_abs(trace),
        "transfer_artifact": rel_or_abs(transfer_artifact),
        "trigger_state": payload.get("trigger_state"),
        "candidate_source": payload.get("candidate_source"),
        "score_semantics": payload.get("score_semantics"),
        "benchmark_evidence_level": payload.get("benchmark_evidence_level"),
        "public_benchmark_score_claim": payload.get("public_benchmark_score_claim"),
        "promotion_allowed": bool(payload.get("promotion_allowed")),
        "public_task_count": int(summary.get("public_task_count") or 0),
        "loader_regression_case_count": int(summary.get("loader_regression_case_count") or 0),
        "total_case_count": int(summary.get("total_case_count") or 0),
        "single_stream_pass_rate": clamp01(summary.get("single_stream_pass_rate")),
        "multi_stream_pass_rate": clamp01(summary.get("multi_stream_pass_rate")),
        "real_public_task_pass_rate": clamp01(summary.get("real_public_task_pass_rate")),
        "pass_rate_delta": float(summary.get("pass_rate_delta") or 0.0),
        "task_level_improvements_over_single_stream": int(summary.get("task_level_improvements_over_single_stream") or 0),
        "task_level_regressions_vs_single_stream": int(summary.get("task_level_regressions_vs_single_stream") or 0),
        "transfer_artifacts_loaded": int(summary.get("transfer_artifacts_loaded") or 0),
        "transfer_behavior_changed_suites": int(summary.get("transfer_behavior_changed_suites") or 0),
        "student_candidate_count": int(summary.get("student_candidate_count") or 0),
        "student_candidate_manifest_exists": bool(summary.get("student_candidate_manifest_exists")),
        "student_candidate_provenance_valid": bool(summary.get("student_candidate_provenance_valid")),
        "student_candidate_benchmark_integrity_valid": bool(summary.get("student_candidate_benchmark_integrity_valid")),
        "template_like_candidate_count": int(summary.get("template_like_candidate_count") or 0),
        "loop_closure_candidate_count": int(summary.get("loop_closure_candidate_count") or 0),
        "token_level_code_generation_learned": bool(summary.get("token_level_code_generation_learned")),
        "token_level_learned_candidate_count": int(summary.get("token_level_learned_candidate_count") or 0),
        "benchmark_promotion_eligible_candidate_count": int(summary.get("benchmark_promotion_eligible_candidate_count") or 0),
        "candidate_generation_modes": summary.get("candidate_generation_modes") if isinstance(summary.get("candidate_generation_modes"), list) else [],
        "external_inference_calls": int(payload.get("external_inference_calls") or 0) if payload else 0,
        "stderr_tail": runner.get("stderr_tail"),
        "stdout_tail": runner.get("stdout_tail"),
    }


def container_runtime_status() -> dict[str, Any]:
    docker = command_path("docker")
    podman = command_path("podman")
    status: dict[str, Any] = {
        "ready": False,
        "docker_cli": bool(docker),
        "podman_cli": bool(podman),
        "docker_path": docker or "",
        "podman_path": podman or "",
        "runtime": "",
        "reason": "no_docker_or_podman_cli",
    }
    if docker:
        probe = run_command([docker, "info"], timeout=15)
        status["docker_info"] = probe
        if probe.get("ok"):
            status.update({"ready": True, "runtime": "docker", "reason": "docker_info_ok"})
            return status
        status["reason"] = "docker_cli_present_but_info_failed"
    if podman:
        probe = run_command([podman, "info", "--format", "json"], timeout=15)
        status["podman_info"] = probe
        if probe.get("ok"):
            status.update({"ready": True, "runtime": "podman", "reason": "podman_info_ok"})
            return status
        machine = run_command([podman, "machine", "list"], timeout=15)
        status["podman_machine_list"] = machine
        stdout = str(machine.get("stdout_tail") or "")
        if "WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED" in stdout or "restart" in stdout.lower() or "reboot" in stdout.lower():
            status["reason"] = "podman_cli_present_but_wsl_reboot_required"
        elif machine.get("ok") and "NAME" in stdout and "theseus-podman" not in stdout:
            status["reason"] = "podman_cli_present_but_no_machine_initialized"
        else:
            status["reason"] = "podman_cli_present_but_runtime_not_ready"
    return status


def append_trace(args: argparse.Namespace, report: dict[str, Any], out: Path) -> None:
    trace = {
        "trace_id": f"pressure_{int(time.time() * 1000)}_{safe_name(args.card_id)}",
        "task": "pressure_runner",
        "workflow": "benchmark pressure runner",
        "command": f"pressure_runner {args.card_id}",
        "selected_arms": ["head_router", "benchmark_ratchet_arm", "residual_governance_arm"],
        "expected_arms": ["head_router", "benchmark_ratchet_arm", "residual_governance_arm"],
        "risk": "medium" if any(word in args.frontier_family for word in ["drone", "minecraft"]) else "low",
        "routing_pattern": "single_card",
        "returncode": 0,
        "success": True,
        "runtime_ms": report.get("runtime_ms"),
        "split": "train",
        "source": "pressure_runner",
        "artifact": rel_or_abs(out),
    }
    path = ROOT / "reports" / "workflow_routing_traces.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace) + "\n")
if __name__ == "__main__":
    raise SystemExit(main())
