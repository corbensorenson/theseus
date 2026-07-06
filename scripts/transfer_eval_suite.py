"""Emit ASI-relevant transfer evaluation surfaces.

These are deliberately small local pressure surfaces that make sure the system
does not overfit BabyLM/Ocean-style loops. They measure whether the surrounding
Theseus machinery has runnable local fixtures for the next broad capability
families: code repair, tool use, web tasks, long-context recovery, RL control,
self-debugging, and native voice I/O.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/transfer_eval_policy.json")
    parser.add_argument("--out", default="reports/transfer_eval_suite.json")
    parser.add_argument("--emit-surfaces", action="store_true", default=True)
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    tasks = [score_task(item) for item in policy.get("task_families", []) if isinstance(item, dict)]
    score = sum(task["score"] for task in tasks) / max(1, len(tasks))
    report = {
        "policy": "project_theseus_transfer_eval_suite_v0",
        "created_utc": now(),
        "config": args.policy,
        "frontier_family": "transfer_eval",
        "pressure_card_id": "transfer_eval_suite",
        "accuracy": score,
        "floor": float(policy.get("floor") or 0.70),
        "summary": {
            "suite": "asi_transfer_suite",
            "accuracy": score,
            "task_count": len(tasks),
            "mastered": sum(1 for task in tasks if task["score"] >= 0.9),
            "frontier": sum(1 for task in tasks if task["score"] < 0.9),
            "total_tool_calls": 0
        },
        "tasks": tasks,
        "residuals": [
            {"task": task["id"], "score": task["score"], "recommended_intervention": task["recommended_intervention"]}
            for task in tasks
            if task["score"] < 0.9
        ],
        "external_inference_calls": 0
    }
    out = ROOT / args.out
    write_json(out, report)
    if args.emit_surfaces:
        for task in tasks:
            write_json(
                ROOT / "reports" / f"transfer_eval_{task['id']}.json",
                {
                    "policy": "project_theseus_transfer_eval_surface_v0",
                    "created_utc": report["created_utc"],
                    "frontier_family": "transfer_eval",
                    "pressure_card_id": f"transfer_eval_{task['id']}",
                    "accuracy": task["score"],
                    "floor": float(policy.get("floor") or 0.70),
                    "summary": {
                        "suite": f"transfer_{task['id']}",
                        "accuracy": task["score"],
                        "total_tool_calls": 0
                    },
                    "task": task,
                    "external_inference_calls": 0
                },
            )
    print(json.dumps(report, indent=2))
    return 0


def score_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("id") or "unknown")
    checks: list[dict[str, Any]] = []
    if task_id == "code_repair":
        compile_ok = run_compile("scripts/pressure_runner.py")
        learner = best_score_report(["reports/code_repair_learner_seed*.json", "reports/pressure_source_bigcodebench_seed*.json"])
        learner_score = float(learner.get("score") or learner.get("accuracy") or 0.0)
        checks.append(check("local_code_fixture_compiles", compile_ok, "scripts/pressure_runner.py"))
        checks.append(check("code_repair_learner_report", learner_score > 0.0, learner.get("path", "missing")))
        score = max(0.45 if compile_ok else 0.15, min(0.68, learner_score))
        intervention = "wire local code-repair candidate generation and unit-test verifier into the pressure runner"
    elif task_id == "tool_use":
        registry = read_json(ROOT / "reports" / "tool_registry.json")
        active = int((registry.get("registry_health") or {}).get("active") or 0)
        promoter = read_json(ROOT / "reports" / "loop_closure_tool_promoter.json")
        promoted = len(promoter.get("promoted", [])) if isinstance(promoter.get("promoted"), list) else 0
        checks.append(check("tool_registry_active", active >= 24, f"active={active}"))
        checks.append(check("loop_closure_promoter_ran", bool(promoter), f"promoted={promoted}"))
        score = min(0.72, 0.2 + active / 100.0 + min(0.12, promoted / 50.0))
        intervention = "promote high-recurrence loop closures into schema-verified tools"
    elif task_id == "web_task":
        card = read_json(ROOT / "benchmarks" / "cards" / "source_webarena.json")
        ready = card.get("status") == "adapter_smoke_passed"
        report = read_json(ROOT / "reports" / "pressure_source_webarena_seed1.json")
        report_score = number(get_path(report, ["summary", "accuracy"], 0.0))
        checks.append(check("webarena_card_smoked", ready, str(card.get("status"))))
        checks.append(check("webarena_pressure_report", bool(report), f"score={report_score:.3f}"))
        score = max(0.30 if ready else 0.10, min(0.52, report_score))
        intervention = "stand up a self-hosted no-account web fixture and add deterministic graders"
    elif task_id == "long_context_recovery":
        packets = read_json(ROOT / "reports" / "context_packet_ledger.json")
        active = int((packets.get("summary") or {}).get("active_packet_count") or 0)
        hard = read_json(ROOT / "reports" / "symliquid_hard_reference_report.json")
        hard_score = long_context_score(hard)
        checks.append(check("context_packets_active", active > 0, f"active={active}"))
        checks.append(check("hard_long_context_reference", hard_score > 0.0, f"score={hard_score:.3f}"))
        score = max(0.48 if active > 0 else 0.12, min(0.66, hard_score * 0.66))
        intervention = "add retrieval-after-compaction challenge cases with hidden-important conclusions"
    elif task_id == "rl_control":
        rl = read_json(ROOT / "reports" / "rl_benchmark_registry.json")
        passed = int((rl.get("summary") or {}).get("adapter_smoke_passed_rl_cards") or 0)
        control = best_score_report(
            [
                "reports/pressure_source_gym_pybullet_drones_seed*.json",
                "reports/pressure_source_pyflyt_seed*.json",
                "reports/pressure_source_mavsdk_python_seed*.json",
                "reports/rl_frontier_ocean_*_train.json",
            ]
        )
        control_score = float(control.get("score") or control.get("accuracy") or 0.0)
        checks.append(check("rl_adapter_smokes", passed >= 8, f"passed={passed}"))
        checks.append(check("local_control_pressure_report", control_score > 0.0, control.get("path", "missing")))
        score = max(min(0.62, 0.25 + passed / 50.0), min(0.76, control_score))
        intervention = "move smoke-passed RL cards into pressure_runner and train local policies"
    elif task_id == "self_debugging":
        watchdog = read_json(ROOT / "reports" / "autonomy_watchdog.json")
        proof = read_json(ROOT / "reports" / "teacher_self_edit_proof.json")
        success_rate = number(get_path(proof, ["summary", "success_rate"], 0.0))
        latest_ok = str(get_path(proof, ["summary", "latest_status"], "")).startswith("checks_passed")
        checks.append(check("watchdog_present", bool(watchdog), str(watchdog.get("trigger_state"))))
        checks.append(check("teacher_self_edit_trace_present", bool(proof), f"success_rate={success_rate:.3f}"))
        score = 0.10
        if watchdog:
            score = 0.42 if watchdog.get("trigger_state") != "RED" else 0.25
        if proof:
            score = max(score, min(0.68, 0.42 + success_rate * 0.65 + (0.10 if latest_ok else 0.0)))
        intervention = "make watchdog red conditions trigger bounded correction cycles and checkpoint repair"
    elif task_id == "voice_io":
        report = ensure_native_voice_report()
        voice_score = number(get_path(report, ["summary", "accuracy"], 0.0))
        report_checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        checks.append(
            check(
                "native_voice_policy_present",
                any(item.get("name") == "native_voice_policy_present" and item.get("passed") for item in report_checks if isinstance(item, dict)),
                "configs/native_voice_policy.json",
            )
        )
        checks.append(
            check(
                "head_router_owns_voice_io",
                get_path(report, ["summary", "voice_is_head_router_io"], False) is True,
                str(report.get("owner") or ""),
            )
        )
        checks.append(
            check(
                "native_audio_packet_contract",
                get_path(report, ["packet_contract", "ok"], False) is True,
                get_path(report, ["packet_contract", "summary"], "missing"),
            )
        )
        checks.append(
            check(
                "no_provider_stt_tts_inference",
                int(report.get("external_inference_calls") or 0) == 0,
                "external_inference_calls=0; installed STT/TTS modules do not count as capability",
            )
        )
        score = min(0.72, voice_score)
        intervention = "train native Theseus STT/TTS components from licensed audio data; do not satisfy this lane by installing speech inference packages"
    else:
        score = 0.05
        intervention = "define local fixture and scorer"
    return {
        "id": task_id,
        "capability": task.get("capability"),
        "runner": task.get("runner"),
        "risk_tier": task.get("risk_tier"),
        "score": round(score, 4),
        "checks": checks,
        "recommended_intervention": intervention,
    }


def best_score_report(patterns: list[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1.0
    for pattern in patterns:
        for raw in glob.glob(str(ROOT / pattern)):
            path = Path(raw)
            data = read_json(path)
            score = extract_score(data)
            if score > best_score:
                best_score = score
                best = {"path": rel(path), "score": score, "data": data}
    return best


def extract_score(data: Any) -> float:
    if not isinstance(data, dict):
        return 0.0
    for path in (
        ["summary", "accuracy"],
        ["summary", "score"],
        ["score"],
        ["evaluation", "score"],
        ["eval_accuracy"],
        ["accuracy"],
        ["normalized_eval_reward"],
        ["ceiling_adjusted_normalized_eval_reward"],
    ):
        value = get_path(data, path, None)
        if value is not None:
            return number(value)
    return 0.0


def long_context_score(report: Any) -> float:
    if not isinstance(report, dict):
        return 0.0
    breakdown = get_path(report, ["summary", "task_breakdown"], {})
    if isinstance(breakdown, dict):
        scores = [
            number(get_path(breakdown, [name, "accuracy"], 0.0))
            for name in ("long_context_retrieval", "long_context_role_filler")
            if isinstance(breakdown.get(name), dict)
        ]
        if scores:
            return sum(scores) / len(scores)
    return extract_score(report)


def ensure_native_voice_report() -> dict[str, Any]:
    path = ROOT / "reports" / "native_voice_io.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/native_voice_io.py",
            "--out",
            "reports/native_voice_io.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    data = read_json(path)
    if not data:
        data = {
            "summary": {"accuracy": 0.05, "voice_is_head_router_io": False},
            "checks": [check("native_voice_report_generated", False, result.stderr[-400:] or result.stdout[-400:])],
            "external_inference_calls": 0,
        }
    return data


def run_compile(path: str) -> bool:
    result = subprocess.run([sys.executable, "-m", "py_compile", path], cwd=ROOT, text=True, capture_output=True, timeout=30)
    return result.returncode == 0


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)[:400]}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math_is_finite(parsed):
        return 0.0
    return parsed


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
