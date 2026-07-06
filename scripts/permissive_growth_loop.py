#!/usr/bin/env python3
"""Run a governed permissive local growth cycle.

This loop is intentionally boring: refresh the gates, materialize the admitted
training pool, run the current strongest local comparator on a bounded slice,
and produce one concise report that says what actually trained and what stayed
blocked. It does not call a teacher or execute public calibration unless those
actions are explicitly requested through flags.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN_ROOT = REPORTS / "permissive_growth_loop"
DEFAULT_OUT = REPORTS / "permissive_growth_loop_report.json"
DEFAULT_MD = REPORTS / "permissive_growth_loop_report.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=0)
    parser.add_argument("--max-broad-train-rows", type=int, default=1024)
    parser.add_argument("--max-broad-eval-rows", type=int, default=96)
    parser.add_argument("--broad-epochs", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=71)
    parser.add_argument("--execute-public-calibration", action="store_true")
    parser.add_argument("--allow-teacher-live", action="store_true")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    if not args.execute:
        report = planned_report(args)
    else:
        report = run_loop(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(compact_report(report), indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


def planned_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_permissive_growth_loop_v1",
        "created_utc": now(),
        "ok": True,
        "execute": False,
        "planned_cycles": max(1, int(args.cycles)),
        "contract": contract(args),
        "next_command": (
            "python3 scripts/permissive_growth_loop.py --execute "
            f"--cycles {max(1, int(args.cycles))} "
            f"--max-broad-train-rows {max(1, int(args.max_broad_train_rows))} "
            f"--max-broad-eval-rows {max(1, int(args.max_broad_eval_rows))} "
            f"--broad-epochs {max(1, int(args.broad_epochs))}"
        ),
        "external_inference_calls": 0,
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cycles: list[dict[str, Any]] = []
    hard_stop: dict[str, Any] | None = None
    for index in range(max(1, int(args.cycles))):
        cycle = run_cycle(index, args)
        cycles.append(cycle)
        append_jsonl(RUN_ROOT / "ledger.jsonl", cycle)
        if cycle.get("hard_violation"):
            hard_stop = {"reason": "hard_no_cheat_violation", "cycle_index": index, "detail": cycle["hard_violation"]}
            break
        if cycle.get("failed_steps"):
            hard_stop = {"reason": "step_failed", "cycle_index": index, "failed_steps": cycle["failed_steps"]}
            break
        if index < int(args.cycles) - 1:
            time.sleep(max(0, int(args.sleep_seconds)))
    latest = cycles[-1] if cycles else {}
    report = {
        "policy": "project_theseus_permissive_growth_loop_v1",
        "created_utc": now(),
        "ok": hard_stop is None,
        "execute": True,
        "contract": contract(args),
        "cycles_requested": max(1, int(args.cycles)),
        "cycles_completed": len(cycles),
        "hard_stop": hard_stop,
        "latest_summary": latest.get("summary", {}),
        "improvement_summary": improvement_summary(cycles),
        "cycles": cycles,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    return report


def run_cycle(index: int, args: argparse.Namespace) -> dict[str, Any]:
    cycle_started = time.perf_counter()
    cycle_dir = RUN_ROOT / f"cycle_{index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.base_seed) + index
    steps: list[dict[str, Any]] = []

    steps.append(
        run_step(
            "external_inference_audit",
            [sys.executable, "scripts/external_inference_audit.py", "--no-scan-reports", "--out", rel(cycle_dir / "external_inference_audit.json")],
            cycle_dir,
            timeout_seconds=240,
        )
    )
    steps.append(
        run_step(
            "teacher_distillation_gate",
            [
                sys.executable,
                "scripts/teacher_distillation_gate.py",
                "--out",
                rel(cycle_dir / "teacher_distillation_gate.json"),
                "--markdown-out",
                rel(cycle_dir / "teacher_distillation_gate.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )
    steps.append(
        run_step(
            "training_data_admission",
            [
                sys.executable,
                "scripts/training_data_admission_v1.py",
                "--out",
                rel(cycle_dir / "training_data_admission_v1.json"),
                "--markdown-out",
                rel(cycle_dir / "training_data_admission_v1.md"),
                "--manifest-out",
                rel(cycle_dir / "training_data_admission_manifest_v1.json"),
            ],
            cycle_dir,
            timeout_seconds=900,
        )
    )
    public_cmd = [
        sys.executable,
        "scripts/operator_bounded_public_calibration.py",
        "--out",
        rel(cycle_dir / "operator_bounded_public_calibration.json"),
        "--markdown-out",
        rel(cycle_dir / "operator_bounded_public_calibration.md"),
    ]
    if args.execute_public_calibration:
        public_cmd.append("--execute")
    steps.append(run_step("public_benchmark_measurement_runner", public_cmd, cycle_dir, timeout_seconds=7200 if args.execute_public_calibration else 120))

    steps.append(
        run_step(
            "resource_governor",
            [sys.executable, "scripts/resource_governor.py", "--out", rel(cycle_dir / "resource_governor.json")],
            cycle_dir,
            timeout_seconds=120,
            allowed_returncodes={0, 1},
        )
    )
    if platform.system() == "Darwin":
        steps.append(
            run_step(
                "macos_mlx_environment_diagnosis",
                [
                    sys.executable,
                    "scripts/macos_mlx_environment_diagnosis.py",
                    "--out",
                    rel(cycle_dir / "macos_mlx_environment_diagnosis.json"),
                    "--markdown-out",
                    rel(cycle_dir / "macos_mlx_environment_diagnosis.md"),
                ],
                cycle_dir,
                timeout_seconds=180,
                allowed_returncodes={0, 1},
            )
        )
        steps.append(
            run_step(
                "macos_metal_production_route_readiness",
                [
                    sys.executable,
                    "scripts/macos_metal_production_route_readiness.py",
                    "--out",
                    rel(cycle_dir / "macos_metal_production_route_readiness.json"),
                    "--markdown-out",
                    rel(cycle_dir / "macos_metal_production_route_readiness.md"),
                ],
                cycle_dir,
                timeout_seconds=180,
                allowed_returncodes={0, 1},
            )
        )

    steps.append(
        run_step(
            "broad_capability_survival_lane_execute",
            [
                sys.executable,
                "scripts/broad_capability_survival_lane_run_v1.py",
                "--execute",
                "--seed",
                str(seed),
                "--max-train-rows",
                str(max(1, int(args.max_broad_train_rows))),
                "--max-eval-rows",
                str(max(1, int(args.max_broad_eval_rows))),
                "--epochs",
                str(max(1, int(args.broad_epochs))),
                "--out",
                rel(cycle_dir / "broad_capability_survival_lane_run_v1.json"),
                "--markdown-out",
                rel(cycle_dir / "broad_capability_survival_lane_run_v1.md"),
                "--train-out",
                rel(cycle_dir / "broad_capability_survival_lane_train.jsonl"),
                "--eval-out",
                rel(cycle_dir / "broad_capability_survival_lane_eval.jsonl"),
                "--config-out",
                rel(cycle_dir / "broad_capability_survival_lane_config.json"),
                "--candidate-manifest-out",
                rel(cycle_dir / "broad_capability_survival_lane_candidates.jsonl"),
                "--comparator-out",
                rel(cycle_dir / "broad_capability_survival_lane_comparator.json"),
                "--comparator-markdown-out",
                rel(cycle_dir / "broad_capability_survival_lane_comparator.md"),
            ],
            cycle_dir,
            timeout_seconds=3600,
        )
    )

    steps.append(
        run_step(
            "permissive_growth_mode_report",
            [
                sys.executable,
                "scripts/permissive_growth_mode_report.py",
                "--training-admission",
                rel(cycle_dir / "training_data_admission_v1.json"),
                "--teacher-gate",
                rel(cycle_dir / "teacher_distillation_gate.json"),
                "--public-runner",
                rel(cycle_dir / "operator_bounded_public_calibration.json"),
                "--growth-loop",
                rel(cycle_dir / "permissive_growth_loop_pending.json"),
                "--out",
                rel(cycle_dir / "permissive_growth_mode_report.json"),
                "--markdown-out",
                rel(cycle_dir / "permissive_growth_mode_report.md"),
            ],
            cycle_dir,
            timeout_seconds=120,
        )
    )

    cycle_report = summarize_cycle(index, seed, cycle_dir, steps, args, cycle_started)
    write_json(cycle_dir / "cycle_report.json", cycle_report)
    write_text(cycle_dir / "cycle_report.md", render_cycle_markdown(cycle_report))
    return cycle_report


def summarize_cycle(
    index: int,
    seed: int,
    cycle_dir: Path,
    steps: list[dict[str, Any]],
    args: argparse.Namespace,
    started: float,
) -> dict[str, Any]:
    admission = read_json(cycle_dir / "training_data_admission_v1.json")
    teacher = read_json(cycle_dir / "teacher_distillation_gate.json")
    public_runner = read_json(cycle_dir / "operator_bounded_public_calibration.json")
    broad = read_json(cycle_dir / "broad_capability_survival_lane_run_v1.json")
    growth = read_json(cycle_dir / "permissive_growth_mode_report.json")
    external = read_json(cycle_dir / "external_inference_audit.json")
    resource = read_json(cycle_dir / "resource_governor.json")
    mlx = read_json(cycle_dir / "macos_mlx_environment_diagnosis.json")
    metal = read_json(cycle_dir / "macos_metal_production_route_readiness.json")

    admission_summary = object_field(admission, "summary")
    teacher_summary = object_field(teacher, "summary")
    public_summary = object_field(public_runner, "summary")
    broad_summary = object_field(broad, "summary")
    broad_comparator = object_field(broad, "comparator_report")
    broad_comparisons = object_field(broad_comparator, "comparisons")
    resource_summary = object_field(resource, "summary")
    failed_steps = [
        {"label": step.get("label"), "returncode": step.get("returncode")}
        for step in steps
        if not step.get("ok")
    ]
    no_cheat = no_cheat_summary(admission_summary, teacher_summary, public_summary, broad_summary, external)
    hard_violation = {key: value for key, value in no_cheat.items() if key.endswith("_violation") and value}
    summary = {
        "growth_report_state": growth.get("trigger_state"),
        "admitted_training_source_count": admission_summary.get("allowed_training_source_count"),
        "admitted_open_public_source_count": admission_summary.get("admitted_open_public_source_count"),
        "admitted_open_public_row_count": admission_summary.get("admitted_open_public_row_count"),
        "admitted_dogfood_source_count": count_sources(admission, "dogfood_metadata"),
        "admitted_dogfood_row_count": sum_source_rows(admission, "dogfood_metadata"),
        "teacher_training_enabled_by_policy": teacher_summary.get("governed_teacher_training_rows_enabled_by_policy"),
        "teacher_distillation_allowed": teacher_summary.get("distillation_allowed"),
        "teacher_accepted_rows": teacher_summary.get("teacher_accepted_rows"),
        "teacher_accepted_row_share": teacher_summary.get("teacher_accepted_row_share"),
        "teacher_share_cap": teacher_summary.get("teacher_share_cap"),
        "public_runner_authorization_mode": public_summary.get("authorization_mode"),
        "public_runner_run_registry_allowed": public_summary.get("run_registry_allowed"),
        "public_runner_would_execute": public_summary.get("would_execute"),
        "public_runner_executed": public_summary.get("executed"),
        "broad_train_rows": broad_summary.get("train_rows"),
        "broad_eval_rows": broad_summary.get("eval_rows"),
        "broad_comparator_state": broad_summary.get("comparator_state"),
        "architecture_winner": broad_summary.get("winner_by_sts_on"),
        "transformer_sts_on_pass_rate": broad_summary.get("transformer_sts_on_pass_rate"),
        "symliquid_sts_on_pass_rate": broad_summary.get("symliquid_sts_on_pass_rate"),
        "symliquid_minus_transformer": broad_summary.get("symliquid_minus_transformer"),
        "architecture_comparisons": broad_comparisons,
        "resource_execution_owner": resource_summary.get("execution_owner") or resource_summary.get("resource_execution_owner"),
        "mac_machine": platform.machine(),
        "mlx_state": mlx.get("trigger_state"),
        "metal_state": metal.get("trigger_state"),
        "no_cheat": no_cheat,
        "allow_teacher_live": bool(args.allow_teacher_live),
        "public_calibration_execute_requested": bool(args.execute_public_calibration),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return {
        "policy": "project_theseus_permissive_growth_loop_cycle_v1",
        "created_utc": now(),
        "cycle_index": index,
        "seed": seed,
        "cycle_dir": rel(cycle_dir),
        "ok": not failed_steps and not hard_violation,
        "summary": summary,
        "failed_steps": failed_steps,
        "hard_violation": hard_violation or None,
        "steps": steps,
    }


def no_cheat_summary(
    admission_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
    public_summary: dict[str, Any],
    broad_summary: dict[str, Any],
    external_report: dict[str, Any],
) -> dict[str, Any]:
    runtime_external_calls = (
        int_number(admission_summary.get("external_inference_calls"))
        + int_number(broad_summary.get("external_inference_calls"))
        + int_number(public_summary.get("external_inference_calls"))
        + int_number(get_path(external_report, ["summary", "total_violations"], 0))
    )
    teacher_training_external_calls = int_number(teacher_summary.get("external_inference_calls"))
    public_benchmark_training = bool(admission_summary.get("public_benchmark_training_allowed")) or bool(
        admission_summary.get("public_benchmark_payload_admitted")
    )
    return {
        "fallback_return_count": int_number(broad_summary.get("fallback_return_count")),
        "public_benchmark_training_allowed": bool(admission_summary.get("public_benchmark_training_allowed")),
        "public_benchmark_payload_admitted": bool(admission_summary.get("public_benchmark_payload_admitted")),
        "public_benchmark_training_rows": int_number(broad_summary.get("public_benchmark_training_rows")),
        "runtime_external_inference_calls": runtime_external_calls,
        "teacher_training_external_inference_calls": teacher_training_external_calls,
        "teacher_runtime_serving_forbidden": bool(teacher_summary.get("runtime_external_tokens_forbidden")),
        "teacher_used_in_broad_training": bool(broad_summary.get("teacher_used")),
        "fallback_return_violation": int_number(broad_summary.get("fallback_return_count")) != 0,
        "public_eval_leakage_violation": public_benchmark_training or int_number(broad_summary.get("public_benchmark_training_rows")) != 0,
        "runtime_external_inference_violation": runtime_external_calls != 0,
    }


def run_step(
    label: str,
    command: list[str],
    cycle_dir: Path,
    *,
    timeout_seconds: int,
    allowed_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    allowed = allowed_returncodes or {0}
    started = time.perf_counter()
    stdout_path = cycle_dir / f"{label}.stdout.txt"
    stderr_path = cycle_dir / f"{label}.stderr.txt"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
        stdout_path.write_text((result.stdout or "")[-20000:], encoding="utf-8")
        stderr_path.write_text((result.stderr or "")[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "ok": result.returncode in allowed,
            "returncode": result.returncode,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(stdout_path),
            "stderr_log": rel(stderr_path),
            "stdout_tail": (result.stdout or "")[-1000:],
            "stderr_tail": (result.stderr or "")[-1000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        stdout_path.write_text(stdout[-20000:], encoding="utf-8")
        stderr_path.write_text(stderr[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "ok": False,
            "returncode": 124,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(stdout_path),
            "stderr_log": rel(stderr_path),
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
        }


def count_sources(admission: dict[str, Any], source_kind: str) -> int:
    return sum(
        1
        for row in admission.get("source_admissions", [])
        if isinstance(row, dict) and row.get("allowed_for_training") and row.get("source_kind") == source_kind
    )


def sum_source_rows(admission: dict[str, Any], source_kind: str) -> int:
    return sum(
        int_number(row.get("row_count"))
        for row in admission.get("source_admissions", [])
        if isinstance(row, dict) and row.get("allowed_for_training") and row.get("source_kind") == source_kind
    )


def improvement_summary(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    if not cycles:
        return {}
    first = object_field(cycles[0], "summary")
    last = object_field(cycles[-1], "summary")
    keys = [
        "admitted_open_public_row_count",
        "admitted_dogfood_row_count",
        "broad_train_rows",
        "transformer_sts_on_pass_rate",
        "symliquid_sts_on_pass_rate",
        "symliquid_minus_transformer",
    ]
    return {
        key: {
            "first": first.get(key),
            "last": last.get(key),
            "delta": round(number(last.get(key)) - number(first.get(key)), 6),
        }
        for key in keys
    }


def contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "public_open_training": "allowed_when_license_provenance_hash_and_decontamination_pass",
        "public_eval_payloads": "heldout; prompts/tests/solutions/traces/answer_templates_not_training_rows",
        "public_calibration": "governed run registry; execute only when --execute-public-calibration is set",
        "teacher": "proposal/distillation enabled by policy; live call only when --allow-teacher-live is set; runtime serving forbidden",
        "teacher_live_requested": bool(getattr(args, "allow_teacher_live", False)),
        "public_calibration_execute_requested": bool(getattr(args, "execute_public_calibration", False)),
        "architecture_selection": "run matched broad survival comparator and report winner without protecting losing substrate",
        "mac_execution": "resource governor plus MLX/Metal readiness on Darwin",
        "fallback_returns": "forbidden",
    }


def render_cycle_markdown(cycle: dict[str, Any]) -> str:
    summary = object_field(cycle, "summary")
    lines = [
        f"# Permissive Growth Cycle {cycle.get('cycle_index')}",
        "",
        f"- ok: `{cycle.get('ok')}`",
        f"- admitted open/public rows: `{summary.get('admitted_open_public_row_count')}`",
        f"- admitted dogfood rows: `{summary.get('admitted_dogfood_row_count')}`",
        f"- teacher accepted rows/share/cap: `{summary.get('teacher_accepted_rows')}` / `{summary.get('teacher_accepted_row_share')}` / `{summary.get('teacher_share_cap')}`",
        f"- public run registry allowed/would execute: `{summary.get('public_runner_run_registry_allowed')}` / `{summary.get('public_runner_would_execute')}`",
        f"- broad train/eval rows: `{summary.get('broad_train_rows')}` / `{summary.get('broad_eval_rows')}`",
        f"- architecture winner: `{summary.get('architecture_winner')}`",
        f"- transformer vs SymLiquid: `{summary.get('transformer_sts_on_pass_rate')}` / `{summary.get('symliquid_sts_on_pass_rate')}`",
        f"- no-cheat: `{summary.get('no_cheat')}`",
    ]
    if cycle.get("failed_steps"):
        lines.append(f"- failed steps: `{cycle.get('failed_steps')}`")
    if cycle.get("hard_violation"):
        lines.append(f"- hard violation: `{cycle.get('hard_violation')}`")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    latest = object_field(report, "latest_summary")
    lines = [
        "# Permissive Growth Loop",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- execute: `{report.get('execute')}`",
        f"- cycles completed: `{report.get('cycles_completed')}` / `{report.get('cycles_requested')}`",
        f"- admitted open/public rows: `{latest.get('admitted_open_public_row_count')}`",
        f"- admitted dogfood rows: `{latest.get('admitted_dogfood_row_count')}`",
        f"- teacher accepted rows/share/cap: `{latest.get('teacher_accepted_rows')}` / `{latest.get('teacher_accepted_row_share')}` / `{latest.get('teacher_share_cap')}`",
        f"- public run registry allowed/would execute: `{latest.get('public_runner_run_registry_allowed')}` / `{latest.get('public_runner_would_execute')}`",
        f"- broad train/eval rows: `{latest.get('broad_train_rows')}` / `{latest.get('broad_eval_rows')}`",
        f"- architecture winner: `{latest.get('architecture_winner')}`",
        f"- transformer vs SymLiquid: `{latest.get('transformer_sts_on_pass_rate')}` / `{latest.get('symliquid_sts_on_pass_rate')}`",
        f"- hard stop: `{report.get('hard_stop')}`",
        "",
        "## Improvement Summary",
    ]
    for key, row in object_field(report, "improvement_summary").items():
        lines.append(f"- `{key}`: {row}")
    return "\n".join(lines) + "\n"


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "ok": report.get("ok"),
        "execute": report.get("execute"),
        "cycles_completed": report.get("cycles_completed"),
        "hard_stop": report.get("hard_stop"),
        "latest_summary": report.get("latest_summary"),
        "external_inference_calls": report.get("external_inference_calls", 0),
    }


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key) if isinstance(row, dict) else None
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_number(value: Any) -> int:
    return int(number(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
