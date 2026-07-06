"""Runnable command-contract executor for VIEA.

The executor turns a first-class command contract into a deterministic route
plan, specialist work packets, digital-runtime packets, verification gates,
residuals, and feedback entries. It is intentionally conservative: high-risk
matter/chip/robotic targets remain planning-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_BUNDLE = REPORTS / "reality_manipulator" / "latest_world"
DEFAULT_DB = REPORTS / "viea_artifact_kernel.sqlite"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default="reports/viea_command_executor.json")
    parser.add_argument("--markdown-out", default="reports/viea_command_executor.md")
    parser.add_argument("--packets-dir", default="reports/digital_runtime/latest_execution")
    args = parser.parse_args()

    bundle_dir = resolve(args.bundle_dir)
    command = read_json(bundle_dir / "command_contract.json")
    world = read_json(bundle_dir / "world.json")
    lifecycle = read_json(bundle_dir / "specialist_lifecycle.json")
    workflow = read_json(bundle_dir / "workflow_tool_metrics.json")
    if not command:
        payload = failure_payload("missing_command_contract", bundle_dir)
    else:
        payload = execute_contract(
            command=command,
            world=world,
            lifecycle=lifecycle,
            workflow=workflow,
            bundle_dir=bundle_dir,
            packets_dir=resolve(args.packets_dir),
            db_path=resolve(args.db),
        )
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def execute_contract(
    *,
    command: dict[str, Any],
    world: dict[str, Any],
    lifecycle: dict[str, Any],
    workflow: dict[str, Any],
    bundle_dir: Path,
    packets_dir: Path,
    db_path: Path,
) -> dict[str, Any]:
    packets_dir.mkdir(parents=True, exist_ok=True)
    arms = lifecycle.get("arms") if isinstance(lifecycle.get("arms"), list) else []
    compile_targets = world.get("compile_targets") if isinstance(world.get("compile_targets"), list) else []
    route_plan = build_route_plan(command, arms, compile_targets)
    specialist_calls = build_specialist_calls(command, arms, route_plan)
    runtime_packets = write_runtime_packets(
        packets_dir=packets_dir,
        command=command,
        world=world,
        route_plan=route_plan,
        specialist_calls=specialist_calls,
    )
    gates = verify_execution(command, route_plan, specialist_calls, runtime_packets, workflow)
    residuals = [
        {
            "id": stable_id("residual", gate["gate"], gate["detail"]),
            "source": "viea_command_executor",
            "failure_type": gate["gate"],
            "severity": gate["severity"],
            "cluster": "command_execution",
            "promotion_status": "blocks_execution" if gate["severity"] == "hard" else "track",
        }
        for gate in gates
        if not gate["passed"]
    ]
    trigger_state = "GREEN"
    if any((not gate["passed"]) and gate["severity"] == "hard" for gate in gates):
        trigger_state = "RED"
    elif residuals or any(call["status"] != "completed" for call in specialist_calls):
        trigger_state = "YELLOW"
    db_write = write_execution_to_db(
        db_path=db_path,
        command=command,
        route_plan=route_plan,
        specialist_calls=specialist_calls,
        runtime_packets=runtime_packets,
        gates=gates,
        residuals=residuals,
    )
    return {
        "policy": "project_theseus_viea_command_executor_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "command_contract_id": command.get("id"),
        "world_id": world.get("id"),
        "route_plan": route_plan,
        "specialist_calls": specialist_calls,
        "runtime_packets": runtime_packets,
        "verification_gates": gates,
        "residuals": residuals,
        "db_write": db_write,
        "artifacts": {
            "packets_dir": rel_or_abs(packets_dir),
            "code_patch_packet": rel_or_abs(packets_dir / "code_patch_packet.json"),
            "test_packet": rel_or_abs(packets_dir / "test_packet.json"),
            "release_manifest": rel_or_abs(packets_dir / "release_manifest.json"),
            "rollback_plan": rel_or_abs(packets_dir / "rollback_plan.md"),
            "repo_repair_trace": rel_or_abs(packets_dir / "repo_repair_trace.jsonl"),
            "dashboard_actions": rel_or_abs(packets_dir / "dashboard_actions.json"),
        },
        "score_semantics": "command execution substrate only; not student learning proof",
        "external_inference_calls": 0,
    }


def build_route_plan(command: dict[str, Any], arms: list[dict[str, Any]], compile_targets: list[str]) -> list[dict[str, Any]]:
    arm_names = [str(arm.get("name")) for arm in arms if isinstance(arm, dict)]
    required = [
        ("validate_command", "head_router"),
        ("load_artifact_context", "memory_arm"),
        ("audit_claims", "claim_auditor"),
        ("red_team_contract", "skeptic_arm"),
        ("plan_digital_runtime", "coding_arm" if "coding_arm" in arm_names else "head_router"),
        ("verify_runtime_gates", "safety_arm"),
        ("write_feedback", "benchmark_arm" if "benchmark_arm" in arm_names else "head_router"),
    ]
    route_plan = []
    for index, (stage, arm) in enumerate(required, start=1):
        route_plan.append(
            {
                "step": index,
                "stage": stage,
                "arm": arm,
                "status": "planned",
                "permission_envelope": permission_for_arm(arm),
                "input_artifacts": [str(command.get("id"))],
                "output_artifacts": [stable_id("artifact", command.get("id"), stage)],
            }
        )
    for target in compile_targets:
        if str(target) in {"chip", "matter", "robotic"}:
            route_plan.append(
                {
                    "step": len(route_plan) + 1,
                    "stage": f"{target}_planning_boundary",
                    "arm": "safety_arm",
                    "status": "planning_only",
                    "permission_envelope": permission_for_arm("safety_arm"),
                    "input_artifacts": [str(command.get("id"))],
                    "output_artifacts": [stable_id("artifact", command.get("id"), target, "boundary")],
                }
            )
    return route_plan


def build_specialist_calls(command: dict[str, Any], arms: list[dict[str, Any]], route_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arm_by_name = {str(arm.get("name")): arm for arm in arms if isinstance(arm, dict)}
    calls = []
    for route in route_plan:
        arm = arm_by_name.get(str(route["arm"]), {"name": route["arm"], "scope": "fallback route"})
        planning_only = route["status"] == "planning_only"
        calls.append(
            {
                "id": stable_id("call", command.get("id"), route["stage"], route["arm"]),
                "stage": route["stage"],
                "arm": route["arm"],
                "scope": arm.get("scope"),
                "input_contract": "command_contract_plus_world_bundle",
                "output_contract": "typed_artifact_or_residual",
                "status": "blocked_pending_gate" if planning_only else "completed",
                "side_effects": "none" if planning_only else "local_report_write_only",
                "output_artifacts": route["output_artifacts"],
                "notes": "High-risk runtime remains planning-only." if planning_only else "Deterministic local execution packet emitted.",
            }
        )
    return calls


def write_runtime_packets(
    *,
    packets_dir: Path,
    command: dict[str, Any],
    world: dict[str, Any],
    route_plan: list[dict[str, Any]],
    specialist_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packets = {
        "code_patch_packet.json": {
            "kind": "code_patch_packet",
            "command_contract_id": command.get("id"),
            "allowed_side_effects": "local_patch_and_tests_only",
            "patch_policy": "minimal_verified_patch_no_public_benchmark_answers",
            "write_scope": "operator_approved_repo_scope",
            "requires_tests": True,
        },
        "test_packet.json": {
            "kind": "test_packet",
            "command_contract_id": command.get("id"),
            "test_policy": "same_seed_before_after_and_no_regressions",
            "public_benchmarks": "calibration_only",
            "private_hidden_tests": "allowed_for_private_curriculum_only",
        },
        "release_manifest.json": {
            "kind": "digital_release_manifest",
            "command_contract_id": command.get("id"),
            "world": world.get("name"),
            "route_plan_steps": len(route_plan),
            "specialist_calls": len(specialist_calls),
            "release_state": "internal_draft",
        },
        "dashboard_actions.json": {
            "kind": "dashboard_actions",
            "actions": [
                "show_viea_kernel_state",
                "show_command_executor_state",
                "show_broad_transfer_closure",
                "show_feedback_ratchet",
            ],
        },
    }
    rollback_text = "\n".join(
        [
            "# Digital Runtime Rollback Plan",
            "",
            "- Do not execute high-risk runtime targets from this packet.",
            "- Revert any local patch through normal git review if tests regress.",
            "- Preserve residuals and feedback before retrying.",
            "",
        ]
    )
    trace_rows = [
        {
            "event": "viea_command_route",
            "command_contract_id": command.get("id"),
            "stage": call["stage"],
            "arm": call["arm"],
            "status": call["status"],
            "external_inference_calls": 0,
        }
        for call in specialist_calls
    ]
    for name, packet in packets.items():
        write_json(packets_dir / name, packet)
    write_text(packets_dir / "rollback_plan.md", rollback_text)
    write_jsonl(packets_dir / "repo_repair_trace.jsonl", trace_rows)
    out = []
    for name in list(packets) + ["rollback_plan.md", "repo_repair_trace.jsonl"]:
        out.append({"file": name, "path": rel_or_abs(packets_dir / name), "exists": (packets_dir / name).exists()})
    return out


def verify_execution(
    command: dict[str, Any],
    route_plan: list[dict[str, Any]],
    specialist_calls: list[dict[str, Any]],
    runtime_packets: list[dict[str, Any]],
    workflow: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        gate("command_contract_loaded", bool(command.get("id") and command.get("fields")), "hard", command.get("id")),
        gate("route_plan_written", len(route_plan) >= 5, "hard", f"steps={len(route_plan)}"),
        gate("specialist_calls_written", len(specialist_calls) >= 5, "hard", f"calls={len(specialist_calls)}"),
        gate("high_risk_calls_planning_only", all(call["status"] != "completed" for call in specialist_calls if call["stage"].endswith("_planning_boundary")), "hard", "matter/chip/robotic remain planning-only"),
        gate("digital_packets_written", all(packet["exists"] for packet in runtime_packets), "hard", runtime_packets),
        gate("workflow_metrics_available", bool(workflow.get("metrics")), "soft", workflow.get("metrics", {})),
        gate("external_inference_zero", True, "hard", "local deterministic executor"),
    ]


def write_execution_to_db(
    *,
    db_path: Path,
    command: dict[str, Any],
    route_plan: list[dict[str, Any]],
    specialist_calls: list[dict[str, Any]],
    runtime_packets: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
) -> dict[str, Any]:
    if not db_path.exists():
        return {"attempted": False, "reason": "artifact_kernel_db_missing"}
    conn = sqlite3.connect(str(db_path))
    stamp = now()
    try:
        for call in specialist_calls:
            object_id = str(call["id"])
            conn.execute(
                """
                INSERT INTO objects (id, type, title, content_json, source_path, provenance_json, version, verification_state, release_state, created_utc, updated_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET content_json=excluded.content_json, verification_state=excluded.verification_state, updated_utc=excluded.updated_utc
                """,
                (
                    object_id,
                    "Artifact",
                    f"executor call {call['stage']}",
                    json.dumps(call, sort_keys=True),
                    "reports/viea_command_executor.json",
                    json.dumps({"command_contract_id": command.get("id")}, sort_keys=True),
                    "0.1.0",
                    call["status"],
                    "internal",
                    stamp,
                    stamp,
                ),
            )
        feedback_id = stable_id("feedback", command.get("id"), "executor")
        feedback = {
            "route_plan": route_plan,
            "runtime_packets": runtime_packets,
            "gates": gates,
            "residuals": residuals,
        }
        conn.execute(
            """
            INSERT INTO objects (id, type, title, content_json, source_path, provenance_json, version, verification_state, release_state, created_utc, updated_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET content_json=excluded.content_json, verification_state=excluded.verification_state, updated_utc=excluded.updated_utc
            """,
            (
                feedback_id,
                "Feedback",
                "command executor feedback",
                json.dumps(feedback, sort_keys=True),
                "reports/viea_command_executor.json",
                json.dumps({"command_contract_id": command.get("id")}, sort_keys=True),
                "0.1.0",
                "planned",
                "internal",
                stamp,
                stamp,
            ),
        )
        conn.commit()
        return {"attempted": True, "written": len(specialist_calls) + 1, "db": rel_or_abs(db_path)}
    finally:
        conn.close()


def permission_for_arm(arm: str) -> dict[str, Any]:
    high = arm in {"safety_arm", "cad_arm", "chip_arm", "fabrication_arm"}
    return {
        "memory_access": "world_local_plus_approved_imports",
        "tool_access": "bounded_registered_tools",
        "runtime_access": "planning_only" if high else "local_artifact_runtime",
        "side_effect_allowance": "none" if high else "local_report_write_only",
        "risk_tier": "high" if high else "medium",
    }


def failure_payload(reason: str, bundle_dir: Path) -> dict[str, Any]:
    return {
        "policy": "project_theseus_viea_command_executor_v1",
        "created_utc": now(),
        "trigger_state": "RED",
        "reason": reason,
        "bundle_dir": rel_or_abs(bundle_dir),
        "external_inference_calls": 0,
    }


def gate(name: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "detail": detail}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VIEA Command Executor",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- command_contract_id: `{payload.get('command_contract_id', '')}`",
        f"- route_steps: `{len(payload.get('route_plan', []))}`",
        f"- specialist_calls: `{len(payload.get('specialist_calls', []))}`",
        "",
        "## Gates",
        "",
    ]
    for row in payload.get("verification_gates", []):
        lines.append(f"- {'PASS' if row['passed'] else 'FAIL'} `{row['gate']}` ({row['severity']}): {row['detail']}")
    lines.extend(["", "## Runtime Packets", ""])
    for packet in payload.get("runtime_packets", []):
        lines.append(f"- `{packet['path']}`")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix.lower()}_{digest}"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
