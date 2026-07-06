"""Shared helpers for legacy port mechanism reports."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "legacy_port_policy.json"
__all__ = [
    "ROOT",
    "REPORTS",
    "DEFAULT_POLICY",
    "aggregate_summary",
    "node",
    "world_model_slot",
    "readiness_lane",
    "campaign_node",
    "runtime_request",
    "maybe_rows",
    "count_values",
    "count_glob",
    "project_inventory",
    "critical_path",
    "check",
    "scenario",
    "adapter",
    "interference_matrix",
    "world_adapter",
    "capability_flags_for_world",
    "runtime_identity",
    "artifact_verdict",
    "has_raw_outputs",
    "failed_gates",
    "trace_quality",
    "trace_utility",
    "write_report",
    "write_markdown",
    "read_json",
    "write_json",
    "read_jsonl_tail",
    "sha256_file",
    "stable_hash",
    "resolve",
    "path_exists",
    "rel",
    "get_path",
    "clamp01",
    "safe",
    "now",
    "today",
]


def aggregate_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    states = {}
    red = 0
    yellow = 0
    for key, payload in reports.items():
        state = payload.get("trigger_state") or payload.get("status") or "UNKNOWN"
        states[key] = state
        if state == "RED":
            red += 1
        elif state in {"YELLOW", "DEGRADED"}:
            yellow += 1
    return {
        "mechanisms": len(reports),
        "red": red,
        "yellow_or_degraded": yellow,
        "green_or_ready": len(reports) - red - yellow,
        "states": states,
        "top_blocker": next((key for key, value in states.items() if value == "RED"), None),
    }


def node(node_id: str, deps: list[str], ready: bool, goal: str, cost: str) -> dict[str, Any]:
    return {"id": node_id, "deps": deps, "ready": ready, "goal": goal, "cost": cost}


def world_model_slot(
    lane: str,
    active_family: str,
    prediction_error: float,
    observation_fields: list[str],
    action_candidates: list[str],
) -> dict[str, Any]:
    active = lane == active_family or (lane == "tool_use" and active_family.startswith("transfer_tool"))
    return {
        "lane": lane,
        "status": "active" if active else "planned",
        "observation_fields": observation_fields,
        "action_candidates": action_candidates,
        "prediction_error_proxy": prediction_error if active else 0.0,
        "resource_bound": True,
        "trainable_local_state": True,
    }


def readiness_lane(lane: str, state: dict[str, Any], has_reward: bool, has_done: bool, has_replay: bool) -> dict[str, Any]:
    contamination_gate = get_path(state, ["external_inference_audit", "ok"], True) is not False
    source_contract = bool(state.get("benchmark_ledger")) or bool(state.get("training_data_inventory"))
    reward_ok = has_reward or lane in {"language", "voice"}
    ready = bool(source_contract and reward_ok and has_done and has_replay and contamination_gate)
    return {
        "lane": lane,
        "ready": ready,
        "asset_rule_separation": source_contract,
        "reward_contract": has_reward,
        "done_or_boundary_contract": has_done,
        "deterministic_replay": has_replay,
        "contamination_gate": contamination_gate,
        "external_inference_forbidden": True,
    }


def campaign_node(node_id: str, deps: list[str], status: str, refs: list[Any]) -> dict[str, Any]:
    return {
        "id": node_id,
        "deps": deps,
        "status": status,
        "refs": [str(item) for item in refs if item],
    }


def runtime_request(
    request_id: str,
    allowed_modes: list[str],
    nvidia: dict[str, Any],
    mlx: dict[str, Any],
    cpu: dict[str, Any],
) -> dict[str, Any]:
    if "cuda" in allowed_modes and nvidia.get("available"):
        selected = "cuda"
        reason = "nvidia_available"
    elif "mlx" in allowed_modes and mlx.get("available"):
        selected = "mlx"
        reason = "mlx_available"
    elif "cpu" in allowed_modes and cpu.get("logical_cores"):
        selected = "cpu"
        reason = "cpu_fallback_available"
    elif "hive" in allowed_modes:
        selected = "hive"
        reason = "defer_to_authenticated_hive"
    else:
        selected = "blocked"
        reason = "no_allowed_runtime_available"
    return {
        "request_id": request_id,
        "allowed_modes": allowed_modes,
        "selected_mode": selected,
        "fallback_reason": reason,
        "bad_fallback": selected not in allowed_modes,
    }


def maybe_rows(payload: Any, keys: list[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value[:8]):
            return [row for row in value if isinstance(row, dict)]
    return []


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def count_glob(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def project_inventory(path: Path) -> dict[str, Any]:
    skip = {
        ".cache",
        ".cargo",
        ".dart_tool",
        ".git",
        ".gradle",
        ".next",
        ".tmp",
        ".venv",
        "__MACOSX",
        "__pycache__",
        "build",
        "cache",
        "data",
        "datasets",
        "deprecated",
        "dist",
        "logs",
        "node_modules",
        "outputs",
        "Pods",
        "reports",
        "site-packages",
        "target",
        "third_party",
        "tmp",
        "vendor",
        "vendors",
        "venv",
    }
    file_count = 0
    doc_count = 0
    source_count = 0
    key_docs: list[str] = []
    if path.exists():
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name not in skip and not name.endswith(".xcbuilddata")]
            for name in filenames:
                file_count += 1
                suffix = Path(name).suffix.lower()
                rel_path = Path(dirpath, name)
                rel_text = str(rel_path.relative_to(path))
                if suffix in {".md", ".txt", ".toml", ".json", ".yaml", ".yml"}:
                    doc_count += 1
                if suffix in {".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".swift", ".kt", ".sh"}:
                    source_count += 1
                lower = rel_text.lower()
                if len(key_docs) < 24 and (
                    "readme" in lower
                    or "whitepaper" in lower
                    or "architecture" in lower
                    or "contract" in lower
                    or "benchmark" in lower
                    or "dataset" in lower
                    or "environment" in lower
                    or "manifest" in lower
                ):
                    key_docs.append(rel_text)
    return {
        "project": path.name,
        "path": str(path),
        "exists": path.exists(),
        "file_count": file_count,
        "doc_like_count": doc_count,
        "source_like_count": source_count,
        "key_docs_sample": key_docs,
    }


def critical_path(nodes: list[dict[str, Any]]) -> list[str]:
    ready = [row["id"] for row in nodes if row.get("ready") and row.get("cost") != "low"]
    order = ["proxy_truth_audit", "coherence_delirium", "taskspell_lock", "active_frontier_pressure", "world_job_runtime", "adapter_bank_transfer", "trace_fabric_exchange", "teacher_self_edit", "checkpoint_and_backup"]
    return [item for item in order if item in ready]


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)}


def scenario(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "evidence": evidence}


def adapter(adapter_id: str, source_lane: str, target_lane: str, features: list[str], state: dict[str, Any]) -> dict[str, Any]:
    active_family = get_path(state, ["arm_transfer_plan", "summary", "frontier_family"], "")
    status = "active" if active_family == source_lane else "planned"
    risk = 0.015 if status == "active" else 0.025
    return {
        "id": adapter_id,
        "source_lane": source_lane,
        "target_lane": target_lane,
        "rank": 8,
        "status": status,
        "features": features,
        "interference_risk": risk,
        "evidence": ["regression floors required", "router loads by lane and task"],
    }


def interference_matrix(adapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left in adapters:
        for right in adapters:
            if left["id"] >= right["id"]:
                continue
            overlap = len(set(left.get("features", [])) & set(right.get("features", [])))
            rows.append(
                {
                    "a": left["id"],
                    "b": right["id"],
                    "estimated_interference": round(0.005 + 0.01 * overlap, 4),
                    "requires_ablation": overlap > 0,
                }
            )
    return rows


def world_adapter(world_type: str, adapter_kind: str, status: str, reason: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "world_type": world_type,
        "adapter_id": f"adapter_{world_type}",
        "adapter_kind": adapter_kind,
        "status": status,
        "reason": reason,
        "deterministic_replay_id": stable_hash({"world_type": world_type, "adapter_kind": adapter_kind})[:16],
        "capability_flags": capability_flags_for_world(world_type, state),
    }


def capability_flags_for_world(world_type: str, state: dict[str, Any]) -> list[str]:
    if world_type == "drone_rl":
        return ["sim_reset", "step", "reward", "residuals", "no_live_hardware"]
    if world_type == "emulator_rl":
        return ["byo_rom_metadata", "episode_trace_planned", "no_rom_download"]
    if world_type == "web_agent_local":
        return ["local_service", "no_real_account", "residuals"]
    if world_type == "coding_local_sandbox":
        return ["unit_tests", "sandbox", "patch_dry_run"]
    return ["jsonl_endpoint_contract", "approval_required"]


def runtime_identity(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    identity = {}
    for key in ["policy", "methodology", "frontier_family", "card_id", "runner_family", "seed", "created_utc"]:
        if key in payload:
            identity[key] = payload.get(key)
    if "summary" in payload and isinstance(payload["summary"], dict):
        identity["summary_suite"] = payload["summary"].get("suite")
    eval_payload = payload.get("eval")
    eval_summary = eval_payload.get("summary") if isinstance(eval_payload, dict) else {}
    if isinstance(eval_summary, dict):
        if eval_summary.get("suite"):
            identity.setdefault("summary_suite", eval_summary.get("suite"))
        if eval_summary.get("model_id"):
            identity.setdefault("model_id", eval_summary.get("model_id"))
        if eval_summary.get("mode"):
            identity.setdefault("runner_family", eval_summary.get("mode"))
        if eval_summary.get("cases") is not None:
            identity.setdefault("case_count", eval_summary.get("cases"))
    if payload.get("input_path") and payload.get("eval_input_path") and payload.get("eval"):
        identity.setdefault("policy", "project_theseus_babylm_probe_train_report_v1")
        identity.setdefault("methodology", "local_babylm_probe_train_eval")
        identity.setdefault("frontier_family", "babylm_mutated" if "mutated" in str(payload.get("eval_input_path")).lower() else "babylm_local")
        identity.setdefault("seed", payload.get("eval_seed") if payload.get("eval_seed") is not None else payload.get("train_seed"))
    return identity


def artifact_verdict(artifact_key: str, path: Path, payload: Any) -> str:
    if not path_exists(path):
        return "missing"
    if not isinstance(payload, dict):
        return "unreadable"
    if payload.get("external_inference_calls", 0) not in {0, None}:
        return "external_inference_present"
    identity = runtime_identity(payload)
    if not identity:
        if artifact_key == "active_frontier":
            return "active_runtime_identity_missing"
        if artifact_key in {"public_report", "seed49_regression", "seed55_frontier", "runtime_report"}:
            return "runtime_identity_warning"
        return "runtime_identity_missing"
    return "real"


def has_raw_outputs(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("raw_outputs") or payload.get("examples") or payload.get("cases"):
        return True
    eval_payload = payload.get("eval")
    if isinstance(eval_payload, dict):
        results = eval_payload.get("results")
        if isinstance(results, list) and results:
            return True
        summary = eval_payload.get("summary")
        if isinstance(summary, dict) and int(summary.get("cases") or 0) > 0:
            return True
    if isinstance(payload.get("residuals"), list) and payload.get("residuals"):
        return True
    if isinstance(payload.get("metrics"), dict) and payload["metrics"].get("evaluation"):
        return True
    return False


def failed_gates(candidate: dict[str, Any]) -> list[str]:
    if isinstance(candidate.get("failed_gates"), list):
        return [str(item) for item in candidate["failed_gates"]]
    return [str(row.get("gate")) for row in candidate.get("checks", []) if isinstance(row, dict) and not row.get("passed")]


def trace_quality(kind: str, path: Path) -> float:
    if kind == "pressure_runner":
        payload = read_json(path)
        score = float(get_path(payload, ["summary", "accuracy"], 0.0) or 0.0)
        residual_count = len(payload.get("residuals", [])) if isinstance(payload, dict) else 0
        return round(clamp01(0.55 + 0.30 * score + min(0.15, residual_count * 0.02)), 4)
    if kind == "teacher_self_edit":
        payload = read_json(path)
        return 0.85 if payload else 0.0
    if path.suffix == ".jsonl":
        return 0.78 if path.stat().st_size > 0 else 0.0
    return 0.7


def trace_utility(kind: str, path: Path) -> float:
    if kind in {"pressure_runner", "teacher_self_edit"}:
        return 0.86
    if kind == "routing":
        return 0.80
    return 0.75


def write_report(name: str, payload: dict[str, Any]) -> str:
    path = REPORTS / name
    write_json(path, payload)
    return rel(path)


def write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    summary = aggregate.get("summary", {})
    lines = [
        "# Legacy Port Mechanisms",
        "",
        f"Updated: {aggregate.get('created_utc')}",
        "",
        "## Summary",
        "",
        f"- mechanisms: {summary.get('mechanisms')}",
        f"- red: {summary.get('red')}",
        f"- yellow_or_degraded: {summary.get('yellow_or_degraded')}",
        f"- top_blocker: {summary.get('top_blocker')}",
        "",
        "## Reports",
        "",
    ]
    for key, report_path in aggregate.get("reports", {}).items():
        state = get_path(aggregate, ["summary", "states", key], "UNKNOWN")
        lines.append(f"- `{key}`: `{report_path}` ({state})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path_exists(path):
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    if not path_exists(path) or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve(path: str | Path) -> Path:
    text = str(path or "").strip()
    if ":" in text and "://" not in text:
        prefix, suffix = text.split(":", 1)
        if suffix and not (len(prefix) == 1 and prefix.isalpha()):
            text = suffix
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def safe(value: Any) -> str:
    text = str(value or "unknown").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
