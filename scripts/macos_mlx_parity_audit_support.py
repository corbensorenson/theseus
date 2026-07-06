"""Support helpers for the macOS MLX parity audit."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# macOS MLX Parity Audit",
        "",
        f"- State: `{report.get('state')}`",
        f"- MLX available: `{get_path(report, ['mlx', 'available'], False)}`",
        f"- Missing or CPU fallback rows: `{get_path(report, ['summary', 'missing_or_cpu_fallback'], 0)}`",
        f"- Runnable MLX bridges: `{get_path(report, ['summary', 'runnable_mlx_bridge_count'], 0)}`",
        f"- Native kernel parity ready: `{get_path(report, ['summary', 'kernel_parity_ready_count'], 0)}`",
        f"- Native guarded proofs ready: `{get_path(report, ['summary', 'kernel_parity_guarded_proof_count'], 0)}`",
        f"- Native guarded canaries ready: `{get_path(report, ['summary', 'kernel_parity_guarded_canary_count'], 0)}`",
        f"- Native kernel parity pending: `{get_path(report, ['summary', 'kernel_parity_pending_count'], 0)}`",
        f"- Production route pending: `{get_path(report, ['summary', 'production_route_pending_count'], 0)}`",
        f"- CUDA hot-loop routes still required: `{get_path(report, ['summary', 'cuda_hot_loop_route_required_count'], 0)}`",
        "",
        "## Hive Task Parity",
        "",
        "| CUDA | MLX/Mac | Status | Route | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.get("hive_task_parity", []) or []:
        lines.append(f"| `{row.get('cuda')}` | `{row.get('mlx')}` | `{row.get('status')}` | `{get_path(row, ['route', 'scheduler_route'], '')}` | {row.get('notes')} |")
    lines.extend(
        [
            "",
            "## Rust CLI Parity",
            "",
            "| CUDA command | Mac command | Bridge status | Latest Evidence | Bridge route | Kernel route until ported | Kernel Port | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("rust_cli_parity", []) or []:
        evidence = row.get("latest_evidence") if isinstance(row.get("latest_evidence"), dict) else {}
        lines.append(
            f"| `{row.get('cuda')}` | `{row.get('mac') or ''}` | `{row.get('status')}` | "
            f"`{evidence.get('report_path') or evidence.get('reason') or 'none'}` | "
            f"`{get_path(row, ['route', 'bridge_route'], '')}` | "
            f"`{get_path(row, ['route', 'kernel_route_until_ported'], '')}` | "
            f"`{row.get('kernel_port_status') or ''}` | {row.get('notes')} |"
        )
    routing = report.get("routing_decisions") if isinstance(report.get("routing_decisions"), dict) else {}
    lines.extend(["", "## Routing Truth", ""])
    lines.append(f"- {routing.get('default_training_route') or 'No routing summary recorded.'}")
    lines.append(f"- {routing.get('safe_claim') or 'No safe-claim summary recorded.'}")
    native = report.get("native_hot_loop_proof") if isinstance(report.get("native_hot_loop_proof"), dict) else {}
    native_summary = native.get("summary") if isinstance(native.get("summary"), dict) else {}
    lines.extend(["", "## Native Hot-Loop Proof", ""])
    lines.append(f"- Native parity claim allowed: `{native.get('parity_claim_allowed')}`")
    lines.append(f"- Rust Metal/MLX source files: `{native_summary.get('rust_metal_source_files', 0) + native_summary.get('rust_mlx_source_files', 0)}`")
    lines.append(f"- Python MLX bridge commands ready: `{native_summary.get('python_mlx_bridge_commands_ready')}`")
    lines.append(f"- Native hot-loop targets guarded proof ready: `{native_summary.get('native_hot_loop_targets_guarded_proof_ready')}`")
    lines.append(f"- Native hot-loop targets guarded canary ready: `{native_summary.get('native_hot_loop_targets_guarded_canary_ready')}`")
    lines.append(f"- Native hot-loop production routes pending: `{native_summary.get('production_route_pending_native_hot_loop_targets')}`")
    lines.append(f"- Native subkernel proofs: `{native_summary.get('native_subkernel_proof_count')}`")
    lines.append(f"- Native CLI proofs: `{native_summary.get('native_cli_proof_count')}`")
    lines.append(f"- Native feature proofs: `{native_summary.get('native_feature_proof_count')}`")
    lines.append(f"- Native readout proofs: `{native_summary.get('native_readout_proof_count')}`")
    lines.append(f"- Native readout training proofs: `{native_summary.get('native_readout_training_proof_count')}`")
    lines.append(f"- Native train-path proofs: `{native_summary.get('native_train_path_proof_count')}`")
    lines.append(f"- Native train-standalone CLI contract proofs: `{native_summary.get('native_train_standalone_cli_contract_count')}`")
    lines.append(f"- Native train-standalone artifact equivalence proofs: `{native_summary.get('native_train_standalone_artifact_equivalence_count')}`")
    lines.append(f"- Native train-standalone scheduler canary proofs: `{native_summary.get('native_train_standalone_scheduler_canary_count')}`")
    lines.append(f"- Native train-rollout CLI contract proofs: `{native_summary.get('native_train_rollout_cli_contract_count')}`")
    lines.append(f"- Native train-rollout artifact equivalence proofs: `{native_summary.get('native_train_rollout_artifact_equivalence_count')}`")
    lines.append(f"- Native train-rollout scheduler guardrail proofs: `{native_summary.get('native_train_rollout_scheduler_guardrail_count')}`")
    lines.append(f"- Native train-rollout scheduler dry-run proofs: `{native_summary.get('native_train_rollout_scheduler_dry_run_count')}`")
    lines.append(f"- Native train-rollout bounded ladder proofs: `{native_summary.get('native_train_rollout_parity_ladder_count')}`")
    lines.append(f"- Native train-rollout scheduler canary proofs: `{native_summary.get('native_train_rollout_scheduler_canary_count')}`")
    lines.append(f"- Native train-rollout sweep CLI contract proofs: `{native_summary.get('native_train_rollout_sweep_cli_contract_count')}`")
    lines.append(f"- Native train-rollout sweep child artifacts: `{native_summary.get('native_train_rollout_sweep_child_artifact_count')}`")
    lines.append(f"- Native train-rollout sweep scheduler canary proofs: `{native_summary.get('native_train_rollout_sweep_scheduler_canary_count')}`")
    lines.append(f"- Native train-rollout state-training proofs: `{native_summary.get('native_train_rollout_state_training_proof_count')}`")
    lines.append(f"- Native token-superposition readout proofs: `{native_summary.get('native_token_superposition_readout_proof_count')}`")
    lines.append(f"- Native token-superposition CLI contract proofs: `{native_summary.get('native_train_token_superposition_cli_contract_count')}`")
    lines.append(f"- Native token-superposition artifact equivalence proofs: `{native_summary.get('native_train_token_superposition_artifact_equivalence_count')}`")
    lines.append(f"- Native token-superposition bounded ladder proofs: `{native_summary.get('native_train_token_superposition_ladder_count')}`")
    lines.append(f"- Native token-superposition scheduler canary proofs: `{native_summary.get('native_train_token_superposition_scheduler_canary_count')}`")
    lines.append(f"- Native Metal production-route readiness reviews: `{native_summary.get('native_metal_production_route_readiness_review_count')}`")
    lines.append(f"- Native Metal production routes ready: `{native_summary.get('native_metal_production_route_ready_count')}`")
    lines.append(f"- Native Metal production route blockers: `{native_summary.get('native_metal_production_route_blocker_count')}`")
    lines.append(f"- Accelerator parity manifests: `{get_path(report, ['summary', 'accelerator_parity_manifest_count'], 0)}`")
    lines.append(f"- Native hot-loop targets pending: `{native_summary.get('pending_native_hot_loop_targets')}`")
    next_native = native.get("next_native_step") if isinstance(native.get("next_native_step"), dict) else {}
    if next_native:
        lines.append(f"- Next native step: {next_native.get('target')}")
    lines.extend(["", "## Runnable Evidence", ""])
    lines.append(f"- Work proof: `{get_path(report, ['latest_work_proof', 'path'], '')}` state `{get_path(report, ['latest_work_proof', 'state'], 'missing')}`")
    lines.append(f"- Worker evidence OK: `{get_path(report, ['summary', 'latest_worker_evidence_ok'], 0)}`")
    lines.append(f"- CLI evidence OK: `{get_path(report, ['summary', 'latest_cli_evidence_ok'], 0)}`")
    lines.extend(["", "## Next Targets", ""])
    for row in report.get("next_implementation_targets", []) or []:
        lines.append(f"- `{row.get('priority')}` {row.get('target')}: {row.get('why')}")
    lines.append("")
    return "\n".join(lines)


def platform_report() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_macos": platform.system() == "Darwin",
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"},
        "is_intel_mac": platform.system() == "Darwin" and platform.machine().lower() in {"x86_64", "amd64"},
    }


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def int_value(value: Any, path: list[str], default: int) -> int:
    raw = get_path(value, path, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
