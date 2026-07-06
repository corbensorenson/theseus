"""Build a no-cheat accelerator parity manifest for CUDA, MLX, and Metal lanes.

The manifest is audit evidence only. It compares currently available Mac MLX
bridge reports and Rust/Metal proof reports against the CUDA-equivalent command
surface without enabling production routing or making a full parity claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "accelerator_parity_manifest.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "accelerator_parity_manifest.md"


SURFACES = [
    {
        "surface": "eval_chunk",
        "cuda": "cuda_eval_chunk",
        "mlx": "mlx_eval_chunk",
        "mlx_report": "reports/macos_mlx_work_proof/mlx_eval_chunk.json",
        "class": "hive_worker_chunk",
    },
    {
        "surface": "training_chunk",
        "cuda": "cuda_training_chunk",
        "mlx": "mlx_training_chunk",
        "mlx_report": "reports/macos_mlx_work_proof/mlx_training_chunk.json",
        "class": "hive_worker_chunk",
    },
    {
        "surface": "rollout_chunk",
        "cuda": "cuda_rollout_chunk",
        "mlx": "mlx_rollout_chunk",
        "mlx_report": "reports/macos_mlx_work_proof/mlx_rollout_chunk.json",
        "class": "hive_worker_chunk",
    },
    {
        "surface": "standalone_readout_cli",
        "cuda": "train-standalone-cuda",
        "mlx": "train-standalone-mlx",
        "mlx_report": "reports/macos_mlx_work_proof/cli_train_standalone_mlx.json",
        "metal": "train-standalone-metal",
        "metal_report": "reports/symliquid_standalone_metal_train_report.json",
        "metal_canary": "reports/macos_metal_standalone_scheduler_canary.json",
        "class": "rust_cli_bridge_plus_native_metal",
    },
    {
        "surface": "rollout_cli",
        "cuda": "train-rollout-cuda",
        "mlx": "train-rollout-mlx",
        "mlx_report": "reports/macos_mlx_work_proof/cli_train_rollout_mlx.json",
        "metal": "train-rollout-metal",
        "metal_report": "reports/symliquid_rollout_metal_train_report.json",
        "metal_canary": "reports/macos_metal_scheduler_canary.json",
        "class": "rust_cli_bridge_plus_native_metal",
    },
    {
        "surface": "rollout_sweep_cli",
        "cuda": "train-rollout-cuda-sweep",
        "mlx": "train-rollout-mlx-sweep",
        "mlx_report": "reports/macos_mlx_work_proof/cli_train_rollout_mlx_sweep.json",
        "metal": "train-rollout-metal-sweep",
        "metal_report": "reports/symliquid_rollout_metal_sweep.json",
        "metal_canary": "reports/macos_metal_rollout_sweep_scheduler_canary.json",
        "class": "rust_cli_bridge_plus_native_metal",
    },
    {
        "surface": "token_superposition_cli",
        "cuda": "train-token-superposition-cuda",
        "mlx": "train-token-superposition-mlx",
        "mlx_report": "reports/macos_mlx_work_proof/cli_train_token_superposition_mlx.json",
        "metal": "train-token-superposition-metal",
        "metal_report": "reports/token_superposition_metal_training.json",
        "metal_canary": "reports/macos_metal_token_superposition_scheduler_canary.json",
        "class": "rust_cli_bridge_plus_native_metal",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an accelerator parity manifest.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    rows = [surface_row(surface) for surface in SURFACES]
    hard_failures = [
        issue
        for row in rows
        for issue in row.get("issues", [])
        if isinstance(issue, dict) and issue.get("severity") == "hard"
    ]
    explicit_guardrail_gaps = [
        issue
        for row in rows
        for issue in row.get("issues", [])
        if isinstance(issue, dict) and issue.get("severity") == "evidence"
    ]
    summary = {
        "surface_count": len(rows),
        "surface_ok_count": sum(1 for row in rows if row.get("ok")),
        "mlx_report_count": sum(1 for row in rows if row.get("mlx", {}).get("present")),
        "mlx_report_ok_count": sum(1 for row in rows if row.get("mlx", {}).get("ok")),
        "metal_report_count": sum(1 for row in rows if row.get("metal", {}).get("present")),
        "metal_report_ok_count": sum(1 for row in rows if row.get("metal", {}).get("ok")),
        "artifact_manifest_count": sum(
            1
            for row in rows
            for lane in ("mlx", "metal")
            if row.get(lane, {}).get("artifact", {}).get("present")
        ),
        "scheduler_canary_count": sum(1 for row in rows if row.get("metal_canary", {}).get("ok")),
        "hard_failure_count": len(hard_failures),
        "explicit_guardrail_gap_count": len(explicit_guardrail_gaps),
        "external_inference_calls": sum(
            int(row.get(lane, {}).get("external_inference_calls") or 0)
            for row in rows
            for lane in ("mlx", "metal")
        ),
        "teacher_used_count": sum(
            1
            for row in rows
            for lane in ("mlx", "metal")
            if row.get(lane, {}).get("teacher_used") is True
        ),
        "public_training_rows": sum(
            int(row.get(lane, {}).get("public_training_rows") or 0)
            for row in rows
            for lane in ("mlx", "metal")
        ),
        "model_promotion_allowed_count": sum(
            1
            for row in rows
            for lane in ("mlx", "metal")
            if row.get(lane, {}).get("model_promotion_allowed") is True
        ),
        "production_routing_enabled_count": sum(
            1
            for row in rows
            for lane in ("mlx", "metal")
            if row.get(lane, {}).get("scheduler_routing_enabled") is True
        ),
    }
    report = {
        "ok": not hard_failures and not explicit_guardrail_gaps and summary["surface_ok_count"] == len(rows),
        "policy": "project_theseus_accelerator_parity_manifest_v0",
        "created_utc": now(),
        "trigger_state": "GREEN"
        if not hard_failures and not explicit_guardrail_gaps and summary["surface_ok_count"] == len(rows)
        else "YELLOW"
        if not hard_failures
        else "RED",
        "score_semantics": (
            "Audit manifest only. CUDA-equivalent surfaces are compared against current MLX bridge and "
            "Rust/Metal evidence using local reports; this does not spend public calibration, call a "
            "teacher, enable production scheduler routing, promote a model, or claim full parity."
        ),
        "summary": summary,
        "rows": rows,
        "guardrails": {
            "public_calibration_run": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "teacher_used": False,
            "model_promotion_allowed": False,
            "production_scheduler_routing_enabled": False,
            "full_parity_claim_allowed": False,
        },
        "hard_failures": hard_failures,
        "explicit_guardrail_gaps": explicit_guardrail_gaps,
        "next_action": next_action(hard_failures, explicit_guardrail_gaps),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = resolve(args.markdown_out)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] != "RED" else 2


def surface_row(surface: dict[str, str]) -> dict[str, Any]:
    mlx = lane_report("mlx", surface.get("mlx_report", ""))
    metal = lane_report("metal", surface.get("metal_report", ""))
    canary = canary_report(surface.get("metal_canary", ""))
    issues: list[dict[str, str]] = []
    if not mlx.get("present"):
        issues.append({"severity": "hard", "lane": "mlx", "issue": "missing_mlx_report"})
    elif not mlx.get("ok"):
        issues.append({"severity": "hard", "lane": "mlx", "issue": "mlx_report_not_ok"})
    if surface.get("metal_report"):
        if not metal.get("present"):
            issues.append({"severity": "hard", "lane": "metal", "issue": "missing_metal_report"})
        elif not metal.get("ok"):
            issues.append({"severity": "hard", "lane": "metal", "issue": "metal_report_not_ok"})
    for lane_name, lane in (("mlx", mlx), ("metal", metal)):
        if not lane.get("present"):
            continue
        for check, value in lane.get("guardrail_checks", {}).items():
            if value is False:
                severity = "hard" if check.endswith("_zero") or check.endswith("_disabled") else "evidence"
                issues.append({"severity": severity, "lane": lane_name, "issue": check})
    metrics = comparable_metrics(mlx, metal)
    return {
        "ok": not issues,
        "surface": surface["surface"],
        "class": surface["class"],
        "cuda_equivalent": surface["cuda"],
        "mlx_equivalent": surface["mlx"],
        "metal_equivalent": surface.get("metal"),
        "full_parity_claim_allowed": False,
        "production_scheduler_routing_enabled": False,
        "mlx": mlx,
        "metal": metal,
        "metal_canary": canary,
        "comparable_metrics": metrics,
        "issues": issues,
    }


def lane_report(lane: str, path_text: str) -> dict[str, Any]:
    if not path_text:
        return {"present": False, "lane": lane}
    path = resolve(path_text)
    report = read_json(path, {})
    if not report:
        return {"present": False, "lane": lane, "path": rel(path)}
    checks = guardrail_checks(report, lane)
    return {
        "present": True,
        "lane": lane,
        "path": rel(path),
        "ok": bool(report.get("ok")),
        "policy": report.get("policy"),
        "command": report.get("command"),
        "backend": report.get("backend"),
        "implementation": report.get("implementation"),
        "parity_for": report.get("parity_for"),
        "metrics": extract_metrics(report),
        "artifact": artifact_manifest(report),
        "external_inference_calls": int(report.get("external_inference_calls") or 0),
        "teacher_used": report.get("teacher_used"),
        "public_training_rows": int(report.get("public_training_rows") or 0),
        "model_promotion_allowed": report.get("model_promotion_allowed"),
        "scheduler_routing_enabled": scheduler_routing_enabled(report),
        "guardrail_checks": checks,
    }


def guardrail_checks(report: dict[str, Any], lane: str) -> dict[str, bool]:
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    guardrail_text = str(report.get("guardrail") or "")
    return {
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": report.get("teacher_used") is False or guardrails.get("no_teacher") is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and ("public_training_rows" in report or guardrails.get("no_public_training_rows") is True),
        "no_public_calibration": guardrails.get("no_public_calibration") is True
        or "no_public_benchmark_training" in guardrail_text,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True,
        "model_promotion_locked": report.get("model_promotion_allowed") is False
        or get_path(report, ["promotion_decision", "promote_to_training_lane"], False) is False,
        "scheduler_routing_disabled": scheduler_routing_enabled(report) is False,
        "parity_claim_locked": not parity_claim_allowed(report, lane),
    }


def scheduler_routing_enabled(report: dict[str, Any]) -> bool | None:
    contract_value = get_path(report, ["report_contract", "scheduler_routing_enabled"], None)
    if contract_value is not None:
        return bool(contract_value)
    guardrail_value = get_path(report, ["guardrails", "scheduler_routing_enabled"], None)
    if guardrail_value is not None:
        return bool(guardrail_value)
    return None


def parity_claim_allowed(report: dict[str, Any], lane: str) -> bool:
    if lane == "metal":
        return bool(
            report.get("full_cli_parity_claim_allowed")
            or report.get("train_standalone_parity_claim_allowed")
            or report.get("train_rollout_parity_claim_allowed")
            or report.get("train_rollout_sweep_parity_claim_allowed")
            or report.get("train_token_superposition_parity_claim_allowed")
        )
    return bool(report.get("full_cli_parity_claim_allowed"))


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("metrics"), dict) and report["metrics"]:
        return keep_metrics(report["metrics"])
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    best = report.get("best_variant") if isinstance(report.get("best_variant"), dict) else {}
    if baseline or best:
        return keep_metrics(
            {
                "baseline_combined_loss": get_path(baseline, ["eval", "combined_ar_loss"], None),
                "baseline_train_examples_per_second": baseline.get("train_examples_per_second"),
                "best_variant_combined_loss": get_path(best, ["eval", "combined_ar_loss"], None),
                "best_variant_id": best.get("id"),
                "best_variant_train_examples_per_second": best.get("train_examples_per_second"),
                "best_variant_nominal_speedup": best.get("nominal_speedup_vs_baseline"),
                "best_variant_measured_train_speedup": best.get("measured_train_speedup_vs_baseline"),
                "best_variant_loss_delta_vs_baseline": best.get("combined_loss_delta_vs_baseline"),
                "kernel_launches": int(baseline.get("kernel_launches") or 0) + int(best.get("kernel_launches") or 0),
            }
        )
    if isinstance(report.get("best_run"), dict):
        return keep_metrics(report["best_run"].get("metrics", {}) if isinstance(report["best_run"].get("metrics"), dict) else {})
    return {}


def keep_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "train_accuracy",
        "eval_accuracy",
        "train_return_proxy",
        "eval_return_proxy",
        "loss_initial",
        "loss_final",
        "train_loss",
        "eval_loss",
        "baseline_combined_loss",
        "best_variant_combined_loss",
        "best_variant_id",
        "best_variant_train_examples_per_second",
        "best_variant_nominal_speedup",
        "best_variant_measured_train_speedup",
        "best_variant_loss_delta_vs_baseline",
        "examples_per_second",
        "train_examples_per_second",
        "mlx_train_ms",
        "mlx_eval_ms",
        "kernel_launches",
        "run_count",
        "best_eval_accuracy",
        "best_eval_loss",
        "mean_eval_accuracy",
        "std_eval_accuracy",
        "mean_eval_loss",
        "std_eval_loss",
        "train_rows",
        "eval_rows",
        "train_cases",
        "eval_cases",
        "feature_dim",
    ]
    return {key: metrics.get(key) for key in keep if key in metrics and metrics.get(key) is not None}


def artifact_manifest(report: dict[str, Any]) -> dict[str, Any]:
    artifact_write = report.get("artifact_write") if isinstance(report.get("artifact_write"), dict) else {}
    path_text = artifact_write.get("path") if isinstance(artifact_write.get("path"), str) else ""
    children = report.get("children") if isinstance(report.get("children"), list) else []
    if not path_text and children:
        child_artifacts = []
        for child in children:
            if not isinstance(child, dict):
                continue
            child_write = child.get("artifact_write") if isinstance(child.get("artifact_write"), dict) else {}
            child_path = child_write.get("path") if isinstance(child_write.get("path"), str) else ""
            child_artifact = read_json(resolve(child_path), {}) if child_path else {}
            if child_write.get("attempted") and child_path and child_artifact:
                child_artifacts.append((child_write, child_path, child_artifact))
        production_compatible = bool(child_artifacts) and all(
            bool(write.get("production_checkpoint_compatible")) for write, _, _ in child_artifacts
        )
        return {
            "present": bool(child_artifacts),
            "path": child_artifacts[0][1] if child_artifacts else "",
            "kind": "canonical_readout_artifact_children",
            "schema": "symliquid_core::benchmarks::ReadoutArtifact",
            "production_checkpoint_compatible": production_compatible,
            "child_artifact_count": len(child_artifacts),
            "hv_dim": child_artifacts[0][2].get("hv_dim") if child_artifacts else None,
            "output_dim": child_artifacts[0][2].get("output_dim") if child_artifacts else None,
            "weights": sum(len(artifact.get("weights") if isinstance(artifact.get("weights"), list) else []) for _, _, artifact in child_artifacts),
            "bias": sum(len(artifact.get("bias") if isinstance(artifact.get("bias"), list) else []) for _, _, artifact in child_artifacts),
            "labels": child_artifacts[0][2].get("output_dim") if child_artifacts else None,
        }
    artifact = read_json(resolve(path_text), {}) if path_text else {}
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    return {
        "present": bool(artifact_write.get("attempted") and path_text and artifact),
        "path": path_text,
        "kind": artifact_write.get("kind"),
        "schema": artifact_write.get("schema"),
        "production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "hv_dim": artifact.get("hv_dim") or artifact_write.get("hv_dim"),
        "output_dim": artifact.get("output_dim") or artifact_write.get("output_dim"),
        "weights": len(weights) or artifact_write.get("weights_written"),
        "bias": len(bias) or artifact_write.get("bias_written"),
        "labels": len(labels) or artifact_write.get("labels_written"),
    }


def comparable_metrics(mlx: dict[str, Any], metal: dict[str, Any]) -> dict[str, Any]:
    mlx_metrics = mlx.get("metrics") if isinstance(mlx.get("metrics"), dict) else {}
    metal_metrics = metal.get("metrics") if isinstance(metal.get("metrics"), dict) else {}
    common = sorted(set(mlx_metrics).intersection(metal_metrics))
    return {
        "common_metric_keys": common,
        "mlx_only_metric_keys": sorted(set(mlx_metrics).difference(metal_metrics)),
        "metal_only_metric_keys": sorted(set(metal_metrics).difference(mlx_metrics)),
        "values": {
            key: {
                "mlx": mlx_metrics.get(key),
                "metal": metal_metrics.get(key),
            }
            for key in common
        },
    }


def canary_report(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {"present": False}
    path = resolve(path_text)
    report = read_json(path, {})
    if not report:
        return {"present": False, "path": rel(path)}
    return {
        "present": True,
        "path": rel(path),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "task_kind": report.get("task_kind") or get_path(report, ["planned_placement", "task_kind"], None),
        "local_only": get_path(report, ["guardrails", "local_only"], None),
        "remote_task_submitted": get_path(report, ["guardrails", "remote_task_submitted"], None),
        "production_scheduler_routing_enabled": get_path(
            report, ["guardrails", "production_scheduler_routing_enabled"], None
        ),
        "model_promotion_allowed": get_path(report, ["guardrails", "model_promotion_allowed"], None),
    }


def next_action(hard_failures: list[dict[str, str]], evidence_gaps: list[dict[str, str]]) -> str:
    if hard_failures:
        return "Fix hard accelerator guardrail/report failures before using this manifest for route or parity review."
    if evidence_gaps:
        return "Refresh MLX/Metal reports with explicit structured no-cheat guardrails before treating metrics as comparable."
    return "Use this manifest as audit input only; keep production routing and full parity claims locked pending operator-reviewed enablement."


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Accelerator Parity Manifest",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Surfaces OK: `{get_path(report, ['summary', 'surface_ok_count'], 0)}/{get_path(report, ['summary', 'surface_count'], 0)}`",
        f"- MLX reports OK: `{get_path(report, ['summary', 'mlx_report_ok_count'], 0)}/{get_path(report, ['summary', 'mlx_report_count'], 0)}`",
        f"- Metal reports OK: `{get_path(report, ['summary', 'metal_report_ok_count'], 0)}/{get_path(report, ['summary', 'metal_report_count'], 0)}`",
        f"- Artifact manifests: `{get_path(report, ['summary', 'artifact_manifest_count'], 0)}`",
        f"- External inference calls: `{get_path(report, ['summary', 'external_inference_calls'], 0)}`",
        f"- Public training rows: `{get_path(report, ['summary', 'public_training_rows'], 0)}`",
        f"- Promotion-enabled rows: `{get_path(report, ['summary', 'model_promotion_allowed_count'], 0)}`",
        f"- Production-routing rows: `{get_path(report, ['summary', 'production_routing_enabled_count'], 0)}`",
        "",
        "| Surface | CUDA | MLX | Metal | OK | Common Metrics | Artifact | Issues |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        issues = ", ".join(issue.get("issue", "") for issue in row.get("issues", []) if isinstance(issue, dict)) or "none"
        common = ", ".join(get_path(row, ["comparable_metrics", "common_metric_keys"], []) or []) or "none"
        artifacts = []
        for lane in ("mlx", "metal"):
            artifact = row.get(lane, {}).get("artifact", {}) if isinstance(row.get(lane), dict) else {}
            if artifact.get("present"):
                artifacts.append(f"{lane}:{artifact.get('path')}")
        lines.append(
            f"| `{row.get('surface')}` | `{row.get('cuda_equivalent')}` | `{row.get('mlx_equivalent')}` | "
            f"`{row.get('metal_equivalent') or ''}` | `{row.get('ok')}` | {common} | {', '.join(artifacts) or 'none'} | {issues} |"
        )
    lines.extend(["", f"Next action: {report.get('next_action')}", ""])
    return "\n".join(lines)


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


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
