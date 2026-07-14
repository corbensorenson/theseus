#!/usr/bin/env python3
"""Publish the frozen v8 exact-recovery comparison without utility claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "configs/neural_seed_functional_utility_freeze.json"
DEFAULT_OUT = ROOT / "reports/moecot_dense_exact_recovery_diagnostic_v8.json"
DEFAULT_MARKDOWN = ROOT / "reports/moecot_dense_exact_recovery_diagnostic_v8.md"
ARMS = ("english", "python", "javascript_typescript", "html_css", "rust")
CONTROLS = ("dense_active_parameter", "dense_total_parameter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    freeze = read_json(resolve(args.freeze))
    report = build_diagnostic(ROOT, freeze)
    write_json(resolve(args.out), report)
    resolve(args.markdown_out).write_text(render_markdown(report), encoding="utf-8")
    summary = {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "publication_ready": report["publication_ready"],
        "architecture_verdict": report["architecture_verdict"],
        "hard_gaps": report["hard_gaps"],
    }
    print(json.dumps(summary if args.gate else report, indent=2, sort_keys=True))
    return 0 if report["publication_ready"] else 2


def build_diagnostic(root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    plan_sha = str(freeze.get("v8_plan_sha256") or "")
    stage_signature = str(freeze.get("v8_stage_signature") or "")
    gaps = []
    target_rows: dict[str, dict[str, Any]] = {}
    for target_id in ("shared_trunk", *ARMS, *CONTROLS):
        row, faults = collect_target(root, target_id, plan_sha, stage_signature)
        target_rows[target_id] = row
        gaps.extend(f"{target_id}:{fault}" for fault in faults)
    sparse_by_arm = {
        arm: target_rows[arm].get("evaluation_by_arm", {}).get(arm)
        for arm in ARMS
    }
    controls_by_arm = {
        control: target_rows[control].get("evaluation_by_arm", {})
        for control in CONTROLS
    }
    if any(value is None for value in sparse_by_arm.values()):
        gaps.append("moecot_per_arm_evaluation_incomplete")
    for control, by_arm in controls_by_arm.items():
        if set(by_arm) != set(ARMS):
            gaps.append(f"{control}:per_arm_evaluation_incomplete")
    sparse_summary = aggregate_metrics([value for value in sparse_by_arm.values() if value])
    control_summaries = {
        control: aggregate_metrics(list(by_arm.values()))
        for control, by_arm in controls_by_arm.items()
    }
    resources = {
        "moecot_system": sparse_resource_summary(target_rows),
        **{control: target_rows[control].get("resource") for control in CONTROLS},
    }
    return {
        "policy": "project_theseus_moecot_dense_exact_recovery_diagnostic_v8",
        "created_utc": now(),
        "trigger_state": "GREEN" if not gaps else "YELLOW",
        "publication_ready": not gaps,
        "freeze_identity": {
            "plan_sha256": plan_sha,
            "stage_signature": stage_signature,
            "functional_case_contract_sha256": freeze.get("case_contract_sha256"),
            "functional_evaluation_state": freeze.get("evaluation_state"),
        },
        "score_semantics": "exact target-string recovery and serialization diagnostics only; not functional utility",
        "moecot": {"by_arm": sparse_by_arm, "summary": sparse_summary},
        "dense_controls": {
            control: {"by_arm": controls_by_arm[control], "summary": control_summaries[control]}
            for control in CONTROLS
        },
        "preregistered_views": {
            "equal_active_parameters_active_compute_control": {
                "moecot": sparse_summary,
                "control": control_summaries["dense_active_parameter"],
                "control_id": "dense_active_parameter",
            },
            "equal_unique_system_positions_total_parameter_control": {
                "moecot": sparse_summary,
                "control": control_summaries["dense_total_parameter"],
                "control_id": "dense_total_parameter",
            },
        },
        "resources": resources,
        "targets": target_rows,
        "architecture_verdict": "DEFER_TO_FROZEN_FUNCTIONAL_UTILITY",
        "hard_gaps": sorted(set(gaps)),
        "boundaries": {
            "functional_utility_claimed": False,
            "architecture_winner_selected": False,
            "accounting_view_selected_after_results": False,
            "public_benchmark_payload_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
        },
    }


def collect_target(
    root: Path, target_id: str, plan_sha: str, stage_signature: str
) -> tuple[dict[str, Any], list[str]]:
    directory = root / "checkpoints/moecot_language_seed_v8" / target_id
    receipt_path = directory / "training_receipt.json"
    evaluation_path = directory / "evaluation_private_dev_receipt.json"
    faults = []
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    if not receipt:
        faults.append("training_receipt_missing")
    if not receipt.get("complete"):
        faults.append("training_incomplete")
    if receipt.get("plan_sha256") != plan_sha:
        faults.append("plan_identity_mismatch")
    if receipt.get("stage_signature") != stage_signature:
        faults.append("stage_identity_mismatch")
    for key in (
        "public_training_rows_written",
        "external_inference_calls",
        "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    ):
        if int(receipt.get(key) or 0) != 0:
            faults.append(f"nonzero_boundary:{key}")
    checkpoint = resolve_from(root, str(receipt.get("checkpoint") or ""))
    checkpoint_hash = ""
    checkpoint_bytes = 0
    if not checkpoint.is_file():
        faults.append("checkpoint_missing")
    else:
        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        if checkpoint_hash != receipt.get("checkpoint_sha256"):
            faults.append("checkpoint_identity_mismatch")
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else {}
    needs_evaluation = target_id != "shared_trunk"
    if needs_evaluation:
        if not evaluation:
            faults.append("evaluation_missing")
        else:
            if evaluation.get("checkpoint_sha256") != checkpoint_hash:
                faults.append("evaluation_checkpoint_mismatch")
            if evaluation.get("candidate_family") != "direct_autoregressive_model_text":
                faults.append("candidate_family_mismatch")
            if evaluation.get("target_visible_to_generator") is not False:
                faults.append("target_visibility_violation")
            for key in (
                "public_training_rows_written",
                "public_benchmark_payload_count",
                "external_inference_calls",
                "fallback_return_count",
                "templates_renderers_routers_tools_credit",
            ):
                if int(evaluation.get(key) or 0) != 0:
                    faults.append(f"evaluation_nonzero_boundary:{key}")
    resource = {
        "parameter_count": int(receipt.get("parameter_count") or 0),
        "trainable_parameter_count": int(receipt.get("trainable_parameter_count") or 0),
        "optimizer_steps": int(receipt.get("optimizer_steps") or 0),
        "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
        "wall_seconds": receipt.get("wall_seconds"),
        "energy_joules": receipt.get("energy_joules"),
        "energy_measurement_state": receipt.get("energy_measurement_state"),
        "checkpoint_bytes": checkpoint_bytes,
        "peak_memory_bytes": None,
        "peak_memory_measurement_state": "NOT_RECORDED_BY_FROZEN_V8_TRAINER",
        "checkpoint_load_time_ms": None,
        "checkpoint_load_measurement_state": "MEASURE_WITH_FROZEN_FUNCTIONAL_GENERATOR",
        "phase_tokens_per_second": {
            key: value.get("tokens_per_second")
            for key, value in (receipt.get("phases") or {}).items()
            if isinstance(value, dict)
        },
    }
    return (
        {
            "target_id": target_id,
            "state": "GREEN" if not faults else "INCOMPLETE",
            "training_receipt": relative_to(root, receipt_path),
            "training_receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else "",
            "evaluation_receipt": relative_to(root, evaluation_path) if evaluation_path.is_file() else "",
            "evaluation_receipt_sha256": sha256_file(evaluation_path) if evaluation_path.is_file() else "",
            "checkpoint": relative_to(root, checkpoint) if checkpoint.is_file() else "",
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_summary": evaluation.get("summary"),
            "evaluation_by_arm": evaluation.get("by_arm") or {},
            "resource": resource,
            "faults": faults,
        },
        faults,
    )


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("row_count") or 0) for row in rows)
    exact = sum(int(row.get("exact_match_count") or 0) for row in rows)
    nonempty = sum(int(row.get("nonempty_count") or 0) for row in rows)
    serialization = sum(int(row.get("byte_serialization_valid_count") or 0) for row in rows)
    syntax_checked = sum(int(row.get("syntax_checked_count") or 0) for row in rows)
    syntax_valid = sum(int(row.get("syntax_valid_count") or 0) for row in rows)
    return {
        "row_count": total,
        "exact_match_count": exact,
        "exact_target_match_rate": exact / total if total else None,
        "nonempty_count": nonempty,
        "nonempty_rate": nonempty / total if total else None,
        "byte_serialization_valid_count": serialization,
        "byte_serialization_valid_rate": serialization / total if total else None,
        "syntax_checked_count": syntax_checked,
        "syntax_valid_count": syntax_valid,
        "syntax_valid_rate_when_checked": syntax_valid / syntax_checked if syntax_checked else None,
    }


def sparse_resource_summary(targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shared = targets["shared_trunk"]["resource"]
    experts = [targets[arm]["resource"] for arm in ARMS]
    return {
        "active_parameter_count_per_request": max((int(row["parameter_count"]) for row in experts), default=0),
        "total_parameter_count": int(shared["parameter_count"]) + sum(int(row["trainable_parameter_count"]) for row in experts),
        "optimizer_positions": int(shared["optimizer_positions"]) + sum(int(row["optimizer_positions"]) for row in experts),
        "optimizer_steps": int(shared["optimizer_steps"]) + sum(int(row["optimizer_steps"]) for row in experts),
        "wall_seconds": sum(float(row.get("wall_seconds") or 0) for row in [shared, *experts]),
        "checkpoint_bytes": int(shared["checkpoint_bytes"]) + sum(int(row["checkpoint_bytes"]) for row in experts),
        "energy_joules": None,
        "energy_measurement_state": "NOT_AVAILABLE_FROM_MLX_RUNTIME",
        "peak_memory_bytes": None,
        "peak_memory_measurement_state": "NOT_RECORDED_BY_FROZEN_V8_TRAINER",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v8 Exact-Recovery Diagnostic",
        "",
        f"State: **{report['trigger_state']}**",
        "",
        "This report measures exact target-string recovery and serialization diagnostics only. It is not functional utility and does not select an architecture.",
        "",
        "| Target | Rows | Exact | Exact rate | Nonempty rate |",
        "|---|---:|---:|---:|---:|",
    ]
    entries = [("MoECOT", report["moecot"]["summary"])] + [
        (target, row["summary"]) for target, row in report["dense_controls"].items()
    ]
    for name, row in entries:
        lines.append(
            f"| {name} | {row['row_count']} | {row['exact_match_count']} | {format_rate(row['exact_target_match_rate'])} | {format_rate(row['nonempty_rate'])} |"
        )
    if report["hard_gaps"]:
        lines.extend(["", "## Pending", *[f"- `{gap}`" for gap in report["hard_gaps"]]])
    lines.extend(["", "Architecture verdict: `DEFER_TO_FROZEN_FUNCTIONAL_UTILITY`", ""])
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_from(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
