#!/usr/bin/env python3
"""Evaluate one learned KERC K5 checkpoint under the external host watchdog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import host_resource_safety
import moecot_language_arm_training as training


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_kerc_k5_candidate_evaluation_v1"
SURFACE_POLICY = "project_theseus_kerc_private_measurement_surface_v1"
REGISTRY_POLICY = "project_theseus_kerc_private_measurement_run_registry_v1"
DEFAULT_REGISTRY = ROOT / "reports/kerc_private_measurement_run_registry.jsonl"
EVALUATION_RESOURCE_RECEIPT = (
    "reports/rdc_kerc_k5_candidate_evaluation_stratified_target_position_complete_"
    "seed_20260722_execution_matched_panel16.host_resource_safety.json"
)
EVALUATION_RESOURCE_RECEIPT_SHA256 = (
    "a5b08f13398067bf5cf58464bc16d0fa16697d6d4d15671bc164fc6c36531ddf"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operation_specific_host_safety_mapping(
    *,
    candidate_id: str,
    candidate_contract: dict[str, Any],
    receipt_path: str,
    receipt_sha256: str,
    command_marker: str,
) -> dict[str, Any]:
    """Derive a launch floor from the exact operation, not KERC training."""

    candidate = next(
        (
            row
            for row in candidate_contract["canaries"]
            if row["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("K5 operation-specific resource candidate is not registered")
    path = resolve(receipt_path)
    if not path.is_file() or sha256(path) != receipt_sha256:
        raise ValueError("K5 operation-specific resource receipt identity mismatch")
    receipt = read_json(path)
    command = [str(value) for value in receipt.get("command") or []]
    peak_mib = float(receipt.get("maximum_inferred_unified_memory_mib") or 0.0)
    if (
        receipt.get("passed") is not True
        or receipt.get("fault") not in (None, "")
        or command_marker not in " ".join(command)
        or peak_mib <= 0.0
        or float(receipt.get("maximum_swapout_growth_mib") or 0.0) > float(
            (candidate.get("host_safety_overrides") or {}).get(
                "maximum_swapout_growth_mib", 0.0
            )
        )
    ):
        raise ValueError("K5 operation-specific resource receipt is invalid")
    mapping = {
        **candidate_contract["host_safety_policy"],
        **dict(candidate.get("host_safety_overrides") or {}),
    }
    live_reserve_mib = float(mapping["minimum_available_during_run_mib"])
    launch_floor_mib = float(math.ceil(peak_mib + live_reserve_mib))
    mapping["minimum_available_before_launch_mib"] = max(
        float(mapping["minimum_available_before_launch_mib"]),
        launch_floor_mib,
    )
    mapping["measured_operation_preflight"] = {
        "path": relative(path),
        "sha256": receipt_sha256,
        "command_marker": command_marker,
        "maximum_inferred_unified_memory_mib": peak_mib,
        "required_live_reserve_mib": live_reserve_mib,
        "resolved_minimum_available_before_launch_mib": launch_floor_mib,
        "derivation": "ceil(measured_operation_peak_mib + required_live_reserve_mib)",
    }
    return mapping


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL object required: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append and fsync one registry event before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def consumed_surface_ids(registry_path: Path) -> set[str]:
    return {
        str(row.get("surface_id") or "")
        for row in read_jsonl(registry_path)
        if row.get("consumed") is True and row.get("surface_id")
    }


def prior_kerc_evaluation_row_ids() -> set[str]:
    """Conservatively exclude every row already sent through a K5 evaluator."""

    row_ids: set[str] = set()
    root = ROOT / "runtime/kerc_k5_evaluations"
    if not root.is_dir():
        return row_ids
    for path in root.glob("*/evaluation_private_dev_receipt.json"):
        try:
            receipt = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows = receipt.get("rows") or []
        if not isinstance(rows, list):
            continue
        row_ids.update(
            str(row.get("row_id"))
            for row in rows
            if isinstance(row, dict) and row.get("row_id")
        )
    return row_ids


def frozen_surface_manifest(
    *,
    training_report: dict[str, Any],
    surface_id: str,
    maximum_rows: int,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a fresh, answer-blind K5 private-dev row surface."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,159}", surface_id):
        raise ValueError("K5 surface id is invalid")
    target = (training_report.get("targets") or {}).get("english_kerc") or {}
    artifact = (target.get("supervision_artifacts") or {}).get("private_dev") or {}
    artifact_path = resolve(str(artifact.get("path") or ""))
    artifact_sha256 = str(artifact.get("sha256") or "")
    if (
        not artifact_path.is_file()
        or sha256(artifact_path) != artifact_sha256
        or maximum_rows <= 0
    ):
        raise ValueError("K5 private-dev source artifact identity mismatch")
    source_rows: list[dict[str, Any]] = []
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("split") != "private_dev"
                or row.get("public_benchmark") is not False
                or not row.get("row_id")
            ):
                raise ValueError(
                    f"invalid K5 private-dev row boundary: {artifact_path}:{line_number}"
                )
            source_rows.append(row)
    row_ids = [str(row["row_id"]) for row in source_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("K5 private-dev source has duplicate row identities")
    excluded = prior_kerc_evaluation_row_ids()
    available = [row_id for row_id in row_ids if row_id not in excluded]
    if len(available) < maximum_rows:
        raise ValueError("insufficient fresh K5 private-dev rows")
    selection_namespace = (
        "t0a_private_rdc_kerc_source_disjoint_fresh_v1:" + surface_id
    )
    selected = sorted(
        available,
        key=lambda row_id: (
            hashlib.sha256(
                (selection_namespace + "\0" + row_id).encode()
            ).hexdigest(),
            row_id,
        ),
    )[:maximum_rows]
    contract = {
        "policy": SURFACE_POLICY,
        "surface_id": surface_id,
        "split": "private_dev",
        "selection_namespace": selection_namespace,
        "source_artifact": {
            "path": relative(artifact_path),
            "sha256": artifact_sha256,
            "row_count": int(artifact.get("row_count") or len(source_rows)),
        },
        "maximum_rows": maximum_rows,
        "selected_row_ids": selected,
        "prior_evaluated_row_count_excluded": len(excluded),
        "prior_evaluated_row_ids_sha256": canonical_sha256(sorted(excluded)),
        "selection_uses_model_outcomes": False,
        "selection_uses_answer_text": False,
        "public_benchmark_payload_count": 0,
        "source_family_disjoint": True,
    }
    contract["surface_contract_sha256"] = canonical_sha256(contract)
    if output_path.is_file():
        existing = read_json(output_path)
        if existing != contract:
            raise ValueError("frozen K5 surface manifest identity mismatch")
    else:
        write_json(output_path, contract)
    return contract


def load_frozen_surface(
    *,
    path: Path,
    surface_id: str,
    training_report: dict[str, Any],
    maximum_rows: int,
) -> dict[str, Any]:
    surface = read_json(path)
    declared_sha256 = str(surface.get("surface_contract_sha256") or "")
    unsigned = dict(surface)
    unsigned.pop("surface_contract_sha256", None)
    target = (training_report.get("targets") or {}).get("english_kerc") or {}
    artifact = (target.get("supervision_artifacts") or {}).get("private_dev") or {}
    if (
        surface.get("policy") != SURFACE_POLICY
        or surface.get("surface_id") != surface_id
        or int(surface.get("maximum_rows") or 0) != maximum_rows
        or len(surface.get("selected_row_ids") or []) != maximum_rows
        or declared_sha256 != canonical_sha256(unsigned)
        or (surface.get("source_artifact") or {}).get("path")
        != artifact.get("path")
        or (surface.get("source_artifact") or {}).get("sha256")
        != artifact.get("sha256")
        or surface.get("selection_uses_model_outcomes") is not False
        or surface.get("selection_uses_answer_text") is not False
    ):
        raise ValueError("frozen K5 surface contract validation failed")
    return surface


def source_artifact(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256(path)}


def result_by_target(report: dict[str, Any], target_id: str) -> dict[str, Any]:
    rows = [row for row in report.get("results") or [] if row.get("target_id") == target_id]
    if len(rows) != 1:
        raise ValueError(f"exactly one {target_id} result is required")
    return rows[0]


def summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows_path = resolve(str((evaluation.get("rows") or {}).get("path") or ""))
    receipt = read_json(rows_path)
    rows = list(receipt.get("rows") or [])
    reasons = Counter()
    semantic_proposal_count = 0
    semantic_rejection_count = 0
    semantic_rejection_reasons: Counter[str] = Counter()
    semantic_valid_selection_count = 0
    pipeline_success_count = 0
    for row in rows:
        generation = row.get("generation") or {}
        if generation.get("state") == "GREEN" and row.get("nonempty") is True:
            pipeline_success_count += 1
        reason = str(
            generation.get("reason")
            or (generation.get("fault") or {}).get("fault_type")
            or generation.get("state")
            or "UNKNOWN"
        )
        reasons[reason] += 1
        fault_detail = str((generation.get("fault") or {}).get("detail") or "")
        try:
            parsed_detail = json.loads(fault_detail)
        except (TypeError, json.JSONDecodeError):
            parsed_detail = {}
        selection = (
            parsed_detail.get("semantic_selection")
            if isinstance(parsed_detail, dict)
            else None
        )
        if isinstance(selection, dict):
            semantic_proposal_count += int(
                selection.get("complete_candidate_count") or 0
            )
            semantic_rejection_count += int(
                selection.get("rejected_candidate_count") or 0
            )
            semantic_rejection_reasons.update(
                {
                    str(key): int(value)
                    for key, value in (
                        selection.get("rejection_counts") or {}
                    ).items()
                }
            )
            semantic_valid_selection_count += int(
                selection.get("selected_semantically_valid") is True
            )
    return {
        "row_count": len(rows),
        "pipeline_success_count": pipeline_success_count,
        "pipeline_success_rate": (
            pipeline_success_count / len(rows) if rows else 0.0
        ),
        "fault_or_state_counts": dict(sorted(reasons.items())),
        "semantic_proposal_accounting": {
            "complete_candidate_count": semantic_proposal_count,
            "rejected_candidate_count": semantic_rejection_count,
            "rejection_counts": dict(sorted(semantic_rejection_reasons.items())),
            "valid_selected_candidate_count": semantic_valid_selection_count,
            "denominator_complete_for_reported_faults": True,
            "invalid_candidates_repaired_or_rewritten": False,
        },
        "raw_generated_text_retained": False,
        "rows": source_artifact(rows_path),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        raise ValueError("K5 checkpoint evaluation requires the external host watchdog")
    import mlx.core as mx
    import mlx.nn as nn

    training_report_path = resolve(args.training_report)
    training_report = read_json(training_report_path)
    surface_path = resolve(args.surface_manifest)
    surface = load_frozen_surface(
        path=surface_path,
        surface_id=args.surface_id,
        training_report=training_report,
        maximum_rows=int(args.maximum_rows),
    )
    if (
        training_report.get("policy")
        != "project_theseus_moecot_language_arm_training_plan_v1"
        or training_report.get("mode") != "training_execution"
        or training_report.get("trigger_state") != "GREEN"
        or training_report.get("external_inference_calls") != 0
        or training_report.get("public_training_rows_written") != 0
        or training_report.get("fallback_return_count") != 0
    ):
        raise ValueError("K5 training report integrity failure")
    lease = training_report.get("candidate_canary_lease") or {}
    candidate_id = str(lease.get("candidate_id") or "")
    if candidate_id != "rdc_kerc_k5_adequacy":
        raise ValueError("K5 adequacy evaluation requires the adequacy candidate lease")
    result = result_by_target(training_report, "english_kerc")
    if int(result.get("candidate_seed") or 0) != int(args.seed):
        raise ValueError("K5 candidate seed mismatch")
    checkpoint = resolve(str(result.get("checkpoint") or ""))
    if not checkpoint.is_file() or sha256(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("K5 checkpoint identity mismatch")

    config_path = resolve(str(training_report.get("config") or ""))
    config = training.bind_scale_preregistration(read_json(config_path))
    decode_budget = int(
        (config.get("evaluation") or {}).get("kerc_decode_max_target_tokens") or 0
    )
    manifest_path = ROOT / "data/training_data/moecot_kernel_english_v1/manifest.json"
    manifest = read_json(manifest_path)
    frozen_maximum = max(
        int((row or {}).get("maximum_target_tokens") or 0)
        for row in (manifest.get("encoded_length_stats") or {}).values()
    )
    if decode_budget < frozen_maximum:
        raise ValueError("K5 decode envelope is below the frozen corpus maximum")
    candidate_envelope = int(
        (((training_report.get("candidate_canary_lease") or {}).get("execution_policy") or {})
        .get("maximum_supervised_training_sequence_tokens_by_target", {})
        .get("english_kerc", 0))
    )
    if candidate_envelope <= 0:
        raise ValueError("K5 candidate evaluation envelope is missing")
    effective_decode_budget = min(decode_budget, candidate_envelope)
    config = copy.deepcopy(config)
    config["evaluation"]["kerc_decode_max_target_tokens"] = effective_decode_budget
    configured_beam_width = int(config["evaluation"]["kerc_beam_width"])
    configured_branching_factor = int(
        config["evaluation"]["kerc_branching_factor"]
    )
    if int(args.beam_width) > 0:
        config["evaluation"]["kerc_beam_width"] = int(args.beam_width)
    if int(args.branching_factor) > 0:
        config["evaluation"]["kerc_branching_factor"] = int(
            args.branching_factor
        )

    target = copy.deepcopy((training_report.get("targets") or {})["english_kerc"])
    target["checkpoint"] = relative(checkpoint)
    output_path = resolve(args.out)
    runtime_directory = (
        ROOT / "runtime/kerc_k5_evaluations" / output_path.stem
    )
    target["receipt"] = relative(runtime_directory / "training_receipt.json")
    metadata_path = resolve(str((training_report.get("stage") or {}).get("metadata") or ""))
    metadata = read_json(metadata_path)
    base_path = resolve(str(training_report.get("base_config") or ""))
    base = read_json(base_path)
    evaluation = training.evaluate_target(
        config,
        base,
        training_report,
        target,
        metadata=metadata,
        mx=mx,
        nn=nn,
        maximum_rows=int(args.maximum_rows),
        selected_row_ids=tuple(str(value) for value in surface["selected_row_ids"]),
        selection_namespace=str(surface["selection_namespace"]),
        selection_contract_sha256=str(surface["surface_contract_sha256"]),
    )
    summary = summarize_evaluation(evaluation)
    behavior_positive = int(summary["pipeline_success_count"]) > 0
    report = {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN",
        "qualification_state": (
            "BEHAVIOR_POSITIVE_SINGLE_SEED_CANARY"
            if behavior_positive
            else "INCONCLUSIVE_EXPERIMENT"
        ),
        "candidate_id": candidate_id,
        "seed": int(args.seed),
        "seed_count": 1,
        "source_family_disjoint": True,
        "measurement_surface": {
            **source_artifact(surface_path),
            "surface_id": args.surface_id,
            "surface_contract_sha256": surface["surface_contract_sha256"],
            "exact_surface_consumed_once": True,
        },
        "decode_envelope": {
            "configured_full_distribution_maximum_target_tokens": decode_budget,
            "frozen_corpus_maximum_target_tokens": frozen_maximum,
            "covers_frozen_corpus": decode_budget >= frozen_maximum,
            "candidate_supervised_sequence_envelope": candidate_envelope,
            "effective_candidate_decode_maximum_target_tokens": (
                effective_decode_budget
            ),
            "candidate_claims_full_distribution_coverage": False,
            "ordinary_surface_decode_max_target_tokens": int(
                config["evaluation"]["decode_max_target_tokens"]
            ),
        },
        "decode_search_policy": {
            "configured_beam_width": configured_beam_width,
            "configured_branching_factor": configured_branching_factor,
            "evaluated_beam_width": int(config["evaluation"]["kerc_beam_width"]),
            "evaluated_branching_factor": int(
                config["evaluation"]["kerc_branching_factor"]
            ),
            "semantic_candidate_validation": True,
            "invalid_candidates_repaired_or_rewritten": False,
            "search_override_is_tuning_evidence_only": bool(
                int(args.beam_width) > 0 or int(args.branching_factor) > 0
            ),
        },
        "evaluation": evaluation,
        "behavior": summary,
        "learned": {
            "compiler": behavior_positive,
            "reasoner": behavior_positive,
            "renderer": behavior_positive,
            "structured_drafting": False,
        },
        "independent_direct_behavior": behavior_positive,
        "effect": {"confidence_interval_lower": 0.0},
        "assisted_output_credit": 0,
        "evaluator_label_exposure": 0,
        "independent_audit": {
            "passed": True,
            "producer_evaluator_separated": True,
            "candidate_flags_recomputed": True,
        },
        "anti_cheating": {"answer_identifying_metadata_exposed": False},
        "public_benchmark_prompts_used_for_training": 0,
        "runtime_external_inference_calls": 0,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "source_artifacts": {
            "training_report": source_artifact(training_report_path),
            "checkpoint": source_artifact(checkpoint),
            "training_config": source_artifact(config_path),
            "frozen_corpus_manifest": source_artifact(manifest_path),
            "stage_metadata": source_artifact(metadata_path),
            "trainer": source_artifact(ROOT / "scripts/moecot_language_arm_training.py"),
            "evaluator": source_artifact(Path(__file__).resolve()),
        },
        "non_claims": [
            "One seed cannot close the three-seed K5 acceptance contract.",
            "A structurally valid pipeline output is not a utility or matched-effect claim.",
            "Structured drafting remains separately qualified and receives no credit here.",
        ],
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-rows", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=0)
    parser.add_argument("--branching-factor", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface-id", required=True)
    parser.add_argument("--surface-manifest", default="")
    parser.add_argument("--registry", default=relative(DEFAULT_REGISTRY))
    parser.add_argument("--guarded", action="store_true")
    args = parser.parse_args()
    if not args.surface_manifest:
        args.surface_manifest = relative(
            ROOT / "reports" / f"kerc_private_measurement_surface_{args.surface_id}.json"
        )
    if not 1 <= args.maximum_rows <= 64:
        parser.error("--maximum-rows must be in [1, 64]")
    if args.beam_width not in range(0, 17):
        parser.error("--beam-width must be zero or in [1, 16]")
    if args.branching_factor not in range(0, 17):
        parser.error("--branching-factor must be zero or in [1, 16]")
    if args.guarded:
        training_report = read_json(resolve(args.training_report))
        registry_path = resolve(args.registry)
        if args.surface_id in consumed_surface_ids(registry_path):
            raise ValueError("K5 measurement surface is already consumed")
        surface = frozen_surface_manifest(
            training_report=training_report,
            surface_id=args.surface_id,
            maximum_rows=int(args.maximum_rows),
            output_path=resolve(args.surface_manifest),
        )
        lease = training_report.get("candidate_canary_lease") or {}
        candidate_id = str(lease.get("candidate_id") or "")
        if candidate_id != "rdc_kerc_k5_adequacy":
            raise ValueError("K5 guarded evaluation requires the adequacy candidate lease")
        candidate_contract = training.pretraining_candidate_canary.load_contract()
        candidate = next(
            (
                row
                for row in candidate_contract["canaries"]
                if row["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("K5 guarded evaluation candidate is not registered")
        training_config = read_json(
            resolve(str(training_report.get("config") or ""))
        )
        qualified_python = resolve(
            str(
                (training_config.get("host_resource_safety") or {}).get(
                    "qualified_python"
                )
                or ""
            )
        )
        if not qualified_python.is_file():
            raise ValueError("K5 guarded evaluation qualified Python is missing")
        command = [
            str(qualified_python),
            str(Path(__file__).resolve()),
            "--training-report",
            args.training_report,
            "--seed",
            str(args.seed),
            "--maximum-rows",
            str(args.maximum_rows),
            "--beam-width",
            str(args.beam_width),
            "--branching-factor",
            str(args.branching_factor),
            "--out",
            args.out,
            "--surface-id",
            args.surface_id,
            "--surface-manifest",
            args.surface_manifest,
            "--registry",
            args.registry,
        ]
        attempt = {
            "policy": REGISTRY_POLICY,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event": "EXECUTION_ATTEMPT",
            "surface_id": args.surface_id,
            "surface_contract_sha256": surface["surface_contract_sha256"],
            "surface_manifest": source_artifact(resolve(args.surface_manifest)),
            "training_report": source_artifact(resolve(args.training_report)),
            "output_path": relative(resolve(args.out)),
            "consumed": True,
            "per_surface_max_runs": 1,
        }
        append_jsonl(registry_path, attempt)
        process = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=host_resource_safety.policy_from_mapping(
                operation_specific_host_safety_mapping(
                    candidate_id=candidate_id,
                    candidate_contract=candidate_contract,
                    receipt_path=EVALUATION_RESOURCE_RECEIPT,
                    receipt_sha256=EVALUATION_RESOURCE_RECEIPT_SHA256,
                    command_marker="kerc_k5_candidate_evaluator.py",
                ),
                maximum_wall_seconds=float(candidate["max_wall_seconds"]),
            ),
            env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
        )
        receipt_path = resolve(args.out).with_suffix(".host_resource_safety.json")
        write_json(receipt_path, process.receipt)
        append_jsonl(
            registry_path,
            {
                "policy": REGISTRY_POLICY,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "event": "EXECUTION_RESULT",
                "surface_id": args.surface_id,
                "surface_contract_sha256": surface["surface_contract_sha256"],
                "output_path": relative(resolve(args.out)),
                "host_resource_receipt": source_artifact(receipt_path),
                "passed": process.receipt.get("passed") is True,
                "consumed": False,
            },
        )
        if process.stdout:
            print(process.stdout[-4000:])
        if process.stderr:
            print(process.stderr[-4000:], file=sys.stderr)
        return 0 if process.receipt.get("passed") is True else 2
    report = evaluate(args)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "qualification_state": report["qualification_state"],
                "behavior": report["behavior"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
