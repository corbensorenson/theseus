#!/usr/bin/env python3
"""Freeze and execute the private functional-utility contract for neural seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import moecot_language_arm_training as training

from neural_seed_functional_cases import ARMS, materialize_cases, stable_hash
from neural_seed_functional_verifiers import (
    english_candidate_binding,
    score_english_judgments,
    verify_candidate,
)
from neural_seed_local_english_raters import validate_config as validate_local_rater_config
from neural_seed_functional_consumption import (
    complete_reservation,
    fail_reservation,
    require_completed_artifact,
    reserve_once,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/neural_seed_functional_utility.json"
DEFAULT_FREEZE = ROOT / "configs/neural_seed_functional_utility_freeze.json"
DEFAULT_MANIFEST = ROOT / "reports/private_functional_utility_manifest.json"
DEFAULT_PACKET = ROOT / "reports/private_functional_utility_candidate_packet.json"
DEFAULT_RESULT = ROOT / "reports/private_functional_utility_qualification.json"
TRAINING_SCRIPT = ROOT / "scripts/moecot_language_arm_training.py"
TRAINING_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
GENERATION_WRAPPER = ROOT / "scripts/neural_seed_functional_generate.py"
LOCAL_RATER_CONFIG = ROOT / "configs/neural_seed_local_english_raters.json"
LOCAL_RATER_IMPLEMENTATION = ROOT / "scripts/neural_seed_local_english_raters.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--supersede-freeze-before-results", action="store_true")
    parser.add_argument("--supersede-reason", default="")
    parser.add_argument("--freeze-out", default=str(DEFAULT_FREEZE))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--packet-out", default=str(DEFAULT_PACKET))
    parser.add_argument("--evaluate-candidates", default="")
    parser.add_argument("--blind-english-packet-out", default="")
    parser.add_argument("--judgments", default="")
    parser.add_argument("--judgment-receipt", default="")
    parser.add_argument("--judgment-label", default="")
    parser.add_argument("--compare-results", nargs=3, default=[])
    parser.add_argument("--exact-diagnostic", default="")
    parser.add_argument("--out", default=str(DEFAULT_RESULT))
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    manifest = build_manifest(config, config_path)
    write_json(resolve(args.manifest_out), manifest)
    write_json(resolve(args.packet_out), manifest["candidate_packet"])
    if args.freeze:
        freeze_path = resolve(args.freeze_out)
        if freeze_path.exists():
            immutable = read_json(freeze_path)
            identity_gaps = validate_freeze(manifest, immutable)
            if identity_gaps and not args.supersede_freeze_before_results:
                print(json.dumps({"trigger_state": "RED", "hard_gaps": identity_gaps}, indent=2))
                return 2
            if identity_gaps:
                if not args.supersede_reason.strip():
                    raise ValueError("superseding a freeze requires --supersede-reason")
                if manifest["training_state_at_materialization"]["dense_controls_complete"]:
                    raise ValueError("cannot supersede functional contract after dense-control completion")
                freeze = build_freeze(
                    manifest,
                    config_path,
                    predecessor_sha256=stable_hash(immutable),
                    supersede_reason=args.supersede_reason.strip(),
                )
                write_json(freeze_path, freeze)
        else:
            freeze = build_freeze(manifest, config_path)
            write_json(freeze_path, freeze)
    if args.evaluate_candidates:
        freeze_path = resolve(args.freeze_out)
        if not freeze_path.is_file():
            raise ValueError("functional utility must be frozen before candidate evaluation")
        bundle_path = resolve(args.evaluate_candidates)
        bundle = read_json(bundle_path)
        active_freeze = read_json(freeze_path)
        stage = "final_functional_qualification" if args.judgments else "code_verification_and_blind_packet"
        registry_path = consumption_registry_path(config, active_freeze)
        bundle_identity = {
            "path": relative(bundle_path),
            "sha256": sha256_file(bundle_path),
        }
        require_completed_artifact(
            registry_path,
            stage="candidate_generation",
            artifact_sha256=bundle_identity["sha256"],
        )
        output_path = resolve(args.out)
        prior_code_evaluation: dict[str, Any] | None = None
        prior_code_identity: dict[str, Any] = {}
        judgment_receipt_path = resolve(args.judgment_receipt) if args.judgment_receipt else None
        judgment_receipt_identity: dict[str, Any] = {}
        if args.judgments:
            if not output_path.is_file():
                raise ValueError("final qualification requires the consumed code evaluation")
            prior_code_identity = {
                "path": relative(output_path),
                "sha256": sha256_file(output_path),
            }
            require_completed_artifact(
                registry_path,
                stage="code_verification_and_blind_packet",
                artifact_sha256=prior_code_identity["sha256"],
            )
            prior_code_evaluation = read_json(output_path)
            if not judgment_receipt_path or not judgment_receipt_path.is_file():
                raise ValueError("final qualification requires a consumed judgment receipt")
            judgment_receipt_identity = {
                "path": relative(judgment_receipt_path),
                "sha256": sha256_file(judgment_receipt_path),
            }
            require_completed_artifact(
                registry_path,
                stage="blind_english_local_scoring",
                artifact_sha256=judgment_receipt_identity["sha256"],
            )
        elif output_path.exists():
            raise ValueError("preliminary functional evaluation output already exists")
        reservation = reserve_once(
            registry_path,
            stage=stage,
            identity={
                "freeze_sha256": stable_hash(active_freeze),
                "candidate_bundle_sha256": bundle_identity["sha256"],
                "target_id": bundle.get("target_id"),
                "case_contract_sha256": active_freeze.get("case_contract_sha256"),
                "prior_code_evaluation_sha256": prior_code_identity.get("sha256", ""),
                "judgment_receipt_sha256": judgment_receipt_identity.get("sha256", ""),
            },
        )
        try:
            blind_packet_path = resolve(args.blind_english_packet_out) if args.blind_english_packet_out else None
            if blind_packet_path:
                blind_packet = build_blind_english_packet(config, manifest, bundle, active_freeze)
                write_json(blind_packet_path, blind_packet)
                if blind_packet["trigger_state"] != "GREEN":
                    raise ValueError("blind English packet failed its frozen contract")
            judgments_path = resolve(args.judgments) if args.judgments else None
            judgments = read_jsonl(judgments_path) if judgments_path else []
            judgment_receipt = read_json(judgment_receipt_path) if judgment_receipt_path else {}
            judgments_identity = (
                {
                    "path": relative(judgments_path),
                    "sha256": sha256_file(judgments_path),
                    "row_count": len(judgments),
                }
                if judgments_path
                else {}
            )
            result = evaluate_bundle(
                config,
                manifest,
                bundle,
                active_freeze,
                judgments,
                judgment_receipt=judgment_receipt,
                judgments_identity=judgments_identity,
                judgment_label=args.judgment_label,
                candidate_bundle_identity=bundle_identity,
                precomputed_code_evaluation=prior_code_evaluation,
                precomputed_code_identity=prior_code_identity,
            )
            write_json(output_path, result)
            complete_reservation(
                registry_path,
                reservation,
                artifact={
                    "path": relative(output_path),
                    "sha256": sha256_file(output_path),
                    "trigger_state": result["trigger_state"],
                    "evaluation_complete": result["evaluation_complete"],
                    "blind_packet": (
                        {
                            "path": relative(blind_packet_path),
                            "sha256": sha256_file(blind_packet_path),
                        }
                        if blind_packet_path
                        else {}
                    ),
                },
            )
        except BaseException as exc:
            close_failed_reservation(registry_path, reservation, exc)
            raise
        print(json.dumps(summary_view(result), indent=2, sort_keys=True))
        return 0 if result["trigger_state"] != "RED" else 2
    if args.compare_results:
        if not args.exact_diagnostic:
            raise ValueError("--compare-results requires --exact-diagnostic")
        freeze_path = resolve(args.freeze_out)
        if not freeze_path.is_file():
            raise ValueError("functional utility must be frozen before architecture comparison")
        result_paths = [resolve(value) for value in args.compare_results]
        exact_path = resolve(args.exact_diagnostic)
        active_freeze = read_json(freeze_path)
        registry_path = consumption_registry_path(config, active_freeze)
        result_sources = [{"path": relative(path), "sha256": sha256_file(path)} for path in result_paths]
        exact_source = {"path": relative(exact_path), "sha256": sha256_file(exact_path)}
        for source in result_sources:
            require_completed_artifact(
                registry_path,
                stage="final_functional_qualification",
                artifact_sha256=source["sha256"],
            )
        reservation = reserve_once(
            registry_path,
            stage="architecture_verdict",
            identity={
                "freeze_sha256": stable_hash(active_freeze),
                "result_sha256s": [row["sha256"] for row in result_sources],
                "exact_diagnostic_sha256": exact_source["sha256"],
            },
        )
        output_path = resolve(args.out)
        try:
            result = compare_qualifications(
                config,
                [read_json(path) for path in result_paths],
                read_json(exact_path),
                active_freeze,
                result_sources=result_sources,
                exact_source=exact_source,
                contract_gaps=validate_freeze(manifest, active_freeze),
            )
            write_json(output_path, result)
            complete_reservation(
                registry_path,
                reservation,
                artifact={
                    "path": relative(output_path),
                    "sha256": sha256_file(output_path),
                    "trigger_state": result["trigger_state"],
                    "decision": result.get("decision"),
                },
            )
        except BaseException as exc:
            close_failed_reservation(registry_path, reservation, exc)
            raise
        print(json.dumps(summary_view(result), indent=2, sort_keys=True))
        return 0 if result["trigger_state"] != "RED" else 2
    print(json.dumps(summary_view(manifest), indent=2, sort_keys=True))
    return 0 if manifest["trigger_state"] == "GREEN" else 2


def build_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    gaps = validate_config(config)
    cases = materialize_cases(config)
    expected = len(ARMS) * len(next(iter(config["arms"].values()))["families"]) * int(config["variants_per_family"])
    if len(cases) != expected:
        gaps.append("case_count_mismatch")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        gaps.append("duplicate_case_id")
    for arm in ARMS:
        if sum(case["arm_id"] == arm for case in cases) != int(config["expected_cases_per_arm"]):
            gaps.append(f"arm_case_count_mismatch:{arm}")
    overlap = source_disjoint_audit(config, cases)
    gaps.extend(overlap["hard_gaps"])
    packet_rows = [case["model_visible"] for case in cases]
    visible_keys = set(config["generator_view"])
    if any(set(row) != visible_keys for row in packet_rows):
        gaps.append("generator_packet_field_mismatch")
    forbidden = set(config["evaluator_only_fields"]) | {
        "task_family", "category", "solution", "solution_body", "tests", "hidden_tests",
        "expected", "answer", "source_task_id", "return_shape", "type_family", "required_constructs",
    }
    leaked = sorted({key for row in packet_rows for key in row if key in forbidden})
    if leaked:
        gaps.append("generator_packet_evaluator_metadata_leak:" + ",".join(leaked))
    case_contract_sha = stable_hash(
        [{key: value for key, value in case.items() if key != "model_visible"} for case in cases]
    )
    candidate_packet = {
        "policy": "project_theseus_private_functional_candidate_packet_v1",
        "contract_sha256": case_contract_sha,
        "generator_visible_fields": list(config["generator_view"]),
        "rows": packet_rows,
        "row_count": len(packet_rows),
        "evaluator_metadata_present": False,
        "public_benchmark_payload_count": 0,
    }
    campaign = campaign_binding(config)
    training_state = current_training_state(config, campaign)
    toolchains = toolchain_identity()
    return {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "frozen_contract_readiness",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "compiler": relative(Path(__file__).resolve()),
        "compiler_sha256": sha256_file(Path(__file__).resolve()),
        "case_compiler_sha256": sha256_file(ROOT / "scripts/neural_seed_functional_cases.py"),
        "verifier_sha256": sha256_file(ROOT / "scripts/neural_seed_functional_verifiers.py"),
        "generation_wrapper_sha256": sha256_file(GENERATION_WRAPPER),
        "training_generator_sha256": sha256_file(TRAINING_SCRIPT),
        "local_english_rater_config_sha256": sha256_file(LOCAL_RATER_CONFIG),
        "local_english_rater_implementation_sha256": sha256_file(LOCAL_RATER_IMPLEMENTATION),
        "toolchain_identity": toolchains,
        "toolchain_identity_sha256": stable_hash(toolchains),
        "candidate_id": campaign["candidate_id"],
        "training_config": campaign["training_config"],
        "training_config_sha256": campaign["training_config_sha256"],
        "training_base_config_sha256": campaign["training_base_config_sha256"],
        "training_stage_signature": campaign["training_stage_signature"],
        "checkpoint_root": campaign["checkpoint_root"],
        "case_contract_sha256": case_contract_sha,
        "case_count": len(cases),
        "cases_by_arm": {arm: sum(case["arm_id"] == arm for case in cases) for arm in ARMS},
        "candidate_packet": candidate_packet,
        "candidate_packet_sha256": stable_hash(candidate_packet),
        "consumption": dict(config.get("consumption") or {}),
        "source_disjoint_audit": overlap,
        "training_state_at_materialization": training_state,
        "evaluator_cases": cases,
        "hard_gaps": gaps,
        "boundaries": {
            **config["boundaries"],
            "generator_sees_verifier": False,
            "generator_sees_task_family": False,
            "public_benchmark_payload_count": 0,
            "capability_claim": "NOT_EVALUATED",
        },
    }


def build_blind_english_packet(
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    gaps = validate_freeze(manifest, freeze)
    cases = {case["case_id"]: case for case in manifest["evaluator_cases"]}
    rows = bundle.get("candidates") if isinstance(bundle.get("candidates"), list) else []
    candidate_rows = {str(row.get("case_id") or ""): row for row in rows}
    if len(candidate_rows) != len(rows) or set(candidate_rows) != set(cases):
        gaps.append("candidate_case_set_invalid_for_blind_packet")
    provenance = audit_candidate_provenance(bundle, freeze, cases)
    gaps.extend(provenance["hard_gaps"])
    items = []
    if not gaps:
        for case in cases.values():
            if case["arm_id"] != "english":
                continue
            candidate = str(candidate_rows[case["case_id"]].get("output") or "")
            binding = english_candidate_binding(case["case_id"], candidate)
            items.append(
                {
                    "blind_item_id": binding["blind_item_id"],
                    "case_id": case["case_id"],
                    "prompt": case["prompt"],
                    "candidate_output": candidate,
                    "candidate_sha256": binding["candidate_sha256"],
                    "dimensions": list(config["english_scoring"]["dimensions"]),
                    "score_scale": list(config["english_scoring"]["score_scale"]),
                }
            )
        order_key = stable_hash(
            {
                "policy": "project_theseus_blind_english_packet_order_v1",
                "freeze_sha256": stable_hash(freeze),
                "candidate_hashes": sorted(item["candidate_sha256"] for item in items),
            }
        )
        items.sort(
            key=lambda item: hashlib.sha256(
                f"{order_key}:{item['blind_item_id']}".encode("utf-8")
            ).hexdigest()
        )
    packet_core = {
        "policy": "project_theseus_blind_english_judgment_packet_v1",
        "freeze_sha256": stable_hash(freeze),
        "item_count": len(items),
        "items": items,
        "judgment_required_fields": [
            "case_id",
            "blind_item_id",
            "candidate_sha256",
            "rater_id",
            "scores",
        ],
        "model_identity_present": False,
        "checkpoint_identity_present": False,
        "reference_answer_present": False,
    }
    return {
        **packet_core,
        "created_utc": now(),
        "trigger_state": "GREEN" if not gaps and len(items) == 32 else "RED",
        "packet_sha256": stable_hash(packet_core),
        "candidate_provenance_state": provenance["state"],
        "hard_gaps": sorted(set(gaps)),
    }


def build_freeze(
    manifest: dict[str, Any],
    config_path: Path,
    *,
    predecessor_sha256: str = "",
    supersede_reason: str = "",
) -> dict[str, Any]:
    if manifest["trigger_state"] != "GREEN":
        raise ValueError("cannot freeze an invalid functional contract")
    training = manifest["training_state_at_materialization"]
    return {
        "policy": "project_theseus_private_functional_utility_freeze_v2",
        "frozen_utc": now(),
        "immutable": True,
        "config": relative(config_path),
        "config_sha256": manifest["config_sha256"],
        "compiler_sha256": manifest["compiler_sha256"],
        "case_compiler_sha256": manifest["case_compiler_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "generation_wrapper_sha256": manifest["generation_wrapper_sha256"],
        "training_generator_sha256": manifest["training_generator_sha256"],
        "local_english_rater_config_sha256": manifest["local_english_rater_config_sha256"],
        "local_english_rater_implementation_sha256": manifest["local_english_rater_implementation_sha256"],
        "toolchain_identity_sha256": manifest["toolchain_identity_sha256"],
        "case_contract_sha256": manifest["case_contract_sha256"],
        "candidate_packet_sha256": manifest["candidate_packet_sha256"],
        "candidate_id": manifest["candidate_id"],
        "training_config": manifest["training_config"],
        "training_config_sha256": manifest["training_config_sha256"],
        "training_base_config_sha256": manifest["training_base_config_sha256"],
        "training_stage_signature": manifest["training_stage_signature"],
        "checkpoint_root": manifest["checkpoint_root"],
        "case_count": manifest["case_count"],
        "cases_by_arm": manifest["cases_by_arm"],
        "dense_controls_complete_at_freeze": training["dense_controls_complete"],
        "training_state_at_freeze": training,
        "evaluation_state": "NOT_EVALUATED",
        "source_disjoint": True,
        "consumed_case_count": 0,
        "capability_claim": "NOT_EVALUATED",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "templates_renderers_routers_tools_credit": 0,
        "consumption_registry": (manifest.get("consumption") or {}).get("registry"),
        "consumption_policy_sha256": stable_hash(manifest.get("consumption") or {}),
        "supersedes_freeze_sha256": predecessor_sha256,
        "supersede_reason": supersede_reason,
    }


def source_disjoint_audit(config: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    root = resolve(config["source_disjoint"]["supervision_root"])
    prompt_hashes: dict[str, str] = {}
    normalized: dict[str, str] = {}
    target_hashes: set[str] = set()
    prior_prompt_hashes: set[str] = set()
    prior_normalized_prompts: set[str] = set()
    files = []
    rows_scanned = 0
    for split in config["source_disjoint"]["scan_splits"]:
        for path in sorted((root / split).glob("*.jsonl")):
            files.append({"path": relative(path), "sha256": sha256_file(path)})
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rows_scanned += 1
                    prompt = str(row.get("prompt") or "")
                    prompt_hashes[hashlib.sha256(prompt.encode()).hexdigest()] = str(row.get("row_id") or "")
                    normalized[normalize_text(prompt)] = str(row.get("row_id") or "")
                    target_hash = str(row.get("target_sha256") or "")
                    if target_hash:
                        target_hashes.add(target_hash)
    gaps = []
    exact = []
    normalized_hits = []
    target_as_prompt = []
    prior_surface_overlap = []
    for packet_value in config["source_disjoint"].get("prior_candidate_packets") or []:
        packet_ref = packet_value if isinstance(packet_value, dict) else {
            "path": str(packet_value), "sha256": ""
        }
        packet_path = resolve(str(packet_ref.get("path") or ""))
        if (
            not packet_path.is_file()
            or sha256_file(packet_path) != str(packet_ref.get("sha256") or "")
        ):
            gaps.append("prior_functional_surface_identity_mismatch")
            continue
        packet = read_json(packet_path)
        for row in packet.get("rows") or []:
            prompt = str(row.get("prompt") or "")
            if prompt:
                prior_prompt_hashes.add(hashlib.sha256(prompt.encode()).hexdigest())
                prior_normalized_prompts.add(normalize_text(prompt))
    for case in cases:
        prompt_hash = case["prompt_sha256"]
        if prompt_hash in prompt_hashes:
            exact.append({"case_id": case["case_id"], "row_id": prompt_hashes[prompt_hash]})
        norm = normalize_text(case["prompt"])
        if norm in normalized:
            normalized_hits.append({"case_id": case["case_id"], "row_id": normalized[norm]})
        if prompt_hash in target_hashes:
            target_as_prompt.append(case["case_id"])
        if prompt_hash in prior_prompt_hashes or norm in prior_normalized_prompts:
            prior_surface_overlap.append(case["case_id"])
    if exact:
        gaps.append("exact_supervision_prompt_overlap")
    if normalized_hits:
        gaps.append("normalized_supervision_prompt_overlap")
    if target_as_prompt:
        gaps.append("supervision_target_reused_as_prompt")
    if prior_surface_overlap:
        gaps.append("prior_functional_surface_prompt_overlap")
    return {
        "state": "GREEN" if not gaps else "RED",
        "rows_scanned": rows_scanned,
        "files": files,
        "exact_prompt_overlaps": exact,
        "normalized_prompt_overlaps": normalized_hits,
        "target_hash_as_prompt": target_as_prompt,
        "prior_surface_prompt_overlap": prior_surface_overlap,
        "hard_gaps": gaps,
    }


def campaign_binding(config: dict[str, Any]) -> dict[str, Any]:
    binding = config.get("campaign_binding") or {}
    training_config_path = resolve(str(binding.get("training_config") or TRAINING_CONFIG))
    training_config = training.bind_scale_preregistration(read_json(training_config_path))
    base_path = resolve(str(training_config["base_config"]))
    stage_metadata_path = resolve(str(training_config["stage_dir"])) / "stage_metadata_v1.json"
    stage_metadata = read_json(stage_metadata_path) if stage_metadata_path.is_file() else {}
    candidate_id = str(binding.get("candidate_id") or "")
    if not candidate_id or candidate_id != str(
        (training_config.get("scale_preregistration") or {}).get("candidate_id") or ""
    ):
        raise ValueError("functional campaign candidate does not match training owner")
    return {
        "candidate_id": candidate_id,
        "training_config": relative(training_config_path),
        "training_config_sha256": sha256_file(training_config_path),
        "training_base_config_sha256": sha256_file(base_path),
        "training_stage_signature": str(stage_metadata.get("stage_signature") or ""),
        "checkpoint_root": relative(resolve(str(training_config["checkpoint_root"]))),
    }


def current_training_state(
    config: dict[str, Any], campaign: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_root = resolve(str(campaign["checkpoint_root"]))
    controls = {}
    for target in ("dense_total_parameter", "dense_active_parameter"):
        directory = checkpoint_root / target
        receipt = directory / "training_receipt.json"
        heartbeat = directory / "training_heartbeat.json"
        receipt_row = read_json(receipt) if receipt.is_file() else {}
        controls[target] = {
            "receipt": relative(receipt),
            "receipt_sha256": sha256_file(receipt) if receipt.is_file() else "",
            "heartbeat": relative(heartbeat),
            "heartbeat_sha256": sha256_file(heartbeat) if heartbeat.is_file() else "",
            "complete": bool(receipt_row.get("complete")),
            "optimizer_steps": int(receipt_row.get("optimizer_steps") or 0),
            "plan_sha256": receipt_row.get("plan_sha256"),
        }
    return {
        "candidate_id": campaign["candidate_id"],
        "checkpoint_root": relative(checkpoint_root),
        "controls": controls,
        "dense_controls_complete": all(row["complete"] for row in controls.values()),
        "functional_contract_frozen_before_control_completion": not all(row["complete"] for row in controls.values()),
    }


def evaluate_bundle(
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    freeze: dict[str, Any],
    judgments: list[dict[str, Any]],
    *,
    judgment_receipt: dict[str, Any] | None = None,
    judgments_identity: dict[str, Any] | None = None,
    judgment_label: str = "",
    candidate_bundle_identity: dict[str, Any] | None = None,
    precomputed_code_evaluation: dict[str, Any] | None = None,
    precomputed_code_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gaps = validate_freeze(manifest, freeze)
    cases = {case["case_id"]: case for case in manifest["evaluator_cases"]}
    rows = bundle.get("candidates") if isinstance(bundle.get("candidates"), list) else []
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        gaps.append("duplicate_candidate_case")
    if set(ids) != set(cases):
        gaps.append("candidate_case_set_mismatch")
    provenance = audit_candidate_provenance(bundle, freeze, cases)
    gaps.extend(provenance["hard_gaps"])
    outputs = {str(row.get("case_id") or ""): str(row.get("output") or "") for row in rows}
    generation_ms_by_case = {
        str(row.get("case_id") or ""): float(row.get("generation_duration_ms") or 0.0)
        for row in rows
    }
    candidate_bundle_identity = dict(candidate_bundle_identity or {})
    verifier_rows: list[dict[str, Any]] = []
    code_reused = precomputed_code_evaluation is not None
    if code_reused:
        verifier_rows, reuse_gaps = validate_precomputed_code_evaluation(
            precomputed_code_evaluation or {},
            precomputed_code_identity or {},
            candidate_bundle_identity,
            freeze,
            cases,
            outputs,
            generation_ms_by_case,
        )
        gaps.extend(reuse_gaps)
    elif not gaps:
        verifier_rows = [
            verify_candidate(cases[case_id], outputs[case_id], config)
            for case_id in sorted(cases)
            if cases[case_id]["arm_id"] != "english"
        ]
        for verifier_row in verifier_rows:
            case_id = str(verifier_row["case_id"])
            verifier_row["candidate_sha256"] = hashlib.sha256(
                outputs[case_id].encode("utf-8")
            ).hexdigest()
            verifier_row["generation_duration_ms"] = generation_ms_by_case[
                case_id
            ]
            verifier_row["end_to_end_duration_ms"] = round(
                float(verifier_row["generation_duration_ms"])
                + float(verifier_row.get("duration_ms") or 0.0),
                6,
            )
    judgment_audit = audit_local_english_judgments(
        config,
        manifest,
        bundle,
        freeze,
        judgments,
        judgment_receipt or {},
        judgments_identity or {},
        judgment_label,
    )
    gaps.extend(judgment_audit["hard_gaps"])
    english = score_english_judgments(list(cases.values()), outputs, judgments, config) if judgments else {
        "valid": False, "faults": ["blind_english_judgments_pending"], "results": [], "quadratic_weighted_kappa": None, "passed": 0, "total": 32
    }
    code_rows = verifier_rows
    by_arm = {}
    for arm in ARMS:
        arm_rows = english["results"] if arm == "english" else [row for row in code_rows if row["arm_id"] == arm]
        by_arm[arm] = metric_summary(arm_rows, expected=int(config["expected_cases_per_arm"]))
    all_complete = not gaps and english["valid"] and len(code_rows) == 128
    passed = sum(row["passed"] for row in code_rows) + int(english["passed"])
    total = len(code_rows) + (int(english["total"]) if english["valid"] else 0)
    code_generation_ms = sum(float(row.get("generation_duration_ms") or 0.0) for row in code_rows)
    code_verification_ms = sum(float(row.get("duration_ms") or 0.0) for row in code_rows)
    code_load_ms = code_checkpoint_load_duration_ms(bundle)
    code_passed = sum(bool(row.get("passed")) for row in code_rows)
    warm_duration_seconds = (code_generation_ms + code_verification_ms) / 1000.0
    cold_duration_seconds = (code_load_ms + code_generation_ms + code_verification_ms) / 1000.0
    return {
        "policy": "project_theseus_private_functional_utility_qualification_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all_complete else ("RED" if gaps else "YELLOW"),
        "evaluation_complete": all_complete,
        "evaluation_stage": (
            "final_functional_qualification"
            if judgments
            else "code_verification_and_blind_packet"
        ),
        "candidate_bundle_identity": candidate_bundle_identity,
        "code_evaluation_reused": code_reused,
        "code_evaluation_source": dict(precomputed_code_identity or {}),
        "freeze_sha256": stable_hash(freeze),
        "candidate_provenance": provenance,
        "english_judgment_provenance": judgment_audit,
        "summary": {
            "functional_pass_rate": passed / total if total else None,
            "passed": passed,
            "total_scored": total,
            "tail_floor": min((row["functional_pass_rate"] for row in by_arm.values() if row["functional_pass_rate"] is not None), default=None),
            "invalid_rate": sum(row.get("fault") in {"syntax_error", "markdown_fence", "candidate_too_large"} for row in code_rows) / len(code_rows) if code_rows else None,
            "timeout_rate": sum(row.get("fault") == "timeout" for row in code_rows) / len(code_rows) if code_rows else None,
            "no_output_rate": sum(row.get("fault") == "no_output" for row in code_rows) / len(code_rows) if code_rows else None,
            "english_inter_rater_agreement": english["quadratic_weighted_kappa"],
            "candidate_budget_per_case": 1,
            "pass_if_any_rate": passed / total if total else None,
            "selected_pass_rate": passed / total if total else None,
            "accepted_verified_output_per_second": (
                code_passed / cold_duration_seconds
                if code_rows and cold_duration_seconds > 0
                else None
            ),
            "accepted_verified_output_per_second_warm": (
                code_passed / warm_duration_seconds
                if code_rows and warm_duration_seconds > 0
                else None
            ),
            "code_checkpoint_load_duration_ms": round(code_load_ms, 6),
            "code_generation_duration_ms": round(code_generation_ms, 6),
            "code_verification_duration_ms": round(code_verification_ms, 6),
            "code_end_to_end_duration_ms_cold": round(
                code_load_ms + code_generation_ms + code_verification_ms, 6
            ),
        },
        "by_arm": by_arm,
        "english": english,
        "rows": verifier_rows,
        "hard_gaps": gaps,
        "boundaries": {
            "candidate_self_declared_flags_trusted": False,
            "candidate_self_declared_timing_trusted": False,
            "timing_source": "freeze_bound_generation_wrapper_monotonic_clock",
            "local_evaluator_inference_calls": int(
                judgment_audit.get("local_evaluator_inference_calls") or 0
            ),
            "local_evaluator_output_admitted_to_training": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "templates_renderers_routers_tools_credit": 0,
            "exact_recovery_is_functional_utility": False,
        },
    }


def audit_candidate_provenance(
    bundle: dict[str, Any],
    freeze: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    gaps = []
    if bundle.get("policy") != "project_theseus_direct_model_candidate_bundle_v1":
        gaps.append("candidate_bundle_policy_mismatch")
    if bundle.get("case_contract_sha256") != freeze["case_contract_sha256"]:
        gaps.append("candidate_bundle_contract_mismatch")
    if bundle.get("candidate_packet_sha256") != freeze["candidate_packet_sha256"]:
        gaps.append("candidate_packet_identity_mismatch")
    if bundle.get("generation_function") != "moecot_language_arm_training.generate_model_text":
        gaps.append("candidate_generation_function_mismatch")
    if bundle.get("generation_wrapper_sha256") != freeze.get("generation_wrapper_sha256"):
        gaps.append("candidate_generation_wrapper_mismatch")
    if bundle.get("training_generator_sha256") != freeze.get("training_generator_sha256"):
        gaps.append("candidate_training_generator_mismatch")
    for key in (
        "templates_renderers_routers_tools_credit",
        "public_training_rows_written",
        "external_inference_calls",
        "fallback_return_count",
    ):
        if int(bundle.get(key, -1)) != 0:
            gaps.append(f"candidate_bundle_nonzero_or_missing_boundary:{key}")
    target_id = str(bundle.get("target_id") or "")
    if target_id == "moecot_system":
        expected_targets = {"shared_trunk", "english", "python", "javascript_typescript", "html_css", "rust"}
    elif target_id in {"dense_active_parameter", "dense_total_parameter"}:
        expected_targets = {target_id}
    else:
        expected_targets = set()
        gaps.append("candidate_bundle_target_invalid")
    artifacts = bundle.get("checkpoint_artifacts") if isinstance(bundle.get("checkpoint_artifacts"), list) else []
    if not artifacts:
        gaps.append("checkpoint_artifacts_missing")
    artifact_ids = [str(row.get("target_id") or "") for row in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        gaps.append("duplicate_checkpoint_artifact_target")
    if set(artifact_ids) != expected_targets:
        gaps.append("checkpoint_artifact_set_mismatch")
    for row in artifacts:
        artifact_target = str(row.get("target_id") or "")
        path = resolve_from(root, str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"checkpoint_identity_mismatch:{artifact_target}")
            continue
        checkpoint_root = resolve_from(
            root,
            str(freeze.get("checkpoint_root") or "checkpoints/moecot_language_seed_v8"),
        )
        receipt_path = checkpoint_root / artifact_target / "training_receipt.json"
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        if not receipt.get("complete"):
            gaps.append(f"checkpoint_receipt_incomplete:{artifact_target}")
        expected_plan = bundle.get("training_plan_sha256") or freeze.get(
            "v8_plan_sha256"
        )
        expected_stage = freeze.get("training_stage_signature") or freeze.get(
            "v8_stage_signature"
        )
        if receipt.get("plan_sha256") != expected_plan:
            gaps.append(f"checkpoint_receipt_plan_mismatch:{artifact_target}")
        if receipt.get("stage_signature") != expected_stage:
            gaps.append(f"checkpoint_receipt_stage_mismatch:{artifact_target}")
        receipt_checkpoint = resolve_from(root, str(receipt.get("checkpoint") or ""))
        if receipt_checkpoint.resolve() != path.resolve() or receipt.get("checkpoint_sha256") != row.get("sha256"):
            gaps.append(f"checkpoint_receipt_artifact_mismatch:{artifact_target}")
    if target_id == "moecot_system":
        artifact_map = {str(row.get("target_id") or ""): row for row in artifacts}
        shared_row = artifact_map.get("shared_trunk") or {}
        shared_path = resolve_from(root, str(shared_row.get("path") or ""))
        for arm in ("english", "python", "javascript_typescript", "html_css", "rust"):
            receipt_path = checkpoint_root / arm / "training_receipt.json"
            receipt = read_json(receipt_path) if receipt_path.is_file() else {}
            declared_path = resolve_from(root, str(receipt.get("shared_trunk_checkpoint") or ""))
            if declared_path.resolve() != shared_path.resolve() or receipt.get("shared_trunk_checkpoint_sha256") != shared_row.get("sha256"):
                gaps.append(f"expert_shared_trunk_binding_mismatch:{arm}")
    candidates = bundle.get("candidates") if isinstance(bundle.get("candidates"), list) else []
    timing = bundle.get("timing") if isinstance(bundle.get("timing"), dict) else {}
    if timing.get("clock") != "time.perf_counter":
        gaps.append("candidate_timing_clock_invalid")
    load_by_target = (
        timing.get("checkpoint_load_duration_ms_by_target")
        if isinstance(timing.get("checkpoint_load_duration_ms_by_target"), dict)
        else {}
    )
    if set(load_by_target) != expected_targets - ({"shared_trunk"} if target_id == "moecot_system" else set()):
        gaps.append("checkpoint_load_timing_target_set_mismatch")
    if not all(nonnegative_finite(value) for value in load_by_target.values()):
        gaps.append("checkpoint_load_timing_invalid")
    for row in candidates:
        case_id = str(row.get("case_id") or "")
        output = str(row.get("output") or "")
        if row.get("output_sha256") != hashlib.sha256(output.encode()).hexdigest():
            gaps.append(f"candidate_output_identity_mismatch:{case_id}")
        case = cases.get(case_id) or {}
        expected_target = case.get("arm_id") if target_id == "moecot_system" else target_id
        if row.get("target_id") != expected_target:
            gaps.append(f"candidate_target_binding_mismatch:{case_id}")
        if not nonnegative_finite(row.get("generation_duration_ms")):
            gaps.append(f"candidate_generation_timing_invalid:{case_id}")
    candidate_duration_total = sum(
        float(row.get("generation_duration_ms") or 0.0) for row in candidates
    )
    load_duration_total = sum(float(value) for value in load_by_target.values())
    if not close_timing_total(timing.get("generation_duration_ms_total"), candidate_duration_total):
        gaps.append("candidate_generation_timing_total_mismatch")
    if not close_timing_total(timing.get("checkpoint_load_duration_ms_total"), load_duration_total):
        gaps.append("checkpoint_load_timing_total_mismatch")
    if not nonnegative_finite(timing.get("wall_duration_ms")) or float(
        timing.get("wall_duration_ms") or 0.0
    ) + 1.0 < candidate_duration_total + load_duration_total:
        gaps.append("candidate_wall_timing_invalid")
    return {
        "state": "GREEN" if not gaps else "RED",
        "bundle_target_id": target_id,
        "checkpoint_artifacts": artifacts,
        "hard_gaps": sorted(set(gaps)),
    }


def audit_local_english_judgments(
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    freeze: dict[str, Any],
    judgments: list[dict[str, Any]],
    receipt: dict[str, Any],
    judgments_identity: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if not judgments:
        return {
            "state": "NOT_EVALUATED",
            "local_evaluator_inference_calls": 0,
            "hard_gaps": [],
        }
    gaps = []
    if receipt.get("policy") != "project_theseus_local_blind_english_judgment_receipt_v1":
        gaps.append("local_judgment_receipt_policy_mismatch")
    if receipt.get("trigger_state") != "GREEN":
        gaps.append("local_judgment_receipt_not_green")
    if receipt.get("config_sha256") != freeze.get("local_english_rater_config_sha256"):
        gaps.append("local_judgment_config_mismatch")
    if receipt.get("implementation_sha256") != freeze.get("local_english_rater_implementation_sha256"):
        gaps.append("local_judgment_implementation_mismatch")
    for key in ("external_inference_calls", "public_training_rows_written"):
        if int(receipt.get(key, -1)) != 0:
            gaps.append(f"local_judgment_nonzero_boundary:{key}")
    if receipt.get("judgments_admitted_to_training") is not False:
        gaps.append("local_judgments_training_boundary_missing")
    if receipt.get("raw_model_responses_retained") is not False:
        gaps.append("local_judgment_raw_response_retained")
    local_calls = int(receipt.get("local_evaluator_inference_calls") or 0)
    if local_calls <= 0:
        gaps.append("local_judgment_inference_calls_missing")
    configured = read_json(LOCAL_RATER_CONFIG)
    primary_ids = {str(row["rater_id"]) for row in configured["primary_raters"]}
    observed_primary = {
        str(row.get("rater_id") or "")
        for row in judgments
        if row.get("adjudicator") is not True
    }
    if observed_primary != primary_ids:
        gaps.append("local_judgment_primary_rater_set_mismatch")
    observed_adjudicators = {
        str(row.get("rater_id") or "")
        for row in judgments
        if row.get("adjudicator") is True
    }
    allowed_adjudicator = str(configured["adjudicator"]["rater_id"])
    if observed_adjudicators - {allowed_adjudicator}:
        gaps.append("local_judgment_adjudicator_mismatch")
    model_receipts = receipt.get("model_receipts") if isinstance(receipt.get("model_receipts"), list) else []
    model_by_id = {str(row.get("rater_id") or ""): row for row in model_receipts}
    for card in [*configured["primary_raters"], configured["adjudicator"]]:
        rater_id = str(card["rater_id"])
        if rater_id == allowed_adjudicator and not observed_adjudicators:
            continue
        model_row = model_by_id.get(rater_id) or {}
        if model_row.get("repo_id") != card["repo_id"] or model_row.get("revision") != card["revision"]:
            gaps.append(f"local_judgment_model_identity_mismatch:{rater_id}")
        if (model_row.get("snapshot_identity") or {}).get("manifest_sha256") in (None, ""):
            gaps.append(f"local_judgment_snapshot_identity_missing:{rater_id}")
    if not label:
        gaps.append("local_judgment_label_missing")
    file_rows = receipt.get("judgment_files") if isinstance(receipt.get("judgment_files"), list) else []
    matches = [row for row in file_rows if row.get("label") == label]
    if len(matches) != 1:
        gaps.append("local_judgment_file_binding_missing")
    else:
        file_row = matches[0]
        for key in ("path", "sha256", "row_count"):
            if file_row.get(key) != judgments_identity.get(key):
                gaps.append(f"local_judgment_file_identity_mismatch:{key}")
        blind_packet = build_blind_english_packet(config, manifest, bundle, freeze)
        if blind_packet.get("trigger_state") != "GREEN":
            gaps.append("local_judgment_blind_packet_invalid")
        if file_row.get("blind_packet_contract_sha256") != blind_packet.get("packet_sha256"):
            gaps.append("local_judgment_blind_packet_mismatch")
    return {
        "state": "GREEN" if not gaps else "RED",
        "receipt_policy": receipt.get("policy"),
        "judgment_label": label,
        "judgment_file": judgments_identity,
        "primary_rater_ids": sorted(observed_primary),
        "adjudicator_ids": sorted(observed_adjudicators),
        "local_evaluator_inference_calls": local_calls,
        "external_inference_calls": 0,
        "hard_gaps": sorted(set(gaps)),
    }


def code_checkpoint_load_duration_ms(bundle: dict[str, Any]) -> float:
    timing = bundle.get("timing") if isinstance(bundle.get("timing"), dict) else {}
    by_target = timing.get("checkpoint_load_duration_ms_by_target")
    if not isinstance(by_target, dict):
        return 0.0
    if str(bundle.get("target_id") or "") == "moecot_system":
        return sum(
            float(value)
            for target, value in by_target.items()
            if target != "english"
        )
    return sum(float(value) for value in by_target.values())


def compare_qualifications(
    config: dict[str, Any],
    qualifications: list[dict[str, Any]],
    exact_diagnostic: dict[str, Any],
    freeze: dict[str, Any],
    *,
    result_sources: list[dict[str, str]] | None = None,
    exact_source: dict[str, str] | None = None,
    contract_gaps: list[str] | None = None,
) -> dict[str, Any]:
    policy = config.get("architecture_verdict") or {}
    expected_targets = tuple(policy.get("required_targets") or ())
    code_arms = tuple(policy.get("required_code_arms") or ())
    expected_freeze = stable_hash(freeze)
    gaps: list[str] = list(contract_gaps or [])
    by_target: dict[str, dict[str, Any]] = {}
    for row in qualifications:
        target = str((row.get("candidate_provenance") or {}).get("bundle_target_id") or "")
        if not target or target in by_target:
            gaps.append("qualification_target_missing_or_duplicate")
            continue
        by_target[target] = row
        if row.get("policy") != "project_theseus_private_functional_utility_qualification_v1":
            gaps.append(f"qualification_policy_mismatch:{target}")
        if row.get("trigger_state") != "GREEN" or row.get("evaluation_complete") is not True:
            gaps.append(f"qualification_incomplete:{target}")
        if row.get("evaluation_stage") != "final_functional_qualification":
            gaps.append(f"qualification_stage_mismatch:{target}")
        if row.get("code_evaluation_reused") is not True:
            gaps.append(f"qualification_code_reuse_missing:{target}")
        code_source = row.get("code_evaluation_source") or {}
        if not code_source.get("path") or not re.fullmatch(
            r"[0-9a-f]{64}", str(code_source.get("sha256") or "")
        ):
            gaps.append(f"qualification_code_source_invalid:{target}")
        if row.get("freeze_sha256") != expected_freeze:
            gaps.append(f"qualification_freeze_mismatch:{target}")
        if (row.get("candidate_provenance") or {}).get("state") != "GREEN":
            gaps.append(f"qualification_provenance_not_green:{target}")
        boundaries = row.get("boundaries") or {}
        for key in (
            "public_training_rows_written",
            "external_inference_calls",
            "templates_renderers_routers_tools_credit",
        ):
            if int(boundaries.get(key, -1)) != 0:
                gaps.append(f"qualification_nonzero_boundary:{target}:{key}")
        for arm in ARMS:
            arm_row = (row.get("by_arm") or {}).get(arm) or {}
            if int(arm_row.get("scored") or 0) != int(config["expected_cases_per_arm"]):
                gaps.append(f"qualification_arm_incomplete:{target}:{arm}")
            if not unit_interval(arm_row.get("functional_pass_rate")):
                gaps.append(f"qualification_arm_rate_invalid:{target}:{arm}")
        summary = row.get("summary") or {}
        for metric in policy.get("cost_dimensions") or []:
            if not nonnegative_finite(summary.get(metric)):
                gaps.append(f"qualification_cost_invalid:{target}:{metric}")
    if set(by_target) != set(expected_targets):
        gaps.append("qualification_target_set_mismatch")
    if exact_diagnostic.get("policy") != "project_theseus_moecot_dense_exact_recovery_diagnostic_v8":
        gaps.append("exact_diagnostic_policy_mismatch")
    if exact_diagnostic.get("trigger_state") != "GREEN" or exact_diagnostic.get("publication_ready") is not True:
        gaps.append("exact_diagnostic_incomplete")
    exact_freeze = exact_diagnostic.get("freeze_identity") or {}
    if exact_freeze.get("functional_case_contract_sha256") != freeze.get("case_contract_sha256"):
        gaps.append("exact_diagnostic_functional_contract_mismatch")
    exact_boundaries = exact_diagnostic.get("boundaries") or {}
    for key in (
        "public_benchmark_payload_count",
        "public_training_rows_written",
        "external_inference_calls",
        "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    ):
        if int(exact_boundaries.get(key, -1)) != 0:
            gaps.append(f"exact_diagnostic_nonzero_boundary:{key}")
    if gaps:
        decision = "INVALID_EVIDENCE"
        recommendation = "REPAIR_EVIDENCE_WITHOUT_REGENERATING_CANDIDATES"
        pareto = {}
    else:
        pareto = {
            "dense_active_over_moecot": pareto_dominates(
                by_target["dense_active_parameter"], by_target["moecot_system"], policy
            ),
            "dense_total_over_moecot": pareto_dominates(
                by_target["dense_total_parameter"], by_target["moecot_system"], policy
            ),
            "moecot_over_dense_active": pareto_dominates(
                by_target["moecot_system"], by_target["dense_active_parameter"], policy
            ),
            "moecot_over_dense_total": pareto_dominates(
                by_target["moecot_system"], by_target["dense_total_parameter"], policy
            ),
        }
        all_code_zero = all(
            int((row.get("by_arm") or {}).get(arm, {}).get("passed") or 0) == 0
            for row in by_target.values()
            for arm in code_arms
        )
        if all_code_zero:
            decision = "FALSIFY_10_8M_ACTIVE_SCALE_RUNG"
            recommendation = "BUILD_DATA_SUPPORTED_50M_TO_100M_ACTIVE_RUNG"
        elif pareto["moecot_over_dense_active"] and pareto["moecot_over_dense_total"]:
            decision = "MOECOT_CONFIRMATION_REQUIRED"
            recommendation = "SPEND_ONE_UNTOUCHED_CONFIRMATION_ON_MOECOT"
        elif pareto["dense_active_over_moecot"] or pareto["dense_total_over_moecot"]:
            decision = "DENSE_HYBRID_CONFIRMATION_REQUIRED"
            recommendation = "SPEND_ONE_UNTOUCHED_CONFIRMATION_ON_DENSE_HYBRID"
        else:
            decision = "UNRESOLVED_CONFIRMATION_REQUIRED"
            recommendation = "USE_ONE_UNTOUCHED_CONFIRMATION_WITHOUT_POST_HOC_WEIGHTING"
    return {
        "policy": "project_theseus_private_functional_architecture_verdict_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "decision": decision,
        "recommendation": recommendation,
        "architecture_selected": False,
        "route_replacement_authorized": False,
        "confirmation_surface_spent": False,
        "functional_results": {
            target: compact_qualification(row) for target, row in by_target.items()
        },
        "pareto": pareto,
        "result_sources": result_sources or [],
        "exact_diagnostic_source": exact_source or {},
        "exact_recovery_semantics": "diagnostic_only_not_functional_utility",
        "hard_gaps": sorted(set(gaps)),
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
            "post_hoc_metric_weighting": False,
            "model_only_and_assisted_channels_separate": True,
        },
    }


def validate_precomputed_code_evaluation(
    prior: dict[str, Any],
    prior_identity: dict[str, Any],
    candidate_bundle_identity: dict[str, Any],
    freeze: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    outputs: dict[str, str],
    generation_ms_by_case: dict[str, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    gaps = []
    if prior.get("policy") != "project_theseus_private_functional_utility_qualification_v1":
        gaps.append("precomputed_code_policy_mismatch")
    if prior.get("evaluation_stage") != "code_verification_and_blind_packet":
        gaps.append("precomputed_code_stage_mismatch")
    if prior.get("evaluation_complete") is not False or prior.get("trigger_state") != "YELLOW":
        gaps.append("precomputed_code_state_mismatch")
    if prior.get("code_evaluation_reused") is not False:
        gaps.append("precomputed_code_must_be_original_execution")
    if prior.get("freeze_sha256") != stable_hash(freeze):
        gaps.append("precomputed_code_freeze_mismatch")
    if prior.get("candidate_bundle_identity") != candidate_bundle_identity:
        gaps.append("precomputed_code_candidate_bundle_mismatch")
    if (prior.get("candidate_provenance") or {}).get("state") != "GREEN":
        gaps.append("precomputed_code_candidate_provenance_invalid")
    if prior.get("hard_gaps"):
        gaps.append("precomputed_code_has_hard_gaps")
    if not prior_identity.get("path") or not re.fullmatch(
        r"[0-9a-f]{64}", str(prior_identity.get("sha256") or "")
    ):
        gaps.append("precomputed_code_source_identity_invalid")
    for key in (
        "public_training_rows_written",
        "external_inference_calls",
        "templates_renderers_routers_tools_credit",
    ):
        if int((prior.get("boundaries") or {}).get(key, -1)) != 0:
            gaps.append(f"precomputed_code_boundary_nonzero:{key}")
    expected = {
        case_id: row
        for case_id, row in cases.items()
        if row["arm_id"] != "english"
    }
    rows = prior.get("rows") if isinstance(prior.get("rows"), list) else []
    by_id = {str(row.get("case_id") or ""): row for row in rows}
    if len(rows) != 128 or len(by_id) != len(rows) or set(by_id) != set(expected):
        gaps.append("precomputed_code_case_set_mismatch")
    for case_id, case in expected.items():
        row = by_id.get(case_id) or {}
        candidate_sha256 = hashlib.sha256(outputs.get(case_id, "").encode("utf-8")).hexdigest()
        if row.get("arm_id") != case["arm_id"]:
            gaps.append(f"precomputed_code_arm_mismatch:{case_id}")
        if row.get("candidate_sha256") != candidate_sha256:
            gaps.append(f"precomputed_code_candidate_mismatch:{case_id}")
        if row.get("passed") not in (True, False):
            gaps.append(f"precomputed_code_pass_state_invalid:{case_id}")
        if not nonnegative_finite(row.get("duration_ms")):
            gaps.append(f"precomputed_code_duration_invalid:{case_id}")
        if not close_timing_total(
            row.get("generation_duration_ms"), generation_ms_by_case.get(case_id, -1.0)
        ):
            gaps.append(f"precomputed_code_generation_timing_mismatch:{case_id}")
    return ([] if gaps else rows), sorted(set(gaps))


def pareto_dominates(
    candidate: dict[str, Any], baseline: dict[str, Any], policy: dict[str, Any]
) -> bool:
    candidate_values = [
        float((candidate.get("by_arm") or {})[arm]["functional_pass_rate"])
        for arm in ARMS
    ]
    baseline_values = [
        float((baseline.get("by_arm") or {})[arm]["functional_pass_rate"])
        for arm in ARMS
    ]
    for metric in policy.get("cost_dimensions") or []:
        candidate_values.append(float((candidate.get("summary") or {})[metric]))
        baseline_values.append(float((baseline.get("summary") or {})[metric]))
    return all(a >= b for a, b in zip(candidate_values, baseline_values)) and any(
        a > b for a, b in zip(candidate_values, baseline_values)
    )


def compact_qualification(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("candidate_provenance") or {}
    english = row.get("english") or {}
    return {
        "target_id": provenance.get("bundle_target_id"),
        "trigger_state": row.get("trigger_state"),
        "evaluation_complete": row.get("evaluation_complete"),
        "evaluation_stage": row.get("evaluation_stage"),
        "code_evaluation_reused": row.get("code_evaluation_reused"),
        "code_evaluation_source": row.get("code_evaluation_source") or {},
        "freeze_sha256": row.get("freeze_sha256"),
        "summary": row.get("summary") or {},
        "by_arm": row.get("by_arm") or {},
        "candidate_provenance": {
            "state": provenance.get("state"),
            "checkpoint_artifacts": provenance.get("checkpoint_artifacts") or [],
            "hard_gaps": provenance.get("hard_gaps") or [],
        },
        "english": {
            "valid": english.get("valid"),
            "faults": english.get("faults") or [],
            "quadratic_weighted_kappa": english.get("quadratic_weighted_kappa"),
            "passed": english.get("passed"),
            "total": english.get("total"),
        },
        "boundaries": row.get("boundaries") or {},
    }


def unit_interval(value: Any) -> bool:
    return nonnegative_finite(value) and float(value) <= 1.0


def nonnegative_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def close_timing_total(value: Any, expected: float) -> bool:
    return nonnegative_finite(value) and abs(float(value) - expected) <= 1e-3


def validate_freeze(manifest: dict[str, Any], freeze: dict[str, Any]) -> list[str]:
    gaps = []
    for key in ("config_sha256", "compiler_sha256", "case_compiler_sha256", "verifier_sha256", "generation_wrapper_sha256", "training_generator_sha256", "local_english_rater_config_sha256", "local_english_rater_implementation_sha256", "toolchain_identity_sha256", "case_contract_sha256", "candidate_packet_sha256", "candidate_id", "training_config", "training_config_sha256", "training_base_config_sha256", "training_stage_signature", "checkpoint_root"):
        if manifest.get(key) != freeze.get(key):
            gaps.append(f"freeze_identity_mismatch:{key}")
    if freeze.get("consumption_registry") != (manifest.get("consumption") or {}).get("registry"):
        gaps.append("freeze_identity_mismatch:consumption_registry")
    if freeze.get("consumption_policy_sha256") != stable_hash(manifest.get("consumption") or {}):
        gaps.append("freeze_identity_mismatch:consumption_policy_sha256")
    return gaps


def validate_config(config: dict[str, Any]) -> list[str]:
    gaps = []
    if config.get("policy") != "project_theseus_private_functional_utility_contract_v1":
        gaps.append("policy_mismatch")
    if tuple(config.get("arms", {}).keys()) != ARMS:
        gaps.append("arm_contract_mismatch")
    if "task_family" in config.get("generator_view", []):
        gaps.append("task_family_visible_to_generator")
    english = config.get("english_scoring") or {}
    if int(english.get("minimum_raters") or 0) != 2:
        gaps.append("english_primary_rater_count_must_equal_two")
    for key in (
        "model_identity_hidden_from_raters",
        "reference_answer_hidden_from_raters",
        "candidate_content_binding_required",
        "distinct_primary_raters_required",
        "adjudication_only_after_threshold_disagreement",
    ):
        if english.get(key) is not True:
            gaps.append(f"english_scoring_boundary_missing:{key}")
    if english.get("blind_packet_policy") != "project_theseus_blind_english_judgment_packet_v1":
        gaps.append("english_blind_packet_policy_mismatch")
    if english.get("local_rater_config") != relative(LOCAL_RATER_CONFIG):
        gaps.append("english_local_rater_config_mismatch")
    if english.get("local_rater_implementation") != relative(LOCAL_RATER_IMPLEMENTATION):
        gaps.append("english_local_rater_implementation_mismatch")
    if english.get("local_judgment_receipt_required") is not True:
        gaps.append("english_local_judgment_receipt_boundary_missing")
    local_rater = read_json(LOCAL_RATER_CONFIG)
    gaps.extend(
        f"english_local_rater:{gap}" for gap in validate_local_rater_config(local_rater)
    )
    local_scoring = local_rater.get("scoring") or {}
    if local_scoring.get("dimensions") != english.get("dimensions"):
        gaps.append("english_local_rater_dimension_mismatch")
    if int(local_scoring.get("adjudication_required_score_delta") or -1) != int(
        english.get("adjudication_required_score_delta") or -2
    ):
        gaps.append("english_local_rater_adjudication_delta_mismatch")
    verdict = config.get("architecture_verdict") or {}
    if tuple(verdict.get("required_targets") or ()) != (
        "moecot_system",
        "dense_active_parameter",
        "dense_total_parameter",
    ):
        gaps.append("architecture_verdict_target_contract_mismatch")
    if tuple(verdict.get("required_code_arms") or ()) != tuple(ARMS[1:]):
        gaps.append("architecture_verdict_code_arm_contract_mismatch")
    if verdict.get("route_replacement_before_confirmation_allowed") is not False:
        gaps.append("architecture_verdict_confirmation_boundary_missing")
    consumption = config.get("consumption") or {}
    if consumption.get("registry") != "reports/private_functional_consumption_registry.jsonl":
        gaps.append("functional_consumption_registry_mismatch")
    for key in (
        "append_only",
        "reserve_before_execution",
        "failed_reservation_remains_consumed",
        "duplicate_identity_refused",
    ):
        if consumption.get(key) is not True:
            gaps.append(f"functional_consumption_boundary_missing:{key}")
    for key, value in config.get("boundaries", {}).items():
        if key.endswith("count") or key in {"public_training_rows_written", "external_inference_calls", "templates_renderers_routers_tools_credit"}:
            if isinstance(value, int) and value != 0:
                gaps.append(f"nonzero_boundary:{key}")
    return gaps


def metric_summary(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    return {
        "passed": sum(bool(row.get("passed")) for row in rows),
        "scored": len(rows),
        "expected": expected,
        "functional_pass_rate": sum(bool(row.get("passed")) for row in rows) / len(rows) if rows else None,
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def consumption_registry_path(config: dict[str, Any], freeze: dict[str, Any]) -> Path:
    configured = str((config.get("consumption") or {}).get("registry") or "")
    frozen = str(freeze.get("consumption_registry") or "")
    if not configured or configured != frozen:
        raise ValueError("functional consumption registry is not freeze-bound")
    return resolve(configured)


def close_failed_reservation(
    registry_path: Path,
    reservation: dict[str, Any],
    exc: BaseException,
) -> None:
    try:
        fail_reservation(
            registry_path,
            reservation,
            fault=f"{type(exc).__name__}:{exc}",
        )
    except Exception:
        pass


def toolchain_identity() -> dict[str, Any]:
    commands = {
        "python": [shutil.which("python3") or "/usr/bin/python3", "--version"],
        "deno": [shutil.which("deno") or "deno", "--version"],
        "cargo": [shutil.which("cargo") or "cargo", "--version"],
        "rustc": [shutil.which("rustc") or "rustc", "--version"],
        "tidy": [shutil.which("tidy") or "/usr/bin/tidy", "-v"],
        "chrome": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
    }
    rows = {}
    for name, command in commands.items():
        executable = Path(command[0])
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
        rows[name] = {
            "executable": str(executable.resolve()) if executable.exists() else command[0],
            "version": completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    return rows


def summary_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "policy",
            "created_utc",
            "trigger_state",
            "mode",
            "case_count",
            "cases_by_arm",
            "decision",
            "recommendation",
            "summary",
            "hard_gaps",
            "boundaries",
        )
        if key in report
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
