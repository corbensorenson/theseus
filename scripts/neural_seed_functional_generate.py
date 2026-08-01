#!/usr/bin/env python3
"""Generate direct model-only candidates for the frozen functional suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import moecot_language_arm_training as training
from neural_seed_functional_utility import read_json, resolve, sha256_file, stable_hash
from neural_seed_functional_consumption import (
    complete_reservation,
    fail_reservation,
    reserve_once,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_FREEZE = ROOT / "configs/neural_seed_57m_functional_utility_freeze.json"
DEFAULT_PACKET = ROOT / "reports/private_functional_utility_candidate_packet.json"
TARGETS = ("moecot_system", "dense_active_parameter", "dense_total_parameter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    contract = build_generation_contract(
        target_id=args.target,
        config_path=resolve(args.config),
        freeze_path=resolve(args.freeze),
        packet_path=resolve(args.packet),
    )
    if args.gate:
        print(json.dumps(contract_summary(contract), indent=2, sort_keys=True))
        return 0 if contract["trigger_state"] == "GREEN" else 2
    if contract["trigger_state"] != "GREEN":
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 2
    output_path = resolve(args.out)
    if output_path.exists():
        raise ValueError(f"candidate output already exists: {relative(output_path)}")
    registry_path = resolve(str(contract["consumption_registry"]))
    reservation = reserve_once(
        registry_path,
        stage="candidate_generation",
        identity={
            "freeze_sha256": contract["freeze_sha256"],
            "target_id": contract["target_id"],
            "case_contract_sha256": contract["case_contract_sha256"],
            "candidate_packet_sha256": contract["candidate_packet_sha256"],
            "checkpoint_artifacts": contract["checkpoint_artifacts"],
        },
    )
    try:
        bundle = generate(contract)
        write_json(output_path, bundle)
        complete_reservation(
            registry_path,
            reservation,
            artifact={
                "path": relative(output_path),
                "sha256": sha256_file(output_path),
                "candidate_count": bundle["candidate_count"],
            },
        )
    except BaseException as exc:
        try:
            fail_reservation(
                registry_path,
                reservation,
                fault=f"{type(exc).__name__}:{exc}",
            )
        except Exception:
            pass
        raise
    print(
        json.dumps(
            {
                key: bundle[key]
                for key in (
                    "policy",
                    "created_utc",
                    "target_id",
                    "candidate_count",
                    "case_contract_sha256",
                    "hard_gaps",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_generation_contract(
    *, target_id: str, config_path: Path, freeze_path: Path, packet_path: Path
) -> dict[str, Any]:
    gaps: list[str] = []
    freeze = read_json(freeze_path) if freeze_path.is_file() else {}
    packet = read_json(packet_path) if packet_path.is_file() else {}
    config = read_json(config_path)
    base = read_json(resolve(str(config["base_config"])))
    utility_config_path = resolve(
        str(
            freeze.get("config") or ROOT / "configs/neural_seed_functional_utility.json"
        )
    )
    utility_config = read_json(utility_config_path)
    completion = utility_config.get("candidate_generation") or {}
    if completion.get("normal_completion") != ["model_eos"]:
        gaps.append("candidate_generation_normal_completion_mismatch")
    if completion.get("project_selected_quality_token_cap") is not None:
        gaps.append("candidate_generation_project_token_cap_present")
    if completion.get("serialization_complete_prefix_is_completion") is not False:
        gaps.append("candidate_generation_prefix_completion_not_forbidden")
    declared_context_window = int(
        (base.get("tokenization") or {}).get("max_sequence_tokens") or 0
    )
    if declared_context_window <= 0:
        gaps.append("candidate_generation_declared_context_window_missing")
    plan = training.build_plan(config, config_path=config_path)
    if freeze.get("policy") not in {
        "project_theseus_private_functional_utility_freeze_v1",
        "project_theseus_private_functional_utility_freeze_v2",
    }:
        gaps.append("functional_freeze_missing_or_invalid")
    if freeze.get("candidate_packet_sha256") != stable_hash(packet):
        gaps.append("candidate_packet_freeze_mismatch")
    if freeze.get("generation_wrapper_sha256") != sha256_file(Path(__file__).resolve()):
        gaps.append("generation_wrapper_freeze_mismatch")
    if freeze.get("training_generator_sha256") != sha256_file(
        ROOT / "scripts/moecot_language_arm_training.py"
    ):
        gaps.append("training_generator_freeze_mismatch")
    consumption_registry = str(freeze.get("consumption_registry") or "")
    if consumption_registry != "reports/private_functional_consumption_registry.jsonl":
        gaps.append("functional_consumption_registry_missing_or_invalid")
    if packet.get("generator_visible_fields") != ["case_id", "arm_id", "prompt"]:
        gaps.append("candidate_packet_visible_fields_mismatch")
    rows = packet.get("rows") if isinstance(packet.get("rows"), list) else []
    if len(rows) != int(freeze.get("case_count") or 0):
        gaps.append("candidate_packet_case_count_mismatch")
    if any(set(row) != {"case_id", "arm_id", "prompt"} for row in rows):
        gaps.append("candidate_packet_contains_evaluator_metadata")
    expected_stage = freeze.get("training_stage_signature") or freeze.get(
        "v8_stage_signature"
    )
    if (plan.get("stage") or {}).get("stage_signature") != expected_stage:
        gaps.append("training_stage_freeze_mismatch")
    if freeze.get("candidate_id") and (
        (plan.get("scale_preregistration") or {}).get("candidate_id")
        != freeze.get("candidate_id")
    ):
        gaps.append("training_candidate_freeze_mismatch")
    target_ids = list(training.ARM_IDS) if target_id == "moecot_system" else [target_id]
    targets = []
    checkpoint_artifacts = []
    for item in target_ids:
        target = (plan.get("targets") or {}).get(item) or {}
        receipt_path = resolve(str(target.get("receipt") or ""))
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        if not receipt.get("complete"):
            gaps.append(f"checkpoint_incomplete:{item}")
        if receipt.get("plan_sha256") != plan.get("plan_sha256"):
            gaps.append(f"checkpoint_plan_mismatch:{item}")
        if receipt.get("stage_signature") != expected_stage:
            gaps.append(f"checkpoint_stage_mismatch:{item}")
        checkpoint = resolve(
            str(receipt.get("checkpoint") or target.get("checkpoint") or "")
        )
        if not checkpoint.is_file() or sha256_file(checkpoint) != str(
            receipt.get("checkpoint_sha256") or ""
        ):
            gaps.append(f"checkpoint_identity_mismatch:{item}")
        else:
            checkpoint_artifacts.append(
                {
                    "target_id": item,
                    "path": relative(checkpoint),
                    "sha256": sha256_file(checkpoint),
                }
            )
        shared = (
            resolve(str(target.get("shared_trunk_checkpoint") or ""))
            if target.get("shared_trunk_checkpoint")
            else None
        )
        if shared:
            shared_receipt_path = shared.parent / "training_receipt.json"
            shared_receipt = (
                read_json(shared_receipt_path) if shared_receipt_path.is_file() else {}
            )
            if (
                not shared_receipt.get("complete")
                or not shared.is_file()
                or sha256_file(shared)
                != str(shared_receipt.get("checkpoint_sha256") or "")
            ):
                gaps.append(f"shared_checkpoint_identity_mismatch:{item}")
            elif not any(
                row["target_id"] == "shared_trunk" for row in checkpoint_artifacts
            ):
                checkpoint_artifacts.append(
                    {
                        "target_id": "shared_trunk",
                        "path": relative(shared),
                        "sha256": sha256_file(shared),
                    }
                )
        targets.append(target)
    return {
        "policy": "project_theseus_direct_model_candidate_generation_contract_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "target_id": target_id,
        "config": relative(config_path),
        "freeze": relative(freeze_path),
        "packet": relative(packet_path),
        "case_contract_sha256": freeze.get("case_contract_sha256"),
        "freeze_sha256": stable_hash(freeze),
        "consumption_registry": consumption_registry,
        "candidate_packet_sha256": stable_hash(packet) if packet else "",
        "rows": rows,
        "plan": plan,
        "targets": targets,
        "checkpoint_artifacts": checkpoint_artifacts,
        "candidate_generation": {
            **completion,
            "model_declared_context_window_tokens": declared_context_window,
        },
        "hard_gaps": gaps,
        "boundaries": {
            "generator_visible_fields": ["case_id", "arm_id", "prompt"],
            "target_or_verifier_visible": False,
            "postprocessing_allowed": False,
            "templates_renderers_routers_tools_credit": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "project_selected_quality_token_cap": None,
        },
    }


def generate(contract: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    config = read_json(resolve(contract["config"]))
    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    plan = contract["plan"]
    models: dict[str, Any] = {}
    checkpoint_load_duration_ms_by_target: dict[str, float] = {}
    candidates = []
    bundle_started = time.perf_counter()
    try:
        for row in contract["rows"]:
            arm_id = str(row["arm_id"])
            target_id = (
                arm_id
                if contract["target_id"] == "moecot_system"
                else contract["target_id"]
            )
            if target_id not in models:
                load_started = time.perf_counter()
                target = (plan["targets"] or {})[target_id]
                target_vocab_size = int(
                    target.get("vocab_size") or plan["models"]["vocab_size"]
                )
                model = training.build_model(
                    training.CausalTransformerConfig(
                        vocab_size=target_vocab_size, **target["model"]
                    ),
                    mx=mx,
                    nn=nn,
                    state_role_lookup=None,
                    source_to_target_lookup=training.build_source_to_target_lookup(
                        base,
                        metadata,
                        vocab_size=target_vocab_size,
                        identity_ranges=training.target_copy_identity_ranges(target),
                    ),
                )
                checkpoint = resolve(str(target["checkpoint"]))
                receipt = read_json(resolve(str(target["receipt"])))
                checkpoint = resolve(str(receipt.get("checkpoint") or checkpoint))
                if target.get("role") == "language_expert":
                    model.load_weights(
                        str(resolve(str(target["shared_trunk_checkpoint"]))),
                        strict=False,
                    )
                    model.load_weights(str(checkpoint), strict=False)
                else:
                    model.load_weights(str(checkpoint))
                mx.eval(model.parameters())
                model.eval()
                models[target_id] = model
                checkpoint_load_duration_ms_by_target[target_id] = round(
                    (time.perf_counter() - load_started) * 1000.0, 6
                )
            generation_started = time.perf_counter()
            context_bound = context_bound_for_prompt(
                str(row["prompt"]),
                source_vocab,
                target_vocab,
                base,
                max_source_tokens=int(
                    config["supervision"]["maximum_source_encoded_tokens"]
                ),
            )
            if (
                context_bound.get("fault")
                and context_bound.get("fault") != "physical_context_boundary"
            ):
                raise RuntimeError(
                    f"candidate_prompt_not_encodable:{row['case_id']}:{context_bound['fault']}"
                )
            prompt_tokens = int(context_bound["exact_encoded_prompt_tokens"])
            declared_context_window = int(
                context_bound["model_declared_context_window_tokens"]
            )
            context_residual_tokens = int(
                context_bound["effective_context_residual_tokens"]
            )
            if context_bound.get("state") != "GREEN" or context_residual_tokens <= 0:
                raise RuntimeError(
                    "INVALID_OBSERVATION:physical_context_boundary:"
                    f"{row['case_id']}:prompt_tokens={prompt_tokens}:"
                    f"declared_context_window={declared_context_window}"
                )
            output, generation = training.generate_model_text(
                models[target_id],
                str(row["prompt"]),
                source_vocab,
                target_vocab,
                base,
                max_tokens=context_residual_tokens,
                max_source_tokens=int(
                    config["supervision"]["maximum_source_encoded_tokens"]
                ),
                beam_width=int(config["evaluation"]["beam_width"]),
                branching_factor=int(config["evaluation"]["branching_factor"]),
                length_penalty=float(config["evaluation"]["length_penalty"]),
                mx=mx,
            )
            generation.update(
                {
                    "completion_policy": "model_eos_only_v1",
                    "project_selected_quality_token_cap": None,
                    "model_declared_context_window_tokens": declared_context_window,
                    "exact_encoded_prompt_tokens": prompt_tokens,
                    "effective_context_residual_tokens": context_residual_tokens,
                    "numeric_ceiling_source": (
                        "model_declared_context_window_minus_exact_encoded_prompt_tokens"
                    ),
                }
            )
            if generation.get("stop_reason") != "eos":
                raise RuntimeError(
                    "INVALID_OBSERVATION:physical_context_boundary:"
                    f"{row['case_id']}:termination={generation.get('stop_reason') or generation.get('reason')}"
                )
            generation_duration_ms = round(
                (time.perf_counter() - generation_started) * 1000.0, 6
            )
            candidates.append(
                {
                    "case_id": row["case_id"],
                    "output": output,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "generation": generation,
                    "generation_duration_ms": generation_duration_ms,
                    "target_id": target_id,
                }
            )
    finally:
        models.clear()
    generation_duration_ms_total = round(
        sum(float(row["generation_duration_ms"]) for row in candidates), 6
    )
    checkpoint_load_duration_ms_total = round(
        sum(checkpoint_load_duration_ms_by_target.values()), 6
    )
    return {
        "policy": "project_theseus_direct_model_candidate_bundle_v1",
        "created_utc": now(),
        "target_id": contract["target_id"],
        "case_contract_sha256": contract["case_contract_sha256"],
        "candidate_packet_sha256": contract["candidate_packet_sha256"],
        "generation_function": "moecot_language_arm_training.generate_model_text",
        "generation_wrapper": relative(Path(__file__).resolve()),
        "generation_wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "training_generator_sha256": sha256_file(
            ROOT / "scripts/moecot_language_arm_training.py"
        ),
        "checkpoint_artifacts": contract["checkpoint_artifacts"],
        "training_plan_sha256": contract["plan"]["plan_sha256"],
        "training_stage_signature": (contract["plan"].get("stage") or {}).get(
            "stage_signature"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "timing": {
            "clock": "time.perf_counter",
            "checkpoint_load_duration_ms_by_target": checkpoint_load_duration_ms_by_target,
            "checkpoint_load_duration_ms_total": checkpoint_load_duration_ms_total,
            "generation_duration_ms_total": generation_duration_ms_total,
            "wall_duration_ms": round(
                (time.perf_counter() - bundle_started) * 1000.0, 6
            ),
        },
        "hard_gaps": [],
        "candidate_generation_completion": {
            "normal_completion": ["model_eos"],
            "project_selected_quality_token_cap": None,
            "physical_context_boundary_hits": 0,
            "model_eos_count": len(candidates),
        },
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def context_bound_for_prompt(
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    max_source_tokens: int,
) -> dict[str, Any]:
    """Derive the only decode ceiling from the pinned model context contract."""

    prepared = training.prepare_model_text_prompt(
        prompt,
        source_vocab,
        target_vocab,
        base,
        max_source_tokens=max_source_tokens,
    )
    declared = int((base.get("tokenization") or {}).get("max_sequence_tokens") or 0)
    if prepared.get("fault"):
        return {
            "state": "INVALID_OBSERVATION",
            "fault": str(prepared["fault"]),
            "project_selected_quality_token_cap": None,
        }
    prompt_tokens = len(prepared["prompt_ids"])
    residual = declared - prompt_tokens
    return {
        "state": "GREEN" if declared > 0 and residual > 0 else "INVALID_OBSERVATION",
        "fault": "" if declared > 0 and residual > 0 else "physical_context_boundary",
        "model_declared_context_window_tokens": declared,
        "exact_encoded_prompt_tokens": prompt_tokens,
        "effective_context_residual_tokens": residual,
        "numeric_ceiling_source": (
            "model_declared_context_window_minus_exact_encoded_prompt_tokens"
        ),
        "project_selected_quality_token_cap": None,
        "physical_boundary_disposition": (
            "INVALIDATE_OBSERVATION_NOT_MODEL_OR_ARCHITECTURE_FAILURE"
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def contract_summary(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": contract["policy"],
        "created_utc": contract["created_utc"],
        "trigger_state": contract["trigger_state"],
        "target_id": contract["target_id"],
        "case_contract_sha256": contract["case_contract_sha256"],
        "candidate_count": len(contract["rows"]),
        "checkpoint_artifacts": contract["checkpoint_artifacts"],
        "hard_gaps": contract["hard_gaps"],
        "boundaries": contract["boundaries"],
    }


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
