#!/usr/bin/env python3
"""Generate the frozen K4 packet with process-isolated learned KERC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import host_resource_safety
from neural_seed_functional_consumption import (
    complete_reservation,
    fail_reservation,
    reserve_once,
)
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from theseus_archive_resolver import read_json_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_kerc_k4_candidate_producer_v2"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_packet(packet: dict[str, Any]) -> dict[int, list[dict[str, str]]]:
    if (
        packet.get("policy") != "project_theseus_kerc_k4_prompt_packet_v2"
        or not str(packet.get("surface_id") or "")
        or (packet.get("information_flow") or {}).get(
            "answer_identifying_metadata_exposed"
        )
        is not False
        or (packet.get("information_flow") or {}).get("candidate_visible_fields")
        != ["prompt"]
        or (packet.get("information_flow") or {}).get("packet_row_fields")
        != ["opaque_id", "prompt"]
    ):
        raise ValueError("K4 prompt packet integrity failure")
    groups = packet.get("checkpoint_groups") or []
    by_seed: dict[int, list[dict[str, str]]] = {}
    identities: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"checkpoint_seed", "rows"}:
            raise ValueError("K4 checkpoint group schema invalid")
        seed = group.get("checkpoint_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in by_seed:
            raise ValueError("K4 checkpoint seed invalid or duplicated")
        rows = group.get("rows") or []
        if any(
            not isinstance(row, dict)
            or set(row) != {"opaque_id", "prompt"}
            or not str(row.get("opaque_id") or "")
            or not str(row.get("prompt") or "")
            for row in rows
        ):
            raise ValueError("K4 candidate row schema invalid")
        row_ids = {str(row["opaque_id"]) for row in rows}
        if len(row_ids) != len(rows) or identities & row_ids:
            raise ValueError("K4 opaque identities are missing or duplicated")
        identities.update(row_ids)
        by_seed[int(seed)] = [
            {"opaque_id": str(row["opaque_id"]), "prompt": str(row["prompt"])}
            for row in rows
        ]
    if len(identities) != int(packet.get("row_count") or 0):
        raise ValueError("K4 packet row count mismatch")
    return by_seed


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        raise ValueError("K4 candidate generation requires the external host watchdog")
    import mlx.core as mx
    import mlx.nn as nn
    import moecot_language_arm_training as training

    packet_path = resolve(args.packet)
    packet = read_json(packet_path)
    packet_rows_by_seed = validate_packet(packet)
    adequacy_path = resolve(args.adequacy_config)
    adequacy = read_json(adequacy_path)
    configured_seeds = [int(value) for value in adequacy["seeds"]]
    if set(packet_rows_by_seed) != set(configured_seeds):
        raise ValueError("K4 packet/checkpoint seed set mismatch")
    training_config = read_json(resolve(adequacy["trainer_config"]))
    base = read_json(resolve(training_config["base_config"]))
    metadata = read_json(
        resolve(training_config["stage_dir"]) / "stage_metadata_v1.json"
    )
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    checkpoints: list[dict[str, Any]] = []
    seed_contexts: list[dict[str, Any]] = []
    for seed in configured_seeds:
        seed_path = resolve(str(adequacy["report_pattern"]).replace("{seed}", str(seed)))
        seed_report = read_json_follow_pointer(seed_path)
        target = dict(seed_report["targets"]["english_kerc"])
        result = next(
            row for row in seed_report["results"] if row["target_id"] == "english_kerc"
        )
        checkpoint = resolve(result["checkpoint"])
        if sha256(checkpoint) != result.get("checkpoint_sha256"):
            raise ValueError(f"K4 checkpoint drift: {seed}")
        checkpoints.append(
            {"seed": seed, "path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint)}
        )
        seed_contexts.append(
            {
                "seed": seed,
                "target": target,
                "checkpoint": checkpoint,
            }
        )
    registry_path = resolve(str(packet.get("consumption_registry") or ""))
    if (
        not str(packet.get("consumption_registry") or "")
        or registry_path != ROOT / "reports/private_functional_consumption_registry.jsonl"
    ):
        raise ValueError("K4 exact-once consumption registry missing or invalid")
    reservation = reserve_once(
        registry_path,
        stage="kerc_k4_candidate_generation",
        identity={
            "surface_id": packet["surface_id"],
            "packet_sha256": sha256(packet_path),
            "adequacy_config_sha256": sha256(adequacy_path),
            "producer_sha256": sha256(Path(__file__).resolve()),
            "generator_sha256": sha256(
                ROOT / "scripts/moecot_language_arm_training.py"
            ),
            "checkpoints": checkpoints,
        },
    )
    outputs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    output_path = resolve(args.outputs)
    report_path = resolve(args.out)
    try:
        for context in seed_contexts:
            seed = int(context["seed"])
            target = dict(context["target"])
            checkpoint = Path(context["checkpoint"])
            model = build_model(
                CausalTransformerConfig(
                    vocab_size=int(target["vocab_size"]), **target["model"]
                ),
                mx=mx,
                nn=nn,
                state_role_lookup=None,
                source_to_target_lookup=training.build_source_to_target_lookup(
                    base,
                    metadata,
                    vocab_size=int(target["vocab_size"]),
                    identity_ranges=training.target_copy_identity_ranges(target),
                ),
                attention_query_chunk_size=128,
                attention_key_chunk_size=128,
                compact_encoder_decoder_partitions=True,
            )
            model.load_weights(str(checkpoint))
            mx.eval(model.parameters())
            model.eval()
            for row in packet_rows_by_seed[seed]:
                output, receipt = training.generate_kerc_pipeline_text(
                    model,
                    row["prompt"],
                    source_vocab,
                    target_vocab,
                    base,
                    target=target,
                    max_tokens=int(
                        training_config["evaluation"][
                            "kerc_decode_max_target_tokens"
                        ]
                    ),
                    max_source_tokens=int(
                        training_config["kernel_english_training"][
                            "maximum_sequence_tokens"
                        ]
                    ),
                    beam_width=int(
                        training_config["evaluation"]["kerc_beam_width"]
                    ),
                    branching_factor=int(
                        training_config["evaluation"]["kerc_branching_factor"]
                    ),
                    length_penalty=float(
                        training_config["evaluation"]["length_penalty"]
                    ),
                    interaction_id=f"kerc-k4:{seed}:{row['opaque_id']}",
                    mx=mx,
                )
                outputs.append({"opaque_id": row["opaque_id"], "output": output})
                observations.append(
                    {
                        "opaque_id": row["opaque_id"],
                        "checkpoint_seed": seed,
                        "state": receipt.get("state"),
                        "reason": receipt.get("reason")
                        or (receipt.get("fault") or {}).get("fault_type")
                        or "",
                        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "output_nonempty": bool(output),
                        "fallback_return_count": int(
                            receipt.get("fallback_return_count") or 0
                        ),
                    }
                )
                mx.clear_cache()
            del model
            mx.clear_cache()
        if len(outputs) != int(packet["row_count"]):
            raise ValueError(
                "K4 producer did not cover the complete opaque identity set"
            )
        write_jsonl(output_path, outputs)
        source_artifacts = {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in {
                "producer": Path(__file__).resolve(),
                "generator": ROOT / "scripts/moecot_language_arm_training.py",
                "adequacy_config": adequacy_path,
            }.items()
        }
        report = {
            "policy": POLICY,
            "surface_id": packet["surface_id"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "trigger_state": "GREEN",
            "packet": {
                "path": str(packet_path.relative_to(ROOT)),
                "sha256": sha256(packet_path),
            },
            "outputs": {
                "path": str(output_path.relative_to(ROOT)),
                "sha256": sha256(output_path),
            },
            "source_artifacts": source_artifacts,
            "checkpoints": checkpoints,
            "row_count": len(outputs),
            "nonempty_output_count": sum(
                row["output_nonempty"] for row in observations
            ),
            "state_counts": dict(
                Counter(str(row["state"]) for row in observations)
            ),
            "reason_counts": dict(
                Counter(str(row["reason"]) for row in observations)
            ),
            "observations": observations,
            "candidate_visible_fields": ["prompt"],
            "controller_visible_fields": ["checkpoint_seed", "opaque_id"],
            "answer_identifying_metadata_exposed": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit": sum(
                int(row["fallback_return_count"]) for row in observations
            ),
            "consumption": {
                "registry": str(registry_path.relative_to(ROOT)),
                "reservation_id": reservation["reservation_id"],
                "consumption_key": reservation["consumption_key"],
            },
        }
        write_json(report_path, report)
        complete_reservation(
            registry_path,
            reservation,
            artifact={
                "report": {
                    "path": str(report_path.relative_to(ROOT)),
                    "sha256": sha256(report_path),
                },
                "outputs": {
                    "path": str(output_path.relative_to(ROOT)),
                    "sha256": sha256(output_path),
                },
                "trigger_state": report["trigger_state"],
            },
        )
        return report
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", default="reports/kerc_k4_prompt_packet_v2.json")
    parser.add_argument("--adequacy-config", default="configs/rdc_kerc_matched_adequacy.json")
    parser.add_argument("--outputs", default="reports/kerc_k4_candidate_outputs_v2.jsonl")
    parser.add_argument("--out", default="reports/kerc_k4_candidate_producer_v2.json")
    parser.add_argument("--guarded", action="store_true")
    args = parser.parse_args()
    if args.guarded:
        command = [sys.executable, str(Path(__file__).resolve())]
        for key in ("packet", "adequacy_config", "outputs", "out"):
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        policy = host_resource_safety.HostSafetyPolicy(
            max_process_memory_mib=1280,
            minimum_available_before_launch_mib=3072,
            minimum_available_during_run_mib=2048,
            maximum_swapout_growth_mib=16,
            maximum_wall_seconds=1200,
            poll_interval_seconds=0.1,
            terminate_grace_seconds=2.0,
        )
        process = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=policy,
            env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
        )
        output_report = resolve(args.out)
        receipt_path = output_report.with_name(
            output_report.stem + ".host_resource_safety.json"
        )
        write_json(receipt_path, process.receipt)
        if process.stdout:
            print(process.stdout[-4000:])
        if process.stderr:
            print(process.stderr[-4000:], file=sys.stderr)
        return 0 if process.receipt.get("passed") is True else 2
    report = run_child(args)
    print(json.dumps({"trigger_state": report["trigger_state"], "row_count": report["row_count"], "nonempty_output_count": report["nonempty_output_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
