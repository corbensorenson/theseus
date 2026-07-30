#!/usr/bin/env python3
"""Qualify exact prompt-boundary reuse on the local Qwen3.5 MLX runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core_evidence_worker_v2 as worker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_qwen35_9b_worker.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    report = run(config_path)
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "first": report["turns"][0]["generation_metrics"],
        "second": report["turns"][1]["generation_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = worker.LocalMlxModel(config["model"])
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object and no Markdown. The only "
                "allowed actions for this mechanics probe are "
                '{"action":"read","path":"scripts/x.py","start_line":1,'
                '"end_line":80} or {"action":"abstain"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose the read action for scripts/x.py. This tests local "
                "function-action formatting only."
            ),
        },
    ]
    turns = []
    started = time.perf_counter()
    first = model.generate(messages)
    turns.append(turn_row(first, model.last_generation_metrics))
    messages.extend([
        {"role": "assistant", "content": first},
        {
            "role": "user",
            "content": (
                'TOOL_RESULT {"ok":true,"path":"scripts/x.py",'
                '"content":"1: value = 1","state":{"required_next_phase":'
                '"edit_now_or_abstain_with_specific_missing_information"}} '
                "Return one allowed JSON action."
            ),
        },
    ])
    second = model.generate(messages)
    turns.append(turn_row(second, model.last_generation_metrics))
    first_metrics = turns[0]["generation_metrics"]
    second_metrics = turns[1]["generation_metrics"]
    checks = {
        "first_turn_starts_without_prior_prefix": (
            first_metrics["prefix_cache_reused"] is False
        ),
        "second_turn_reuses_prefix": (
            second_metrics["prefix_cache_reused"] is True
        ),
        "second_turn_only_prefills_new_context_suffix": (
            int(second_metrics["uncached_context_tokens"])
            < int(second_metrics["context_boundary_tokens"])
        ),
        "generation_prefill_is_only_template_suffix": (
            int(second_metrics["generation_uncached_prompt_tokens"]) <= 32
        ),
        "both_turns_use_boundary_cache": all(
            row["generation_metrics"]["prompt_boundary_cache_used"] is True
            for row in turns
        ),
        "both_outputs_parse_as_actions": all(
            row["parseable_action"] for row in turns
        ),
    }
    return {
        "policy": "project_theseus_qwen35_prompt_cache_qualification_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(checks.values()) else "RED",
        "scope": "runtime_mechanics_only_no_repository_competence_claim",
        "model_identity": {
            "repo_id": config["model"]["repo_id"],
            "revision": config["model"]["revision"],
        },
        "source_identities": {
            "config_sha256": sha256_file(config_path),
            "worker_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "qualification_sha256": sha256_file(Path(__file__).resolve()),
        },
        "checks": checks,
        "turns": turns,
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "maximum_inference": (
            "This probe establishes only local MLX action formatting and exact "
            "prompt-boundary cache reuse. It cannot establish repository "
            "competence or any subsystem effect."
        ),
    }


def turn_row(raw: str, metrics: dict[str, Any]) -> dict[str, Any]:
    try:
        action = worker.parse_action(raw)
        parseable = True
        action_kind = str(action.get("action") or "")
    except Exception:
        parseable = False
        action_kind = None
    return {
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_characters": len(raw),
        "parseable_action": parseable,
        "action_kind": action_kind,
        "generation_metrics": dict(metrics),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
