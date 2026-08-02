#!/usr/bin/env python3
"""Count complete chat-templated prompts with the pinned local tokenizer only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_exact_local_prompt_addressability_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = count_prompts(
        resolve(args.worker_config),
        resolve(args.prompts),
    )
    write_json(resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "prompt_count": report["prompt_count"],
                "minimum_context_residual_tokens": report[
                    "minimum_context_residual_tokens"
                ],
                "faults": report["faults"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


def count_prompts(worker_path: Path, prompts_path: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import theseus_local_inference_backend as backend

    worker = read_json(worker_path)
    card = mapping(worker.get("model"))
    snapshot = backend.local_snapshot(card)
    prompts = mapping(read_json(prompts_path).get("prompts"))
    faults: list[str] = []
    if not backend.complete_model_snapshot(snapshot):
        faults.append("complete_local_model_snapshot_missing")
    context = backend.snapshot_context_window(snapshot)
    declared = int(
        mapping(worker.get("generation_boundary")).get(
            "model_declared_context_window_tokens"
        )
        or 0
    )
    if context < 1 or context != declared:
        faults.append("model_declared_context_binding_invalid")
    if not prompts or any(not str(value).strip() for value in prompts.values()):
        faults.append("prompt_set_invalid")
    counts: dict[str, dict[str, Any]] = {}
    if not faults:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=True,
        )
        for arm, prompt in sorted(prompts.items()):
            encoded = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": backend.SYSTEM_PROMPT},
                    {"role": "user", "content": str(prompt)},
                ],
                tokenize=True,
                add_generation_prompt=True,
                **mapping(card.get("chat_template_kwargs")),
            )
            if hasattr(encoded, "get"):
                tokens = encoded.get("input_ids", [])
            else:
                tokens = encoded
            shape = getattr(tokens, "shape", None)
            if shape is not None and len(shape) >= 1:
                count = int(shape[-1])
            elif (
                isinstance(tokens, (list, tuple))
                and tokens
                and isinstance(tokens[0], (list, tuple))
            ):
                if len(tokens) != 1:
                    faults.append(f"unexpected_prompt_batch_size:{arm}")
                count = len(tokens[0])
            else:
                count = len(tokens)
            residual = context - count
            if count < 1:
                faults.append(f"exact_prompt_token_count_missing:{arm}")
            if residual <= 0:
                faults.append(f"prompt_not_physically_addressable:{arm}")
            counts[str(arm)] = {
                "prompt_sha256": sha256_text(str(prompt)),
                "exact_chat_templated_prompt_tokens": count,
                "model_declared_context_window_tokens": context,
                "physical_context_residual_tokens": residual,
                "project_selected_quality_token_cap": None,
            }
    residuals = [
        int(row.get("physical_context_residual_tokens") or 0)
        for row in counts.values()
    ]
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "worker_config_sha256": sha256_file(worker_path),
        "snapshot_manifest_sha256": (
            backend.snapshot_manifest(snapshot)
            if backend.complete_model_snapshot(snapshot)
            else ""
        ),
        "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(snapshot / "tokenizer_config.json"),
        "system_prompt_sha256": sha256_text(backend.SYSTEM_PROMPT),
        "prompt_count": len(counts),
        "prompts": counts,
        "minimum_context_residual_tokens": min(residuals, default=0),
        "project_selected_quality_token_cap": None,
        "candidate_or_control_calls": 0,
        "maximum_inference": "Pre-model physical prompt addressability only; no generation, task-quality, model, or mechanism inference.",
    }


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
