#!/usr/bin/env python3
"""Resident, identity-bound MLX runtime for the governed neural seed.

The runtime keeps one independently verified model in memory and provides
bounded prompt-prefill and deterministic-completion reuse. Unqualified
checkpoints are available only to private evaluation; serving fails closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
DEFAULT_PACKET = ROOT / "reports" / "neural_seed_57m_functional_utility_candidate_packet.json"
DEFAULT_REPORT = ROOT / "reports" / "neural_seed_resident_runtime_qualification.json"


class BoundedPromptPrefixCache:
    """Model-local LRU for evaluated prompt logits and immutable MLX KV arrays."""

    def __init__(self, maximum_entries: int = 32) -> None:
        if maximum_entries < 1:
            raise ValueError("prompt prefix cache requires a positive entry bound")
        self.maximum_entries = int(maximum_entries)
        self._entries: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> tuple[Any, Any] | None:
        value = self._entries.get(key)
        if value is None:
            self.miss_count += 1
            return None
        self._entries.move_to_end(key)
        self.hit_count += 1
        return value

    def put(self, key: str, value: tuple[Any, Any]) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.maximum_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)


class NeuralSeedResidentRuntime:
    """Load once, verify continuously, and generate through the canonical decoder."""

    def __init__(
        self,
        *,
        config_path: Path = DEFAULT_CONFIG,
        target_id: str = "shared_trunk",
        mode: str = "evaluation",
        prefix_cache_entries: int = 32,
        completion_cache_entries: int = 64,
        identity_rehash_interval_seconds: float = 30.0,
    ) -> None:
        if mode not in {"evaluation", "serving"}:
            raise ValueError("resident runtime mode must be evaluation or serving")
        if completion_cache_entries < 1:
            raise ValueError("completion cache requires a positive entry bound")
        if identity_rehash_interval_seconds < 0.0:
            raise ValueError("identity rehash interval cannot be negative")

        import mlx.core as mx
        import mlx.nn as nn
        import moecot_language_arm_training as training

        self.mx = mx
        self.training = training
        self.mode = mode
        self.config_path = config_path.resolve()
        self.target_id = target_id
        self.prefix_cache = BoundedPromptPrefixCache(prefix_cache_entries)
        self.completion_cache_entries = int(completion_cache_entries)
        self.completion_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.identity_rehash_interval_seconds = float(identity_rehash_interval_seconds)
        self._lock = threading.RLock()
        self.request_count = 0
        self.completion_hit_count = 0
        self.completion_miss_count = 0
        self._last_full_identity_check = 0.0

        load_started = time.perf_counter()
        self.config = training.bind_scale_preregistration(
            read_json(self.config_path)
        )
        self.plan = training.build_plan(self.config, config_path=self.config_path)
        if self.plan.get("trigger_state") == "RED":
            raise ValueError("resident runtime denied by the canonical training plan")
        self.target = (self.plan.get("targets") or {}).get(target_id)
        if not isinstance(self.target, dict):
            raise ValueError(f"resident runtime target is unavailable: {target_id}")
        self.receipt_path = resolve(str(self.target.get("receipt") or ""))
        self.receipt = read_json(self.receipt_path)
        self.checkpoint = resolve(
            str(self.receipt.get("checkpoint") or self.target.get("checkpoint") or "")
        )
        self.optimizer = resolve(
            str(
                self.receipt.get("optimizer_state")
                or self.target.get("optimizer_state")
                or ""
            )
        )
        self.registered_plan_migration = training.validate_resume(
            self.receipt,
            self.plan,
            self.target,
            self.checkpoint,
            self.optimizer,
        )
        if mode == "serving" and (
            self.receipt.get("runtime_serving_allowed") is not True
            or self.receipt.get("capability_claim") in {None, "", "NOT_EVALUATED", "NONE"}
        ):
            raise PermissionError(
                "resident serving denied until direct model-only capability grants runtime authority"
            )

        stage_dir = resolve(str(self.config["stage_dir"]))
        self.metadata = read_json(stage_dir / "stage_metadata_v1.json")
        self.base = read_json(resolve(str(self.config["base_config"])))
        self.source_vocab = dict(self.metadata["source_vocab"])
        self.target_vocab = dict(self.metadata["target_vocab"])
        vocab_size = int(
            self.target.get("vocab_size") or self.plan["models"]["vocab_size"]
        )
        self.model = training.build_model(
            training.CausalTransformerConfig(
                vocab_size=vocab_size, **self.target["model"]
            ),
            mx=mx,
            nn=nn,
            state_role_lookup=None,
            source_to_target_lookup=training.build_source_to_target_lookup(
                self.base,
                self.metadata,
                vocab_size=vocab_size,
                identity_ranges=training.target_copy_identity_ranges(self.target),
            ),
        )
        self.model.load_weights(str(self.checkpoint))
        mx.eval(self.model.parameters())
        self.model.eval()
        self.checkpoint_sha256 = str(self.receipt["checkpoint_sha256"])
        self.optimizer_sha256 = str(self.receipt["optimizer_state_sha256"])
        self.receipt_plan_sha256 = str(self.receipt["plan_sha256"])
        self._checkpoint_stat = artifact_stat(self.checkpoint)
        self._receipt_stat = artifact_stat(self.receipt_path)
        self._assert_identity(force_hash=True)
        self.runtime_identity = digest_json(
            {
                "policy": "project_theseus_neural_seed_resident_runtime_v1",
                "target_id": target_id,
                "plan_sha256": self.plan["plan_sha256"],
                "receipt_plan_sha256": self.receipt_plan_sha256,
                "registered_plan_migration": self.registered_plan_migration,
                "checkpoint_sha256": self.checkpoint_sha256,
                "stage_signature": self.receipt.get("stage_signature"),
                "source_vocab_sha256": digest_json(self.source_vocab),
                "target_vocab_sha256": digest_json(self.target_vocab),
                "decoder_source_sha256": file_sha256(
                    ROOT / "scripts" / "moecot_language_arm_training.py"
                ),
            }
        )
        self.load_seconds = time.perf_counter() - load_started

    def _assert_identity(self, *, force_hash: bool = False) -> None:
        if (
            artifact_stat(self.checkpoint) != self._checkpoint_stat
            or artifact_stat(self.receipt_path) != self._receipt_stat
        ):
            self.prefix_cache.clear()
            self.completion_cache.clear()
            raise RuntimeError("resident runtime identity changed; explicit reload required")
        current_receipt = read_json(self.receipt_path)
        if (
            current_receipt.get("checkpoint") != relative(self.checkpoint)
            or current_receipt.get("checkpoint_sha256") != self.checkpoint_sha256
            or current_receipt.get("optimizer_state_sha256") != self.optimizer_sha256
            or current_receipt.get("plan_sha256") != self.receipt_plan_sha256
        ):
            self.prefix_cache.clear()
            self.completion_cache.clear()
            raise RuntimeError("resident runtime receipt identity changed; explicit reload required")
        monotonic = time.monotonic()
        if force_hash or (
            monotonic - self._last_full_identity_check
            >= self.identity_rehash_interval_seconds
        ):
            if file_sha256(self.checkpoint) != self.checkpoint_sha256:
                self.prefix_cache.clear()
                self.completion_cache.clear()
                raise RuntimeError("resident runtime checkpoint hash changed")
            self._last_full_identity_check = monotonic

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 32,
        beam_width: int = 2,
        branching_factor: int = 2,
        length_penalty: float = 0.6,
        use_prefix_cache: bool = True,
        use_completion_cache: bool = True,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("resident generation requires a nonempty prompt")
        if max_tokens < 1 or beam_width < 1 or branching_factor < 1:
            raise ValueError("resident generation bounds must be positive")
        request_contract = {
            "runtime_identity": self.runtime_identity,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_tokens": int(max_tokens),
            "beam_width": int(beam_width),
            "branching_factor": int(branching_factor),
            "length_penalty": float(length_penalty),
        }
        request_key = digest_json(request_contract)
        with self._lock:
            self._assert_identity()
            self.request_count += 1
            started = time.perf_counter()
            cached = self.completion_cache.get(request_key) if use_completion_cache else None
            if cached is not None:
                self.completion_cache.move_to_end(request_key)
                self.completion_hit_count += 1
                result = copy.deepcopy(cached)
                result["runtime_receipt"].update(
                    {
                        "completion_cache_state": "HIT",
                        "prompt_prefix_cache_state": "NOT_USED_COMPLETION_HIT",
                        "request_seconds": round(time.perf_counter() - started, 6),
                    }
                )
                return result

            self.completion_miss_count += 1
            text, generation = self.training.generate_model_text(
                self.model,
                prompt,
                self.source_vocab,
                self.target_vocab,
                self.base,
                max_tokens=max_tokens,
                max_source_tokens=int(
                    self.config["supervision"]["maximum_source_encoded_tokens"]
                ),
                beam_width=beam_width,
                branching_factor=branching_factor,
                length_penalty=length_penalty,
                prompt_prefix_cache=self.prefix_cache if use_prefix_cache else None,
                mx=self.mx,
            )
            result = {
                "text": text,
                "generation": generation,
                "runtime_receipt": {
                    "policy": "project_theseus_neural_seed_resident_request_v1",
                    "runtime_identity": self.runtime_identity,
                    "request_sha256": request_key,
                    "prompt_sha256": request_contract["prompt_sha256"],
                    "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "completion_cache_state": (
                        "MISS" if use_completion_cache else "DISABLED"
                    ),
                    "prompt_prefix_cache_state": generation.get(
                        "prompt_prefix_cache_state", "NOT_REACHED"
                    ),
                    "request_seconds": round(time.perf_counter() - started, 6),
                    "model_loads_for_runtime": 1,
                    "external_inference_calls": 0,
                    "public_training_rows_written": 0,
                    "fallback_return_count": 0,
                    "raw_prompt_or_output_retained": False,
                    "capability_claim": "NONE" if self.mode == "evaluation" else "SERVING",
                },
            }
            if use_completion_cache:
                self.completion_cache[request_key] = copy.deepcopy(result)
                self.completion_cache.move_to_end(request_key)
                while len(self.completion_cache) > self.completion_cache_entries:
                    self.completion_cache.popitem(last=False)
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._assert_identity()
            return {
                "policy": "project_theseus_neural_seed_resident_runtime_v1",
                "mode": self.mode,
                "runtime_identity": self.runtime_identity,
                "target_id": self.target_id,
                "checkpoint": relative(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "optimizer_state_sha256": self.optimizer_sha256,
                "plan_sha256": self.plan["plan_sha256"],
                "receipt_plan_sha256": self.receipt_plan_sha256,
                "registered_plan_migration": self.registered_plan_migration,
                "load_seconds": round(self.load_seconds, 6),
                "model_load_count": 1,
                "request_count": self.request_count,
                "prefix_cache": {
                    "entries": self.prefix_cache.entry_count,
                    "maximum_entries": self.prefix_cache.maximum_entries,
                    "hits": self.prefix_cache.hit_count,
                    "misses": self.prefix_cache.miss_count,
                },
                "completion_cache": {
                    "entries": len(self.completion_cache),
                    "maximum_entries": self.completion_cache_entries,
                    "hits": self.completion_hit_count,
                    "misses": self.completion_miss_count,
                },
                "runtime_serving_allowed": self.mode == "serving",
                "external_inference_calls": 0,
            }


def qualify_resident_runtime(
    *, config_path: Path, packet_path: Path, max_tokens: int
) -> dict[str, Any]:
    packet = read_json(packet_path)
    rows = packet.get("rows") if isinstance(packet.get("rows"), list) else []
    row = next((item for item in rows if item.get("arm_id") == "python"), None)
    if not isinstance(row, dict) or not str(row.get("prompt") or ""):
        raise ValueError("resident qualification requires one private Python prompt")
    prompt = str(row["prompt"])
    runtime = NeuralSeedResidentRuntime(config_path=config_path, mode="evaluation")

    uncached = runtime.generate(
        prompt,
        max_tokens=max_tokens,
        use_prefix_cache=False,
        use_completion_cache=False,
    )
    seeded = runtime.generate(prompt, max_tokens=max_tokens)
    repeated = runtime.generate(prompt, max_tokens=max_tokens)
    prefix_reference = runtime.generate(
        prompt,
        max_tokens=max(1, max_tokens - 1),
        use_prefix_cache=False,
        use_completion_cache=False,
    )
    prefix_reuse = runtime.generate(
        prompt,
        max_tokens=max(1, max_tokens - 1),
        use_prefix_cache=True,
        use_completion_cache=False,
    )
    parity = bool(
        uncached["text"] == seeded["text"]
        and uncached["generation"].get("generated_token_sha256")
        == seeded["generation"].get("generated_token_sha256")
        and prefix_reference["text"] == prefix_reuse["text"]
        and prefix_reference["generation"].get("generated_token_sha256")
        == prefix_reuse["generation"].get("generated_token_sha256")
    )
    seeded_seconds = float(seeded["runtime_receipt"]["request_seconds"])
    repeated_seconds = float(repeated["runtime_receipt"]["request_seconds"])
    reference_prefill = float(prefix_reference["generation"].get("prompt_prefill_seconds") or 0.0)
    reused_prefill = float(prefix_reuse["generation"].get("prompt_prefill_seconds") or 0.0)
    repeated_speedup = seeded_seconds / max(1e-9, repeated_seconds)
    prefix_speedup = reference_prefill / max(1e-9, reused_prefill)
    gaps = []
    if not parity:
        gaps.append("resident_cache_output_or_token_parity_failed")
    if repeated["runtime_receipt"].get("completion_cache_state") != "HIT":
        gaps.append("completion_cache_did_not_hit")
    if prefix_reuse["generation"].get("prompt_prefix_cache_state") != "HIT":
        gaps.append("prompt_prefix_cache_did_not_hit")
    if repeated_speedup < 5.0:
        gaps.append("repeated_prompt_speedup_below_5x")
    return {
        "policy": "project_theseus_neural_seed_resident_runtime_qualification_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not gaps else "RED",
        "case_id": row.get("case_id"),
        "arm_id": row.get("arm_id"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_prompt_or_output_retained": False,
        "max_tokens": max_tokens,
        "exact_output_and_token_parity": parity,
        "uncached_request_seconds": uncached["runtime_receipt"]["request_seconds"],
        "cache_seed_request_seconds": seeded_seconds,
        "repeated_request_seconds": repeated_seconds,
        "repeated_prompt_speedup": round(repeated_speedup, 6),
        "uncached_prefill_seconds": reference_prefill,
        "cached_prefill_seconds": reused_prefill,
        "prefix_prefill_speedup": round(prefix_speedup, 6),
        "generation_state": uncached["generation"].get("state"),
        "generation_reason": uncached["generation"].get("reason"),
        "runtime": runtime.status(),
        "boundaries": {
            "private_evaluation_prompt_only": True,
            "target_or_verifier_visible_to_generator": False,
            "runtime_serving_allowed": False,
            "public_benchmark_rows_read": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "capability_claim": "NONE",
        },
        "hard_gaps": gaps,
        "claim_scope": (
            "Resident model, exact prompt-prefill reuse, and deterministic completion "
            "reuse mechanics only; this is not model utility or public-transfer evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--packet", default=relative(DEFAULT_PACKET))
    parser.add_argument("--out", default=relative(DEFAULT_REPORT))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least two")
    if not args.execute:
        report = {
            "policy": "project_theseus_neural_seed_resident_runtime_qualification_v1",
            "trigger_state": "READY",
            "config": relative(resolve(args.config)),
            "packet": relative(resolve(args.packet)),
            "max_tokens": args.max_tokens,
            "capability_claim": "NONE",
        }
    else:
        report = qualify_resident_runtime(
            config_path=resolve(args.config),
            packet_path=resolve(args.packet),
            max_tokens=args.max_tokens,
        )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") != "RED" else 2


def artifact_stat(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
