#!/usr/bin/env python3
"""Prove semantic VCM runtime cache lifecycle behavior.

This is deliberately not a native KV/prefix-cache implementation. It verifies
that accepted VCM semantic materialization claims can be keyed, cached, reused,
and invalidated deterministically without increasing authority or claiming
hardware runtime parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNTIME = ROOT / "runtime" / "vcm_runtime_cache"
DEFAULT_CLAIMS = REPORTS / "vcm_runtime_materialization_claims.jsonl"
DEFAULT_OUT = REPORTS / "vcm_runtime_cache_lifecycle.json"
DEFAULT_MARKDOWN = REPORTS / "vcm_runtime_cache_lifecycle.md"
DEFAULT_INDEX = RUNTIME / "semantic_materialization_cache_index.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default=rel(DEFAULT_CLAIMS))
    parser.add_argument("--cache-index-out", default=rel(DEFAULT_INDEX))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    report, records = build_report(
        claims_path=resolve(args.claims),
        cache_index_path=resolve(args.cache_index_out),
        limit=max(1, int(args.limit)),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.cache_index_out), records)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(*, claims_path: Path, cache_index_path: Path, limit: int, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claims = [
        claim
        for claim in read_jsonl(claims_path)[:limit]
        if isinstance(claim, dict) and claim.get("outcome") == "ACCEPTED_SEMANTIC_DESCRIPTOR"
    ]
    records = [cache_record_for_claim(claim) for claim in claims if claim.get("key_complete") is True]
    key_counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("runtime_cache_key_hash") or "")
        key_counts[key] = key_counts.get(key, 0) + 1
    collisions = {key: count for key, count in key_counts.items() if count > 1}
    cache = {str(record["runtime_cache_key_hash"]): record for record in records}
    reuse_hits = sum(1 for claim in claims if runtime_cache_key_hash(claim) in cache)
    mutated_snapshot_misses = sum(1 for claim in claims if mutated_runtime_cache_key_hash(claim, "snapshot") not in cache)
    mutated_policy_misses = sum(1 for claim in claims if mutated_runtime_cache_key_hash(claim, "policy_hash") not in cache)
    accepted_claims = len(claims)
    complete_key_count = len(records)
    reuse_hit_rate = ratio(reuse_hits, accepted_claims)
    mutated_snapshot_miss_rate = ratio(mutated_snapshot_misses, accepted_claims)
    mutated_policy_miss_rate = ratio(mutated_policy_misses, accepted_claims)
    no_cheat = {
        "external_inference_calls": sum_int(claims, "external_inference_calls"),
        "public_training_rows_written": sum_int(claims, "public_training_rows_written"),
        "fallback_return_count": sum_int(claims, "fallback_return_count"),
        "native_kv_cache_claimed_count": sum(1 for claim in claims if claim.get("native_kv_cache_claimed") is True),
        "runtime_profile_claimed_count": sum(1 for claim in claims if claim.get("runtime_profile_claimed") is True),
    }
    gates = [
        gate("claims_loaded", accepted_claims > 0, {"claims": accepted_claims, "claims_path": rel(claims_path)}),
        gate("complete_runtime_keys", complete_key_count == accepted_claims, {"complete": complete_key_count, "accepted": accepted_claims}),
        gate("cache_records_written", len(records) == accepted_claims, {"records": len(records), "cache_index": rel(cache_index_path)}),
        gate("cache_key_collision_zero", not collisions, collisions),
        gate("reuse_hit_rate_full", reuse_hit_rate >= 1.0, reuse_hit_rate),
        gate("snapshot_invalidation_miss_rate_full", mutated_snapshot_miss_rate >= 1.0, mutated_snapshot_miss_rate),
        gate("policy_invalidation_miss_rate_full", mutated_policy_miss_rate >= 1.0, mutated_policy_miss_rate),
        gate("external_inference_zero", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"]),
        gate("public_training_zero", no_cheat["public_training_rows_written"] == 0, no_cheat["public_training_rows_written"]),
        gate("fallback_return_zero", no_cheat["fallback_return_count"] == 0, no_cheat["fallback_return_count"]),
        gate("native_kv_not_claimed", no_cheat["native_kv_cache_claimed_count"] == 0, no_cheat["native_kv_cache_claimed_count"]),
        gate("runtime_profile_not_claimed", no_cheat["runtime_profile_claimed_count"] == 0, no_cheat["runtime_profile_claimed_count"]),
    ]
    failed = [row for row in gates if not row["passed"]]
    report = {
        "policy": "project_theseus_vcm_runtime_cache_lifecycle_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not failed else "RED",
        "summary": {
            "claims_path": rel(claims_path),
            "cache_index": rel(cache_index_path),
            "accepted_claims": accepted_claims,
            "cache_records": len(records),
            "cache_key_complete_rate": ratio(complete_key_count, accepted_claims),
            "reuse_hit_rate": reuse_hit_rate,
            "snapshot_invalidation_miss_rate": mutated_snapshot_miss_rate,
            "policy_invalidation_miss_rate": mutated_policy_miss_rate,
            "cache_key_collision_count": len(collisions),
            "runtime_profile_claimed": False,
            "native_kv_cache_claimed": False,
            "native_prefix_cache_claimed": False,
            "hardware_aware_runtime_cache_scheduling_claimed": False,
            "external_inference_calls": no_cheat["external_inference_calls"],
            "public_training_rows_written": no_cheat["public_training_rows_written"],
            "fallback_return_count": no_cheat["fallback_return_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "gates": gates,
        "record_sample": records[:5],
        "score_semantics": (
            "Semantic VCM cache lifecycle proof only. This creates deterministic descriptor cache records and "
            "tests reuse/invalidation. It does not implement or claim native KV cache, prefix cache, serving, "
            "public benchmark training, teacher use, or hardware parity."
        ),
        "external_inference_calls": 0,
    }
    return report, records


def cache_record_for_claim(claim: dict[str, Any]) -> dict[str, Any]:
    runtime_key = claim.get("runtime_key") if isinstance(claim.get("runtime_key"), dict) else {}
    descriptor = {
        "claim_id": claim.get("claim_id"),
        "source_address": claim.get("source_address"),
        "runtime_key": runtime_key,
        "materialization_kind": "semantic_resident_descriptor",
        "authority_mode": "same_as_source_no_escalation",
    }
    return {
        "policy": "project_theseus_vcm_runtime_cache_record_v1",
        "created_utc": now(),
        "claim_id": claim.get("claim_id"),
        "source_address": claim.get("source_address"),
        "runtime_cache_key_hash": hash_json(runtime_key),
        "materialized_descriptor_hash": hash_json(descriptor),
        "snapshot": runtime_key.get("snapshot"),
        "policy_hash": runtime_key.get("policy_hash"),
        "permission_view": runtime_key.get("permission_view"),
        "redaction_view": runtime_key.get("redaction_view"),
        "representation_level": runtime_key.get("representation_level"),
        "native_kv_cache_claimed": False,
        "runtime_profile_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def runtime_cache_key_hash(claim: dict[str, Any]) -> str:
    runtime_key = claim.get("runtime_key") if isinstance(claim.get("runtime_key"), dict) else {}
    return hash_json(runtime_key)


def mutated_runtime_cache_key_hash(claim: dict[str, Any], field: str) -> str:
    runtime_key = dict(claim.get("runtime_key") if isinstance(claim.get("runtime_key"), dict) else {})
    runtime_key[field] = f"{runtime_key.get(field) or ''}#mutated-for-invalidating-proof"
    return hash_json(runtime_key)


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        try:
            total += int(row.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return total


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 6)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Runtime Cache Lifecycle",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- accepted_claims: `{summary.get('accepted_claims')}`",
        f"- cache_records: `{summary.get('cache_records')}`",
        f"- reuse_hit_rate: `{summary.get('reuse_hit_rate')}`",
        f"- snapshot_invalidation_miss_rate: `{summary.get('snapshot_invalidation_miss_rate')}`",
        f"- policy_invalidation_miss_rate: `{summary.get('policy_invalidation_miss_rate')}`",
        f"- native_kv_cache_claimed: `{summary.get('native_kv_cache_claimed')}`",
        "",
        "## Failed Gates",
    ]
    failed = [row for row in report.get("gates", []) if not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}`: `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
