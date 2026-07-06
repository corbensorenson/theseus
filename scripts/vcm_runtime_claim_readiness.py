"""Prototype VCM runtime resident-materialization claim readiness.

This is not native KV/prefix-cache integration. It creates deterministic,
auditable semantic materialization claim records with complete runtime keys so
Theseus can review whether the VCM-Runtime boundary is ready for a future real
runtime implementation.
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
DEFAULT_COMPILED = REPORTS / "virtual_context_compiled_context.json"
DEFAULT_PAGES = REPORTS / "virtual_context_memory_pages.jsonl"
DEFAULT_OUT = REPORTS / "vcm_runtime_claim_readiness.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_runtime_claim_readiness.md"
DEFAULT_CLAIMS_OUT = REPORTS / "vcm_runtime_materialization_claims.jsonl"

REQUIRED_RUNTIME_KEY_FIELDS = [
    "source_address",
    "source_content_hash",
    "representation_level",
    "representation_object_hash",
    "certificate_id",
    "model_id",
    "tokenizer_id",
    "adapter_id",
    "policy_hash",
    "principal",
    "permission_view",
    "redaction_view",
    "snapshot",
    "role_layout_hash",
    "materialization_predicate_hash",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", default=rel(DEFAULT_COMPILED))
    parser.add_argument("--pages", default=rel(DEFAULT_PAGES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--claims-out", default=rel(DEFAULT_CLAIMS_OUT))
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    report, claims = build_report(
        compiled_path=resolve(args.compiled),
        pages_path=resolve(args.pages),
        limit=max(1, args.limit),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.claims_out), claims)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, compiled_path: Path, pages_path: Path, limit: int, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    compiled = read_json(compiled_path)
    pages = read_jsonl(pages_path)
    visible = list_value(compiled.get("model_visible_pages"))
    snapshot = str(compiled.get("snapshot") or "")
    page_by_address = {str(page.get("address") or ""): page for page in pages if isinstance(page, dict)}
    claims: list[dict[str, Any]] = []
    for row in visible[:limit]:
        if not isinstance(row, dict):
            continue
        address = str(row.get("address") or "")
        page = page_by_address.get(address)
        claim = make_claim(row=row, page=page or {}, snapshot=snapshot, compiled=compiled)
        claims.append(claim)
    complete_claims = [claim for claim in claims if claim["key_complete"]]
    rejected_claims = [claim for claim in claims if not claim["key_complete"]]
    key_complete_rate = len(complete_claims) / max(1, len(claims))
    blockers = []
    if not compiled:
        blockers.append({"kind": "missing_compiled_context", "detail": rel(compiled_path)})
    if not pages:
        blockers.append({"kind": "missing_page_ledger", "detail": rel(pages_path)})
    if not claims:
        blockers.append({"kind": "no_visible_pages", "detail": "No model-visible pages were available for semantic materialization claim review."})
    if rejected_claims:
        blockers.append(
            {
                "kind": "incomplete_runtime_keys",
                "detail": f"{len(rejected_claims)} semantic materialization claims are missing required runtime key fields.",
            }
        )
    trigger_state = "GREEN" if claims and not blockers else "RED"
    report = {
        "policy": "project_theseus_vcm_runtime_claim_readiness_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "runtime_profile_claimed": False,
            "native_kv_cache_claimed": False,
            "native_prefix_cache_claimed": False,
            "semantic_materialization_claims": len(claims),
            "accepted_semantic_claims": len(complete_claims),
            "rejected_semantic_claims": len(rejected_claims),
            "cache_key_complete_rate": round(key_complete_rate, 6),
            "resident_materialization_claims_ready": bool(claims and not rejected_claims),
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        },
        "claim_scope": {
            "claimed": "semantic resident-materialization descriptors only",
            "not_claimed": [
                "native KV cache reuse",
                "native prefix cache lifecycle",
                "hardware-aware runtime cache scheduling",
                "CUDA/Metal/MLX runtime parity",
            ],
        },
        "required_key_fields": REQUIRED_RUNTIME_KEY_FIELDS,
        "claim_sample": claims[:5],
        "blockers": blockers,
    }
    return report, claims


def make_claim(*, row: dict[str, Any], page: dict[str, Any], snapshot: str, compiled: dict[str, Any]) -> dict[str, Any]:
    address = str(row.get("address") or page.get("address") or "")
    level = str(row.get("representation_level") or "L4")
    rep = dict_value(dict_value(page.get("representations")).get(level))
    cert = dict_value(rep.get("certificate"))
    governance = dict_value(page.get("governance")) or dict_value(row.get("governance"))
    policy_snapshot = dict_value(compiled.get("policy_snapshot"))
    predicate = {
        "address": address,
        "level": level,
        "snapshot": snapshot,
        "allowed_purposes": governance.get("allowed_purposes"),
        "sharing": governance.get("sharing"),
        "training_use_allowed": governance.get("training_use_allowed"),
    }
    runtime_key = {
        "source_address": address,
        "source_content_hash": page.get("content_hash") or row.get("source_hash") or "",
        "representation_level": level,
        "representation_object_hash": rep.get("object_hash") or row.get("object_hash") or "",
        "certificate_id": row.get("certificate_id") or cert.get("certificate_id") or "",
        "model_id": "theseus_local_semantic_runtime_vcm_v1",
        "tokenizer_id": "semantic_pages_not_native_tokens",
        "adapter_id": "vcm_semantic_materialization_adapter_v1",
        "policy_hash": stable_hash(policy_snapshot or {"policy": "sparkstream_local_autonomy_v0"}),
        "principal": "principal:local-theseus-operator",
        "permission_view": str(governance.get("sharing") or "private_local"),
        "redaction_view": "default_private_redacted",
        "snapshot": snapshot,
        "role_layout_hash": stable_hash({"lane": row.get("lane"), "execution_class": row.get("execution_class"), "taints": row.get("taints")}),
        "materialization_predicate_hash": stable_hash(predicate),
    }
    missing = [key for key in REQUIRED_RUNTIME_KEY_FIELDS if not runtime_key.get(key)]
    return {
        "policy": "project_theseus_vcm_runtime_materialization_claim_v1",
        "claim_id": "claim:" + hashlib.sha256(json.dumps(runtime_key, sort_keys=True).encode("utf-8")).hexdigest()[:24],
        "created_utc": now(),
        "source_address": address,
        "runtime_key": runtime_key,
        "key_complete": not missing,
        "missing_key_fields": missing,
        "outcome": "ACCEPTED_SEMANTIC_DESCRIPTOR" if not missing else "REJECTED_INCOMPLETE_KEY",
        "runtime_profile_claimed": False,
        "native_kv_cache_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Runtime Claim Readiness",
        "",
        f"State: `{report['trigger_state']}`",
        "",
        "## Summary",
        "",
        f"- Runtime profile claimed: `{summary['runtime_profile_claimed']}`",
        f"- Native KV cache claimed: `{summary['native_kv_cache_claimed']}`",
        f"- Semantic materialization claims: `{summary['semantic_materialization_claims']}`",
        f"- Accepted semantic claims: `{summary['accepted_semantic_claims']}`",
        f"- Rejected semantic claims: `{summary['rejected_semantic_claims']}`",
        f"- Cache-key complete rate: `{summary['cache_key_complete_rate']}`",
        f"- External inference calls: `{summary['external_inference_calls']}`",
        f"- Public training rows written: `{summary['public_training_rows_written']}`",
        f"- Fallback return count: `{summary['fallback_return_count']}`",
        "",
        "## Not Claimed",
        "",
    ]
    for item in report["claim_scope"]["not_claimed"]:
        lines.append(f"- {item}")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker.get('kind')}`: {blocker.get('detail')}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
