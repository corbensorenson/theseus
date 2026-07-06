"""Security helpers for private Project Theseus Hive task envelopes.

This is intentionally stdlib-only. It gives private hives scoped manifests,
revocation checks, quota accounting, and artifact signatures now, while leaving
public-network Ed25519/sandbox attestation as a stricter later release gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def authorize_task_payload(
    policy: dict[str, Any],
    *,
    kind: str,
    payload: dict[str, Any],
    source: str,
    hive_id: str,
    join_token: str,
    local_node_id: str,
) -> dict[str, Any]:
    security = policy.get("security") if isinstance(policy.get("security"), dict) else {}
    subject = subject_id(payload, source)
    revocation = revocation_check(policy, subject, payload)
    if not revocation.get("ok"):
        return revocation
    scope = scope_check(policy, kind, payload)
    if not scope.get("ok"):
        return scope
    manifest = manifest_check(
        policy,
        kind=kind,
        payload=payload,
        source=source,
        hive_id=hive_id,
        join_token=join_token,
        required=bool(security.get("require_signed_task_manifest_for_remote", False)) and not is_local_source(source),
    )
    if not manifest.get("ok"):
        return manifest
    quota = quota_check_and_record(policy, subject=subject, kind=kind, payload=payload, source=source)
    if not quota.get("ok"):
        return quota
    return {
        "ok": True,
        "policy": "project_theseus_hive_task_security_v0",
        "subject": subject,
        "local_node_id": local_node_id,
        "scope": scope,
        "manifest": manifest,
        "quota": quota,
    }


def build_manifest(kind: str, payload: dict[str, Any], *, hive_id: str, join_token: str, scope: list[str] | None = None) -> dict[str, Any]:
    body = {
        "policy": "project_theseus_hive_task_manifest_v0",
        "kind": kind,
        "hive_id": hive_id,
        "scope": scope or [kind],
        "payload_hash": sha256_json(payload),
        "created_utc": now(),
        "expires_utc": "",
        "nonce": hashlib.sha256(f"{time.time_ns()}:{kind}".encode("utf-8")).hexdigest()[:24],
        "signature": {"alg": "hmac-sha256-private-hive-v0", "value": ""},
    }
    body["signature"]["value"] = sign_manifest(body, join_token)
    return body


def manifest_check(
    policy: dict[str, Any],
    *,
    kind: str,
    payload: dict[str, Any],
    source: str,
    hive_id: str,
    join_token: str,
    required: bool,
) -> dict[str, Any]:
    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
    if not manifest:
        return {"ok": not required, "required": required, "status": "missing_optional" if not required else "missing_required"}
    if str(manifest.get("hive_id") or "") not in {"", hive_id}:
        return {"ok": False, "error": "manifest_hive_mismatch"}
    scope = manifest.get("scope") if isinstance(manifest.get("scope"), list) else []
    if scope and kind not in set(str(item) for item in scope):
        return {"ok": False, "error": "manifest_scope_denied", "kind": kind}
    expected_hash = str(manifest.get("payload_hash") or "")
    payload_without_manifest = {key: value for key, value in payload.items() if key != "manifest"}
    if expected_hash and expected_hash != sha256_json(payload_without_manifest):
        return {"ok": False, "error": "manifest_payload_hash_mismatch"}
    expires = parse_time(str(manifest.get("expires_utc") or ""))
    if expires and time.time() > expires:
        return {"ok": False, "error": "manifest_expired"}
    signature = manifest.get("signature") if isinstance(manifest.get("signature"), dict) else {}
    alg = str(signature.get("alg") or "")
    if alg != "hmac-sha256-private-hive-v0":
        return {"ok": False, "error": "unsupported_manifest_signature", "alg": alg}
    if join_token:
        actual = str(signature.get("value") or "")
        if actual and hmac.compare_digest(actual, sign_manifest(manifest, join_token)):
            return {"ok": True, "status": "verified", "required": required}
    if required:
        return {"ok": False, "error": "manifest_signature_invalid"}
    return {"ok": True, "status": "present_unverified_optional", "required": required}


def sign_manifest(manifest: dict[str, Any], secret: str) -> str:
    unsigned = json.loads(json.dumps(manifest))
    if isinstance(unsigned.get("signature"), dict):
        unsigned["signature"]["value"] = ""
    return hmac.new(secret.encode("utf-8"), canonical_json(unsigned).encode("utf-8"), hashlib.sha256).hexdigest()


def sign_artifact(payload: dict[str, Any], *, secret: str, signer: str) -> dict[str, Any]:
    digest = sha256_json(payload)
    signature = hmac.new(secret.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest() if secret else ""
    return {
        "policy": "project_theseus_hive_artifact_signature_v0",
        "alg": "hmac-sha256-private-hive-v0" if secret else "sha256-unsigned-local-v0",
        "signer": signer,
        "digest": digest,
        "signature": signature,
        "created_utc": now(),
    }


def revocation_check(policy: dict[str, Any], subject: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(get_path(policy, ["security", "revocations_path"], "configs/hive_revocations.local.json"))
    data = read_json(path, {})
    revoked = set(str(item) for item in data.get("revoked_subjects", []) if item)
    revoked.update(str(item) for item in data.get("revoked_invite_ids", []) if item)
    candidates = {subject, str(payload.get("invite_id") or ""), str(payload.get("requester_node_id") or "")}
    hit = sorted(item for item in candidates if item and item in revoked)
    if hit:
        return {"ok": False, "error": "revoked_hive_subject", "revoked": hit}
    return {"ok": True, "revocation_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}


def scope_check(policy: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    scope = payload.get("allowed_task_scope") if isinstance(payload.get("allowed_task_scope"), list) else []
    if scope and kind not in set(str(item) for item in scope):
        return {"ok": False, "error": "payload_scope_denied", "kind": kind}
    return {"ok": True, "scope": scope or "policy_tier_scope"}


def quota_check_and_record(policy: dict[str, Any], *, subject: str, kind: str, payload: dict[str, Any], source: str) -> dict[str, Any]:
    quotas = get_path(policy, ["security", "quotas"], {})
    if not isinstance(quotas, dict) or not quotas.get("enabled", True):
        return {"ok": True, "status": "disabled"}
    path = ROOT / str(get_path(policy, ["security", "quota_state_path"], "reports/hive_quota_state.json"))
    state = read_json(path, {"subjects": {}})
    now_ts = time.time()
    window_seconds = int(quotas.get("window_seconds", 3600))
    max_tasks = int(quotas.get("max_tasks_per_subject_per_window", 128))
    max_worker = int(quotas.get("max_worker_chunks_per_subject_per_window", 24))
    subject_state = state.setdefault("subjects", {}).setdefault(subject, {"events": []})
    events = [row for row in subject_state.get("events", []) if now_ts - float(row.get("ts", 0)) <= window_seconds]
    worker_events = [row for row in events if str(row.get("kind", "")).endswith("_chunk")]
    is_worker = kind.endswith("_chunk")
    if len(events) >= max_tasks:
        return {"ok": False, "error": "quota_tasks_exceeded", "subject": subject, "max": max_tasks}
    if is_worker and len(worker_events) >= max_worker:
        return {"ok": False, "error": "quota_worker_chunks_exceeded", "subject": subject, "max": max_worker}
    events.append({"ts": now_ts, "kind": kind, "source": source, "job_id": payload.get("job_id")})
    subject_state["events"] = events[-max_tasks:]
    subject_state["updated_utc"] = now()
    write_json(path, state)
    return {
        "ok": True,
        "subject": subject,
        "window_seconds": window_seconds,
        "used": len(events),
        "max": max_tasks,
        "worker_used": len(worker_events) + (1 if is_worker else 0),
        "worker_max": max_worker,
    }


def subject_id(payload: dict[str, Any], source: str) -> str:
    return str(payload.get("requester_node_id") or payload.get("invite_id") or source or "local")


def is_local_source(source: str) -> bool:
    return source in {"", "local"} or source.startswith("http:127.") or source.startswith("http:::1") or source.startswith("http:localhost")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_time(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
