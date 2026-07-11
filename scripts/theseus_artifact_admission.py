"""Content-bound admission for model, tokenizer, and derived artifacts.

The verifier runs before deserialization. Human-readable paths are locators,
never authorization identity; admission binds artifact bytes, signer, advisory
snapshot, generation floor, custody observations, and intended purpose.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_payload(payload: dict[str, Any]) -> bytes:
    body = dict(payload)
    body.pop("signature", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_payload(payload)).hexdigest()


def _decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def verify_signature(payload: dict[str, Any], trusted_keys: dict[str, str]) -> dict[str, Any]:
    signature = payload.get("signature") if isinstance(payload.get("signature"), dict) else {}
    key_id = str(signature.get("key_id") or "")
    if signature.get("alg") != "ed25519":
        return {"valid": False, "reason": "unsupported_signature_algorithm", "key_id": key_id}
    public_key_hex = str(trusted_keys.get(key_id) or "")
    if not public_key_hex:
        return {"valid": False, "reason": "untrusted_signing_key", "key_id": key_id}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            _decode(str(signature.get("value") or "")),
            canonical_payload(payload),
        )
    except Exception as exc:  # verifier boundary
        return {"valid": False, "reason": "signature_invalid", "key_id": key_id, "detail": exc.__class__.__name__}
    return {"valid": True, "reason": "signature_valid", "key_id": key_id}


def parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def observe_custody(path: Path) -> dict[str, Any]:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    return {
        "locator": str(path),
        "is_regular_file": stat.S_ISREG(info.st_mode),
        "is_symlink": path.is_symlink(),
        "owner_uid": info.st_uid,
        "current_uid": os.getuid(),
        "owner_matches_current_user": info.st_uid == os.getuid(),
        "mode_octal": oct(mode),
        "group_or_world_writable": bool(mode & 0o022),
        "group_or_world_readable": bool(mode & 0o044),
    }


def admit_artifact(
    path: Path,
    *,
    attestation: dict[str, Any],
    advisory_snapshot: dict[str, Any],
    policy: dict[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    if not path.exists() or not path.is_file():
        return {"admitted": False, "reason": "artifact_missing", "path": str(path), "checks": {}}
    observed_hash = file_sha256(path)
    custody = observe_custody(path)
    trusted_keys = policy.get("trusted_public_keys") if isinstance(policy.get("trusted_public_keys"), dict) else {}
    revoked_keys = {str(value) for value in policy.get("revoked_key_ids", [])}
    revoked_artifacts = {str(value) for value in policy.get("revoked_artifact_hashes", [])}
    attestation_signature = verify_signature(attestation, trusted_keys)
    advisory_signature = verify_signature(advisory_snapshot, trusted_keys)

    def check(name: str, passed: bool, reason: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            reasons.append(reason)

    check("attestation_signature", bool(attestation_signature.get("valid")), str(attestation_signature.get("reason")))
    check("advisory_signature", bool(advisory_signature.get("valid")), str(advisory_signature.get("reason")))
    check("signing_key_active", str(attestation_signature.get("key_id") or "") not in revoked_keys, "signing_key_revoked")
    check("artifact_hash_matches", str(attestation.get("artifact_sha256") or "") == observed_hash, "artifact_hash_mismatch")
    check("artifact_not_revoked", observed_hash not in revoked_artifacts, "artifact_revoked")
    issued = parse_utc(str(attestation.get("issued_utc") or ""))
    expires = parse_utc(str(attestation.get("expires_utc") or ""))
    check("attestation_time_valid", bool(issued and expires and issued <= now_utc <= expires), "attestation_stale_or_not_yet_valid")
    advisory_created = parse_utc(str(advisory_snapshot.get("created_utc") or ""))
    max_age_hours = float(policy.get("max_advisory_age_hours") or 24.0)
    advisory_age = (now_utc - advisory_created).total_seconds() / 3600.0 if advisory_created else float("inf")
    check("advisory_fresh", 0.0 <= advisory_age <= max_age_hours, "advisory_snapshot_stale")
    check(
        "advisory_identity_bound",
        str(attestation.get("advisory_snapshot_sha256") or "") == payload_sha256(advisory_snapshot),
        "advisory_snapshot_identity_mismatch",
    )
    advisory_revoked = {str(value) for value in advisory_snapshot.get("revoked_artifact_hashes", [])}
    check("advisory_allows_artifact", observed_hash not in advisory_revoked, "artifact_revoked_by_advisory")
    logical_id = str(attestation.get("logical_artifact_id") or "")
    generation = int(attestation.get("generation") or 0)
    floors = policy.get("minimum_generation_by_artifact") if isinstance(policy.get("minimum_generation_by_artifact"), dict) else {}
    check("anti_rollback_generation", generation >= int(floors.get(logical_id) or 0), "artifact_generation_rollback")
    check("regular_non_symlink", custody["is_regular_file"] and not custody["is_symlink"], "artifact_locator_not_regular_file")
    check("owner_matches", custody["owner_matches_current_user"], "artifact_owner_mismatch")
    check("not_group_or_world_writable", not custody["group_or_world_writable"], "artifact_permissions_writable_by_others")
    value_class = str(attestation.get("value_class") or "development")
    custody_record = attestation.get("custody") if isinstance(attestation.get("custody"), dict) else {}
    if value_class == "valuable_weight":
        check("private_file_mode", not custody["group_or_world_readable"], "valuable_weight_permissions_too_broad")
        check("encrypted_storage_observed", bool(custody_record.get("encrypted_storage_observed")), "valuable_weight_encrypted_storage_unproven")
        check("key_release_bound", bool(custody_record.get("key_release_record_id")), "valuable_weight_key_release_missing")
        check("anti_rollback_bound", bool(custody_record.get("anti_rollback_state_id")), "valuable_weight_anti_rollback_state_missing")
    purpose = str(policy.get("purpose") or "")
    allowed_purposes = {str(value) for value in attestation.get("allowed_purposes", [])}
    check("purpose_allowed", bool(purpose and purpose in allowed_purposes), "artifact_purpose_not_allowed")
    return {
        "policy": "project_theseus_artifact_admission_v1",
        "admitted": all(checks.values()),
        "reason": "admitted" if all(checks.values()) else reasons[0],
        "reasons": reasons,
        "artifact": {
            "logical_artifact_id": logical_id,
            "locator": str(path),
            "observed_sha256": observed_hash,
            "generation": generation,
            "value_class": value_class,
        },
        "checks": checks,
        "custody_observation": custody,
        "attestation_signature": attestation_signature,
        "advisory_signature": advisory_signature,
        "external_inference_calls": 0,
    }


def admit_from_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    admission = config.get("artifact_admission") if isinstance(config.get("artifact_admission"), dict) else {}
    if not admission.get("required"):
        return {"required": False, "admitted": True, "reason": "development_artifact_admission_not_required"}
    policy_path = Path(str(admission.get("policy") or ""))
    attestation_path = Path(str(admission.get("attestation") or ""))
    advisory_path = Path(str(admission.get("advisory_snapshot") or ""))
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[1] / policy_path
    if not attestation_path.is_absolute():
        attestation_path = Path(__file__).resolve().parents[1] / attestation_path
    if not advisory_path.is_absolute():
        advisory_path = Path(__file__).resolve().parents[1] / advisory_path
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        advisory = json.loads(advisory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"required": True, "admitted": False, "reason": f"artifact_admission_input_invalid:{exc.__class__.__name__}"}
    result = admit_artifact(path, attestation=attestation, advisory_snapshot=advisory, policy=policy)
    result["required"] = True
    return result

