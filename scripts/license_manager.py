"""Project Theseus registration and license gate.

The license manager is intentionally local-first. It supports a free
non-commercial community registration for small private hives and verifies paid
license files with a public-key signature when release verification keys are
configured. It never stores or ships an issuing private key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "license_policy.json"
HIVE_POLICY_PATH = ROOT / "configs" / "hive_policy.json"
DEFAULT_FEATURE = "local_research"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--out", default="")

    register = sub.add_parser("register")
    register.add_argument("--name", default="")
    register.add_argument("--email", default="")
    register.add_argument("--organization", default="")
    register.add_argument("--usage", choices=["personal_homelab", "research", "startup_free", "company", "public_operator"], default="personal_homelab")
    register.add_argument("--seats", type=int, default=1)
    register.add_argument("--commercial", action="store_true")
    register.add_argument("--accept-terms", action="store_true")
    register.add_argument("--out", default="")

    request = sub.add_parser("request")
    request.add_argument("--feature", action="append", default=[])
    request.add_argument("--out", default="")

    import_cmd = sub.add_parser("import")
    import_cmd.add_argument("--file", default="")
    import_cmd.add_argument("--license-json", default="")
    import_cmd.add_argument("--out", default="")

    check = sub.add_parser("check")
    check.add_argument("--feature", default=DEFAULT_FEATURE)
    check.add_argument("--requested-tier", default="")
    check.add_argument("--node-count", type=int, default=0)
    check.add_argument("--seat-count", type=int, default=0)
    check.add_argument("--commercial", action="store_true")
    check.add_argument("--out", default="")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "register":
        report = register_install(policy, args)
    elif args.command == "request":
        report = license_request(policy, args.feature)
    elif args.command == "import":
        report = import_license(policy, file_path=args.file, raw=args.license_json)
    elif args.command == "check":
        report = check_feature(
            args.feature,
            context={
                "requested_tier": args.requested_tier,
                "node_count": args.node_count or None,
                "seat_count": args.seat_count or None,
                "commercial_use": bool(args.commercial),
            },
            policy=policy,
            write_report=True,
        )
    else:
        report = status_report(policy=policy, write_report=True)
    out = getattr(args, "out", "") or ""
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) and report.get("allowed", True) is not False else 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = False) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    registration = read_registration(policy)
    license_file = read_license_file(policy)
    hive_state = current_hive_state()
    verification = verify_license_file(policy, license_file) if license_file else no_license_verification()
    verification = dict(verification)
    verification["license_file_present"] = bool(license_file)
    entitlement = entitlement_from(policy, registration, license_file, verification, hive_state)
    report = {
        "ok": True,
        "policy": "project_theseus_license_status_v0",
        "created_utc": now(),
        "license_policy": policy.get("policy"),
        "registration": public_registration(registration),
        "registration_complete": registration_complete(policy, registration),
        "license_file_present": bool(license_file),
        "license_verification": verification,
        "entitlement": entitlement,
        "hive": hive_state,
        "gates": build_status_gates(policy, registration, verification, entitlement, hive_state),
        "feature_summary": feature_summary(policy, entitlement, hive_state),
        "issuer_public_keys_configured": bool(get_path(policy, ["issuer", "signature_public_keys"], {})),
        "private_issuer_key_in_repo": False,
        "next_action": license_next_action(policy, registration, verification, entitlement),
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(status_path(policy), report)
    return report


def check_feature(
    feature: str,
    *,
    context: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    context = context or {}
    report = status_report(policy=policy, write_report=write_report)
    entitlement = report.get("entitlement") if isinstance(report.get("entitlement"), dict) else {}
    hive = report.get("hive") if isinstance(report.get("hive"), dict) else {}
    rules = get_path(policy, ["feature_rules", feature], {})
    node_count = int(context.get("node_count") or hive.get("node_count") or 1)
    seat_count = int(context.get("seat_count") or entitlement.get("seat_limit") or 1)
    requested_tier = str(context.get("requested_tier") or "")
    commercial = bool(context.get("commercial_use")) or requested_tier == "company"
    allowed_features = set(str(item) for item in entitlement.get("features", []))
    paid_tier = str(entitlement.get("paid_tier") or "")

    gates = [
        gate("license_policy_enabled", bool(policy.get("enabled", True)), "enabled"),
        gate(
            "registration_complete",
            not rules.get("registration_required") or bool(report.get("registration_complete")),
            "registration required" if rules.get("registration_required") else "not required",
        ),
        gate(
            "paid_tier_present",
            not rules.get("requires_paid_tier") or bool(entitlement.get("paid")),
            f"paid_tier={paid_tier or 'none'}",
        ),
        gate(
            "feature_in_entitlement",
            feature in allowed_features or feature == "local_status",
            f"feature={feature} source={entitlement.get('source')}",
        ),
        gate(
            "node_limit_ok",
            node_count <= int(entitlement.get("node_limit") or 0),
            f"nodes={node_count} limit={entitlement.get('node_limit')}",
        ),
        gate(
            "seat_limit_ok",
            seat_count <= int(entitlement.get("user_limit") or entitlement.get("seat_limit") or 0),
            f"seats={seat_count} limit={entitlement.get('user_limit') or entitlement.get('seat_limit')}",
        ),
        gate(
            "commercial_use_allowed",
            not commercial or bool(entitlement.get("commercial_use")),
            f"commercial={commercial} tier={paid_tier or entitlement.get('tier')}",
        ),
    ]
    allowed_paid = rules.get("allowed_paid_tiers")
    if isinstance(allowed_paid, list) and allowed_paid:
        gates.append(gate("allowed_paid_tier", paid_tier in set(str(item) for item in allowed_paid), f"allowed={allowed_paid} actual={paid_tier}"))
    allowed = all(row["ok"] for row in gates)
    result = {
        **report,
        "feature_check": {
            "feature": feature,
            "allowed": allowed,
            "requested_tier": requested_tier,
            "node_count": node_count,
            "seat_count": seat_count,
            "commercial_use": commercial,
            "gates": gates,
        },
        "allowed": allowed,
    }
    if write_report:
        write_json(status_path(policy), result)
    return result


def register_install(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.accept_terms:
        return {
            "ok": False,
            "error": "terms_acceptance_required",
            "message": f"Re-run with --accept-terms after accepting {policy.get('terms_version')}.",
        }
    usage = str(args.usage)
    commercial = bool(args.commercial or usage in {"company", "public_operator"})
    registration = read_registration(policy)
    if not registration:
        registration = {
            "policy": "project_theseus_registration_v0",
            "install_id": f"theseus-install-{uuid.uuid4().hex}",
            "created_utc": now(),
        }
    node = current_node_registration()
    nodes = registration.get("registered_nodes") if isinstance(registration.get("registered_nodes"), list) else []
    if not any(row.get("node_fingerprint") == node.get("node_fingerprint") for row in nodes if isinstance(row, dict)):
        nodes.append(node)
    registrant = registration.get("registrant") if isinstance(registration.get("registrant"), dict) else {}
    registration.update(
        {
            "updated_utc": now(),
            "terms_version": policy.get("terms_version"),
            "terms_accepted": True,
            "registrant": {
                "name": str(args.name or registrant.get("name") or ""),
                "email": str(args.email or registrant.get("email") or ""),
                "organization": str(args.organization or registrant.get("organization") or ""),
            },
            "usage": usage,
            "commercial_use": commercial,
            "declared_seats": max(1, int(args.seats or 1)),
            "registered_nodes": nodes,
        }
    )
    write_json(registration_path(policy), registration)
    append_jsonl(events_path(policy), event("registration_updated", {"usage": usage, "commercial_use": commercial, "declared_seats": registration["declared_seats"]}))
    report = status_report(policy=policy, write_report=True)
    return {**report, "registration_written": str(registration_path(policy).relative_to(ROOT))}


def license_request(policy: dict[str, Any], features: list[str]) -> dict[str, Any]:
    registration = read_registration(policy)
    report = status_report(policy=policy, write_report=True)
    request = {
        "policy": "project_theseus_license_request_v0",
        "created_utc": now(),
        "request_id": f"license-request-{uuid.uuid4().hex}",
        "project": "Project Theseus",
        "terms_version": policy.get("terms_version"),
        "registration": public_registration(registration),
        "hive": report.get("hive"),
        "requested_features": features or ["company_hive", "distributed_worker_chunks"],
        "requested_tier_hint": requested_tier_hint(registration, features),
        "machine": current_node_registration(),
        "privacy_note": "This request excludes join tokens, local datasets, ROM paths, private keys, and teacher credentials.",
    }
    write_json(request_path(policy), request)
    append_jsonl(events_path(policy), event("license_request_created", {"request_id": request["request_id"], "features": request["requested_features"]}))
    return {**request, "ok": True, "request_path": str(request_path(policy).relative_to(ROOT))}


def import_license(policy: dict[str, Any], *, file_path: str, raw: str) -> dict[str, Any]:
    if file_path:
        value = read_json((ROOT / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path), {})
    elif raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "error": "license_json_invalid"}
    else:
        return {"ok": False, "error": "license_file_or_json_required"}
    if not isinstance(value, dict):
        return {"ok": False, "error": "license_payload_must_be_object"}
    verification = verify_license_file(policy, value)
    if not verification.get("valid"):
        return {"ok": False, "error": "license_verification_failed", "license_verification": verification}
    write_json(license_path(policy), value)
    append_jsonl(events_path(policy), event("license_imported", {"license_id": value.get("license_id"), "tier": value.get("tier")}))
    return status_report(policy=policy, write_report=True)


def entitlement_from(
    policy: dict[str, Any],
    registration: dict[str, Any],
    license_file: dict[str, Any],
    verification: dict[str, Any],
    hive_state: dict[str, Any],
) -> dict[str, Any]:
    if license_file and verification.get("valid"):
        tier = str(license_file.get("tier") or "")
        paid_tiers = policy.get("paid_tiers") if isinstance(policy.get("paid_tiers"), dict) else {}
        tier_policy = paid_tiers.get(tier) if isinstance(paid_tiers.get(tier), dict) else {}
        limits = license_file.get("limits") if isinstance(license_file.get("limits"), dict) else {}
        features = license_file.get("features") if isinstance(license_file.get("features"), list) else tier_policy.get("features", [])
        return {
            "source": "signed_license",
            "tier": tier,
            "paid_tier": tier,
            "paid": True,
            "commercial_use": bool(tier_policy.get("commercial_use", True)),
            "node_limit": int(limits.get("max_nodes") or tier_policy.get("default_max_nodes") or 1),
            "seat_limit": int(limits.get("max_users") or limits.get("max_seats") or tier_policy.get("default_max_users") or 1),
            "user_limit": int(limits.get("max_users") or limits.get("max_seats") or tier_policy.get("default_max_users") or 1),
            "features": sorted(set(str(item) for item in features)),
            "expires_utc": license_file.get("expires_utc"),
            "license_id": license_file.get("license_id"),
        }
    community = policy.get("community_registration") if isinstance(policy.get("community_registration"), dict) else {}
    features = community.get("allowed_features") if isinstance(community.get("allowed_features"), list) else []
    registered = registration_complete(policy, registration)
    commercial = bool(registration.get("commercial_use"))
    declared_seats = int(registration.get("declared_seats") or 1)
    node_count = int(hive_state.get("node_count") or 1)
    community_ok = bool(community.get("enabled", True)) and registered and not commercial
    community_ok = community_ok and declared_seats <= int(community.get("max_users") or 0)
    community_ok = community_ok and node_count <= int(community.get("max_nodes") or 0)
    return {
        "source": "community_registration" if community_ok else "unlicensed",
        "tier": "homelab_free" if community_ok else "unregistered",
        "paid_tier": "",
        "paid": False,
        "commercial_use": False,
        "node_limit": int(community.get("max_nodes") or 1) if community_ok else 1,
        "seat_limit": int(community.get("max_users") or 1) if community_ok else 1,
        "user_limit": int(community.get("max_users") or 1) if community_ok else 1,
        "features": sorted(set(str(item) for item in features)) if community_ok else ["local_status"],
        "expires_utc": "",
        "license_id": "",
    }


def verify_license_file(policy: dict[str, Any], license_file: dict[str, Any]) -> dict[str, Any]:
    required = get_path(policy, ["license_file_format", "required_fields"], [])
    missing = [key for key in required if key not in license_file]
    if missing:
        return {"valid": False, "reason": "missing_required_fields", "missing": missing}
    signature = license_file.get("signature") if isinstance(license_file.get("signature"), dict) else {}
    alg = str(signature.get("alg") or "")
    key_id = str(signature.get("key_id") or "")
    value = str(signature.get("value") or "")
    if alg != "ed25519":
        return {"valid": False, "reason": "unsupported_signature_algorithm", "alg": alg}
    keys = get_path(policy, ["issuer", "signature_public_keys"], {})
    public_key_hex = str(keys.get(key_id) or "") if isinstance(keys, dict) else ""
    if not public_key_hex:
        return {"valid": False, "reason": "public_key_not_configured", "key_id": key_id}
    if is_expired(str(license_file.get("expires_utc") or "")):
        return {"valid": False, "reason": "license_expired", "expires_utc": license_file.get("expires_utc")}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional release dependency.
        return {"valid": False, "reason": "cryptography_backend_missing", "message": str(exc)}
    try:
        payload = canonical_license_payload(license_file)
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(base64url_decode(value), payload)
    except Exception as exc:  # noqa: BLE001 - signature verifier boundary.
        return {"valid": False, "reason": "signature_invalid", "message": str(exc), "key_id": key_id}
    return {"valid": True, "reason": "signature_valid", "alg": alg, "key_id": key_id, "license_id": license_file.get("license_id")}


def no_license_verification() -> dict[str, Any]:
    return {"valid": False, "reason": "no_license_file"}


def build_status_gates(
    policy: dict[str, Any],
    registration: dict[str, Any],
    verification: dict[str, Any],
    entitlement: dict[str, Any],
    hive_state: dict[str, Any],
) -> list[dict[str, Any]]:
    community = policy.get("community_registration") if isinstance(policy.get("community_registration"), dict) else {}
    return [
        gate("license_policy_enabled", bool(policy.get("enabled", True)), "enabled"),
        gate("registration_complete", registration_complete(policy, registration), "local app registration"),
        gate("terms_accepted", bool(registration.get("terms_accepted")), f"required={policy.get('terms_version')} actual={registration.get('terms_version')}"),
        gate("commercial_requires_paid_license", not registration.get("commercial_use") or bool(entitlement.get("paid")), f"commercial={registration.get('commercial_use')} paid={entitlement.get('paid')}"),
        gate("node_limit_ok", int(hive_state.get("node_count") or 1) <= int(entitlement.get("node_limit") or community.get("max_nodes") or 1), f"nodes={hive_state.get('node_count')} limit={entitlement.get('node_limit')}"),
        gate("seat_limit_ok", int(registration.get("declared_seats") or 1) <= int(entitlement.get("seat_limit") or community.get("max_users") or 1), f"seats={registration.get('declared_seats') or 1} limit={entitlement.get('seat_limit')}"),
        gate("paid_license_signature_valid_if_present", not bool(verification.get("license_file_present")) or verification.get("valid") is True, str(verification.get("reason"))),
    ]


def feature_summary(policy: dict[str, Any], entitlement: dict[str, Any], hive_state: dict[str, Any]) -> dict[str, Any]:
    features = sorted(str(item) for item in entitlement.get("features", []))
    return {
        "tier": entitlement.get("tier"),
        "source": entitlement.get("source"),
        "features": features,
        "can_create_private_hive": "private_hive" in features,
        "can_create_friends_family_hive": "friends_family_hive" in features,
        "can_create_company_hive": "company_hive" in features,
        "can_run_worker_chunks": "distributed_worker_chunks" in features,
        "can_use_public_contribution_worker": "public_contribution_worker" in features,
        "can_use_compute_market": "compute_market_accounting" in features,
        "can_rent_compute": "compute_rental_client" in features,
        "can_account_public_work": "public_work_accounting" in features,
        "can_install_updates": "update_install" in features or "local_research" in features,
        "can_use_private_update_channel": "private_update_channel" in features,
        "can_operate_public_gateway": "public_hive_gateway" in features,
        "nodes_used": hive_state.get("node_count"),
        "node_limit": entitlement.get("node_limit"),
    }


def license_next_action(policy: dict[str, Any], registration: dict[str, Any], verification: dict[str, Any], entitlement: dict[str, Any]) -> str:
    if not registration_complete(policy, registration):
        return "Register this install and accept the current terms before creating or joining hives."
    if registration.get("commercial_use") and not entitlement.get("paid"):
        return "Commercial/company use requires importing a signed paid license."
    if entitlement.get("source") == "community_registration":
        return "Community registration active for non-commercial private use under the free node/user cap."
    if entitlement.get("source") == "signed_license":
        return "Signed license active."
    if verification.get("reason") == "public_key_not_configured":
        return "Add release public verification keys before importing paid licenses."
    return "Import a signed license for paid or larger-scope use."


def current_hive_state() -> dict[str, Any]:
    peers = read_json(ROOT / "reports" / "hive_peers.json", {})
    status = read_json(ROOT / "reports" / "hive_status.json", {})
    join = read_json(ROOT / "configs" / "hive_join.local.json", {})
    peer_count = int(peers.get("peer_count") or 0) if isinstance(peers, dict) else 0
    return {
        "hive_id": join.get("hive_id") or status.get("hive_id") or "local",
        "tier": join.get("tier") or status.get("federation_tier") or "private",
        "mode": join.get("mode") or "",
        "node_count": 1 + max(0, peer_count),
        "peer_count": max(0, peer_count),
        "node_name": status.get("node_name") or socket.gethostname(),
    }


def registration_complete(policy: dict[str, Any], registration: dict[str, Any]) -> bool:
    return (
        bool(registration)
        and registration.get("policy") == "project_theseus_registration_v0"
        and bool(registration.get("terms_accepted"))
        and registration.get("terms_version") == policy.get("terms_version")
    )


def public_registration(registration: dict[str, Any]) -> dict[str, Any]:
    if not registration:
        return {}
    clean = dict(registration)
    clean["registrant"] = {
        "name": get_path(registration, ["registrant", "name"], ""),
        "email_configured": bool(get_path(registration, ["registrant", "email"], "")),
        "organization": get_path(registration, ["registrant", "organization"], ""),
    }
    return clean


def current_node_registration() -> dict[str, Any]:
    identity = read_json(ROOT / "reports" / "hive_node_identity.json", {})
    raw = "|".join(
        [
            str(identity.get("node_id") or ""),
            socket.gethostname(),
            platform.system(),
            platform.machine(),
            str(ROOT),
        ]
    )
    return {
        "node_id": identity.get("node_id") or "",
        "node_name": identity.get("node_name") or socket.gethostname(),
        "node_fingerprint": hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "registered_utc": now(),
    }


def requested_tier_hint(registration: dict[str, Any], features: list[str]) -> str:
    if "public_hive_gateway" in set(features) or registration.get("usage") == "public_operator":
        return "public_operator"
    if registration.get("commercial_use") or registration.get("usage") == "company":
        return "team"
    return "homelab_free"


def feature_for_hive_tier(tier: str) -> str:
    if tier == "company":
        return "company_hive"
    if tier == "friends_family":
        return "friends_family_hive"
    if tier == "public":
        return "public_hive_gateway"
    return "private_hive"


def feature_for_task_kind(kind: str) -> str:
    if kind in {
        "training_orchestrate",
        "training_smoke",
        "cuda_eval_chunk",
        "cuda_training_chunk",
        "cuda_rollout_chunk",
        "mlx_eval_chunk",
        "mlx_training_chunk",
        "mlx_rollout_chunk",
    }:
        return "distributed_worker_chunks"
    if kind == "compute_market_status":
        return "compute_market_accounting"
    if kind in {"update_status", "update_apply_soft", "hive_version_status", "hive_version_converge"}:
        return "update_install"
    if kind.startswith("public_"):
        return "public_contribution_worker"
    if kind in {"compute_market_quote", "compute_market_settle", "compute_market_rent"}:
        return "compute_market_accounting"
    return "private_hive"


def canonical_license_payload(license_file: dict[str, Any]) -> bytes:
    payload = dict(license_file)
    payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def is_expired(value: str) -> bool:
    if not value:
        return True
    try:
        expires = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    return expires < datetime.now(timezone.utc)


def base64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def gate(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), "kind": kind, **payload}


def read_registration(policy: dict[str, Any]) -> dict[str, Any]:
    return read_json(registration_path(policy), {})


def read_license_file(policy: dict[str, Any]) -> dict[str, Any]:
    return read_json(license_path(policy), {})


def registration_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "registration"], "configs/theseus_registration.local.json"))


def license_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "license"], "configs/theseus_license.local.json"))


def status_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "status"], "reports/license_status.json"))


def events_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "events"], "reports/license_events.jsonl"))


def request_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "request"], "reports/license_request.json"))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
