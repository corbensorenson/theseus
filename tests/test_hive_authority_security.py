from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hive_node  # noqa: E402
import hive_node_federation  # noqa: E402
import hive_node_peer_registry  # noqa: E402
import hive_users  # noqa: E402


def load_policy() -> dict:
    return json.loads(
        (ROOT / "configs" / "hive_policy.json").read_text(encoding="utf-8")
    )


def test_hive_defaults_loopback_signed_and_header_only() -> None:
    policy = load_policy()

    assert policy["node"]["http_host"] == "127.0.0.1"
    assert policy["discovery"]["require_signed_multicast"] is True
    assert policy["discovery"]["trust_unsigned_multicast_without_secret"] is False
    assert policy["security"]["allow_status_without_secret"] is False
    assert policy["security"]["allow_loopback_tasks_without_secret"] is False
    assert policy["security"]["require_signed_task_manifest_for_remote"] is True
    assert policy["security"]["allow_query_credentials"] is False
    assert policy["multi_user"]["legacy_hive_join_token_role"] == "disabled"


def test_query_credentials_are_ignored_by_default() -> None:
    policy = load_policy()

    assert (
        hive_users.token_from_request(
            "",
            "token=leaked-in-url",
            policy=policy,
        )
        == ""
    )
    assert (
        hive_users.token_from_request(
            "header-token",
            "token=query-token",
            policy=policy,
        )
        == "header-token"
    )


def test_coordinator_and_worker_credentials_are_separate_and_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_policy()
    monkeypatch.setenv("THESEUS_HIVE_COORDINATOR_SECRET", "coordinator")
    monkeypatch.setenv("THESEUS_HIVE_WORKER_SECRET", "worker")
    monkeypatch.setenv("THESEUS_HIVE_DISCOVERY_SECRET", "discovery")

    coordinator = hive_users.authorize(
        policy,
        "192.0.2.10",
        "coordinator",
        action="status",
        allow_loopback=False,
    )
    worker = hive_users.authorize(
        policy,
        "192.0.2.11",
        "worker",
        action="task",
        task_kind="resource_probe",
        allow_loopback=False,
    )
    worker_operator_attempt = hive_users.authorize(
        policy,
        "192.0.2.11",
        "worker",
        action="update_apply",
        allow_loopback=False,
    )

    assert coordinator["ok"] is True
    assert coordinator["token_kind"] == "hive_coordinator_secret"
    assert worker["ok"] is True
    assert worker["token_kind"] == "hive_worker_secret"
    assert worker_operator_attempt["ok"] is False
    assert hive_node_federation.worker_secret(policy) == "worker"
    assert hive_node_federation.discovery_secret(policy) == "discovery"


def test_discovery_is_signed_with_a_separate_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_policy()
    monkeypatch.setenv("THESEUS_HIVE_COORDINATOR_SECRET", "coordinator")
    monkeypatch.setenv("THESEUS_HIVE_WORKER_SECRET", "worker")
    monkeypatch.setenv("THESEUS_HIVE_DISCOVERY_SECRET", "discovery")
    payload = hive_node_peer_registry.signed_discovery_payload(
        policy,
        {"node_id": "worker-a", "node_name": "worker-a"},
    )

    assert payload["signature_alg"] == "hmac-sha256"
    assert hive_node_peer_registry.verify_discovery_payload(policy, payload)
    payload["peer"]["node_name"] = "attacker"
    assert not hive_node_peer_registry.verify_discovery_payload(policy, payload)


def test_unsigned_discovery_fails_closed_without_discovery_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_policy()
    monkeypatch.delenv("THESEUS_HIVE_DISCOVERY_SECRET", raising=False)
    monkeypatch.delenv("THESEUS_HIVE_SECRET", raising=False)
    payload = {
        "prefix": "project-theseus-hive",
        "hive_id": "local",
        "peer": {"node_id": "unsigned"},
    }

    assert not hive_node_peer_registry.unsigned_multicast_trusted(
        policy,
        payload,
    )


def test_remote_worker_execution_fails_closed_without_qualified_sandbox() -> None:
    policy = load_policy()
    task = {
        "kind": "resource_probe",
        "payload": {},
        "source": "http:192.0.2.20",
    }

    result = hive_node.run_task(policy, task)

    assert result["status"] == "denied"
    assert result["error"] == "remote_worker_sandbox_unqualified"
    assert result["sandbox"]["execution_attempted"] is False


def test_hive_json_body_and_browser_origin_are_strict() -> None:
    policy = load_policy()
    handler = hive_node.make_handler(policy)
    request = SimpleNamespace(
        headers={
            "Content-Type": "application/json",
            "Content-Length": "2",
        },
        rfile=io.BytesIO(b"{}"),
    )
    assert handler.read_json_body(request) == {}

    oversized = SimpleNamespace(
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(
                policy["security"]["max_request_body_bytes"] + 1
            ),
        },
        rfile=io.BytesIO(b""),
    )
    with pytest.raises(hive_node.HiveRequestFault, match="body_too_large"):
        handler.read_json_body(oversized)

    hive_node.validate_hive_browser_origin(
        {"Origin": "http://127.0.0.1:8791"},
        policy,
    )
    with pytest.raises(hive_node.HiveRequestFault, match="origin_denied"):
        hive_node.validate_hive_browser_origin(
            {"Origin": "http://evil.invalid"},
            policy,
        )


def test_hive_post_schemas_reject_unknown_missing_and_wrong_type_fields() -> None:
    hive_node.validate_hive_request_schema(
        "/api/hive/tasks",
        {"kind": "resource_probe", "payload": {}},
    )
    with pytest.raises(hive_node.HiveRequestFault, match="unknown_request_fields"):
        hive_node.validate_hive_request_schema(
            "/api/hive/tasks",
            {
                "kind": "resource_probe",
                "payload": {},
                "teacher_authority": True,
            },
        )
    with pytest.raises(hive_node.HiveRequestFault, match="missing_request_fields"):
        hive_node.validate_hive_request_schema(
            "/api/hive/tasks",
            {"kind": "resource_probe"},
        )
    with pytest.raises(
        hive_node.HiveRequestFault,
        match="request_fields_must_be_objects",
    ):
        hive_node.validate_hive_request_schema(
            "/api/hive/operator/task",
            {"kind": "resource_probe", "task_payload": "not-an-object"},
        )


def test_hive_handler_applies_limits_to_reads_and_writes() -> None:
    source = (ROOT / "scripts" / "hive_node.py").read_text(encoding="utf-8")

    assert "def do_GET(self)" in source
    assert "return self._do_GET()" in source
    assert "def do_POST(self)" in source
    assert "return self._do_POST()" in source
    assert source.count("self.request_slots.acquire(blocking=False)") == 2
    assert source.count("if not self.rate_allowed()") == 2
