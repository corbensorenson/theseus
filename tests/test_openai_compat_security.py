from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from scripts import openai_compat_server as server


def policy() -> dict:
    return {
        "security": {
            "loopback_only_by_default": True,
            "require_token_by_default": True,
            "allow_unauthenticated_loopback": False,
            "require_token_for_non_loopback": True,
            "cors_disabled_by_default": True,
            "never_call_external_inference": True,
            "teacher_disabled_by_default": True,
        }
    }


def test_safe_defaults_generate_token_and_disable_external_authority() -> None:
    cfg = {
        "host": "0.0.0.0",
        "require_token": False,
        "api_token": "",
        "cors": False,
        "allow_teacher": True,
    }

    server.enforce_safe_defaults(policy(), cfg)

    assert cfg["host"] == "127.0.0.1"
    assert cfg["require_token"] is True
    assert len(cfg["api_token"]) >= 32
    assert cfg["cors"] is False
    assert cfg["allow_teacher"] is False


def test_cors_requires_exact_nonwildcard_origin() -> None:
    cfg = {
        "cors": True,
        "allowed_origins": ["http://127.0.0.1:8789"],
    }

    assert server.origin_allowed(cfg, "http://127.0.0.1:8789")
    assert not server.origin_allowed(cfg, "http://localhost:8789")
    assert not server.origin_allowed(cfg, "*")
    assert not server.origin_allowed({"cors": False}, "http://127.0.0.1:8789")


def test_request_schema_rejects_unknown_fields() -> None:
    server.validate_request_payload(
        "/v1/chat/completions",
        {"model": "theseus-live", "messages": []},
    )

    with pytest.raises(server.RequestFault, match="unknown_fields:solution"):
        server.validate_request_payload(
            "/v1/chat/completions",
            {"messages": [], "solution": "hidden"},
        )


def test_json_reader_requires_type_length_utf8_object_and_bound() -> None:
    valid = SimpleNamespace(
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": "2",
        },
        rfile=io.BytesIO(b"{}"),
        config={"max_request_body_bytes": 8},
    )
    assert server.Handler.read_json_body(valid) == {}

    wrong_type = SimpleNamespace(
        headers={"Content-Type": "text/plain", "Content-Length": "2"},
        rfile=io.BytesIO(b"{}"),
        config={},
    )
    with pytest.raises(server.RequestFault, match="application_json_required"):
        server.Handler.read_json_body(wrong_type)

    oversized = SimpleNamespace(
        headers={"Content-Type": "application/json", "Content-Length": "9"},
        rfile=io.BytesIO(b'{"x": 1}'),
        config={"max_request_body_bytes": 8},
    )
    with pytest.raises(server.RequestFault, match="request_body_too_large"):
        server.Handler.read_json_body(oversized)


def test_bearer_authentication_is_required_and_exact() -> None:
    good = SimpleNamespace(
        config={"require_token": True, "api_token": "secret"},
        headers={"Authorization": "Bearer secret"},
    )
    bad = SimpleNamespace(
        config={"require_token": True, "api_token": "secret"},
        headers={"Authorization": "Bearer secrex"},
    )

    assert server.Handler.authorized(good)
    assert not server.Handler.authorized(bad)
