from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sparkstream_dashboard as dashboard  # noqa: E402


def security() -> dict:
    return {
        "allowed_origins": ["http://127.0.0.1:8787"],
        "session_token": "session-secret",
        "csrf_token": "csrf-secret",
        "max_request_body_bytes": 16,
    }


def headers(**overrides: str) -> dict[str, str]:
    rows = {
        "Origin": "http://127.0.0.1:8787",
        "Cookie": "theseus_dashboard_session=session-secret",
        "X-Theseus-CSRF": "csrf-secret",
    }
    rows.update(overrides)
    return rows


def test_mutation_requires_exact_origin_session_csrf_and_loopback() -> None:
    dashboard.validate_dashboard_mutation(
        headers(),
        security(),
        client="127.0.0.1",
    )

    with pytest.raises(dashboard.DashboardRequestFault, match="origin_denied"):
        dashboard.validate_dashboard_mutation(
            headers(Origin="http://evil.invalid"),
            security(),
            client="127.0.0.1",
        )
    with pytest.raises(dashboard.DashboardRequestFault, match="auth_required"):
        dashboard.validate_dashboard_mutation(
            headers(Cookie=""),
            security(),
            client="127.0.0.1",
        )
    with pytest.raises(dashboard.DashboardRequestFault, match="csrf_denied"):
        dashboard.validate_dashboard_mutation(
            headers(**{"X-Theseus-CSRF": "wrong"}),
            security(),
            client="127.0.0.1",
        )
    with pytest.raises(dashboard.DashboardRequestFault, match="loopback_required"):
        dashboard.validate_dashboard_mutation(
            headers(),
            security(),
            client="192.0.2.8",
        )


def test_dashboard_payload_schema_rejects_cross_route_fields() -> None:
    dashboard.validate_dashboard_payload(
        "/api/teacher/ask",
        {"prompt": "help", "allow_teacher": False},
    )

    with pytest.raises(dashboard.DashboardRequestFault, match="unknown_fields"):
        dashboard.validate_dashboard_payload(
            "/api/teacher/ask",
            {"prompt": "help", "allow_network_fetch": True},
        )


def test_dashboard_json_reader_is_strict_and_bounded() -> None:
    request = SimpleNamespace(
        headers={
            "Content-Type": "application/json",
            "Content-Length": "2",
        },
        rfile=io.BytesIO(b"{}"),
        security=security(),
    )
    assert dashboard.Handler.read_json_body(request) == {}

    malformed = SimpleNamespace(
        headers={
            "Content-Type": "application/json",
            "Content-Length": "1",
        },
        rfile=io.BytesIO(b"{"),
        security=security(),
    )
    with pytest.raises(dashboard.DashboardRequestFault, match="malformed_json"):
        dashboard.Handler.read_json_body(malformed)

    wrong_type = SimpleNamespace(
        headers={"Content-Type": "text/plain", "Content-Length": "2"},
        rfile=io.BytesIO(b"{}"),
        security=security(),
    )
    with pytest.raises(
        dashboard.DashboardRequestFault,
        match="application_json_required",
    ):
        dashboard.Handler.read_json_body(wrong_type)


def test_control_omission_never_inherits_teacher_or_network_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_start_job(name: str, command: list[str]) -> dict:
        captured["name"] = name
        captured["command"] = command
        return {"ok": True}

    monkeypatch.setattr(dashboard, "start_job", fake_start_job)
    result = dashboard.handle_control(
        {"action": "run_cycle", "profile": "smoke"}
    )

    assert result["ok"] is True
    assert "--allow-teacher" not in captured["command"]
    assert "--allow-network-fetch" not in captured["command"]


def test_dashboard_security_tokens_are_random_and_limits_are_bounded() -> None:
    policy = {
        "dashboard_security": {
            "allowed_origins": ["http://127.0.0.1:8787"],
            "max_concurrent_requests": 0,
            "max_active_jobs": 0,
        }
    }

    first = dashboard.build_dashboard_security("127.0.0.1", 8787, policy)
    second = dashboard.build_dashboard_security("127.0.0.1", 8787, policy)

    assert first["session_token"] != second["session_token"]
    assert first["csrf_token"] != second["csrf_token"]
    assert first["max_concurrent_requests"] == 4
    assert first["max_active_jobs"] == 8
