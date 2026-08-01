from __future__ import annotations

import subprocess
from types import SimpleNamespace

import scripts.theseus_workspace_hygiene_audit as hygiene


def test_git_status_failures_are_typed_and_do_not_expose_process_output(
    monkeypatch,
) -> None:
    fault_type = getattr(hygiene, "WorkspaceStatusFault", RuntimeError)

    monkeypatch.setattr(
        hygiene.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=7,
            stdout="private stdout must not escape",
            stderr="private stderr must not escape",
        ),
    )
    try:
        hygiene.git_status_rows()
    except fault_type as exc:
        message = str(exc)
    else:
        message = ""
    assert message == "git_status_failed", (
        "request_contract:typed_git_status_failures"
    )
    assert "private" not in message, (
        "request_contract:typed_git_status_failures"
    )

    monkeypatch.setattr(
        hygiene.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("git status", 10)
        ),
    )
    try:
        hygiene.git_status_rows()
    except fault_type as exc:
        timeout_message = str(exc)
    else:
        timeout_message = ""
    assert timeout_message == "git_status_timeout", (
        "request_contract:typed_git_status_failures"
    )

    monkeypatch.setattr(
        hygiene.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("private operating-system detail")
        ),
    )
    try:
        hygiene.git_status_rows()
    except fault_type as exc:
        unavailable_message = str(exc)
    else:
        unavailable_message = ""
    assert unavailable_message == "git_status_unavailable", (
        "request_contract:typed_git_status_failures"
    )


def test_dirty_workspace_reports_collection_failure_as_high_priority(
    monkeypatch,
) -> None:
    fault_type = getattr(hygiene, "WorkspaceStatusFault", RuntimeError)

    def unavailable() -> list[str]:
        raise fault_type("git_status_timeout")

    monkeypatch.setattr(hygiene, "git_status_rows", unavailable)
    try:
        candidates = hygiene.dirty_workspace_candidates()
    except Exception:
        candidates = []

    assert len(candidates) == 1, (
        "request_contract:workspace_status_unavailable_candidate"
    )
    candidate = candidates[0]
    assert candidate["kind"] == "workspace_status_unavailable", (
        "request_contract:workspace_status_unavailable_candidate"
    )
    assert candidate["id"] == "dirty_workspace_status_unavailable", (
        "request_contract:workspace_status_unavailable_candidate"
    )
    assert candidate["priority"] == "high", (
        "request_contract:workspace_status_unavailable_candidate"
    )
    serialized = repr(candidate)
    assert "private stdout" not in serialized
    assert "private stderr" not in serialized


def test_normal_git_status_rows_remain_trimmed(monkeypatch) -> None:
    monkeypatch.setattr(
        hygiene.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=" M scripts/x.py\n?? tests/test_x.py\n\n",
            stderr="",
        ),
    )

    assert hygiene.git_status_rows() == [
        " M scripts/x.py",
        "?? tests/test_x.py",
    ]
