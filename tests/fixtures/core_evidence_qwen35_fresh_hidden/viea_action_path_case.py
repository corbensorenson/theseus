from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import viea_action_executor as executor  # noqa: E402


def action(script: str) -> dict:
    return {
        "kind": "write_repo_repair_tasks",
        "public_data_rule": "public_benchmarks_calibration_only",
        "command": [sys.executable, script],
    }


def test_relative_parent_traversal_is_rejected() -> None:
    result = executor.validate_action(
        action(
            "scripts/../outside/long_horizon_programming_curriculum.py"
        ),
        allow_teacher=False,
    )
    assert result["allowed"] is False, (
        "request_contract:viea_relative_script_traversal_rejected"
    )


def test_absolute_parent_traversal_is_rejected() -> None:
    escaped = (
        executor.ROOT
        / "scripts"
        / ".."
        / "outside"
        / "long_horizon_programming_curriculum.py"
    )
    result = executor.validate_action(
        action(str(escaped)),
        allow_teacher=False,
    )
    assert result["allowed"] is False, (
        "request_contract:viea_absolute_script_traversal_rejected"
    )


def test_symlink_resolved_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_script = outside / "long_horizon_programming_curriculum.py"
    escaped_script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    link = executor.ROOT / "scripts" / "qualification-escape-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
        result = executor.validate_action(
            action(
                "scripts/qualification-escape-link/"
                "long_horizon_programming_curriculum.py"
            ),
            allow_teacher=False,
        )
        assert result["allowed"] is False, (
            "request_contract:viea_symlink_script_escape_rejected"
        )
    finally:
        link.unlink(missing_ok=True)


def test_real_allowlisted_script_remains_allowed() -> None:
    result = executor.validate_action(
        action("scripts/long_horizon_programming_curriculum.py"),
        allow_teacher=False,
    )
    assert result["allowed"] is True
