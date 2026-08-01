from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402


def protocol() -> dict:
    return {
        "maximum_actions_per_candidate": 8,
        "maximum_replacement_bytes": 65536,
    }


def task() -> dict:
    return {"allowed_effect_paths": ["src/example.py"]}


def test_frozen_p2a_instrument_audit_is_green() -> None:
    report = p2a.audit_instrument(ROOT / "configs" / "theseus_assistant_p2a_instrument.json")

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["model_identity"]["decoder"]["maximum_tokens"] == 1536


def test_prospectively_bound_p2b_instrument_audit_is_green() -> None:
    report = p2a.audit_instrument(ROOT / "configs" / "theseus_assistant_p2b_instrument.json")

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["model_identity"]["repo_id"] == "mlx-community/Qwen3.5-9B-MLX-4bit"


def test_typed_line_edits_are_concise_authorized_and_deterministically_applied(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "example.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    text = "THESEUS_EDIT_V1\nREPLACE src/example.py 2 2\n<<<\n    return 2\n>>>\nEND"

    actions, faults = p2a.parse_actions(text, task(), protocol())
    apply_faults = p2a.apply_actions(tmp_path, actions)

    assert faults == []
    assert apply_faults == []
    assert target.read_text(encoding="utf-8") == "def value():\n    return 2\n"


def test_typed_edits_reject_path_escape_and_unparsed_prose() -> None:
    escaped = "THESEUS_EDIT_V1\nREPLACE ../outside.py 1 1\n<<<\nx\n>>>\nEND"
    prose = "THESEUS_EDIT_V1\nI changed it.\nREPLACE src/example.py 1 1\n<<<\nx\n>>>\nEND"

    _, escaped_faults = p2a.parse_actions(escaped, task(), protocol())
    _, prose_faults = p2a.parse_actions(prose, task(), protocol())

    assert "typed_action_path_unauthorized" in escaped_faults
    assert "typed_action_unparsed_text" in prose_faults


def test_multi_action_application_is_atomic_on_invalid_range(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "example.py"
    original = "first\nsecond\n"
    target.write_text(original, encoding="utf-8")
    actions = [
        {"op": "REPLACE", "path": "src/example.py", "start_line": 1, "end_line": 1, "replacement": "changed"},
        {"op": "REPLACE", "path": "src/example.py", "start_line": 99, "end_line": 99, "replacement": "bad"},
    ]

    faults = p2a.apply_actions(tmp_path, actions)

    assert faults == ["action_1_line_range_out_of_bounds"]
    assert target.read_text(encoding="utf-8") == original


def test_task_audit_binds_license_archive_and_visible_verifier(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n", encoding="utf-8")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source / "example.py", arcname="example.py")
    payload = {
        "policy": "project_theseus_p2a_licensed_task_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": "fixture",
        "natural_request": "Change the value.",
        "source_archive": str(archive),
        "source_archive_sha256": p2a.sha256_file(archive),
        "source_provenance": {
            "url": "https://example.invalid/repository",
            "revision": "abc123",
            "license_spdx": "MIT",
        },
        "allowed_effect_paths": ["example.py"],
        "candidate_visible_context": {
            "reads": [{"path": "example.py", "start_line": 1, "end_line": 1}],
            "searches": [],
        },
        "visible_verifier": {"command": ["python3", "-m", "compileall", "-q", "."]},
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = p2a.audit_task(path)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []


def test_declared_archive_root_extracts_to_repository_relative_namespace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "github-prefix" / "src"
    nested.mkdir(parents=True)
    (nested / "example.py").write_text("value = 1\n", encoding="utf-8")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source / "github-prefix", arcname="github-prefix")
    destination = tmp_path / "candidate"

    p2a.extract_source_archive(archive, destination, "github-prefix")

    assert (destination / "src" / "example.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not (destination / "github-prefix").exists()
