from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dogfood_trace_event.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("dogfood_event_contract", SCRIPT)
assert SPEC and SPEC.loader
logger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logger)


def arguments(config: Path, *, execute: bool) -> Namespace:
    return Namespace(
        config=str(config),
        trace_jsonl=str(config.parent / "events.jsonl"),
        out=str(config.parent / "check.json"),
        markdown_out=str(config.parent / "check.md"),
        surface="local_cli",
        assistant_lane="code_assistant",
        outcome="completed",
        intent_summary_redacted="bounded local dry-run receipt check",
        artifact_ref=[],
        error_family="",
        duration_ms=1,
        execute=execute,
    )


def consent_config(tmp_path: Path) -> Path:
    path = tmp_path / "dogfood.json"
    path.write_text(
        json.dumps({
            "capture_enabled": True,
            "explicit_capture_consent_utc": "2026-07-31T00:00:00Z",
            "training_enabled": False,
            "raw_text_capture_enabled": False,
            "trace_jsonl": str(tmp_path / "events.jsonl"),
        }),
        encoding="utf-8",
    )
    return path


def test_dry_run_names_execute_as_the_only_blocker(tmp_path: Path) -> None:
    report = logger.build_report(
        arguments(consent_config(tmp_path), execute=False),
        time.perf_counter(),
    )
    assert report["trigger_state"] == "YELLOW"
    assert report["event_written"] is False
    assert report["write_blocker"] == "execute_not_requested", (
        "request_contract:execute_not_requested"
    )


def test_event_uses_metadata_scope_v1(tmp_path: Path) -> None:
    event = logger.build_event(
        arguments(consent_config(tmp_path), execute=False)
    )
    assert event["consent_scope"] == "dogfood_metadata_only_v1", (
        "request_contract:metadata_scope_v1"
    )


def test_dry_run_preserves_safety_boundaries(tmp_path: Path) -> None:
    report = logger.build_report(
        arguments(consent_config(tmp_path), execute=False),
        time.perf_counter(),
    )
    hard = {
        row["name"]: row["passed"]
        for row in report["gates"]
        if row["severity"] == "hard"
    }
    assert all(hard.values()), "request_contract:safety_boundaries_preserved"
    assert report["summary"]["trained_on_user_text"] is False, (
        "request_contract:safety_boundaries_preserved"
    )
    assert report["external_inference_calls"] == 0, (
        "request_contract:safety_boundaries_preserved"
    )
