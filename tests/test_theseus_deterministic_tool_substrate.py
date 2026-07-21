from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_deterministic_tool_substrate as substrate  # noqa: E402


def test_qualification_receipt_binds_exact_tool_card_set(tmp_path: Path) -> None:
    cards = {
        "math.exact": {"replay_checksum": "card-a"},
        "search.local": {"replay_checksum": "card-b"},
    }
    report_path = tmp_path / "qualification.json"
    report_path.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "passed": True,
                "created_utc": "2026-01-01T00:00:00Z",
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
                "tool_results": [{"case_id": "a"}],
                "artifact_graph": {
                    "artifacts": [
                        {"type": "Tool", "content_hash": "card-b"},
                        {"type": "Tool", "content_hash": "card-a"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = substrate.deterministic_tool_qualification_receipt(report_path, cards)

    assert receipt["ready"] is True
    assert receipt["tool_card_identity_matches"] is True
    assert receipt["qualified_case_count"] == 1
    assert len(receipt["report_sha256"]) == 64
    changed = {**cards, "search.local": {"replay_checksum": "changed"}}
    assert (
        substrate.deterministic_tool_qualification_receipt(report_path, changed)[
            "ready"
        ]
        is False
    )


def test_qualification_receipt_rejects_dirty_boundary(tmp_path: Path) -> None:
    report_path = tmp_path / "qualification.json"
    report_path.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "passed": True,
                "public_training_rows_written": 1,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
                "artifact_graph": {
                    "artifacts": [{"type": "Tool", "content_hash": "card-a"}]
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = substrate.deterministic_tool_qualification_receipt(
        report_path, {"math.exact": {"replay_checksum": "card-a"}}
    )

    assert receipt["ready"] is False
    assert receipt["boundaries_clean"] is False


def test_assistant_refresh_uses_registry_only_contract() -> None:
    config = json.loads(
        (ROOT / "configs" / "theseus_assistant_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    command = next(
        row
        for row in config["context_refresh_commands"]
        if row["id"] == "deterministic_tool_registry"
    )

    assert command["command"][-1] == "--registry-only"
    assert command["cache"]["outputs"] == [
        "reports/deterministic_tool_registry.json"
    ]
    assert "reports/deterministic_tool_substrate.json" in command["cache"]["inputs"]


def test_registry_payload_preserves_no_cheat_boundaries() -> None:
    payload = substrate.build_registry_payload(
        {"math.exact": {"replay_checksum": "card-a"}},
        trigger_state="GREEN",
        refresh_mode="qualification_bound_runtime_refresh",
        gates=[],
        qualification={"ready": True},
    )

    assert payload["refresh_mode"] == "qualification_bound_runtime_refresh"
    assert payload["qualification_receipt"]["ready"] is True
    assert payload["strict_no_fallback_returns"] is True
    assert payload["public_training_rows_written"] == 0
    assert payload["external_inference_calls"] == 0
    assert payload["fallback_return_count"] == 0
