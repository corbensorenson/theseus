from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_compact_protocol_audit as audit  # noqa: E402


def test_compact_protocol_audit_is_green_and_call_free() -> None:
    report = p2a.read_json(audit.DEFAULT_OUT)
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "COMPACT_PROTOCOL_MECHANICS_GREEN_FRESH_DENOMINATOR_REQUIRED"
    assert report["green_task_count"] == 18
    assert report["consumed_indices"] == [1, 2, 3, 4]
    assert report["fresh_replacement_required_for_indices"] == [1, 2, 3, 4]
    assert report["unexposed_indices_eligible_for_uniform_protocol_rebind"] == list(range(5, 19))
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())
    assert all(row["inventory_complete"] is True for row in report["rows"])
    assert all(row["inventory_truncated"] is False for row in report["rows"])
    assert all(row["compact_prompt_utf8_bytes"] < row["old_prompt_utf8_bytes"] for row in report["rows"])
    assert all(row["full_node_digest_candidate_visible"] is False for row in report["rows"])
