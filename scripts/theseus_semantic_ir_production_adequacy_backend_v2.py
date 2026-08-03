#!/usr/bin/env python3
"""Successor adequacy backend retaining infrastructure-invalid partial output."""

from __future__ import annotations

from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_backend as base


class DiagnosticAdequacyLocalMlxChatModel(base.AdequacyLocalMlxChatModel):
    """Keep watchdog output for diagnosis without admitting it as a candidate."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = super().generate(messages)
        watchdog = (
            p2a.mapping(getattr(self, "last_generation_metrics", {})).get(
                "host_safety_wall_time_hit"
            )
            is True
        )
        self.last_invalid_partial_text = text if watchdog else ""
        self.last_invalid_partial_sha256 = (
            p2a.sha256_text(text) if watchdog and text else ""
        )
        return text


def attach_invalid_observation_diagnostic(
    report: dict[str, Any], model: Any
) -> dict[str, Any]:
    """Attach non-scoring partial text after a watchdog and hold the response."""

    metrics = p2a.mapping(report.get("metrics"))
    if metrics.get("host_safety_wall_time_hit") is not True:
        return report
    partial = str(getattr(model, "last_invalid_partial_text", "") or "")
    report["invalid_observation_diagnostic"] = {
        "policy": "project_theseus_invalid_local_output_diagnostic_v1",
        "disposition": "RETAINED_NON_CANDIDATE_INFRASTRUCTURE_DIAGNOSTIC",
        "candidate_admission_allowed": False,
        "hidden_evaluation_allowed": False,
        "partial_output_text": partial,
        "partial_output_chars": len(partial),
        "partial_output_sha256": p2a.sha256_text(partial),
        "raw_prompt_stored": False,
    }
    report["faults"] = sorted(
        set(
            p2a.strings(report.get("faults"))
            + ["instrument_inadequate_host_safety_wall_time"]
        )
    )
    report["trigger_state"] = "RED"
    p2a.mapping(report.get("response"))["answer"] = ""
    return report
