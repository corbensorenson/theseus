#!/usr/bin/env python3
"""Aggregate the complete-artifact P4 repair campaign independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_p4v2r2r2_disposition as predecessor
import theseus_p4v2r2r3_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4v2r2r3_terminal_disposition_v1"
INSTRUMENT = (
    ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"
)
PROGRESS = ROOT / "reports" / "theseus_p4v2r2r3_attempt1_campaign_progress.json"
OUT = ROOT / "reports" / "theseus_p4v2r2r3_attempt1_terminal_disposition.json"
ZERO_CALL_DISPOSITION = (
    ROOT
    / "reports"
    / "theseus_p4v2r2r2_attempt2_pre_generation_information_cap_disposition.json"
)
EFFECTIVE_RESEAL_COMMIT = "590ceca0dd0b801ed9be64822d093e2cb691e39c"


def build_report() -> dict[str, Any]:
    rebound = {
        "POLICY": POLICY,
        "INSTRUMENT": INSTRUMENT,
        "PROGRESS": PROGRESS,
        "OUT": OUT,
        "PRE_GENERATION_FAILURE": ZERO_CALL_DISPOSITION,
        "EFFECTIVE_RESEAL_COMMIT": EFFECTIVE_RESEAL_COMMIT,
        "campaign": campaign,
    }
    original = {name: getattr(predecessor, name) for name in rebound}
    try:
        for name, value in rebound.items():
            setattr(predecessor, name, value)
        report = predecessor.build_report()
    finally:
        for name, value in original.items():
            setattr(predecessor, name, value)
    report["policy"] = POLICY
    report["source_identities"]["zero_call_prompt_cap_disposition"] = (
        predecessor.base.source_identity(ZERO_CALL_DISPOSITION)
    )
    report["prompt_continuity"] = {
        "complete_first_call_artifact_retained": True,
        "same_rule_all_learned_arms": True,
        "project_selected_first_artifact_character_cap": None,
        "project_selected_first_artifact_token_cap": None,
        "exact_repair_prompt_token_count_required": True,
        "non_addressable_prompt_is_negative_evidence": False,
    }
    report["maximum_inference"] = (
        "This report can decide only whether the exact frozen TMax plus Semantic-IR "
        "v2r2 implementation survives the unchanged ten-task P4 development surface "
        "with complete first-call artifact continuity. It cannot establish or "
        "falsify cognitive compilation generally, qualify serving or training, "
        "decide D2, or promote ASI Stack support."
    )
    return report


def classify_status(**kwargs: Any) -> str:
    return predecessor.classify_status(**kwargs)


def next_stage(status: str) -> dict[str, Any]:
    return predecessor.next_stage(status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(OUT))
    args = parser.parse_args()
    report = build_report()
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "scientific_status": report["scientific_status"],
                "complete_tasks": report["denominators"]["tasks"],
                "learned_model_calls": report["denominators"][
                    "learned_model_calls"
                ],
                "faults": report["faults"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
