from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r4_cognitive_compilation as runner  # noqa: E402


INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"


def test_repair_instrument_is_green_zero_call_and_uncapped() -> None:
    report = runner.audit_instrument(INSTRUMENT)
    value = json.loads(INSTRUMENT.read_text(encoding="utf-8"))

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["runtime_attempt_namespace"] == "p4v2r2r4_attempt1"
    assert value["generation_budget"]["project_selected_quality_token_cap"] is None
    repair = value["prompt_continuity_repair"]
    assert repair["project_selected_first_artifact_character_cap"] is None
    assert repair["project_selected_first_artifact_token_cap"] is None
    assert repair["same_rule_all_learned_arms"] is True


def test_full_repair_prompt_retains_prefix_beyond_historical_tail_for_every_arm() -> None:
    prefix = "BEGIN-OF-PROVISIONAL-ARTIFACT\n"
    first = prefix + ("x" * 20000) + "\nEND-OF-PROVISIONAL-ARTIFACT"
    verification = {
        "apply_faults": [],
        "visible_verifier": {
            "returncode": 1,
            "stdout_tail": "failed",
            "stderr_tail": "",
        },
    }
    for original in (
        "direct prompt",
        "plan prompt",
        runner.causal.SEMANTIC_PROMPT_MARKER + "\nsemantic prompt",
    ):
        prompt = runner.render_full_final_prompt(
            original,
            first,
            {"faults": []},
            verification,
            {"O1"},
        )
        assert first in prompt
        assert prefix in prompt


def test_projected_instrument_changes_only_attempt_and_prompt_repair_fields() -> None:
    overlay = json.loads(INSTRUMENT.read_text(encoding="utf-8"))
    projected = runner.projected_instrument(overlay)
    base = json.loads(
        (ROOT / overlay["base_instrument"]).read_text(encoding="utf-8")
    )

    ignored = {
        "runtime_attempt_namespace",
        "state",
        "prompt_continuity_repair",
        "pre_generation_prompt_cap_disposition",
    }
    assert {key: value for key, value in projected.items() if key not in ignored} == {
        key: value for key, value in base.items() if key not in ignored
    }
