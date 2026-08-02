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
    assert repair["project_selected_verifier_feedback_character_cap"] is None
    assert repair["complete_visible_verifier_feedback_visible_to_second_call"] is True
    assert repair["same_rule_all_learned_arms"] is True
    runner_path = ROOT / value["candidate_runner"]
    assert runner.p2a.sha256_file(runner_path) == value["candidate_runner_sha256"]


def test_full_repair_prompt_retains_prefix_beyond_historical_tail_for_every_arm() -> None:
    prefix = "BEGIN-OF-PROVISIONAL-ARTIFACT\n"
    first = prefix + ("x" * 20000) + "\nEND-OF-PROVISIONAL-ARTIFACT"
    verifier_prefix = "BEGIN-OF-VERIFIER-FEEDBACK\n"
    verifier_output = verifier_prefix + ("v" * 5000) + "\nEND-OF-VERIFIER-FEEDBACK"
    verification = {
        "apply_faults": [],
        "visible_verifier": {
            "returncode": 1,
            "stdout_tail": verifier_output,
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
        assert json.dumps(verifier_output) in prompt
        assert "BEGIN-OF-VERIFIER-FEEDBACK" in prompt
        assert "END-OF-VERIFIER-FEEDBACK" in prompt


def test_visible_verifier_retains_complete_output(tmp_path: Path) -> None:
    marker = "BEGIN" + ("z" * 5000) + "END"
    report = runner.run_complete_visible_verifier(
        tmp_path,
        {
            "visible_verifier": {
                "command": [sys.executable, "-c", f"print({marker!r})"],
                "timeout_seconds": 10,
            }
        },
    )
    assert report["passed"] is True
    assert marker in report["stdout_tail"]
    assert report["stdout_complete"] is True
    assert report["project_selected_character_cap"] is None


def test_projected_instrument_changes_only_attempt_and_prompt_repair_fields() -> None:
    overlay = json.loads(INSTRUMENT.read_text(encoding="utf-8"))
    projected = runner.projected_instrument(overlay)
    base = json.loads(
        (ROOT / overlay["base_instrument"]).read_text(encoding="utf-8")
    )

    ignored = {
        "base_local_instrument",
        "base_local_instrument_sha256",
        "runtime_attempt_namespace",
        "state",
        "prompt_continuity_repair",
        "pre_generation_prompt_cap_disposition",
    }
    assert {key: value for key, value in projected.items() if key not in ignored} == {
        key: value for key, value in base.items() if key not in ignored
    }
    assert projected["base_local_instrument"] == overlay["release_local_instrument"]
    assert (
        projected["base_local_instrument_sha256"]
        == overlay["release_local_instrument_sha256"]
    )


def test_pre_inference_custody_is_durable_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    call_directory = tmp_path / "call-starts"
    monkeypatch.setattr(runner, "CALL_START_DIRECTORY", call_directory)
    runtime_report = tmp_path / "runtime-report.json"
    runtime_report.write_text('{"trigger_state":"GREEN"}\n', encoding="utf-8")

    def fake_runtime_call(*args, **kwargs):
        return {
            "assistant_text": "answer",
            "runtime_report": {"trigger_state": "GREEN"},
            "receipt": {
                "report_path": str(runtime_report),
                "report_sha256": runner.p2a.sha256_file(runtime_report),
                "candidate_output_sha256": runner.p2a.sha256_text("answer"),
                "runtime_trigger_state": "GREEN",
                "route_integrity_ready": True,
            },
        }

    governed = runner.bind_pre_inference_custody(fake_runtime_call)
    governed(
        "direct",
        "task_typed_semantic_ir_treatment_p4v2r2r4_attempt1",
        1,
        "prompt",
        262144,
        "configs/theseus_assistant_runtime.json",
    )
    path = runner.call_start_path(
        "task_typed_semantic_ir_treatment_p4v2r2r4_attempt1", 1
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["state"] == "RETURNED_WITH_RUNTIME_RECEIPT"
    assert receipt["runtime_report_binding_valid"] is True
    assert receipt["prompt_retained"] is False
    assert receipt["candidate_output_retained"] is False

    try:
        governed(
            "direct",
            "task_typed_semantic_ir_treatment_p4v2r2r4_attempt1",
            1,
            "prompt",
            262144,
            "configs/theseus_assistant_runtime.json",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate inference call was not rejected")


def test_semantic_prompt_advertises_every_parser_reachable_operation() -> None:
    prompt = runner.causal.render_arm_prompt(
        runner.p4.SEMANTIC, {"natural_request": "task"}, "context", {}
    )
    assert "OP <REPLACE|INSERT_BEFORE|INSERT_AFTER>" in prompt
