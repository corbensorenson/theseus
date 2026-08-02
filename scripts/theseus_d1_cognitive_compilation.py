#!/usr/bin/env python3
"""Run the exact repaired P4 causal mechanism on D1 with sandboxed verifiers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_evaluator_sandbox as sandbox  # noqa: E402
import theseus_d1_evaluator_seal as seal  # noqa: E402
import theseus_p4v2r2r4_cognitive_compilation as causal  # noqa: E402


POLICY = "project_theseus_d1_cognitive_compilation_run_v1"
RUNTIME_ATTEMPT_NAMESPACE = "d1_fresh_qualification_attempt1"


def audit_instrument(path: Path) -> dict[str, Any]:
    report = causal.audit_instrument(path)
    faults = p2a.strings(report.get("faults"))
    instrument = p2a.read_json(path)
    generation = p2a.mapping(instrument.get("generation_budget"))
    if generation.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    continuity = p2a.mapping(instrument.get("prompt_continuity_repair"))
    if (
        continuity.get("complete_first_call_artifact_visible_to_second_call")
        is not True
        or continuity.get("complete_visible_verifier_feedback_visible_to_second_call")
        is not True
        or continuity.get("same_rule_all_learned_arms") is not True
    ):
        faults.append("complete_prompt_continuity_missing")
    if (
        continuity.get("project_selected_first_artifact_character_cap") is not None
        or continuity.get("project_selected_first_artifact_token_cap") is not None
        or continuity.get("project_selected_verifier_feedback_character_cap")
        is not None
    ):
        faults.append("project_selected_artifact_cap_present")
    report["policy"] = "project_theseus_d1_causal_instrument_adapter_audit_v1"
    report["trigger_state"] = "GREEN" if not faults else "RED"
    report["faults"] = sorted(set(faults))
    report["D1_runtime_attempt_namespace"] = RUNTIME_ATTEMPT_NAMESPACE
    report["causal_mechanism_changed"] = False
    report["candidate_or_evaluator_information_changed"] = False
    return report


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    evaluator_path: Path,
    *,
    session_factory: Any = causal.predecessor.persistent_v2_session,
) -> dict[str, Any]:
    instrument_audit = audit_instrument(instrument_path)
    if instrument_audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY,
            "trigger_state": "RED",
            "faults": ["D1_causal_instrument_audit_red"],
        }
    task = p2a.read_json(task_path)
    evaluator = p2a.read_json(evaluator_path)
    sandbox_config = seal.read_json(
        seal.resolve("configs/theseus_d1_evaluator_sandbox.json")
    )
    original_visible = p2a.run_visible_verifier
    original_namespace = causal.predecessor.RUNTIME_ATTEMPT_NAMESPACE
    original_state = causal.predecessor.INSTRUMENT_STATE
    original_render = causal.causal.render_final_prompt

    def visible(root: Path, task_value: dict[str, Any]) -> dict[str, Any]:
        return run_verifier_on_candidate(
            root,
            p2a.strings(
                p2a.mapping(task_value.get("visible_verifier")).get("command")
            )[2:],
            evaluator,
            sandbox_config,
        )

    overlay = p2a.read_json(instrument_path)
    projected = causal.projected_instrument(overlay)
    projected["runtime_attempt_namespace"] = RUNTIME_ATTEMPT_NAMESPACE
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=ROOT / "runtime" / "control",
        delete=False,
    ) as handle:
        projected_path = Path(handle.name)
        json.dump(projected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    p2a.run_visible_verifier = visible
    try:
        causal.predecessor.RUNTIME_ATTEMPT_NAMESPACE = RUNTIME_ATTEMPT_NAMESPACE
        causal.predecessor.INSTRUMENT_STATE = causal.INSTRUMENT_STATE
        causal.causal.render_final_prompt = causal.render_full_final_prompt
        report = causal.predecessor.run_experiment(
            projected_path,
            task_path,
            session_factory=session_factory,
        )
    finally:
        causal.predecessor.RUNTIME_ATTEMPT_NAMESPACE = original_namespace
        causal.predecessor.INSTRUMENT_STATE = original_state
        causal.causal.render_final_prompt = original_render
        p2a.run_visible_verifier = original_visible
        projected_path.unlink(missing_ok=True)
    report["policy"] = POLICY
    report["D1_runtime_attempt_namespace"] = RUNTIME_ATTEMPT_NAMESPACE
    report["D1_task_sha256"] = p2a.sha256_file(task_path)
    report["D1_evaluator_sha256"] = p2a.sha256_file(evaluator_path)
    report["D1_sandboxed_visible_verifier"] = True
    report["prompt_continuity"] = {
        "complete_first_call_artifact_retained": True,
        "complete_visible_verifier_feedback_retained": True,
        "same_rule_all_learned_arms": True,
        "project_selected_first_artifact_character_cap": None,
        "project_selected_first_artifact_token_cap": None,
        "project_selected_verifier_feedback_character_cap": None,
    }
    report["scope"] = (
        "One sealed fresh D1 task under the exact repaired P4 causal runner; no "
        "serving, training, D2, hosted inference, or book-support authority."
    )
    return report


def run_verifier_on_candidate(
    candidate_root: Path,
    nodeids: list[str],
    evaluator: dict[str, Any],
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="theseus-d1-visible-", dir="/private/tmp"
    ) as temporary:
        root = Path(temporary)
        shutil.copytree(candidate_root, root, dirs_exist_ok=True)
        overlay_target_tests(root, evaluator)
        return seal.run_pytest_sandboxed(root, nodeids, sandbox_config)


def overlay_target_tests(root: Path, evaluator: dict[str, Any]) -> None:
    archive = seal.resolve(str(evaluator.get("target_archive") or ""))
    archive_root = str(evaluator.get("target_archive_root") or "")
    if seal.sha256_file(archive) != evaluator.get("target_archive_sha256"):
        raise ValueError("D1_target_archive_binding_invalid")
    for path in p2a.strings(evaluator.get("test_overlay_paths")):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            seal.archive_text(archive, archive_root, path), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_experiment(
        p2a.resolve(args.instrument),
        p2a.resolve(args.task),
        p2a.resolve(args.evaluator),
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report.get("trigger_state"),
                "faults": report.get("faults"),
                "denominators": report.get("denominators"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
