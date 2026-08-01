from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a_disposition as disposition  # noqa: E402


def test_consumed_tmax_p2a_is_scoped_as_inconclusive_implementation() -> None:
    report = disposition.build_disposition(
        ROOT / "reports" / "theseus_assistant_p2a_typing_extensions_677_run.json",
        ROOT / "reports" / "theseus_assistant_p2a_typing_extensions_677_evaluation.json",
        ROOT / "configs" / "theseus_p2a_task_typing_extensions_677.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["terminal_disposition"] == "P2A_TERMINAL_INCONCLUSIVE_INSTRUMENT_AND_TASK_NAMESPACE"
    assert report["recomputed_checks"]["matched_pair_ready"] is True
    assert report["recomputed_checks"]["one_persistent_model_load"] is True
    assert report["recomputed_checks"]["parseable_candidates"] == 0
    assert report["recomputed_checks"]["correctness_evaluated_candidates"] == 0
    assert report["observed_residuals"]["task_path_namespace"]["repo_relative_path_ambiguity_present"] is True
    assert report["observed_residuals"]["integrated_runtime_red_is_separate_from_route_integrity"] is True
    assert "assistant_product_trace_exercises_required_surfaces" in report["observed_residuals"]["integrated_failed_gate_names"]
    assert report["consumption"]["eligible_for_exact_rerun"] is False
    assert report["counters"]["external_inference_calls"] == 0


def test_namespace_audit_accepts_one_canonical_repo_relative_path() -> None:
    report = disposition.namespace_audit({
        "natural_request": "Modify only src/example.py.",
        "allowed_effect_paths": ["src/example.py"],
    })

    assert report["repo_relative_path_ambiguity_present"] is False
    assert report["request_paths_rejected_but_allowed_with_archive_prefix"] == []
