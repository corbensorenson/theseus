from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_fresh_v6_acquisition as acquisition  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v6_evaluator as evaluator  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v6_task_pool as task_pool  # noqa: E402


def test_v6_source_preflight_is_green_and_call_free() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is False
    assert report["candidate_packet_materialized"] is False
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())


def test_v6_sources_are_frozen_licensed_and_disjoint() -> None:
    report = acquisition.materialize.read_json(acquisition.DEFAULT_OUT)
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is True
    assert all(row["trigger_state"] == "GREEN" for row in report["rows"])
    assert [row["repository"] for row in report["rows"]] == [
        "ModelTC/LightLLM",
        "WeblateOrg/translation-finder",
        "durandtibo/feu",
        "statsmodels/statsmodels",
    ]
    assert all(row["pr_base_head_authoritative"] is True for row in report["rows"])
    assert all(row["merge_parent_is_lineage_only"] is True for row in report["rows"])


def test_v6_hidden_evaluators_qualify_before_packet_creation() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["green_task_count"] == 4
    assert report["candidate_packet_materialized"] is False
    assert report["candidate_or_model_exposure_authorized"] is False
    assert report["faults"] == []
    assert all(all(row["checks"].values()) for row in report["rows"])


def test_v6_pool_is_uniform_fresh_balanced_and_call_free() -> None:
    report = task_pool.p2a.read_json(task_pool.DEFAULT_OUT)
    assert report["trigger_state"] == "GREEN"
    assert report["task_count"] == 18
    assert report["sealed_packet_count"] == 18
    assert report["repository_count"] == 18
    assert report["replacement_indices"] == [1, 2, 3, 4]
    assert report["uniformly_rebound_unexposed_indices"] == list(range(5, 19))
    assert all(count == 3 for count in report["stratum_counts"].values())
    assert report["information_flow"]["uniform_compact_protocol_for_all_tasks"] is True
    assert report["consumed_v5_prompt_or_candidate_reused"] is False
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())
    assert all(row["compact_integrity_abi"]["handle_bits"] == 128 for row in report["rows"])
