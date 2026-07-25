from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rdc_kerc_matched_adequacy as adequacy


def evaluation(exact: int, similarity: float) -> dict:
    return {
        "row_count": 16,
        "bounded_candidate_evaluation": {"active": True, "selection_uses_model_outcomes": False, "selection_uses_answer_text": False},
        "generator_visible_fields": ["prompt"],
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "summary": {"exact_match_count": exact, "mean_target_sequence_similarity": similarity, "nonempty_rate": 1.0, "byte_serialization_valid_rate": 1.0},
    }


def seed_report(seed: int, candidate_exact: int, control_exact: int) -> dict:
    common = {
        "sha256": f"{seed:064x}"[-64:],
        "tensor_count": 12,
        "element_count": 256,
        "payload_bytes": 1024,
    }
    candidate_initialization = {
        "policy": "project_theseus_candidate_common_initialization_v1",
        "role": "reference",
        "seed": seed,
        "exact_alignment": True,
        "common_tensor_manifest": common,
        "architecture_specific_tensor_manifest": {"tensor_count": 3},
        "architecture_specific_tensors_unchanged": True,
    }
    control_initialization = {
        **candidate_initialization,
        "role": "aligned",
        "architecture_specific_tensor_manifest": {"tensor_count": 2},
    }
    return {
        "policy": "project_theseus_moecot_language_arm_training_plan_v1",
        "trigger_state": "GREEN",
        "candidate_canary_lease": {"candidate_id": "rdc_kerc_adequacy", "selected_seed": seed, "seed_execution_mode": "single_bound_seed", "targets": ["english_kerc", "english_surface_control"]},
        "candidate_canary_resource_receipt": {"passed": True, "faults": []},
        "host_resource_safety_receipt": {
            "policy": "project_theseus_host_resource_safety_v1",
            "passed": True,
            "terminated_by_guard": False,
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "templates_renderers_routers_tools_credit": 0,
        "results": [
            {"target_id": "english_kerc", "candidate_seed": seed, "candidate_initialization": candidate_initialization, "parameter_count": 72_534_757, "optimizer_steps": 128, "phases": {"kernel_english": {"device_step_seconds_total": 300.0}}, "evaluation": evaluation(candidate_exact, 0.5)},
            {"target_id": "english_surface_control", "candidate_seed": seed, "candidate_initialization": control_initialization, "parameter_count": 72_534_538, "optimizer_steps": 128, "phases": {"kernel_english": {"device_step_seconds_total": 100.0}}, "evaluation": evaluation(control_exact, 0.3)},
        ],
    }


def test_seed_summary_requires_bound_seed_and_direct_behavior() -> None:
    config = adequacy.load_config()
    row = adequacy.summarize_seed(config, 20260722, seed_report(20260722, 4, 1))
    assert row["exact_match_gain_count"] == 3
    assert row["training_wall_time_ratio"] == 3.0


def test_three_seed_aggregate_adopts_only_repeated_gain() -> None:
    config = adequacy.load_config()
    work = ROOT / "runtime" / "test-rdc-kerc-aggregate"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    reports = []
    for seed in config["seeds"]:
        path = work / f"{seed}.json"
        path.write_text("{}", encoding="utf-8")
        reports.append((seed, path, seed_report(seed, 4, 1)))
    try:
        result = adequacy.aggregate(config, reports)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    assert result["disposition"] == "ADOPT_RDC_KERC_FIRST_CAMPAIGN"
    assert result["scientific_falsification_claimed"] is False


def test_empty_kerc_generation_below_known_k5_budget_is_inconclusive() -> None:
    config = adequacy.load_config()
    reports = []
    for seed in config["seeds"]:
        report = seed_report(seed, 0, 0)
        candidate_summary = report["results"][0]["evaluation"]["summary"]
        candidate_summary["nonempty_rate"] = 0.0
        candidate_summary["byte_serialization_valid_rate"] = 0.0
        reports.append((seed, ROOT / "roadmap.md", report))
    result = adequacy.aggregate(config, reports)
    assert result["disposition"] == "INCONCLUSIVE_EXPERIMENT_REPAIR_K5_BEFORE_K8"
    assert result["adequacy_assessment"]["matched_budget_below_known_k5_profile"] is True


def test_isolated_assembly_treats_requested_steps_as_canary_completion(
    tmp_path: Path, monkeypatch
) -> None:
    config = adequacy.load_config()
    seed = int(config["seeds"][0])
    pair = seed_report(seed, 0, 0)
    components = []
    for target_id, result in zip(config["targets"], pair["results"]):
        component = copy.deepcopy(pair)
        component["results"] = [copy.deepcopy(result)]
        component["executed_targets"] = [target_id]
        components.append((target_id, component))
    scratch = tmp_path / "scratch"
    receipt_path = scratch / "english_kerc" / "training_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {"candidate_initialization": pair["results"][0]["candidate_initialization"]}
        ),
        encoding="utf-8",
    )
    lease = copy.deepcopy(pair["candidate_canary_lease"])
    lease.update(
        {
            "authorized": True,
            "lease_digest": "a" * 64,
            "budgets": {"max_peak_memory_mib": 14336},
        }
    )
    monkeypatch.setattr(
        adequacy.pretraining_candidate_canary,
        "candidate_lease",
        lambda **_kwargs: lease,
    )
    output = tmp_path / "assembled.json"
    adequacy.assemble_seed_report(
        config,
        seed,
        report_path=output,
        scratch=scratch,
        candidate_contract={},
        components=components,
    )
    assembled = json.loads(output.read_text(encoding="utf-8"))
    assert assembled["trigger_state"] == "GREEN"
    assert assembled["bounded_candidate_steps_complete"] is True
    assert assembled["all_requested_targets_complete"] is False
