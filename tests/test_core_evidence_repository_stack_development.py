from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "core_evidence_repository_stack_development.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stack_development", SCRIPT)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_compact_receipt_preserves_pre_generation_decision() -> None:
    packet = {
        "policy": "adapter",
        "variant_id": "vcm_stale",
        "dispatch_allowed": False,
        "typed_faults": ["CONTEXT_REQUIRED_STALE"],
        "audit": {"target_patch_consulted": False},
        "authority_receipt": {},
        "vcm_receipt": {},
        "compiled_plan": {},
        "route_receipt": {"selected_route": "conservative_hold"},
        "procedural_reuse_receipt": {},
        "counters": {"external_inference_calls": 0},
    }
    receipt = campaign.compact_adapter_receipt(packet)
    assert receipt["dispatch_allowed"] is False
    assert receipt["typed_faults"] == ["CONTEXT_REQUIRED_STALE"]
    assert receipt["audit"]["target_patch_consulted"] is False


def test_default_matrix_contains_quality_and_denial_controls() -> None:
    assert {
        "full_stack",
        "direct",
        "planning_none",
        "vcm_information_matched_untyped",
        "vcm_information_matched_shuffled",
        "vcm_stale",
        "vcm_omission",
        "conservative_hold",
    } == set(campaign.DEFAULT_VARIANTS)


def test_distinct_variants_cannot_share_worker_input_identity() -> None:
    rows = [
        {
            "variant_id": "full_stack",
            "dispatch_allowed": True,
            "candidate": {"worker_input_sha256": "same"},
        },
        {
            "variant_id": "direct",
            "dispatch_allowed": True,
            "candidate": {"worker_input_sha256": "same"},
        },
    ]
    assert campaign.worker_input_aliases(rows) == [{
        "worker_input_sha256": "same",
        "variant_ids": ["direct", "full_stack"],
    }]


def test_distinct_worker_inputs_pass_alias_audit() -> None:
    rows = [
        {
            "variant_id": "full_stack",
            "dispatch_allowed": True,
            "candidate": {"worker_input_sha256": "full"},
        },
        {
            "variant_id": "direct",
            "dispatch_allowed": True,
            "candidate": {"worker_input_sha256": "direct"},
        },
    ]
    assert campaign.worker_input_aliases(rows) == []


def test_campaign_config_is_exact_tmax_with_worker_v4_development_contract() -> None:
    worker = json.loads(campaign.DEFAULT_WORKER_CONFIG.read_text(encoding="utf-8"))
    assert worker["model"]["repo_id"] == "mlx-community/Tmax-9B-MLX-8bit"
    assert (
        worker["model"]["revision"]
        == "33812d6cf04f88856f25eb828de4f3144a194560"
    )
    assert worker["budgets"]["maximum_insert_characters"] == 4000
    assert campaign.DEFAULT_WORKER_CONFIG.name == (
        "core_evidence_tmax_9b_worker_control_v4_development.json"
    )


def test_consumed_development_manifest_is_not_e2_or_public_calibration() -> None:
    manifest = json.loads(
        campaign.DEFAULT_TASK_MANIFEST.read_text(encoding="utf-8")
    )
    assert manifest["policy"] == (
        "project_theseus_repository_stack_consumed_development_public_v1"
    )
    assert manifest["boundaries"]["E2_heldout_cases_consumed"] == 0
    assert manifest["boundaries"]["public_calibration_cases_consumed"] == 0
    assert all(
        task["partition"] == "development"
        and task["denominator"] == "D1_DEVELOPMENT"
        for task in manifest["tasks"]
    )
