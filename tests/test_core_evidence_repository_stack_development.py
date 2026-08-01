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


def test_campaign_config_is_exact_fixed_tmax_worker_v3_contract() -> None:
    worker = json.loads(campaign.DEFAULT_WORKER_CONFIG.read_text(encoding="utf-8"))
    assert worker["model"]["repo_id"] == "mlx-community/Tmax-9B-MLX-8bit"
    assert (
        worker["model"]["revision"]
        == "33812d6cf04f88856f25eb828de4f3144a194560"
    )
    assert worker["budgets"]["maximum_patch_bytes"] == 262144
    assert campaign.DEFAULT_WORKER_CONFIG.name == (
        "core_evidence_tmax_9b_worker_control_v3.json"
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


def test_l0_arm_names_map_to_existing_target_blind_adapter_variants() -> None:
    assert campaign.canonical_arm_variant_id("direct_fixed_worker") == "direct"
    assert campaign.canonical_arm_variant_id("full_theseus") == "full_stack"
    assert campaign.canonical_arm_variant_id("planning_none") == "planning_none"


def test_l0_manifest_selects_only_reusable_development_rows() -> None:
    tasks = [
        {
            "opaque_task_id": "eligible",
            "partition": "development",
            "denominator": "L0_DEVELOPMENT",
        },
        {
            "opaque_task_id": "wrong-denominator",
            "partition": "development",
            "denominator": "D1_DEVELOPMENT",
        },
        {
            "opaque_task_id": "wrong-partition",
            "partition": "qualification",
            "denominator": "L0_DEVELOPMENT",
        },
    ]
    selected = campaign.select_task_rows(
        tasks,
        manifest_policy="project_theseus_l0_real_work_task_manifest_v1",
        task_index=0,
        task_limit=10,
    )
    assert [row["opaque_task_id"] for row in selected] == ["eligible"]


def test_static_model_audit_rejects_arm_identity_or_config_drift() -> None:
    worker = json.loads(campaign.DEFAULT_WORKER_CONFIG.read_text(encoding="utf-8"))
    expected_sha = campaign.sha256_file(campaign.DEFAULT_WORKER_CONFIG)

    def row(arm: str, revision: str, config_sha: str = expected_sha) -> dict:
        return {
            "arm_id": arm,
            "dispatch_allowed": True,
            "candidate": {
                "model_identity": {
                    "repo_id": worker["model"]["repo_id"],
                    "revision": revision,
                    "runtime": "mlx_lm_local_metal",
                },
                "candidate_seal": {"config_sha256": config_sha},
            },
        }

    valid = campaign.fixed_model_identity_audit(
        [
            row("direct_fixed_worker", worker["model"]["revision"]),
            row("full_theseus", worker["model"]["revision"]),
        ],
        worker_config=worker,
        worker_config_sha256=expected_sha,
    )
    assert valid["passed"] is True

    drifted = campaign.fixed_model_identity_audit(
        [
            row("direct_fixed_worker", worker["model"]["revision"]),
            row("full_theseus", "different-revision"),
        ],
        worker_config=worker,
        worker_config_sha256=expected_sha,
    )
    assert drifted["passed"] is False

    timeout_receipt = campaign.fixed_model_identity_audit(
        [
            {
                "arm_id": "full_theseus",
                "dispatch_allowed": True,
                "candidate": None,
                "run_failure": {
                    "model_identity": {
                        "repo_id": worker["model"]["repo_id"],
                        "revision": worker["model"]["revision"],
                        "runtime": "mlx_lm_local_metal",
                    },
                    "worker_config_sha256": expected_sha,
                },
            }
        ],
        worker_config=worker,
        worker_config_sha256=expected_sha,
    )
    assert timeout_receipt["passed"] is True


def test_event_metrics_are_scoped_to_appended_arm_segment(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({
            "action": "read",
            "generated_tokens": 10,
            "prompt_tokens": 100,
            "uncached_prompt_tokens": 20,
            "verification_count": 0,
            "generation_wall_ms": 50,
            "ok": True,
        })
        + "\n",
        encoding="utf-8",
    )
    offset = events.stat().st_size
    with events.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({
                "action": "plan",
                "generated_tokens": 7,
                "prompt_tokens": 70,
                "uncached_prompt_tokens": 12,
                "verification_count": 0,
                "generation_wall_ms": 30,
                "ok": False,
            })
            + "\n"
        )
        handle.write(
            json.dumps({
                "action": "edit",
                "generated_tokens": 11,
                "prompt_tokens": 90,
                "uncached_prompt_tokens": 15,
                "verification_count": 1,
                "generation_wall_ms": 40,
                "ok": True,
            })
            + "\n"
        )

    metrics = campaign.development.appended_event_metrics(events, offset)

    assert metrics["model_calls"] == 2
    assert metrics["generated_tokens"] == 18
    assert metrics["tool_calls"] == 1
    assert metrics["verification_count"] == 1
    assert metrics["failed_actions"] == 1
