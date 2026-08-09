from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_instrument_builder as owner  # noqa: E402

REPORT = owner.build(ROOT / "configs" / "theseus_vcm_instrument_builder.json")


def test_existing_closures_replay_through_one_row_schema() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["replayed_closure_count"] == 6
    assert REPORT["replayed_managers"] == ["cargo", "npm", "pnpm", "uv"]
    assert len({row["closure_id"] for row in REPORT["rows"]}) == 6
    assert {row["task_index"] for row in REPORT["rows"]} == {3, 7, 14, 30, 36}
    assert all(row["state"] == "QUALIFIED_EXISTING_CLOSURE_REPLAY" for row in REPORT["rows"])
    assert all(row["historical_store_topology_reused_for_forward_execution"] is False for row in REPORT["rows"])


def test_forward_store_and_resource_contract_is_generic_and_bounded() -> None:
    store = REPORT["store_contract"]
    assert store["per_task_duplicate_package_cache_authorized"] is False
    assert store["installed_environments_are_disposable"] is True
    assert set(store["manager_roots"]) == {"npm", "pnpm", "cargo", "uv"}
    assert all("/shared/" in path for path in store["manager_roots"].values())
    assert set(REPORT["resource_preflight"]) == {"state", *owner.RESOURCE_FIELDS}
    assert REPORT["resource_preflight"]["state"] == "STATIC_REPLAY_ONLY_K2_03_PROJECTION_PENDING"


def test_builder_executes_nothing_and_grants_no_downstream_authority() -> None:
    assert REPORT["static_evidence_replay_only"] is True
    assert REPORT["network_or_dependency_execution_performed"] is False
    for key in (
        "repository_runner_executions",
        "parent_target_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_reference_calls",
    ):
        assert REPORT[key] == 0


def test_four_risk_classes_are_prospectively_bound_without_execution() -> None:
    plan = REPORT["risk_canary_plan"]
    assert plan["state"] == "PROSPECTIVELY_SEALED_GENERIC_RISK_EXECUTOR_V2_AFTER_SANDBOX_BINDING_REPAIR"
    assert plan["campaign_id"] == "k2_03_generic_ecosystem_risk_canaries_v2"
    assert [row["risk_class"] for row in plan["rows"]] == [
        "bun_real_lock_install",
        "yarn_real_lock_install",
        "typescript_parent_repository_transpilation",
        "rust_parent_repository_untrusted_compilation",
    ]
    assert all(row["execution_authorized"] is True for row in plan["rows"])
    assert plan["host_free_bytes"] - max(row["resource_projection"]["projected_peak_temporary_bytes"] for row in plan["rows"]) >= plan["host_reserve_bytes"]
    assert REPORT["prior_risk_attempts"] == [{"attempt_id":"k2_03_generic_ecosystem_risk_canaries_v1","state":"INCONCLUSIVE_IMPLEMENTATION","fault":"generic_config_missing_required_sandbox_exec_binding","exception":"KeyError:tools","external_commands_started":0,"network_calls":0,"retained_store_writes":0,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0}]
