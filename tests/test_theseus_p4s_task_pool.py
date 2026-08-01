from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4s_task_pool as pool  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p4s_pool_builder_binds_frozen_inputs_and_zero_call_registry() -> None:
    registry = read(pool.SOURCE_REGISTRY)

    assert pool.audit_registry(registry) == []
    assert pool.p2a.sha256_file(pool.SOURCE_REGISTRY) == pool.EXPECTED_SOURCE_REGISTRY_SHA256
    assert pool.p2a.sha256_file(pool.SOURCE_FETCH) == pool.EXPECTED_SOURCE_FETCH_SHA256
    assert pool.p2a.sha256_file(pool.INSTRUMENT) == pool.EXPECTED_INSTRUMENT_SHA256
    assert (
        pool.p2a.sha256_file(pool.INSTRUMENT_AUDIT)
        == pool.EXPECTED_INSTRUMENT_AUDIT_SHA256
    )


def test_p4s_pool_builder_rejects_open_candidate_boundary() -> None:
    registry = read(pool.SOURCE_REGISTRY)
    opened = copy.deepcopy(registry)
    opened["boundaries"]["candidate_generation_opened"] = True

    assert "candidate_generation_already_opened" in pool.audit_registry(opened)


def test_p4s_oracle_uses_each_units_bound_path() -> None:
    registry = read(pool.SOURCE_REGISTRY)
    for source in registry["tasks"]:
        assert all(
            unit["path"] in source["allowed_effect_paths"]
            for unit in source["oracle_units"]
        )
    multi = [row for row in registry["tasks"] if len(row["oracle_units"]) > 1]
    assert {row["case"] for row in multi} == {"httpx", "httpcore", "isort"}


def test_p4s_pool_completion_policy_has_no_project_quality_cap() -> None:
    instrument = read(pool.INSTRUMENT)
    generation = instrument["generation_budget"]

    assert generation["project_selected_quality_token_cap"] is None
    assert generation["normal_completion"] == ["parser_complete", "model_eos"]
    assert generation["ceiling_hit_invalidates_observation"] is True
    assert generation["ceiling_hit_counts_as_model_or_mechanism_failure"] is False
