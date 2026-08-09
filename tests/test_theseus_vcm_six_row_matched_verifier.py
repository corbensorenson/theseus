from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_six_row_matched_verifier as owner  # noqa: E402

CONFIG = ROOT / "configs/theseus_vcm_six_row_matched_verifier.json"


def test_six_row_matched_verifier_preflight_is_sealed_and_call_free() -> None:
    cfg, bound, faults = owner.preflight(CONFIG, verify_store=False)
    assert faults == []
    assert cfg["state"] == owner.STATE
    assert [row["index"] for row in cfg["rows"]] == owner.EXPECTED_INDICES
    assert len(bound["rows"]) == 6
    assert cfg["authority"]["model_or_reference_calls_authorized"] is False
    assert cfg["authority"]["network_enabled_calls_authorized"] is False
    assert cfg["limits"]["minimum_free_bytes_after_execution"] == 10 * 1024**3
    assert cfg["project_selected_output_cap"] is None


def test_task_16_excludes_changed_production_runner_from_hidden_evaluator() -> None:
    cfg, _, faults = owner.preflight(CONFIG, verify_store=False)
    assert faults == []
    row = next(row for row in cfg["rows"] if row["index"] == 16)
    assert row["common_evaluator_paths"] == ["ops-server/dashboard/test_arena_auth.py"]
    assert "ops-server/tools/training_test_runner.py" in row["forbidden_transplant_paths"]
    assert not set(row["common_evaluator_paths"]) & set(row["forbidden_transplant_paths"])


def test_dispositions_preserve_mechanics_and_construct_boundaries() -> None:
    passed = {"returncode": 0, "boundary_hit": False}
    failed = {"returncode": 1, "boundary_hit": False}
    boundary = {"returncode": -15, "boundary_hit": True}
    assert owner.derive_disposition({"parent": failed, "target": passed}, []) == "QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS"
    assert owner.derive_disposition({"parent": passed, "target": passed}, []) == "INCONCLUSIVE_EXPERIMENT_EVALUATOR_NOT_DISCRIMINATIVE"
    assert owner.derive_disposition({"parent": failed, "target": failed}, []) == "INCONCLUSIVE_IMPLEMENTATION_MATCHED_VERIFIER_MECHANICS"
    assert owner.derive_disposition({"parent": boundary, "target": passed}, []) == "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"
    assert owner.derive_disposition({"parent": failed, "target": passed}, ["environment_failed"]) == "INCONCLUSIVE_IMPLEMENTATION_MATCHED_VERIFIER_MECHANICS"


def test_v3_repairs_only_declared_residuals_and_reuses_sealed_rows() -> None:
    cfg, bound, faults = owner.preflight(CONFIG, verify_store=False)
    assert faults == []
    assert cfg["reuse_predecessor_indices"] == [13, 16, 25]
    assert sorted(bound["reuse_indices"]) == [13, 16, 25]
    rows = {row["index"]: row for row in cfg["rows"]}
    assert rows[12]["python_path_roots"] == [".", "src"]
    assert rows[56]["python_path_roots"] == [".", "src"]
    assert "--all-features" in rows[35]["arguments"]
    assert rows[35]["runner_evidence"]["receipt_sha256"] == "6be555d88e8f2321fa4edde4c4575283275dcccea86127b6f61bb5500c45070d"
