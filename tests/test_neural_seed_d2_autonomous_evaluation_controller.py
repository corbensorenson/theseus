from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_d2_autonomous_evaluation_controller as controller  # noqa: E402


CONFIG_PATH = ROOT / "configs/neural_seed_d2_autonomous_evaluation_controller.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_has_machine_only_one_shot_authority() -> None:
    value = config()
    controller.validate_config(value)
    authority = value["authority"]
    assert authority["user_or_operator_approval_required"] is False
    assert authority["rerun_consumed_identity_allowed"] is False
    assert authority["project_selected_quality_token_cap_allowed"] is False
    assert authority["physical_boundary_is_negative_evidence"] is False
    assert authority["external_inference_authorized"] is False
    acquisition = value["local_rater_model_acquisition"]
    assert acquisition["automatic_only_when_every_other_gate_is_green"] is True
    assert acquisition["static_weight_download_only"] is True
    assert acquisition["external_inference_calls"] == 0
    assert acquisition["training_rows_written"] == 0


def test_preflight_authorizes_only_when_every_machine_predicate_passes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(controller.utility, "validate_freeze", lambda *_: [])
    report = controller.preflight(
        config(),
        config_path=CONFIG_PATH,
        source_override={
            "commit": "a" * 40,
            "branch": "main",
            "clean_at_generation": True,
            "dirty_path_count": 0,
            "dirty_paths": [],
        },
        process_override=[],
        freeze_override={
            "evaluation_state": "NOT_EVALUATED",
            "consumed_case_count": 0,
        },
        manifest_override={"trigger_state": "GREEN"},
        checkpoint_override={"trigger_state": "GREEN", "hard_gaps": []},
        registry_rows_override=[],
        local_models_override={"trigger_state": "GREEN", "hard_gaps": []},
    )
    assert report["trigger_state"] == "GREEN"
    assert report["execution_authorized"] is True
    assert report["failed_gates"] == []


def test_dirty_source_competing_process_or_consumed_identity_pauses(
    monkeypatch,
) -> None:
    monkeypatch.setattr(controller.utility, "validate_freeze", lambda *_: [])
    frozen = {"evaluation_state": "NOT_EVALUATED", "consumed_case_count": 0}
    freeze_sha = controller.stable_hash(frozen)
    report = controller.preflight(
        config(),
        config_path=CONFIG_PATH,
        source_override={
            "commit": "a" * 40,
            "branch": "main",
            "clean_at_generation": False,
            "dirty_path_count": 1,
            "dirty_paths": [" M source.py"],
        },
        process_override=[{"pid": 7, "command": "theseus_p4s_campaign.py"}],
        freeze_override=frozen,
        manifest_override={"trigger_state": "GREEN"},
        checkpoint_override={"trigger_state": "GREEN", "hard_gaps": []},
        registry_rows_override=[
            {"identity": {"freeze_sha256": freeze_sha}, "event": "reserved"}
        ],
        local_models_override={"trigger_state": "GREEN", "hard_gaps": []},
    )
    assert report["trigger_state"] == "PAUSED"
    assert "source_clean" in report["failed_gates"]
    assert "no_competing_accelerator_job" in report["failed_gates"]
    assert "surface_identity_unconsumed" in report["failed_gates"]


def test_command_plan_has_no_human_gate_or_project_token_cap() -> None:
    plan = controller.command_plan(config())
    step_ids = [step_id for step_id, _command in plan]
    flattened = [token for _step_id, command in plan for token in command]
    assert (
        len([step for step in step_ids if step.startswith("candidate_generation:")])
        == 3
    )
    assert (
        len([step for step in step_ids if step.startswith("final_qualification:")]) == 3
    )
    assert "blind_local_english_scoring" in step_ids
    assert "architecture_verdict" in step_ids
    assert "--human-audit-receipt" not in flattened
    assert "--max-tokens" not in flattened


def test_checkpoint_audit_honors_exact_registered_plan_migration() -> None:
    value = config()
    freeze = json.loads(
        (ROOT / value["freeze"]).read_text(encoding="utf-8")
    )
    report = controller.checkpoint_audit(value, freeze)
    rows = {row["target_id"]: row for row in report["targets"]}
    shared = rows["shared_trunk"]
    assert shared["plan_binding"]["state"] == (
        "ACCEPTED_EXACT_IDENTITY_MIGRATION"
    )
    assert shared["plan_binding"]["migration"]["migration_id"] == (
        "shared_trunk_step11416_uncapped_d2_contract_rebind_v1"
    )
    assert "plan_mismatch" not in shared["hard_gaps"]
    assert "training_incomplete" in shared["hard_gaps"]


def test_unregistered_plan_mismatch_remains_a_hard_gap() -> None:
    binding = controller.checkpoint_plan_binding(
        {
            "plan_sha256": "legacy",
            "checkpoint_sha256": "checkpoint",
            "optimizer_state_sha256": "optimizer",
            "optimizer_steps": 1,
            "optimizer_positions": 1,
        },
        {
            "plan_sha256": "current",
            "plan_identity": {
                "policy": "project_theseus_semantic_training_plan_identity_v3",
                "legacy_migrations": [],
            },
        },
        {"target_id": "shared_trunk"},
    )
    assert binding["state"] == "UNBOUND_PLAN_MISMATCH"
    assert binding["migration"] == {}


def test_dead_lease_is_archived_without_authorizing_rerun(tmp_path: Path) -> None:
    value = config()
    active = tmp_path / "active.json"
    archive = tmp_path / "archive"
    value["active_lease"] = str(active)
    value["lease_archive_directory"] = str(archive)
    active.write_text(
        json.dumps(
            {
                "policy": controller.POLICY,
                "lease_id": "dead",
                "pid": 999_999_999,
                "state": "RUNNING",
            }
        ),
        encoding="utf-8",
    )

    controller.recover_stale_lease(value)

    assert not active.exists()
    recovered = json.loads(next(archive.glob("*.json")).read_text(encoding="utf-8"))
    assert recovered["state"] == "RECOVERED_STALE_TERMINAL_LEASE"
    assert recovered["rerun_authorized"] is False


def test_local_rater_acquisition_is_revision_bound_static_download(
    tmp_path: Path, monkeypatch
) -> None:
    value = config()
    cache = tmp_path / "cache"
    report = tmp_path / "acquisition.json"
    value["local_rater_model_acquisition"].update(
        {"cache_root": str(cache), "report": str(report)}
    )
    rater_config = json.loads(
        (ROOT / "configs/neural_seed_local_english_raters.json").read_text()
    )
    rater_config["model_cache_root"] = str(cache)
    rater_path = tmp_path / "raters.json"
    rater_path.write_text(json.dumps(rater_config), encoding="utf-8")
    monkeypatch.setattr(controller.utility, "LOCAL_RATER_CONFIG", rater_path)
    observed = []

    def download(*, repo_id, revision, cache_dir, local_files_only):
        assert local_files_only is False
        observed.append((repo_id, revision, cache_dir))
        snapshot = cache / repo_id.replace("/", "_") / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    result = controller.acquire_local_rater_models(value)

    assert result["trigger_state"] == "GREEN"
    assert len(observed) == 3
    assert all(len(revision) == 40 for _repo, revision, _cache in observed)
    assert result["external_inference_calls"] == 0
    assert result["training_rows_written"] == 0
    assert result["user_or_operator_approval_required"] is False
