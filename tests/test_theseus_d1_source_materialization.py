from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_d1_source_materialization as materialization  # noqa: E402


CONFIG_PATH = ROOT / "configs/theseus_d1_source_materialization.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def registry() -> dict:
    tasks = []
    for index in range(1, 45):
        tasks.append(
            {
                "campaign_index": index,
                "repository": f"owner/repo-{index}",
                "parent_revision": f"{index:040x}",
                "target_revision": f"{index + 100:040x}",
                "selection_digest": f"{index + 200:064x}",
                "changed_paths": ["pkg/core.py", "tests/test_core.py", "old.py", "new.py"],
                "changed_files": [
                    {"filename": "pkg/core.py", "status": "modified", "previous_filename": ""},
                    {"filename": "tests/test_core.py", "status": "added", "previous_filename": ""},
                    {"filename": "old.py", "status": "removed", "previous_filename": ""},
                    {"filename": "new.py", "status": "renamed", "previous_filename": "before.py"},
                ],
            }
        )
    return {
        "policy": "project_theseus_d1_online_source_registry_v1",
        "state": "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_ORACLE_EVALUATOR_OR_CANDIDATE_EXECUTION",
        "campaign_id": "theseus_d1_cognitive_compilation_fresh_qualification_v1",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "task_count": 44,
        "tasks": tasks,
        "boundaries": {
            "archive_fetches": 0,
            "parent_target_oracle_or_evaluator_executions": 0,
            "candidate_or_control_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
        },
        "replacement_after_membership_freeze": False,
    }


def write_archive(path: Path, root: str, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        for relative in ["LICENSE", *members]:
            payload = b"fixture\n"
            member = tarfile.TarInfo(f"{root}/{relative}")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))


def test_config_binds_sanitizer_and_forbids_cross_stage_authority() -> None:
    value = config()
    assert materialization.validate_config(value) == []
    assert materialization.audit_bindings(value)["passed"] is True
    authority = value["authority"]
    assert authority["user_or_operator_approval_required"] is False
    assert authority["postfreeze_task_replacement_allowed"] is False
    assert authority["candidate_or_control_calls_authorized"] is False
    assert authority["external_inference_authorized"] is False


def test_missing_registry_pauses_without_network_authority() -> None:
    report = materialization.preflight(
        config(), config_path=CONFIG_PATH, registry_override={}, lease_exists_override=False
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["execution_authorized"] is False
    assert report["network_fetches"] == 0


def test_frozen_registry_opens_only_archive_fetch() -> None:
    report = materialization.preflight(
        config(), config_path=CONFIG_PATH, registry_override=registry(), lease_exists_override=False
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "D1_SOURCE_ARCHIVE_FETCH_READY"
    assert report["execution_authorized"] is True
    assert report["parent_target_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_changed_member_requirements_follow_file_status() -> None:
    task = registry()["tasks"][0]
    assert materialization.required_changed_members(task, "parent") == [
        "before.py",
        "old.py",
        "pkg/core.py",
    ]
    assert materialization.required_changed_members(task, "target") == [
        "new.py",
        "pkg/core.py",
        "tests/test_core.py",
    ]


def test_artifact_audit_requires_revision_root_license_and_changed_members(tmp_path: Path) -> None:
    task = registry()["tasks"][0]
    plan = materialization.artifact_plan(task, config())[0]
    root = f"repo-1-{plan['revision']}"
    archive = tmp_path / "parent.tar.gz"
    write_archive(archive, root, plan["required_relative_members"])
    sanitization = {
        "trigger_state": "GREEN",
        "source_archive_root": root,
    }
    assert materialization.audit_materialized_artifact(task, plan, archive, sanitization) == []
    missing = tmp_path / "missing.tar.gz"
    write_archive(missing, root, [])
    assert any(
        fault.startswith("required_changed_members_missing:")
        for fault in materialization.audit_materialized_artifact(task, plan, missing, sanitization)
    )


def test_materialize_uses_exact_urls_and_keeps_all_tasks_on_failure(tmp_path: Path) -> None:
    value = config()
    value["archive_root"] = str(tmp_path / "archives")
    value["sanitization_report_root"] = str(tmp_path / "sanitization")
    value["report"] = str(tmp_path / "report.json")
    calls: list[str] = []

    def downloader(url: str, destination: Path) -> None:
        calls.append(url)
        revision = url.rsplit("/", 1)[-1]
        root = f"repo-{destination.name.split('_')[0]}-{revision}"
        label = "parent" if "_parent_" in destination.name else "target"
        task = registry()["tasks"][int(destination.name.split("_", 1)[0]) - 1]
        write_archive(
            destination,
            root,
            materialization.required_changed_members(task, label),
        )

    report = materialization.materialize(value, registry(), downloader=downloader)
    assert report["trigger_state"] == "GREEN"
    assert report["task_count"] == 44
    assert report["archive_artifacts"] == 88
    assert report["network_fetches"] == 88
    assert len(calls) == 88
    assert calls[0].startswith("https://codeload.github.com/owner/repo-1/tar.gz/")
    assert report["candidate_or_control_calls"] == 0
    assert report["postfreeze_task_replacement_allowed"] is False
