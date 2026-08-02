from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_evaluator_seal as seal  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "theseus_d1_evaluator_seal.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_archive(path: Path, root: str, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        for name, value in {"LICENSE": "MIT\n", **files}.items():
            payload = value.encode()
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))


def task() -> dict:
    return {
        "campaign_index": 1,
        "repository": "owner/repository",
        "repository_url": "https://github.com/owner/repository",
        "license_spdx": "MIT",
        "pull_request": 7,
        "pull_request_url": "https://github.com/owner/repository/pull/7",
        "pull_request_title": "Fix normalize for negative values",
        "pull_request_body": "Return the absolute magnitude without changing positive values.",
        "parent_revision": "1" * 40,
        "target_revision": "2" * 40,
        "merged_utc": "2026-08-01T00:00:00Z",
        "selection_digest": "3" * 64,
        "changed_files": [
            {
                "filename": "pkg/core.py",
                "status": "modified",
                "previous_filename": "",
            },
            {
                "filename": "tests/test_core.py",
                "status": "modified",
                "previous_filename": "",
            },
        ],
    }


def materialized(tmp_path: Path) -> dict:
    parent = tmp_path / "parent.tar.gz"
    target = tmp_path / "target.tar.gz"
    tests = (
        "from pkg.core import normalize\n\n"
        "def test_negative():\n    assert normalize(-2) == 2\n\n"
        "def test_positive():\n    assert normalize(2) == 2\n"
    )
    write_archive(
        parent,
        "repository-" + "1" * 40,
        {"pkg/core.py": "def normalize(value):\n    return value\n", "tests/test_core.py": tests},
    )
    write_archive(
        target,
        "repository-" + "2" * 40,
        {"pkg/core.py": "def normalize(value):\n    return abs(value)\n", "tests/test_core.py": tests},
    )
    return {
        "campaign_index": 1,
        "artifacts": [
            {
                "label": "parent",
                "normalized": str(parent),
                "normalized_sha256": seal.sha256_file(parent),
                "source_archive_root": "repository-" + "1" * 40,
            },
            {
                "label": "target",
                "normalized": str(target),
                "normalized_sha256": seal.sha256_file(target),
                "source_archive_root": "repository-" + "2" * 40,
            },
        ],
    }


def test_preflight_waits_without_source_and_opens_no_candidate_route() -> None:
    report = seal.preflight(
        config(),
        config_path=CONFIG_PATH,
        registry_override={},
        materialization_override={},
        sandbox_override={
            "policy": "project_theseus_d1_untrusted_evaluator_sandbox_v1",
            "trigger_state": "GREEN",
            "untrusted_execution_authorized": True,
            "faults": [],
        },
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["execution_authorized"] is False
    assert report["candidate_or_control_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None


def test_single_callable_change_requires_identical_signature_and_outside_ast() -> None:
    change, faults = seal.changed_callable(
        "VALUE = 1\ndef repair(value: int = 0):\n    return value\n",
        "VALUE = 1\ndef repair(value: int = 0):\n    return abs(value)\n",
    )
    assert faults == []
    assert change["qualified_name"] == "repair"
    assert change["signature"] == "def repair(value: int=0)"

    _, faults = seal.changed_callable(
        "VALUE = 1\ndef repair(value):\n    return value\n",
        "VALUE = 2\ndef repair(value):\n    return abs(value)\n",
    )
    assert faults == ["AST_outside_changed_callable_differs"]


def test_test_split_is_deterministic_disjoint_and_nonempty() -> None:
    nodeids = ["tests/test_core.py::test_a", "tests/test_core.py::test_b", "tests/test_core.py::test_c"]
    first = seal.split_test_nodeids("identity", nodeids)
    second = seal.split_test_nodeids("identity", list(reversed(nodeids)))
    assert first == second
    assert first["visible"] and first["hidden"]
    assert set(first["visible"]).isdisjoint(first["hidden"])


def test_qualification_rejects_boundaries_without_model_negative_evidence() -> None:
    receipts = {
        "parent_visible": {"passed": False, "boundary_hit": True},
        "parent_hidden": {"passed": False, "boundary_hit": False},
        "target_visible": {"passed": True, "boundary_hit": False},
        "target_hidden": {"passed": True, "boundary_hit": False},
        "transplant_visible": {"passed": True, "boundary_hit": False},
        "transplant_hidden": {"passed": True, "boundary_hit": False},
    }
    assert seal.qualification_faults(receipts) == [
        "sandbox_boundary_hit:parent_visible"
    ]


def test_task_qualification_seals_blind_visible_hidden_surface(tmp_path: Path) -> None:
    source = task()
    artifacts = materialized(tmp_path)

    def runner(root: Path, nodeids: list[str], _: dict) -> dict:
        candidate = (root / "pkg/core.py").read_text(encoding="utf-8")
        passed = "abs(value)" in candidate
        return {
            "passed": passed,
            "returncode": 0 if passed else 1,
            "boundary_hit": False,
            "nodeids_sha256": seal.stable_hash(nodeids),
        }

    def prompt_counter(prompts: dict[str, str]) -> dict:
        assert set(prompts) == {
            "direct_target_generation",
            "natural_language_plan_control",
            "typed_semantic_ir_treatment",
        }
        return {
            "policy": "project_theseus_exact_local_prompt_addressability_v1",
            "trigger_state": "GREEN",
            "faults": [],
            "prompt_count": 3,
            "minimum_context_residual_tokens": 250000,
            "candidate_or_control_calls": 0,
            "project_selected_quality_token_cap": None,
        }

    row = seal.qualify_task(
        config(),
        source,
        artifacts,
        runner=runner,
        prompt_counter=prompt_counter,
    )
    assert row["qualified"] is True
    assert row["faults"] == []
    assert row["execution_count"] == 6
    manifest = row["task_manifest"]
    evaluator = row["evaluator_manifest"]
    assert manifest["candidate_visible_context"]["project_selected_character_or_token_cap"] is None
    assert manifest["natural_request"].startswith("Fix normalize")
    assert evaluator["blindness"]["test_source_candidate_visible"] is False
    assert evaluator["blindness"]["target_source_candidate_visible"] is False
    assert evaluator["hidden_pytest_nodeids"]
    assert "oracle_callable_source" not in manifest
    assert manifest["candidate_visible_context"]["initial_prompt_addressability"][
        "trigger_state"
    ] == "GREEN"


def test_config_forbids_user_gate_cross_stage_authority_and_quality_cap() -> None:
    value = config()
    assert seal.validate_config(value) == []
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["candidate_or_control_calls_authorized"] is False
    assert value["execution"]["project_selected_quality_token_cap"] is None
    assert value["prompt_addressability"]["project_selected_quality_token_cap"] is None
    assert value["cohort_policy"]["all_pre_model_rejections_retained"] is True


def test_final_pool_persists_content_addressed_task_and_evaluator_bindings(
    tmp_path: Path,
) -> None:
    value = config()
    value["sealed_task_root"] = str(tmp_path / "sealed")
    value["final_task_pool"] = str(tmp_path / "pool.json")
    rows = []
    for index in range(1, 45):
        rows.append(
            {
                "campaign_index": index,
                "source_campaign_index": index,
                "repository": f"owner/repo-{index}",
                "selection_digest": f"{index:064x}",
                "task_manifest": {"policy": "task", "index": index},
                "evaluator_manifest": {"policy": "evaluator", "index": index},
            }
        )
    pool = {
        "policy": seal.POOL_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_OR_CONTROL_GENERATION",
        "task_count": 44,
        "distinct_repository_count": 44,
        "tasks": rows,
        "candidate_or_control_calls": 0,
        "post_candidate_task_replacement_allowed": False,
        "project_selected_quality_token_cap": None,
    }
    persisted = seal.persist_final_pool(value, pool)
    seal.write_json(Path(value["final_task_pool"]), persisted)
    audit = seal.audit_existing_pool(value)
    assert audit["passed"] is True
    assert audit["task_count"] == 44
    first = persisted["tasks"][0]
    assert Path(first["task"]).is_file()
    assert Path(first["evaluator"]).is_file()
    evaluator = json.loads(Path(first["evaluator"]).read_text(encoding="utf-8"))
    assert evaluator["task_manifest_sha256"] == first["task_sha256"]
