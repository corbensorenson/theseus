from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public_release_audit  # noqa: E402


def test_forbidden_generated_roots_are_not_misreported_as_allowlist_warnings() -> None:
    roots = ["configs", "reports", "runtime", "unexpected"]
    allowlist = {"configs"}
    forbidden = ("reports/", "runtime/", "data/private/")

    assert public_release_audit.unregistered_tracked_roots(
        roots,
        allowlist,
        forbidden,
    ) == ["unexpected"]


def test_partially_forbidden_root_still_requires_explicit_registration() -> None:
    roots = ["data"]

    assert public_release_audit.unregistered_tracked_roots(
        roots,
        set(),
        ("data/private/",),
    ) == ["data"]


def test_github_repository_from_common_remote_urls() -> None:
    expected = "corbensorenson/theseus"

    assert (
        public_release_audit.github_repository_from_url(
            "https://github.com/corbensorenson/theseus.git"
        )
        == expected
    )
    assert (
        public_release_audit.github_repository_from_url(
            "git@github.com:corbensorenson/theseus.git"
        )
        == expected
    )
    assert (
        public_release_audit.github_repository_from_url(
            "ssh://git@github.com/corbensorenson/theseus"
        )
        == expected
    )
    assert (
        public_release_audit.github_repository_from_url(
            "https://example.com/corbensorenson/theseus.git"
        )
        == ""
    )


def test_visibility_falls_back_to_public_rest_repository(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[:4] == ["git", "remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="https://github.com/corbensorenson/theseus.git\n",
                stderr="",
            )
        if command[:3] == ["gh", "repo", "view"]:
            raise subprocess.CalledProcessError(1, command)
        assert command == [
            "gh",
            "api",
            "repos/corbensorenson/theseus",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "full_name": "corbensorenson/theseus",
                    "visibility": "public",
                    "private": False,
                    "html_url": "https://github.com/corbensorenson/theseus",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(public_release_audit.subprocess, "run", fake_run)

    visibility = public_release_audit.gh_repo_visibility()

    assert visibility["visibility"] == "PUBLIC"
    assert visibility["nameWithOwner"] == "corbensorenson/theseus"
    assert visibility["source"] == "github_rest_repository"
    assert len(calls) == 3


def test_source_release_selection_is_explicit_and_private_safe() -> None:
    config = {
        "forbidden_tracked_prefixes": ["reports/", "tests/fixtures/hidden/"],
        "forbidden_tracked_paths": ["scripts/private_eval.py"],
        "forbidden_suffixes": [".pt"],
        "source_release": {
            "include_paths": ["README.md"],
            "include_prefixes": ["scripts/", "tests/"],
            "exclude_paths": ["scripts/local_only.py"],
            "exclude_prefixes": ["tests/fixtures/archive/"],
            "exclude_suffixes": [".pdf"],
        },
    }
    selected, excluded = public_release_audit.select_source_release_paths(
        [
            "README.md",
            "data/train.jsonl",
            "reports/result.json",
            "scripts/main.py",
            "scripts/model.pt",
            "scripts/private_eval.py",
            "scripts/local_only.py",
            "tests/manual.pdf",
            "tests/test_main.py",
            "tests/fixtures/hidden/case.py",
            "tests/fixtures/archive/old.py",
        ],
        config,
    )

    assert selected == ["README.md", "scripts/main.py", "tests/test_main.py"]
    assert set(excluded) == {
        "data/train.jsonl",
        "reports/result.json",
        "scripts/model.pt",
        "scripts/private_eval.py",
        "scripts/local_only.py",
        "tests/fixtures/hidden/case.py",
        "tests/fixtures/archive/old.py",
        "tests/manual.pdf",
    }


def test_project_manifest_forbids_private_evaluator_surfaces() -> None:
    config = json.loads(
        (ROOT / "configs/public_release_manifest.json").read_text(encoding="utf-8")
    )
    forbidden_paths = set(config["forbidden_tracked_paths"])
    forbidden_prefixes = set(config["forbidden_tracked_prefixes"])

    assert "scripts/neural_seed_functional_cases.py" in forbidden_paths
    assert "configs/neural_seed_functional_utility.json" in forbidden_paths
    assert (
        "configs/core_evidence_repository_stack_development_evaluator.json"
        in forbidden_paths
    )
    assert "tests/fixtures/core_evidence_tmax_fresh_v1_hidden/" in forbidden_prefixes
    source_excludes = set(config["source_release"]["exclude_prefixes"])
    assert {
        "tests/fixtures/theseus_assistant_p2a_online/",
        "tests/fixtures/theseus_assistant_p3_online/",
        "tests/fixtures/theseus_p4_online/",
        "tests/fixtures/theseus_p4r_online/",
        "tests/fixtures/theseus_p4s_online/",
    }.issubset(source_excludes)


def test_prepare_source_tree_writes_content_bound_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").mkdir()
    (source_root / "README.md").write_text("hello\n", encoding="utf-8")
    (source_root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(public_release_audit, "ROOT", source_root)
    report = {
        "trigger_state": "GREEN",
        "source_release": {
            "policy": "test_source_release_v1",
            "source_commit": "a" * 40,
            "source_dirty": False,
        },
    }
    destination = tmp_path / "release"

    prepared = public_release_audit.prepare_source_tree(
        destination,
        ["README.md", "src/main.py"],
        report,
    )
    manifest = json.loads(
        (destination / "SOURCE_RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert prepared["publishable"] is True
    assert manifest["publishable"] is True
    assert manifest["file_count"] == 2
    assert [row["path"] for row in manifest["files"]] == [
        "README.md",
        "src/main.py",
    ]
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])


def test_prepare_source_tree_refuses_existing_or_in_repo_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(public_release_audit, "ROOT", source_root)
    report = {
        "trigger_state": "RED",
        "source_release": {
            "policy": "test_source_release_v1",
            "source_commit": "a" * 40,
            "source_dirty": True,
        },
    }

    with pytest.raises(ValueError, match="outside"):
        public_release_audit.prepare_source_tree(
            source_root / "release",
            [],
            report,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        public_release_audit.prepare_source_tree(existing, [], report)
