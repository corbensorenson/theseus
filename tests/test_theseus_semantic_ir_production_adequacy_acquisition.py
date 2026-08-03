from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_semantic_ir_production_adequacy_acquisition as acquisition  # noqa: E402


class FakeClient:
    def __init__(self, selected: dict) -> None:
        self.selected = selected
        self.request_count = 0
        self.response_digests: list[str] = []

    def get(self, path: str, parameters=None):
        self.request_count += 1
        self.response_digests.append(f"digest-{self.request_count}")
        repo = self.selected["repository"]
        if path.endswith(f"/pulls/{self.selected['pull_request']}"):
            return {
                "number": self.selected["pull_request"],
                "title": self.selected["title"],
                "merged_at": self.selected["merged_utc"],
                "merge_commit_sha": self.selected["merge_revision"],
                "html_url": f"https://github.com/{repo}/pull/{self.selected['pull_request']}",
                "changed_files": len(self.selected["selected_source_paths"]),
                "base": {"repo": {"full_name": repo}},
            }
        if path.endswith(f"/commits/{self.selected['merge_revision']}"):
            return {"parents": [{"sha": "a" * 40}]}
        if path.endswith("/files"):
            return [
                {"filename": value, "status": "modified", "additions": 1, "deletions": 1, "changes": 2, "patch": "@@"}
                for value in self.selected["selected_source_paths"]
            ]
        if path.endswith("/license"):
            return {
                "path": self.selected["license_paths"][0],
                "content": base64.b64encode(b"license text " * 20).decode(),
            }
        if path == f"/repos/{repo}":
            return {"license": {"spdx_id": self.selected["declared_license_spdx"]}}
        raise AssertionError(path)


def test_candidate_registry_preflight_is_green_and_zero_call() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["selection_admitted"] is False
    assert set(report["counters"].values()) == {0}


def test_source_only_amendment_is_exactly_one_path_and_pre_model() -> None:
    registry = json.loads(acquisition.DEFAULT_CANDIDATES.read_text())
    config = json.loads(
        (ROOT / registry["adequacy_preregistration"]).read_text()
    )
    assert acquisition.audit_candidate_registry(registry, config) == []
    tampered = json.loads(json.dumps(registry))
    tampered["candidates"][4]["selected_source_paths"] = ["another.py"]
    assert "amendment_scope_not_single_candidate" in acquisition.audit_candidate_registry(
        tampered, config
    )


def test_independent_metadata_row_recomputes_parent_files_and_license() -> None:
    registry = json.loads(acquisition.DEFAULT_CANDIDATES.read_text())
    selected = registry["candidates"][0]
    row, faults = acquisition.fetch_and_audit_candidate(
        FakeClient(selected), selected
    )
    assert faults == []
    assert row["parent_revision"] == "a" * 40
    assert row["target_revision"] == selected["merge_revision"]
    assert row["license"]["verified"] is True


def test_metadata_row_rejects_source_and_license_mismatch() -> None:
    registry = json.loads(acquisition.DEFAULT_CANDIDATES.read_text())
    selected = dict(registry["candidates"][0])
    selected["selected_source_paths"] = ["missing.py"]
    selected["license_paths"] = ["WRONG"]
    client_selected = registry["candidates"][0]
    row, faults = acquisition.fetch_and_audit_candidate(
        FakeClient(client_selected), selected
    )
    assert "selected_source_path_not_changed" in faults
    assert "license_path_mismatch" in faults
    assert row["metadata_faults"] == faults


def test_gh_cli_transport_hashes_json_without_exposing_credentials(monkeypatch) -> None:
    payload = b'{"ok":true}'
    monkeypatch.setattr(
        acquisition.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=payload, stderr=b""
        ),
    )
    client = acquisition.GitHubCliMetadataClient()
    assert client.get("/example", {"page": 1}) == {"ok": True}
    assert client.request_count == 1
    assert len(client.response_digests[0]) == 64
