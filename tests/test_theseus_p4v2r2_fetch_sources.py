from __future__ import annotations

import copy
import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2_fetch_sources as fetcher  # noqa: E402


def registry() -> dict:
    return json.loads(fetcher.REGISTRY.read_text(encoding="utf-8"))


def test_p4v2r2_fetcher_binds_exact_frozen_registry() -> None:
    source_registry = registry()

    assert fetcher.sha256_file(fetcher.REGISTRY) == fetcher.EXPECTED_REGISTRY_SHA256
    assert fetcher.SOURCE_SELECTION_COMMIT == (
        "8cebe4a65bb03965e9f62efa8249f2f9ddb8fc08"
    )
    assert fetcher.audit_registry(source_registry) == []


def test_p4v2r2_fetch_plan_is_exact_complete_and_registry_derived() -> None:
    rows = registry()["tasks"]
    plans = [fetcher.artifact_plan(row) for row in rows]

    assert len(plans) == 10
    assert sum(map(len, plans)) == 20
    for row, plan in zip(rows, plans):
        assert [artifact["label"] for artifact in plan] == ["parent", "target"]
        for artifact in plan:
            label = artifact["label"]
            assert artifact["revision"] == row[f"{label}_revision"]
            assert artifact["url"] == (
                f"https://codeload.github.com/{row['repository']}/tar.gz/"
                f"{row[f'{label}_revision']}"
            )
            assert artifact["upstream_name"].endswith(
                f"_{label}_upstream.tar.gz"
            )
            assert artifact["normalized_name"].endswith(f"_{label}.tar.gz")


def test_p4v2r2_fetcher_rejects_opened_overlap_or_nonzero_boundaries() -> None:
    source_registry = registry()
    opened = copy.deepcopy(source_registry)
    opened["boundaries"]["candidate_generation_opened"] = True
    nonzero = copy.deepcopy(source_registry)
    nonzero["boundaries"]["local_model_calls"] = 1
    overlap = copy.deepcopy(source_registry)
    overlap["source_disjoint_from_repositories"].append(
        overlap["tasks"][0]["repository"]
    )

    assert "candidate_generation_already_opened" in fetcher.audit_registry(opened)
    assert (
        "source_registry_boundary_nonzero:local_model_calls"
        in fetcher.audit_registry(nonzero)
    )
    assert "source_registry_repository_overlap" in fetcher.audit_registry(overlap)


def test_p4v2r2_required_member_check_binds_effect_and_license_paths(
    tmp_path: Path,
) -> None:
    task = registry()["tasks"][0]
    root = task["source_root"]
    required = fetcher.required_archive_paths(task, root)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in required[:-1]:
            info = tarfile.TarInfo(name)
            payload = b"fixture"
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    assert fetcher.missing_archive_paths(archive, required) == [required[-1]]
