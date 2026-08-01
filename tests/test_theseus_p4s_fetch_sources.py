from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4s_fetch_sources as fetcher  # noqa: E402


def registry() -> dict:
    return json.loads(fetcher.REGISTRY.read_text(encoding="utf-8"))


def test_p4s_fetcher_binds_exact_frozen_registry() -> None:
    source_registry = registry()

    assert fetcher.sha256_file(fetcher.REGISTRY) == fetcher.EXPECTED_REGISTRY_SHA256
    assert fetcher.SOURCE_SELECTION_COMMIT == (
        "560df8f437d470e8eea9fbd927ee7854f1b93e74"
    )
    assert fetcher.audit_registry(source_registry) == []


def test_p4s_fetch_plan_is_exact_complete_and_registry_derived() -> None:
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
            assert artifact["upstream_name"].endswith(f"_{label}_upstream.tar.gz")
            assert artifact["normalized_name"].endswith(f"_{label}.tar.gz")


def test_p4s_fetcher_rejects_opened_or_nonzero_registry_boundaries() -> None:
    source_registry = registry()
    opened = copy.deepcopy(source_registry)
    opened["boundaries"]["candidate_generation_opened"] = True
    nonzero = copy.deepcopy(source_registry)
    nonzero["boundaries"]["local_model_calls"] = 1

    assert "candidate_generation_already_opened" in fetcher.audit_registry(opened)
    assert (
        "source_registry_boundary_nonzero:local_model_calls"
        in fetcher.audit_registry(nonzero)
    )
