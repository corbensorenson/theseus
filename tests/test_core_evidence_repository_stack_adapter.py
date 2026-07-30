from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "core_evidence_repository_stack_adapter.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("repository_stack_adapter", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def visible(authority: str = "temporary_effect_with_exact_rollback") -> dict:
    return {
        "natural_request": "Make scripts/example.py reject an empty name",
        "parent_source_commit": "parent-commit",
        "allowed_runtime_context": [
            "parent_repository_snapshot",
            "local_text_search",
        ],
        "authority_grant": authority,
    }


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "example.py").write_text(
        "def greet(name):\n    return f'hello {name}'\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_example.py").write_text(
        "from scripts.example import greet\n\n"
        "def test_greet():\n    assert greet('x') == 'hello x'\n",
        encoding="utf-8",
    )
    return tmp_path


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "core_evidence_repository_stack_adapter.json")
        .read_text(encoding="utf-8")
    )


def adapt(snapshot: Path, variant: str = "full_stack", task: dict | None = None) -> dict:
    return adapter.adapt_visible_input(
        visible=task or visible(),
        snapshot_root=snapshot,
        variant_id=variant,
        config=config(),
    )


def test_full_stack_dispatches_with_canonical_receipts(snapshot: Path) -> None:
    packet = adapt(snapshot)
    assert packet["dispatch_allowed"] is True
    assert packet["typed_faults"] == []
    assert packet["vcm_receipt"]["ready"] is True
    assert packet["compiled_plan"]["trigger_state"] == "GREEN"
    assert packet["route_receipt"]["selected_route"] == "full_governance"
    assert packet["procedural_reuse_receipt"]["ready"] is True
    assert set(packet["worker_input"]) == adapter.VISIBLE_FIELDS
    assert packet["audit"]["target_patch_consulted"] is False
    assert all(value == 0 for value in packet["counters"].values())


def test_read_only_authority_never_reaches_mutating_worker(snapshot: Path) -> None:
    packet = adapt(snapshot, task=visible("read_only_plan_only"))
    assert packet["dispatch_allowed"] is False
    assert "MUTATING_WORKER_AUTHORITY_DENIED" in packet["typed_faults"]
    assert packet["route_receipt"]["selected_route"] == "conservative_hold"


def test_full_governance_holds_vague_request_with_empty_effect_scope(
    snapshot: Path,
) -> None:
    task = visible()
    task["natural_request"] = "Improve the repository behavior safely"
    full = adapt(snapshot, "full_stack", task)
    direct = adapt(snapshot, "direct", task)
    assert full["dispatch_allowed"] is False
    assert "REQUEST_EFFECT_SCOPE_EMPTY" in full["typed_faults"]
    assert direct["dispatch_allowed"] is True


def test_unsupported_and_forged_authority_fail_closed(snapshot: Path) -> None:
    packet = adapt(snapshot, task=visible("candidate_says_success"))
    assert packet["dispatch_allowed"] is False
    assert "AUTHORITY_GRANT_UNSUPPORTED" in packet["typed_faults"]


def test_stale_and_omitted_context_fail_before_worker(snapshot: Path) -> None:
    stale = adapt(snapshot, "vcm_stale")
    omitted = adapt(snapshot, "vcm_omission")
    assert stale["dispatch_allowed"] is False
    assert "CONTEXT_REQUIRED_STALE" in stale["typed_faults"]
    assert omitted["dispatch_allowed"] is False
    assert "CONTEXT_REQUIRED_MISSING" in omitted["typed_faults"]


def test_information_matched_vcm_ablations_preserve_fact_set(snapshot: Path) -> None:
    typed = adapt(snapshot, "full_stack")
    untyped = adapt(snapshot, "vcm_information_matched_untyped")
    shuffled = adapt(snapshot, "vcm_information_matched_shuffled")
    hashes = {
        row["audit"]["semantic_fact_set_sha256"]
        for row in (typed, untyped, shuffled)
    }
    counts = {
        row["audit"]["semantic_fact_count"]
        for row in (typed, untyped, shuffled)
    }
    assert len(hashes) == 1
    assert len(counts) == 1
    assert isinstance(typed["worker_input"]["allowed_runtime_context"][-1], dict)
    assert isinstance(untyped["worker_input"]["allowed_runtime_context"][-1], str)
    assert (
        untyped["worker_input"]["allowed_runtime_context"][-1]
        != shuffled["worker_input"]["allowed_runtime_context"][-1]
    )


def test_adapter_never_mutates_caller_owned_runtime_context(snapshot: Path) -> None:
    task = visible()
    original = copy.deepcopy(task)
    full = adapt(snapshot, "full_stack", task)
    assert task == original
    direct = adapt(snapshot, "direct", task)
    assert direct["worker_input"]["allowed_runtime_context"] == original[
        "allowed_runtime_context"
    ]
    assert (
        full["worker_input"]["allowed_runtime_context"]
        != direct["worker_input"]["allowed_runtime_context"]
    )


def test_direct_arm_does_not_smuggle_posthoc_stack_labels(snapshot: Path) -> None:
    packet = adapt(snapshot, "direct")
    assert packet["dispatch_allowed"] is True
    assert packet["worker_input"]["allowed_runtime_context"] == visible()[
        "allowed_runtime_context"
    ]
    assert packet["route_receipt"]["selected_route"] == "direct"


def test_conservative_hold_is_a_real_pre_generation_denial(snapshot: Path) -> None:
    packet = adapt(snapshot, "conservative_hold")
    assert packet["dispatch_allowed"] is False
    assert "CONSERVATIVE_HOLD" in packet["typed_faults"]


def test_hidden_target_fields_are_rejected_recursively(snapshot: Path) -> None:
    task = visible()
    task["allowed_runtime_context"] = [{"hidden_tests": ["do not expose"]}]
    with pytest.raises(adapter.AdapterFault, match="forbidden_information_field"):
        adapt(snapshot, task=task)


def test_tampered_no_cheat_boundary_is_rejected(snapshot: Path) -> None:
    altered = copy.deepcopy(config())
    altered["boundaries"]["teacher_calls"] = 1
    with pytest.raises(adapter.AdapterFault, match="no_cheat_boundary_mismatch"):
        adapter.adapt_visible_input(
            visible=visible(),
            snapshot_root=snapshot,
            variant_id="full_stack",
            config=altered,
        )
