from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_worker_v2 as worker  # noqa: E402


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "core_evidence_local_8b_worker.json").read_text()
    )


def permissive_config() -> dict:
    value = config()
    value["budgets"]["minimum_pre_mutation_read_actions"] = 0
    value["budgets"]["minimum_pre_mutation_distinct_paths"] = 0
    value["budgets"]["require_test_read_before_mutation"] = False
    value["budgets"][
        "require_existing_integration_before_new_implementation"
    ] = False
    return value


def visible() -> dict:
    return {
        "natural_request": "Make the policy return bounded and test it",
        "parent_source_commit": "a" * 40,
        "allowed_runtime_context": ["parent_repository_snapshot", "local_text_search"],
        "authority_grant": "temporary_effect_with_exact_rollback",
    }


def test_injected_agent_produces_real_verified_diff(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "policy.py").write_text(
        "def policy():\n    return 'open'\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_policy.py").write_text(
        "from scripts.policy import policy\n\n"
        "def test_policy():\n    assert policy() == 'bounded'\n",
        encoding="utf-8",
    )
    actions = iter([
        json.dumps({
            "action": "replace",
            "path": "scripts/policy.py",
            "old": "return 'open'",
            "new": "return 'bounded'",
        }),
        json.dumps({
            "action": "verify",
            "pytest": [],
            "py_compile": [],
            "json": [],
        }),
        json.dumps({"action": "finish"}),
    ])

    events = []
    result = worker.run_worker(
        visible(), tmp_path, permissive_config(),
        generator=lambda _: next(actions),
        event_sink=events.append,
    )

    assert result["abstained"] is False
    assert result["terminal_reason"] == "finished"
    assert "--- a/scripts/policy.py" in result["patch_unified_diff"]
    assert "+++ b/scripts/policy.py" in result["patch_unified_diff"]
    assert "+    return 'bounded'" in result["patch_unified_diff"]
    assert result["proposed_paths"] == ["scripts/policy.py"]
    assert result["verification_receipts"][-1]["passed"] is True
    assert result["verification_receipts"][-1]["selection"] == {
        "pytest": ["tests/test_policy.py"],
        "py_compile": ["scripts/policy.py"],
        "json": [],
    }
    assert result["learned_generation_credit"] == 1
    assert result["external_inference_calls"] == 0
    assert [row["action"] for row in events] == ["replace", "verify", "finish"]
    assert events[-1]["terminal"] is True


def test_hidden_field_and_git_metadata_are_rejected(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with pytest.raises(worker.WorkerFault, match="visible_fields"):
        worker.validate_inputs({**visible(), "target_commit": "b" * 40}, tmp_path, config())
    with pytest.raises(worker.WorkerFault, match="git_metadata"):
        worker.validate_inputs(visible(), tmp_path, config())


def test_complete_model_snapshot_accepts_single_and_sharded_weights(
    tmp_path: Path,
) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    assert worker.complete_model_snapshot(tmp_path) is False
    (tmp_path / "model.safetensors.index.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    assert worker.complete_model_snapshot(tmp_path) is True
    (tmp_path / "model-00001-of-00002.safetensors").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert worker.complete_model_snapshot(tmp_path) is True


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/tmp/escape.py",
        "runtime/escape.py",
        "scripts/escape.exe",
    ],
)
def test_path_escape_and_unsupported_surfaces_fail_closed(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "scripts").mkdir()
    state = worker.RepositoryState(tmp_path, config())
    with pytest.raises(worker.WorkerFault):
        state.create(path, "bad")


def test_symlink_write_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("safe\n")
    (tmp_path / "scripts" / "link.py").symlink_to(outside)
    state = worker.RepositoryState(tmp_path, config())

    with pytest.raises(worker.WorkerFault, match="symlink"):
        state.replace("scripts/link.py", "safe", "unsafe")
    assert outside.read_text() == "safe\n"


def test_noop_candidate_cannot_finish(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    actions = iter([json.dumps({"action": "finish"})] * 30)

    result = worker.run_worker(
        visible(), tmp_path, permissive_config(),
        generator=lambda _: next(actions)
    )

    assert result["abstained"] is True
    assert result["patch_unified_diff"] == ""
    assert result["terminal_reason"] == "stalled_repeated_denied_action"
    assert "no_effect_capable_patch" in result["residuals"]


def test_abstention_discards_every_provisional_effect(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    actions = iter([
        json.dumps({
            "action": "replace",
            "path": "scripts/x.py",
            "old": "x = 1",
            "new": "x = 2",
        }),
        json.dumps({"action": "abstain"}),
    ])
    result = worker.run_worker(
        visible(), tmp_path, permissive_config(),
        generator=lambda _: next(actions),
    )
    assert result["terminal_reason"] == "explicit_abstention"
    assert result["abstained"] is True
    assert result["patch_unified_diff"] == ""
    assert (tmp_path / "scripts" / "x.py").read_text() == "x = 1\n"


def test_candidate_authored_flags_are_not_an_action(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    state = worker.RepositoryState(tmp_path, config())
    with pytest.raises(worker.WorkerFault, match="unknown_action"):
        state.execute({"action": "claim_success", "passed": True})


def test_first_complete_action_can_stop_generation_before_trailing_text() -> None:
    raw = 'analysis prefix\n{"action":"search","query":"needle"} trailing'
    assert worker.complete_action_json(raw) == (
        '{"action":"search","query":"needle"}'
    )
    assert worker.parse_action(raw) == {
        "action": "search",
        "query": "needle",
    }
    assert worker.complete_action_json('{"action":"read"') is None


def test_observed_replace_concatenation_is_repaired_and_counted(
    tmp_path: Path,
) -> None:
    raw = (
        '{"action":"replace","path":"scripts/x.py",'
        '"old":"x = 1"+"x = 2"}'
    )
    assert worker.repair_common_replace_concatenation(raw) == {
        "action": "replace",
        "path": "scripts/x.py",
        "old": "x = 1",
        "new": "x = 2",
    }
    assert worker.parse_action(raw)["_format_repaired"] is True
    assert worker.repair_common_replace_concatenation(
        '{"action":"create","path":"scripts/x.py","content":"a"+"b"}'
    ) is None

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    actions = iter([
        raw,
        '{"action":"verify","pytest":[],"py_compile":[],"json":[]}',
        '{"action":"finish"}',
    ])
    result = worker.run_worker(
        visible(),
        tmp_path,
        permissive_config(),
        generator=lambda _: next(actions),
    )
    assert result["format_repairs"] == 1
    assert result["terminal_reason"] == "finished"


def test_denied_tool_actions_do_not_consume_format_retry_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    actions = iter(
        [
            json.dumps({"action": f"unsupported_{index}"})
            for index in range(5)
        ]
        + [
            json.dumps({
                "action": "replace",
                "path": "scripts/x.py",
                "old": "x = 1",
                "new": "x = 2",
            }),
            json.dumps({
                "action": "verify",
                "pytest": [],
                "py_compile": [],
                "json": [],
            }),
            json.dumps({"action": "finish"}),
        ]
    )
    result = worker.run_worker(
        visible(), tmp_path, permissive_config(),
        generator=lambda _: next(actions)
    )
    assert result["terminal_reason"] == "finished"
    assert result["action_summary"]["failed_actions"] == 5


def test_phrase_search_falls_back_to_ranked_request_tokens(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "availability.py").write_text(
        "def laptop_state():\n    return 'available'\n",
        encoding="utf-8",
    )
    state = worker.RepositoryState(tmp_path, config())
    result = state.search("laptop availability gate")
    assert result["match_mode"] == "token_ranked_fallback"
    assert result["matches"][0]["path"] == "scripts/availability.py"


def test_mutation_requires_three_distinct_reads_including_a_test(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    (tmp_path / "configs" / "x.json").write_text("{}\n")
    state = worker.RepositoryState(tmp_path, config())
    state.read("scripts/x.py", 1, 10)
    state.read("configs/x.json", 1, 10)
    with pytest.raises(worker.WorkerFault, match="read_actions"):
        state.replace("scripts/x.py", "x = 1", "x = 2")
    state.read("tests/test_x.py", 1, 10)
    assert state.replace("scripts/x.py", "x = 1", "x = 2")["changed"]


def test_duplicate_read_span_is_denied_as_no_new_information(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    state = worker.RepositoryState(tmp_path, config())
    state.read("scripts/x.py", 1, 10)
    with pytest.raises(worker.WorkerFault, match="duplicate_read_span"):
        state.read("scripts/x.py", 1, 10)


def test_duplicate_read_after_complete_inspection_forces_decision(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    (tmp_path / "configs" / "x.json").write_text("{}\n")
    state = worker.RepositoryState(tmp_path, config())
    state.read("scripts/x.py", 1, 10)
    state.read("tests/test_x.py", 1, 10)
    state.read("configs/x.json", 1, 10)

    guidance = state.recovery_instruction(
        "duplicate_read_span_no_new_information"
    )

    assert "NEXT ACTION MUST BE replace/create/delete" in guidance
    assert "Do not read or search again" in guidance


def test_prompt_boundary_cache_reuses_untrimmable_prefix() -> None:
    class FakeCacheStore:
        def __init__(self) -> None:
            self.entries: list[tuple[tuple[str, str], list[int], dict]] = []

        def fetch_nearest_cache(self, model_key, tokens):
            matches = [
                (stored_tokens, value)
                for key, stored_tokens, value in self.entries
                if key == model_key and tokens[:len(stored_tokens)] == stored_tokens
            ]
            if not matches:
                return None, tokens
            stored_tokens, value = max(matches, key=lambda row: len(row[0]))
            return dict(value), tokens[len(stored_tokens):]

        def insert_cache(self, model_key, tokens, value):
            self.entries = [
                row for row in self.entries
                if not (row[0] == model_key and row[1] == tokens)
            ]
            self.entries.append((model_key, list(tokens), dict(value)))

    store = FakeCacheStore()
    prefills = []

    def prefill(value, tokens):
        value["tokens"] = value.get("tokens", []) + list(tokens)
        prefills.append(list(tokens))

    first_cache, first_uncached, first_reused, first_new = (
        worker.prepare_prompt_boundary_cache(
            store,
            ("model", "revision"),
            [1, 2, 3, 4],
            [1, 2, 3],
            make_cache=lambda: {"tokens": []},
            prefill=prefill,
        )
    )
    second_cache, second_uncached, second_reused, second_new = (
        worker.prepare_prompt_boundary_cache(
            store,
            ("model", "revision"),
            [1, 2, 3, 5, 6],
            [1, 2, 3, 5],
            make_cache=lambda: {"tokens": []},
            prefill=prefill,
        )
    )

    assert first_reused is False
    assert first_uncached == [4]
    assert first_new == 3
    assert first_cache["tokens"] == [1, 2, 3]
    assert second_reused is True
    assert second_uncached == [6]
    assert second_new == 1
    assert second_cache["tokens"] == [1, 2, 3, 5]
    assert prefills == [[1, 2, 3], [5]]


def test_tiny_reads_expand_to_stable_context_blocks(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text(
        "".join(f"line_{index} = {index}\n" for index in range(1, 201))
    )
    state = worker.RepositoryState(tmp_path, config())
    result = state.read("scripts/x.py", 114, 119)
    assert result["start_line"] == 81
    assert result["end_line"] == 160
    with pytest.raises(worker.WorkerFault, match="duplicate_read_span"):
        state.read("scripts/x.py", 120, 125)


def test_pre_mutation_read_ceiling_requires_decision(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    local = config()
    local["budgets"]["maximum_pre_mutation_read_actions"] = 1
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "scripts" / "y.py").write_text("y = 1\n")
    state = worker.RepositoryState(tmp_path, local)
    state.read("scripts/x.py", 1, 10)
    with pytest.raises(
        worker.WorkerFault, match="pre_mutation_inspection_ceiling"
    ):
        state.read("scripts/y.py", 1, 10)
    assert "NEXT ACTION MUST BE" in state.recovery_instruction(
        "pre_mutation_inspection_ceiling_reached"
    )


def test_required_next_phase_tracks_edit_verify_and_finish(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    (tmp_path / "configs" / "x.json").write_text("{}\n")
    state = worker.RepositoryState(tmp_path, config())
    state.read("scripts/x.py", 1, 10)
    state.read("tests/test_x.py", 1, 10)
    state.read("configs/x.json", 1, 10)
    assert state.required_next_phase().startswith("edit_now")
    state.replace("scripts/x.py", "x = 1", "x = 2")
    assert state.required_next_phase() == "verify_changed_paths"
    assert state.verify([], [], [])["passed"] is True
    assert state.required_next_phase().startswith("finish_or")


def test_created_file_and_unified_diff_end_with_newline(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    state = worker.RepositoryState(tmp_path, permissive_config())
    state.create("scripts/new.py", "value = 1")
    assert (tmp_path / "scripts" / "new.py").read_bytes().endswith(b"\n")
    assert state.unified_diff().endswith("\n")


def test_verification_before_any_effect_is_denied(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    state = worker.RepositoryState(tmp_path, permissive_config())
    with pytest.raises(worker.WorkerFault, match="changed_paths"):
        state.verify([], ["scripts/x.py"], [])


def test_new_implementation_requires_an_existing_integration_effect(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "existing.py").write_text("enabled = False\n")
    (tmp_path / "tests" / "test_existing.py").write_text("assert True\n")
    (tmp_path / "configs" / "existing.json").write_text("{}\n")
    state = worker.RepositoryState(tmp_path, config())
    state.read("scripts/existing.py", 1, 10)
    state.read("tests/test_existing.py", 1, 10)
    state.read("configs/existing.json", 1, 10)
    with pytest.raises(worker.WorkerFault, match="integration_effect"):
        state.create("scripts/orphan.py", "value = 1")
    state.replace(
        "scripts/existing.py", "enabled = False", "enabled = True"
    )
    assert state.create("scripts/helper.py", "value = 1")["changed"]


def test_repair_budget_is_enforced_after_failed_verification(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text("assert False\n")
    local = permissive_config()
    local["budgets"]["maximum_repair_attempts"] = 1
    state = worker.RepositoryState(tmp_path, local)
    state.replace("scripts/x.py", "x = 1", "x = 2")
    receipt = state.verify(["tests/test_x.py"], [], [])
    assert receipt["passed"] is False
    state.replace("scripts/x.py", "x = 2", "x = 3")
    state.failed_verification_seen = True
    with pytest.raises(worker.WorkerFault, match="repair_budget"):
        state.replace("scripts/x.py", "x = 3", "x = 4")


def test_failed_verification_requires_repair_before_reverification(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text("assert False\n")
    state = worker.RepositoryState(tmp_path, permissive_config())
    state.replace("scripts/x.py", "x = 1", "x = 2")
    assert state.verify(["tests/test_x.py"], [], [])["passed"] is False
    with pytest.raises(
        worker.WorkerFault, match="verification_requires_repair_after_failure"
    ):
        state.verify(["tests/test_x.py"], [], [])
    assert "bounded repair" in state.recovery_instruction(
        "verification_requires_repair_after_failure"
    )


def test_target_blind_priority_inspection_covers_code_config_and_test(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "canary.py").write_text(
        "memory_floor = 8\n"
    )
    (tmp_path / "tests" / "test_canary.py").write_text(
        "def test_memory_floor():\n    assert True\n"
    )
    (tmp_path / "configs" / "canary.json").write_text(
        '{"memory_floor": 8}\n'
    )
    inventory = worker.text_inventory(tmp_path)
    assert worker.priority_inspection_paths(
        inventory, worker.keywords("remove canary memory floor")
    ) == [
        "scripts/canary.py",
        "configs/canary.json",
        "tests/test_canary.py",
    ]
