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
        json.dumps({
            "action": "abstain",
            "reason": "The request needs an unavailable schema.",
        }),
    ])
    result = worker.run_worker(
        visible(), tmp_path, permissive_config(),
        generator=lambda _: next(actions),
    )
    assert result["terminal_reason"] == "explicit_abstention"
    assert result["abstained"] is True
    assert result["patch_unified_diff"] == ""
    assert result["abstention_reason"] == (
        "The request needs an unavailable schema."
    )
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


def test_request_criteria_plan_can_be_required_before_mutation(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    (tmp_path / "configs" / "x.json").write_text("{}\n")
    local = config()
    local["budgets"]["require_plan_before_mutation"] = True
    state = worker.RepositoryState(tmp_path, local)
    state.read("scripts/x.py", 1, 10)
    state.read("tests/test_x.py", 1, 10)
    state.read("configs/x.json", 1, 10)
    assert state.required_next_phase() == (
        "record_request_criteria_plan_before_edit"
    )
    with pytest.raises(worker.WorkerFault, match="plan_required"):
        state.replace("scripts/x.py", "x = 1", "x = 2")

    result = state.execute({
        "action": "plan",
        "criteria": [
            "Return the bounded value.",
            "Preserve the existing test behavior.",
        ],
        "target_paths": ["scripts/x.py"],
        "implementation": "Replace the unbounded constant.",
        "verification": "Run the paired test and compile the module.",
    })

    assert result["advisory_only"] is True
    assert state.required_next_phase().startswith("edit_now")
    assert state.replace("scripts/x.py", "x = 1", "x = 2")["changed"]


def test_recovery_instruction_respects_verify_repair_and_finish_phases(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    state = worker.RepositoryState(tmp_path, permissive_config())
    state.replace("scripts/x.py", "x = 1", "x = 2")
    assert "NEXT ACTION MUST BE verify" in state.recovery_instruction(
        "duplicate_read_span_no_new_information"
    )
    assert state.verify([], [], [])["passed"] is True
    assert "NEXT ACTION MUST BE finish" in state.recovery_instruction(
        "duplicate_read_span_no_new_information"
    )

    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert False\n"
    )
    failing = worker.RepositoryState(tmp_path, permissive_config())
    failing.replace("scripts/x.py", "x = 2", "x = 3")
    assert failing.verify([], [], [])["passed"] is False
    assert "bounded" in failing.recovery_instruction(
        "duplicate_read_span_no_new_information"
    )


def test_verification_targets_ignore_non_string_model_values() -> None:
    assert worker.strings([
        "scripts/x.py",
        {"path": "scripts/y.py"},
        3,
    ]) == ["scripts/x.py"]


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


def test_request_path_anchors_inspection_and_scoped_test_creation(
    tmp_path: Path,
) -> None:
    for directory in ("scripts", "tests", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return -1\n"
    )
    (tmp_path / "scripts" / "unrelated.py").write_text(
        "def age():\n    return 0\n"
    )
    (tmp_path / "tests" / "test_clock_paths.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    (tmp_path / "configs" / "clock.json").write_text("{}\n")
    inventory = worker.text_inventory(tmp_path)
    request = "Fix scripts/peer_clock.py and add focused tests."

    paths = worker.priority_inspection_paths(
        inventory, worker.keywords(request), request=request
    )

    assert paths[0] == "scripts/peer_clock.py"
    assert paths[-1] == "tests/test_clock_paths.py"
    assert worker.request_effect_paths(inventory, request) == [
        "scripts/peer_clock.py",
        "tests/test_peer_clock.py",
    ]


def test_scoped_effect_boundary_denies_unrelated_test_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return -1\n"
    )
    local = permissive_config()
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )

    with pytest.raises(
        worker.WorkerFault,
        match="effect_path_outside_request_scoped_authority",
    ):
        state.create(
            "tests/test_peer_clock_timestamp.py",
            "def test_age():\n    assert True\n",
        )
    assert state.create(
        "tests/test_peer_clock.py",
        "def test_age():\n    assert True\n",
    )["changed"]


def test_navigation_ceiling_counts_failed_reads_and_preserves_edit_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    local = permissive_config()
    local["budgets"]["maximum_pre_mutation_navigation_actions"] = 2
    state = worker.RepositoryState(tmp_path, local)

    with pytest.raises(worker.WorkerFault, match="file_missing"):
        state.execute({
            "action": "read",
            "path": "tests/test_x.py",
            "start_line": 1,
            "end_line": 80,
        })
    assert state.execute({
        "action": "read",
        "path": "scripts/x.py",
        "start_line": 1,
        "end_line": 80,
    })["ok"] is True
    with pytest.raises(
        worker.WorkerFault,
        match="pre_mutation_navigation_ceiling",
    ):
        state.execute({"action": "search", "query": "x"})


def test_final_pre_mutation_read_slot_is_reserved_for_test_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "large.py").write_text(
        "\n".join(f"value_{index} = {index}" for index in range(400)) + "\n"
    )
    (tmp_path / "tests" / "test_large.py").write_text(
        "def test_large():\n    assert True\n"
    )
    local = config()
    local["budgets"]["maximum_pre_mutation_read_actions"] = 3
    local["budgets"]["require_test_read_before_mutation"] = True
    state = worker.RepositoryState(tmp_path, local)
    state.execute({
        "action": "read",
        "path": "scripts/large.py",
        "start_line": 1,
        "end_line": 80,
    })
    state.execute({
        "action": "read",
        "path": "scripts/large.py",
        "start_line": 81,
        "end_line": 160,
    })

    with pytest.raises(
        worker.WorkerFault,
        match="last_inspection_slot_reserved_for_test",
    ):
        state.execute({
            "action": "read",
            "path": "scripts/large.py",
            "start_line": 161,
            "end_line": 240,
        })
    guidance = state.recovery_instruction(
        "last_inspection_slot_reserved_for_test"
    )
    assert "tests/test_large.py" in guidance
    assert state.execute({
        "action": "read",
        "path": "tests/test_large.py",
        "start_line": 1,
        "end_line": 80,
    })["ok"] is True


def test_missing_scoped_creation_path_redirects_to_existing_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return -1\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )
    state.read("scripts/peer_clock.py", 1, 80)

    guidance = state.recovery_instruction(
        "file_missing_or_not_regular"
    )

    assert "tests/test_clock_behavior.py" in guidance
    assert "exists=false" in guidance


def test_navigation_is_denied_once_required_inspection_is_complete(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x():\n    assert True\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["forbid_navigation_after_inspection_complete"] = True
    state = worker.RepositoryState(tmp_path, local)
    state.read("scripts/x.py", 1, 80)
    state.read("tests/test_x.py", 1, 80)

    with pytest.raises(
        worker.WorkerFault,
        match="navigation_forbidden_after_inspection_complete",
    ):
        state.execute({"action": "search", "query": "x"})
    assert "NEXT ACTION MUST BE" in state.recovery_instruction(
        "navigation_forbidden_after_inspection_complete"
    )


def test_plan_accepts_authorized_request_scoped_creation_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return -1\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["require_plan_before_mutation"] = True
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )
    state.read("scripts/peer_clock.py", 1, 80)
    state.read("tests/test_clock_behavior.py", 1, 80)

    result = state.record_plan({
        "criteria": ["Return a bounded age."],
        "target_paths": [
            "scripts/peer_clock.py",
            "tests/test_peer_clock.py",
        ],
        "implementation": "Bound the computed age and add focused tests.",
        "verification": "Run the new focused test.",
    })

    assert result["ok"] is True
    assert result["plan"]["target_paths"] == [
        "scripts/peer_clock.py",
        "tests/test_peer_clock.py",
    ]


def test_phase_contract_requires_all_planned_paths_then_verify_then_finish(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return -1\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["require_plan_before_mutation"] = True
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    local["budgets"]["enforce_phase_action_contract"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )
    state.execute({
        "action": "read",
        "path": "scripts/peer_clock.py",
        "start_line": 1,
        "end_line": 80,
    })
    state.execute({
        "action": "read",
        "path": "tests/test_clock_behavior.py",
        "start_line": 1,
        "end_line": 80,
    })
    assert state.allowed_phase_actions() == {"plan", "abstain"}
    state.execute({
        "action": "plan",
        "criteria": ["Return a bounded age."],
        "target_paths": [
            "scripts/peer_clock.py",
            "tests/test_peer_clock.py",
        ],
        "implementation": "Bound the age and add tests.",
        "verification": "Run the focused tests.",
    })
    state.execute({
        "action": "replace",
        "path": "scripts/peer_clock.py",
        "old": "return -1",
        "new": "return 0",
    })
    assert state.required_next_phase() == (
        "complete_planned_effect_paths:tests/test_peer_clock.py"
    )
    with pytest.raises(
        worker.WorkerFault,
        match="action_not_allowed_in_current_phase",
    ):
        state.execute({
            "action": "verify",
            "pytest": [],
            "py_compile": [],
            "json": [],
        })
    state.execute({
        "action": "create",
        "path": "tests/test_peer_clock.py",
        "content": (
            "from scripts.peer_clock import age\n\n"
            "def test_age():\n    assert age() == 0\n"
        ),
    })
    assert state.allowed_phase_actions() == {"verify", "abstain"}
    state.execute({
        "action": "verify",
        "pytest": [],
        "py_compile": [],
        "json": [],
    })
    assert state.allowed_phase_actions() == {"finish", "abstain"}
    assert state.execute({"action": "finish"})["terminal"] is True


def test_failed_verification_requires_one_bounded_repair_then_reverify(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    (tmp_path / "tests" / "test_x.py").write_text(
        "from scripts.x import x\n\ndef test_x():\n    assert x == 2\n"
    )
    local = permissive_config()
    local["budgets"]["enforce_phase_action_contract"] = True
    state = worker.RepositoryState(tmp_path, local)
    state.execute({
        "action": "replace",
        "path": "scripts/x.py",
        "old": "x = 1",
        "new": "x = 3",
    })
    result = state.execute({
        "action": "verify",
        "pytest": [],
        "py_compile": [],
        "json": [],
    })
    assert result["passed"] is False
    assert state.allowed_phase_actions() == {
        "replace", "create", "create_test", "delete", "read", "abstain"
    }
    reread = state.execute({
        "action": "read",
        "path": "scripts/x.py",
        "start_line": 1,
        "end_line": 80,
    })
    assert "x = 3" in reread["content"]
    with pytest.raises(
        worker.WorkerFault,
        match="duplicate_read_span_no_new_information",
    ):
        state.execute({
            "action": "read",
            "path": "scripts/x.py",
            "start_line": 1,
            "end_line": 80,
        })
    state.execute({
        "action": "replace",
        "path": "scripts/x.py",
        "old": "x = 3",
        "new": "x = 2",
    })
    assert state.allowed_phase_actions() == {"verify", "abstain"}


def test_created_file_character_ceiling_is_enforced(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    local = permissive_config()
    local["budgets"]["maximum_created_file_characters"] = 10
    state = worker.RepositoryState(tmp_path, local)

    with pytest.raises(
        worker.WorkerFault,
        match="created_file_character_ceiling_exceeded",
    ):
        state.create("tests/test_x.py", "x" * 11)


def test_structured_test_action_renders_bounded_model_authored_tests(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return 0\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["require_plan_before_mutation"] = True
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    local["budgets"]["structured_test_creation_required"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )
    state.read("scripts/peer_clock.py", 1, 80)
    state.read("tests/test_clock_behavior.py", 1, 80)
    state.record_plan({
        "criteria": ["Return a nonnegative age."],
        "target_paths": [
            "scripts/peer_clock.py",
            "tests/test_peer_clock.py",
        ],
        "implementation": "Bound the age and add a focused test.",
        "verification": "Run the focused test.",
    })
    state.replace(
        "scripts/peer_clock.py",
        "return 0",
        "return max(0, 0)",
    )

    result = state.execute({
        "action": "create_test",
        "path": "tests/test_peer_clock.py",
        "preamble": "",
        "tests": [{
            "name": "test_age_is_nonnegative",
            "parameters": "",
            "body": "assert target.age() >= 0",
        }],
    })

    rendered = (tmp_path / "tests" / "test_peer_clock.py").read_text()
    assert result["changed"] is True
    assert "import peer_clock as target" in rendered
    assert "def test_age_is_nonnegative():" in rendered
    assert "    assert target.age() >= 0" in rendered


def test_structured_test_preamble_keeps_only_imports_and_drops_future(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return 0\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = permissive_config()
    state = worker.RepositoryState(tmp_path, local)
    state.advisory_plan = {
        "criteria": ["age"],
        "target_paths": [
            "scripts/peer_clock.py",
            "tests/test_peer_clock.py",
        ],
        "implementation": "test",
        "verification": "test",
    }

    result = state.create_structured_test({
        "path": "tests/test_peer_clock.py",
        "preamble": (
            "from __future__ import annotations\n"
            "import os\n"
            "HELPER = 3\n"
            "def hidden_helper():\n    return 3\n"
        ),
        "tests": [{
            "name": "test_age",
            "parameters": "",
            "body": "assert target.age() == 0",
        }],
    })

    rendered = (tmp_path / "tests" / "test_peer_clock.py").read_text()
    assert result["accepted_import_lines"] == 1
    assert rendered.count("from __future__ import annotations") == 1
    assert "import os" in rendered
    assert "HELPER" not in rendered
    assert "hidden_helper" not in rendered


def test_plan_targets_can_be_bound_to_request_scoped_authority(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return 0\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["bind_plan_target_paths_to_request_scope"] = True
    local["budgets"]["enforce_request_scoped_effect_paths"] = True
    state = worker.RepositoryState(
        tmp_path,
        local,
        request="Fix scripts/peer_clock.py and add focused tests.",
    )
    state.read("scripts/peer_clock.py", 1, 80)
    state.read("tests/test_clock_behavior.py", 1, 80)

    result = state.record_plan({
        "criteria": ["Return a nonnegative age."],
        "target_paths": ["scripts/peer_clock.py"],
        "implementation": "Bound the age and add a focused test.",
        "verification": "Run the focused test.",
    })

    assert result["candidate_target_paths_ignored"] is True
    assert result["plan"]["target_paths"] == [
        "scripts/peer_clock.py",
        "tests/test_peer_clock.py",
    ]


def test_plan_verification_can_default_to_controller_policy(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "peer_clock.py").write_text(
        "def age():\n    return 0\n"
    )
    (tmp_path / "tests" / "test_clock_behavior.py").write_text(
        "from scripts.peer_clock import age\n"
    )
    local = config()
    local["budgets"]["minimum_pre_mutation_read_actions"] = 2
    local["budgets"]["minimum_pre_mutation_distinct_paths"] = 2
    local["budgets"]["default_plan_verification_strategy"] = True
    state = worker.RepositoryState(tmp_path, local)
    state.read("scripts/peer_clock.py", 1, 80)
    state.read("tests/test_clock_behavior.py", 1, 80)

    result = state.record_plan({
        "criteria": ["Return a nonnegative age."],
        "target_paths": ["scripts/peer_clock.py"],
        "implementation": "Bound the age.",
        "verification": "",
    })

    assert result["candidate_verification_defaulted"] is True
    assert result["plan"]["verification"] == (
        "controller_selects_checks_from_changed_paths_and_request"
    )


def test_automatic_verification_uses_direct_tests_not_broad_references(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "admission.py").write_text(
        "def allowed():\n    return False\n"
    )
    (tmp_path / "tests" / "test_admission.py").write_text(
        "import admission\n\ndef test_allowed():\n    assert admission.allowed()\n"
    )
    (tmp_path / "tests" / "test_unrelated_campaign.py").write_text(
        "import admission\n\ndef test_runtime_asset():\n    assert False\n"
    )
    local = permissive_config()
    state = worker.RepositoryState(tmp_path, local)
    state.replace(
        "scripts/admission.py",
        "return False",
        "return True",
    )

    selected = state.automatic_verification_targets()

    assert selected["pytest"] == ["tests/test_admission.py"]
