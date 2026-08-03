from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_production_adequacy_runtime as runtime  # noqa: E402
import theseus_semantic_ir_production_adequacy_task_pool as task_pool  # noqa: E402
import theseus_semantic_ir_production_canary as canary  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pool_is_green_sealed_and_call_free_after_preserved_red_boundary() -> None:
    report = read(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
    )
    failure = read(
        ROOT
        / "reports"
        / "theseus_semantic_ir_production_adequacy_task_pool_representation_failure.json"
    )

    assert failure["trigger_state"] == "RED"
    assert failure["production_create_file_supported"] is False
    assert failure["missing_parent_selected_paths"] == {
        "task_16": ["src/openframe/recognize/coords.py"],
        "task_17": ["src/funcsort/ordering.py"],
    }
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert report["task_count"] == report["sealed_packet_count"] == 18
    assert report["production_create_file_supported"] is True
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())
    assert report["completion_boundary"]["project_selected_quality_token_cap"] is None
    assert report["completion_boundary"]["physical_context_boundary_hit_invalidates_observation"] is True
    assert p2a.sha256_file(ROOT / report["task_pool_owner"]["path"]) == report[
        "task_pool_owner"
    ]["sha256"]
    assert p2a.sha256_file(ROOT / report["adequacy_runtime"]["path"]) == report[
        "adequacy_runtime"
    ]["sha256"]


def test_exact_candidate_packets_are_allowlisted_parent_only_and_replayable() -> None:
    report = read(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
    )
    materialization = read(task_pool.MATERIALIZATION)
    materialization_rows = {
        int(row["index"]): row for row in materialization["rows"]
    }

    for receipt in report["rows"]:
        row = materialization_rows[int(receipt["index"])]
        task_path = ROOT / receipt["task_manifest"]
        packet_path = ROOT / receipt["candidate_packet"]
        task = read(task_path)
        packet = read(packet_path)
        selected = list(row["selected_source_paths"])
        parent = task_pool.archive_sources(row["archives"]["parent"], selected)
        target = task_pool.archive_sources(row["archives"]["target"], selected)

        assert set(packet) == {
            "policy", "state", "opaque_task_id", "serialized_prompt"
        }
        assert packet["policy"] == task_pool.PACKET_POLICY
        assert packet["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
        assert packet["opaque_task_id"] == task["opaque_task_id"]
        assert p4.audit_task(task_path)["trigger_state"] == "GREEN"
        assert task["source_provenance"]["license_spdx"] != "UNKNOWN"
        assert "maximum_total_characters" not in task["candidate_visible_context"]
        assert "maximum_units" not in task["semantic_ir_contract"]
        assert task["candidate_visible_context"][
            "project_selected_character_or_token_cap"
        ] is None
        assert all(
            source.rstrip("\n") in packet["serialized_prompt"]
            for source in parent.values()
        )
        assert all(
            source.rstrip("\n") not in packet["serialized_prompt"]
            for source in target.values()
        )
        assert str(row["target_revision"]) not in packet["serialized_prompt"]
        assert str(row["archives"]["target"]["sha256"]) not in packet[
            "serialized_prompt"
        ]
        assert receipt["serialized_prompt_utf8_bytes"] < task_pool.MODEL_CONTEXT_TOKENS
        assert receipt["conservative_minimum_residual_tokens"] > 0

        with tempfile.TemporaryDirectory(prefix="theseus-packet-replay-") as directory:
            root = Path(directory) / "source"
            p2a.extract_source_archive(
                ROOT / task["source_archive"], root, task["source_archive_root"]
            )
            symbols = runtime.semantic_scope_symbol_table(root, task)
            common = canary.render_common_context(root, task, symbols)
            missing = task["candidate_visible_context"][
                "missing_allowed_effect_paths"
            ]
            if missing:
                common += "\n[MISSING_ALLOWED_EFFECT_PATHS]\n" + "\n".join(missing)
            assert runtime.render_prompt(task, common) == packet["serialized_prompt"]


def test_recursive_answer_metadata_injection_is_rejected() -> None:
    report = read(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
    )
    materialization = read(task_pool.MATERIALIZATION)
    receipt = report["rows"][0]
    row = materialization["rows"][0]
    packet = read(ROOT / receipt["candidate_packet"])
    injected = copy.deepcopy(packet)
    injected["nested"] = [{"solution": "answer-bearing"}]

    faults = task_pool.audit_candidate_packet(
        injected, row, row["archives"]["target"]
    )
    assert any(value.startswith("forbidden_candidate_key:") for value in faults)


def test_new_file_capability_is_narrow_and_historical_owner_is_unchanged() -> None:
    assert p2a.sha256_file(
        ROOT / "scripts" / "theseus_semantic_ir_production.py"
    ) == "6a48cbd5b763343f7edbb65292881f186fd4714ae2ab6d12092d6bc2b4295f20"
    assert runtime.CREATE_FILE_SUPPORTED is True
    for index in (16, 17):
        task = read(
            ROOT
            / "configs"
            / f"theseus_semantic_ir_production_adequacy_task_{index:02d}.json"
        )
        missing = task["candidate_visible_context"][
            "missing_allowed_effect_paths"
        ]
        assert len(missing) == 1
        assert missing[0] in task["allowed_effect_paths"]
    for index in [*range(1, 16), 18]:
        task = read(
            ROOT
            / "configs"
            / f"theseus_semantic_ir_production_adequacy_task_{index:02d}.json"
        )
        assert task["candidate_visible_context"][
            "missing_allowed_effect_paths"
        ] == []
