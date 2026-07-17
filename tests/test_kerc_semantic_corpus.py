from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_semantic_corpus as producer  # noqa: E402
import kerc_semantic_corpus_verify as verifier  # noqa: E402
import kernel_english_protocol as kernel  # noqa: E402
import vcm_semantic_memory as memory  # noqa: E402


def source_contract(path: Path) -> dict:
    return {
        "path": str(path),
        "dataset_id": "fixture/dolly",
        "dataset_revision": "fixture-v1",
        "content_sha256": verifier.sha256_file(path),
        "license_spdx": "CC0-1.0",
        "records_by_split": {
            "private_train": 1,
            "private_dev": 1,
            "private_eval": 1,
        },
        "allowed_objectives": ["surface_direct_control_v1"],
    }


def test_dolly_replay_rejects_candidate_target_corruption(tmp_path: Path) -> None:
    path = tmp_path / "dolly.jsonl"
    rows = [
        {
            "instruction": f"Explain deterministic fixture {index} clearly.",
            "context": "",
            "response": f"This is independently authored response {index}.",
            "category": "open_qa",
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source = source_contract(path)
    assignments = verifier.independent_dolly_assignments(source, 2048)
    expected = next(iter(assignments.values()))
    candidate = producer.dolly_record(
        {
            "selection_key": expected["selection_key"],
            "prompt": expected["prompt"],
            "target": expected["target"],
            "annotation": expected["annotation"],
            "category": expected["annotation"]["category"],
        },
        split=expected["split"],
        source=source,
        producer_sha256="sha256:" + "1" * 64,
    )

    verifier.verify_dolly_record(candidate, source, expected)
    candidate["surface_target"] += " corrupted"
    with pytest.raises(ValueError, match="surface replay mismatch"):
        verifier.verify_dolly_record(candidate, source, expected)


def test_dolly_grounded_question_replays_unique_support_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dolly-grounded.jsonl"
    rows = [
        {
            "instruction": f"What is the fixture value for record {index}?",
            "context": (
                f"Record {index} has introductory material. "
                f"The fixture value is value-{index}. End of source context."
            ),
            "response": f"value-{index}",
            "category": "closed_qa",
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source = {
        "path": str(path),
        "dataset_id": "fixture/dolly-grounded",
        "dataset_revision": "fixture-v1",
        "content_sha256": verifier.sha256_file(path),
        "license_spdx": "CC0-1.0",
        "records_by_split": {
            "private_train": 0,
            "private_dev": 0,
            "private_eval": 0,
        },
        "grounded_question_records_by_split": {
            "private_train": 1,
            "private_dev": 1,
            "private_eval": 1,
        },
        "grounded_question_required_forms": ["what"],
        "grounded_question_allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
        "allowed_objectives": ["surface_direct_control_v1"],
    }
    produced, rejects = producer.load_dolly_grounded_question_candidates(
        source, maximum_characters=2048
    )
    selected = producer.select_dolly_grounded_questions(
        produced,
        source["grounded_question_records_by_split"],
        required_question_forms=source["grounded_question_required_forms"],
    )
    independent = verifier.independent_dolly_grounded_assignments(source, 2048)
    assert not rejects["not_unique_bounded_exact_support"]
    assert len(independent) == 3
    for split, split_rows in selected.items():
        candidate = producer.dolly_grounded_question_record(
            split_rows[0],
            split=split,
            source=source,
            producer_sha256="sha256:" + "1" * 64,
        )
        expected = independent[candidate["provenance"]["source_id"]]
        receipt = verifier.verify_dolly_grounded_record(candidate, source, expected)
        assert receipt["support_relation"] == "unique_contiguous_exact_span"
        candidate["verification_receipt"] = {
            "policy": kernel.TRAINING_VERIFICATION_POLICY,
            "receipt_id": "fixture-source-replay",
            "accepted": True,
            "verifier_id": "fixture-independent-source-replay",
            "reviewer_independent_of_record_producer": True,
            "method": "licensed_semantic_dataset_plus_independent_schema_review",
            "evidence_sha256": "sha256:" + "2" * 64,
            "semantic_payload_sha256": kernel.training_semantic_payload_sha256(
                candidate
            ),
        }
        views = {
            view["objective"]: view for view in kernel.compile_training_views(candidate)
        }
        for objective in (
            "surface_direct_control_v1",
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        ):
            assert '"answer_span"' not in views[objective]["prompt"].lower()
            assert "source_annotation" not in views[objective]["prompt"]
            assert "surface_target" not in views[objective]["prompt"]
            assert views[objective]["generator_visible_fields"] == [
                "trusted_source_prefix_tokens",
                "prompt",
            ]
            assert "target" in views[objective]["evaluator_only_fields"]
        forged = json.loads(json.dumps(candidate))
        forged["source_annotation"]["answer_span"][0] += 1
        with pytest.raises(ValueError, match="annotation replay mismatch"):
            verifier.verify_dolly_grounded_record(forged, source, expected)


def test_grounded_question_split_reserves_rare_forms_for_future_splits() -> None:
    rows = [
        {
            "selection_key": f"sha256:{index:064x}",
            "question_form": "who",
        }
        for index in range(3)
    ]
    rows.extend(
        {
            "selection_key": f"sha256:{index + 100:064x}",
            "question_form": "what",
        }
        for index in range(9)
    )

    selected = producer.select_dolly_grounded_questions(
        rows,
        {"private_train": 8, "private_dev": 2, "private_eval": 2},
        required_question_forms=["what", "who"],
    )

    assert {split: len(split_rows) for split, split_rows in selected.items()} == {
        "private_train": 8,
        "private_dev": 2,
        "private_eval": 2,
    }
    assert all(
        {row["question_form"] for row in split_rows} == {"what", "who"}
        for split_rows in selected.values()
    )


def write_oasst_fixture(tmp_path: Path, *, duplicate_responses: bool = False) -> dict:
    labels = {
        "name": ["spam", "lang_mismatch", "pii", "not_appropriate", "quality"],
        "value": [0.0, 0.0, 0.0, 0.0, 1.0],
        "count": [2, 2, 2, 2, 2],
    }

    def row(
        message_id: str,
        parent_id: str | None,
        text: str,
        role: str,
        rank: int | None = None,
    ) -> dict:
        return {
            "message_id": message_id,
            "parent_id": parent_id,
            "text": text,
            "role": role,
            "lang": "en",
            "review_count": 2,
            "review_result": True,
            "deleted": False,
            "rank": rank,
            "synthetic": False,
            "model_name": None,
            "message_tree_id": "tree-1",
            "tree_state": "ready_for_export",
            "labels": labels,
        }

    second = "Use a temporary copy, then replace the destination atomically."
    if duplicate_responses:
        second = "Leave the original intact and report a structured failure."
    rows = [
        row("u0", None, "How should a durable file move work?", "prompter"),
        row(
            "a0",
            "u0",
            "Validate the checksum before committing the move.",
            "assistant",
        ),
        row("u1", "a0", "What should happen if validation fails?", "prompter"),
        row(
            "r0",
            "u1",
            "Leave the original intact and report a structured failure.",
            "assistant",
            0,
        ),
        row("r1", "u1", second, "assistant", 1),
    ]
    train = tmp_path / "oasst-train.parquet"
    validation = tmp_path / "oasst-validation.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, train)
    pq.write_table(table.slice(0, 0), validation)
    file_hashes = {
        "train": verifier.sha256_file(train),
        "validation": verifier.sha256_file(validation),
    }
    return {
        "dataset_id": "fixture/oasst2",
        "dataset_revision": "fixture-v1",
        "content_sha256": kernel.stable_hash(file_hashes),
        "license_spdx": "Apache-2.0",
        "files": {
            "train": {"path": str(train), "content_sha256": file_hashes["train"]},
            "validation": {
                "path": str(validation),
                "content_sha256": file_hashes["validation"],
            },
        },
        "records_by_split": {
            "private_train": 1,
            "private_dev": 0,
            "private_eval": 0,
        },
        "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
        "minimum_quality": 0.5,
        "maximum_label_values": {
            "spam": 0.5,
            "lang_mismatch": 0.5,
            "pii": 0.5,
            "not_appropriate": 0.5,
        },
        "maximum_current_characters": 1024,
        "maximum_response_characters": 1024,
        "maximum_context_characters": 2048,
        "maximum_compiled_context_bytes": 4096,
        "minimum_prior_turns": 2,
        "maximum_prior_turns": 4,
        "required_valid_realization_ranks": [0, 1],
    }


def test_oasst_replay_binds_context_and_two_human_realizations(tmp_path: Path) -> None:
    source = write_oasst_fixture(tmp_path)
    produced, rejects = producer.load_oasst_candidates(source)
    independent = verifier.independent_oasst_assignments(source)
    assert not rejects["duplicate_ranked_realization"]
    assert len(produced["private_train"]) == 1
    expected = next(iter(independent.values()))
    candidate = producer.oasst_record(
        produced["private_train"][0],
        split="private_train",
        source=source,
        producer_sha256="sha256:" + "1" * 64,
    )

    receipt = verifier.verify_oasst_record(candidate, source, expected)
    semantic_sha = kernel.training_semantic_payload_sha256(candidate)
    candidate["verification_receipt"] = {
        "policy": kernel.TRAINING_VERIFICATION_POLICY,
        "receipt_id": "kerc-source-replay:" + "2" * 32,
        "accepted": True,
        "verifier_id": "fixture-independent-oasst-replay",
        "reviewer_independent_of_record_producer": True,
        "method": "licensed_semantic_dataset_plus_independent_schema_review",
        "evidence_sha256": "sha256:" + "3" * 64,
        "semantic_payload_sha256": semantic_sha,
    }
    views = kernel.compile_training_views(candidate)
    counts = {
        objective: sum(view["objective"] == objective for view in views)
        for objective in kernel.TRAINING_OBJECTIVES
    }
    assert receipt == {
        "message_tree_sha256": expected["annotation"]["message_tree_sha256"],
        "context_turn_count": 2,
        "human_valid_realization_count": 2,
    }
    assert counts == {
        "surface_direct_control_v1": 2,
        "surface_to_kernel_program_v1": 1,
        "kernel_program_to_answer_packet_v1": 2,
        "answer_packet_to_surface_v1": 2,
    }
    targets = expected["targets"]
    for view in views:
        if view["objective"] in {
            "surface_direct_control_v1",
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        }:
            assert all(target not in view["prompt"] for target in targets)
    assert all("realization_id" not in view["prompt"] for view in views)

    forged = json.loads(json.dumps(candidate))
    forged["valid_realizations"][1]["surface_target"] = "Forged alternative."
    with pytest.raises(ValueError, match="valid-realization set replay mismatch"):
        verifier.verify_oasst_record(forged, source, expected)


def test_oasst_duplicate_ranked_responses_are_not_admitted(tmp_path: Path) -> None:
    source = write_oasst_fixture(tmp_path, duplicate_responses=True)
    produced, rejects = producer.load_oasst_candidates(source)

    assert produced.get("private_train", []) == []
    assert rejects["duplicate_ranked_realization"] == 1


def write_oasst_behavior_fixture(tmp_path: Path) -> dict:
    labels = {
        "name": ["spam", "lang_mismatch", "pii", "not_appropriate", "quality"],
        "value": [0.0, 0.0, 0.0, 0.0, 1.0],
        "count": [2, 2, 2, 2, 2],
    }

    def row(message_id: str, parent_id: str | None, text: str, role: str, tree: str) -> dict:
        return {
            "message_id": message_id,
            "parent_id": parent_id,
            "text": text,
            "role": role,
            "lang": "en",
            "review_count": 2,
            "review_result": True,
            "deleted": False,
            "rank": 0 if role == "assistant" else None,
            "synthetic": False,
            "model_name": None,
            "message_tree_id": tree,
            "tree_state": "ready_for_export",
            "labels": labels,
        }

    rows = [
        row("u1", None, "Which release should I install?", "prompter", "tree-clarify"),
        row(
            "a1",
            "u1",
            "Could you clarify which operating system and release you mean?",
            "assistant",
            "tree-clarify",
        ),
        row("u2", None, "What is the current local time?", "prompter", "tree-abstain"),
        row(
            "a2",
            "u2",
            "I don't know because I do not have access to your clock.",
            "assistant",
            "tree-abstain",
        ),
    ]
    train = tmp_path / "behavior-train.parquet"
    validation = tmp_path / "behavior-validation.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, train)
    pq.write_table(table.slice(0, 0), validation)
    hashes = {
        "train": verifier.sha256_file(train),
        "validation": verifier.sha256_file(validation),
    }
    return {
        "dataset_id": "fixture/oasst2-behavior",
        "dataset_revision": "fixture-v1",
        "content_sha256": kernel.stable_hash(hashes),
        "license_spdx": "Apache-2.0",
        "files": {
            "train": {"path": str(train), "content_sha256": hashes["train"]},
            "validation": {
                "path": str(validation),
                "content_sha256": hashes["validation"],
            },
        },
        "records_by_split": {
            "private_train": 0,
            "private_dev": 0,
            "private_eval": 0,
        },
        "explicit_behavior_records_by_split": {
            "private_train": {"CLARIFY": 1, "ABSTAIN": 1},
            "private_dev": {"CLARIFY": 0, "ABSTAIN": 0},
            "private_eval": {"CLARIFY": 0, "ABSTAIN": 0},
        },
        "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
        "minimum_quality": 0.5,
        "maximum_label_values": {
            "spam": 0.5,
            "lang_mismatch": 0.5,
            "pii": 0.5,
            "not_appropriate": 0.5,
        },
        "maximum_current_characters": 1024,
        "maximum_response_characters": 1024,
        "maximum_context_characters": 2048,
        "maximum_compiled_context_bytes": 4096,
        "minimum_prior_turns": 0,
        "maximum_prior_turns": 4,
        "required_valid_realization_ranks": [0, 1],
    }


def test_oasst_behavior_stratum_replays_and_rejects_disposition_forgery(
    tmp_path: Path,
) -> None:
    source = write_oasst_behavior_fixture(tmp_path)
    rows, rejects = producer.load_oasst_behavior_candidates(source)
    selected = producer.select_oasst_behavior(
        rows, source["explicit_behavior_records_by_split"]
    )
    independent = verifier.independent_oasst_behavior_assignments(source)

    assert not rejects["bounded_context_or_response_length"]
    assert {row["disposition"] for row in selected["private_train"]} == {
        "CLARIFY",
        "ABSTAIN",
    }
    assert len(independent) == 2
    for row in selected["private_train"]:
        candidate = producer.oasst_behavior_record(
            row,
            split="private_train",
            source=source,
            producer_sha256="sha256:" + "1" * 64,
        )
        expected = independent[candidate["provenance"]["source_id"]]
        receipt = verifier.verify_oasst_behavior_record(candidate, source, expected)
        assert receipt["disposition"] == row["disposition"]
        forged = json.loads(json.dumps(candidate))
        forged["answer_packet"]["decision"]["disposition"] = "ANSWER"
        with pytest.raises(ValueError, match="behavior answer replay mismatch"):
            verifier.verify_oasst_behavior_record(forged, source, expected)


def write_masc_fixture(root: Path) -> Path:
    base = root / "data" / "written" / "fixture" / "sample"
    base.parent.mkdir(parents=True)
    Path(str(base) + ".txt").write_text("Alice runs.", encoding="utf-8")
    Path(str(base) + "-seg.xml").write_text(
        f'''<graph xmlns="{verifier.GRAF[1:-1]}">
        <region xml:id="r1" anchors="0 5"/><region xml:id="r2" anchors="6 10"/>
        </graph>''',
        encoding="utf-8",
    )
    Path(str(base) + "-fntok.xml").write_text(
        f'''<graph xmlns="{verifier.GRAF[1:-1]}">
        <node xml:id="t1"><link targets="r1"/></node>
        <node xml:id="t2"><link targets="r2"/></node>
        </graph>''',
        encoding="utf-8",
    )
    Path(str(base) + "-penn.xml").write_text(
        f'''<graph xmlns="{verifier.GRAF[1:-1]}">
        <node xml:id="p1"><link targets="r1"/></node>
        <node xml:id="p2"><link targets="r2"/></node>
        </graph>''',
        encoding="utf-8",
    )
    Path(str(base) + "-ne.xml").write_text(
        f'''<graph xmlns="{verifier.GRAF[1:-1]}">
        <node xml:id="e1"/><a label="person" ref="e1"><fs><f name="type" value="person"/></fs></a>
        <edge from="e1" to="p1"/>
        </graph>''',
        encoding="utf-8",
    )
    fn = Path(str(base) + "-fn.xml")
    fn.write_text(
        f'''<graph xmlns="{verifier.GRAF[1:-1]}">
        <node xml:id="s"/><a label="sentence" ref="s"><fs><f name="ID" value="1"/></fs></a>
        <node xml:id="a"/><a label="annotationSet" ref="a"><fs>
          <f name="ID" value="2"/><f name="frameName" value="Self_motion"/>
          <f name="luName" value="run.v"/><f name="status" value="MANUAL"/>
        </fs></a>
        <node xml:id="t1"/><a label="FE" ref="t1"><fs><f name="FE" value="Self_mover"/></fs></a>
        <node xml:id="t2"/><a label="Target" ref="t2"><fs/></a>
        <edge from="s" to="a"/><edge from="a" to="t1"/><edge from="a" to="t2"/>
        </graph>''',
        encoding="utf-8",
    )
    return fn


def test_independent_masc_replay_binds_manual_frame_and_roles(tmp_path: Path) -> None:
    fn = write_masc_fixture(tmp_path)
    rows = verifier.independent_masc_document(fn, tmp_path / "data")
    assert len(rows) == 1
    annotation = rows[0]["annotation"]
    assert annotation["frame_name"] == "Self_motion"
    assert annotation["frame_elements"][0]["role"] == "Self_mover"
    assert annotation["protected_spans"] == [
        {
            "start": 0,
            "end": 5,
            "object_type": "PERSON",
            "copy_policy": "EXACT",
            "source_label": "person",
            "source_features": {"type": "person"},
            "text": "Alice",
        }
    ]
    source = {
        "dataset_id": "fixture/masc",
        "dataset_revision": "fixture-v1",
        "content_sha256": "sha256:" + "2" * 64,
        "license_spdx": "LicenseRef-MASC-Unrestricted",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        ],
    }
    candidate = producer.masc_record(
        rows[0],
        split="private_train",
        source=source,
        producer_sha256="sha256:" + "1" * 64,
    )
    expected = {**rows[0], "split": "private_train"}
    verifier.verify_masc_record(candidate, source, expected)
    residual = candidate["kernel_packet"]["residual"]
    assert residual["segment_frame"] == {
        "frame_name": "Self_motion",
        "lexical_unit": "run.v",
        "target_spans": [[6, 10]],
        "frame_roles": ["SELF_MOVER"],
    }
    assert {row["tag"] for row in residual["token_tags"]} == {
        "ENTITY:PERSON",
        "FRAME_ROLE:SELF_MOVER",
        "FRAME_TARGET:SELF_MOTION",
    }
    assert candidate["residual_supervision"]["labels_by_channel"] == {
        "interaction": 0,
        "segment": 1,
        "token": 2,
        "exact": 3,
    }
    value = candidate["kernel_packet"]["program"]["nodes"][0]["arguments"][0]["value"]
    assert value == {"type": "handle", "value": "@E1"}
    candidate["kernel_packet"]["program"]["nodes"][0]["operator"] = "WRONG"
    with pytest.raises(ValueError, match="kernel program replay mismatch"):
        verifier.verify_masc_record(candidate, source, expected)


def test_independent_masc_replay_rejects_forged_residual_annotation(tmp_path: Path) -> None:
    fn = write_masc_fixture(tmp_path)
    row = verifier.independent_masc_document(fn, tmp_path / "data")[0]
    source = {
        "dataset_id": "fixture/masc",
        "dataset_revision": "fixture-v1",
        "content_sha256": "sha256:" + "2" * 64,
        "license_spdx": "LicenseRef-MASC-Unrestricted",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        ],
    }
    candidate = producer.masc_record(
        row,
        split="private_train",
        source=source,
        producer_sha256="sha256:" + "1" * 64,
    )
    expected = {**row, "split": "private_train"}
    candidate["kernel_packet"]["residual"]["token_tags"][0]["tag"] = "FORGED"
    with pytest.raises(ValueError, match="token residual replay mismatch"):
        verifier.verify_masc_record(candidate, source, expected)


def test_independent_masc_replay_rejects_forged_protected_span(tmp_path: Path) -> None:
    fn = write_masc_fixture(tmp_path)
    row = verifier.independent_masc_document(fn, tmp_path / "data")[0]
    source = {
        "dataset_id": "fixture/masc",
        "dataset_revision": "fixture-v1",
        "content_sha256": "sha256:" + "2" * 64,
        "license_spdx": "LicenseRef-MASC-Unrestricted",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        ],
    }
    candidate = producer.masc_record(
        row,
        split="private_train",
        source=source,
        producer_sha256="sha256:" + "1" * 64,
    )
    expected = {
        **row,
        "annotation": json.loads(json.dumps(row["annotation"])),
        "split": "private_train",
    }
    candidate["source_annotation"]["protected_spans"][0]["object_type"] = "PLACE"
    with pytest.raises(ValueError, match="annotation replay mismatch"):
        verifier.verify_masc_record(candidate, source, expected)


def test_independent_masc_replay_binds_and_rejects_forged_interaction_state(
    tmp_path: Path,
) -> None:
    fn = write_masc_fixture(tmp_path)
    row = verifier.independent_masc_document(fn, tmp_path / "data")[0]
    source = {
        "dataset_id": "fixture/masc",
        "dataset_revision": "fixture-v1",
        "content_sha256": "sha256:" + "2" * 64,
        "license_spdx": "LicenseRef-MASC-Unrestricted",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        ],
    }
    interaction = {
        "document_id": row["annotation"]["document_id"],
        "sentence_node": "prior-sentence",
        "annotation_set_node": "prior-frame",
        "frame_name": "Prior_frame",
        "lexical_unit": "prior.v",
        "source_annotation_sha256": "sha256:" + "3" * 64,
    }
    expected = {
        **row,
        "split": "private_train",
        "interaction_annotation": interaction,
    }
    candidate = producer.masc_record(
        row,
        split="private_train",
        source=source,
        producer_sha256="sha256:" + "1" * 64,
        interaction_annotation=interaction,
    )
    receipt = verifier.verify_masc_record(candidate, source, expected)
    assert receipt["interaction_bound"] is True
    assert candidate["residual_supervision"]["labels_by_channel"]["interaction"] == 1

    forged_state = json.loads(json.dumps(candidate))
    forged_state["hrl_state"]["segments"]["previous_turn"]["entries"][
        "frame_name"
    ]["value"] = "Forged"
    with pytest.raises(ValueError, match="VCM interaction state replay mismatch"):
        verifier.verify_masc_record(forged_state, source, expected)

    forged_label = json.loads(json.dumps(candidate))
    forged_label["residual_supervision"]["labels_by_channel"]["interaction"] = 0
    with pytest.raises(ValueError, match="interaction residual label mismatch"):
        verifier.verify_masc_record(forged_label, source, expected)


def test_residual_supervision_matches_packet_mechanics() -> None:
    packet = {
        "residual": {
            "fidelity": "exact",
            "segment_frame": {},
            "token_tags": [],
            "exact_object_handles": ["@E1"],
        }
    }
    state = memory.create_hierarchical_residual_state(
        "fixture", scope=producer.scope("fixture")
    )
    supervision = producer.residual_supervision(
        "fixture", packet=packet, hrl_state=state
    )
    assert supervision["record_fidelity_label"] == 3
    assert supervision["labels_by_channel"] == {
        "interaction": 0,
        "segment": 0,
        "token": 0,
        "exact": 3,
    }
