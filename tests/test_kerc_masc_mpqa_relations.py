from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kerc_masc_mpqa_relations import reconstruct_mpqa_relation_chains  # noqa: E402
from kerc_masc_mpqa_relations_verify import (  # noqa: E402
    independently_reconstruct_mpqa_relation_chains,
)


SOURCE = "Alice strongly likes reliable systems."
ANNOTATIONS = """# fixture
1\t0,5\tstring\tGATE_agent\t nested-source="w" id="alice"
2\t21,37\tstring\tGATE_target\t id="systems"
3\t15,20\tstring\tGATE_attitude\t target-link="systems" intensity="high" id="approval" attitude-type="sentiment-pos"
4\t6,20\tstring\tGATE_direct-subjective\t nested-source="w,alice" polarity="positive" attitude-link="approval"
"""


def fixture(tmp_path: Path, annotations: str = ANNOTATIONS, source: str = SOURCE) -> dict:
    root = tmp_path / "mpqa"
    document = root / "example"
    texts = root / "texts"
    document.mkdir(parents=True, exist_ok=True)
    texts.mkdir(parents=True, exist_ok=True)
    (document / "gateman.mpqa.lre.2.0").write_text(annotations, encoding="utf-8")
    (texts / "example").write_text(source, encoding="utf-8")
    return {
        "original_mpqa_root": root,
        "private_dev_documents": {"example"},
        "private_eval_documents": set(),
        "maximum_characters": 128,
    }


def test_dual_mpqa_relation_reconstruction_is_identical_and_source_bound(
    tmp_path: Path,
) -> None:
    arguments = fixture(tmp_path)
    produced, producer_audit = reconstruct_mpqa_relation_chains(**arguments)
    verified, verifier_audit = independently_reconstruct_mpqa_relation_chains(**arguments)

    assert produced == verified
    assert producer_audit["admitted_relation_count"] == 1
    assert verifier_audit["admitted_relation_count"] == 1
    assert producer_audit["record_count_by_split"] == {
        "private_train": 0,
        "private_dev": 1,
        "private_eval": 0,
    }
    annotation = produced["private_dev"][0]["annotation"]
    assert annotation["complete_relation_alignment"] is True
    assert annotation["source_compaction_contract"] == (
        "uniform_radius_relation_member_source_windows_v1"
    )
    assert annotation["original_source_file_sha256"] == (
        "sha256:" + hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    )
    assert annotation["expression"]["source_text"] == "strongly likes"
    assert [row["annotation_id"] for row in annotation["source_chain"]] == [
        "w",
        "alice",
    ]
    assert annotation["attitudes"][0]["annotation_id"] == "approval"
    assert annotation["attitudes"][0]["targets"][0]["source_text"] == (
        "reliable systems"
    )
    assert {edge["edge_type"] for edge in annotation["edges"]} == {
        "nested_source_member",
        "attitude_link",
        "target_link",
    }
    assert producer_audit["partial_relation_admission_count"] == 0
    assert producer_audit["inferred_relation_count"] == 0


def test_missing_or_ambiguous_member_rejects_the_complete_relation(tmp_path: Path) -> None:
    missing_target = ANNOTATIONS.replace('id="systems"', 'id="other"', 1)
    arguments = fixture(tmp_path, missing_target)
    produced, producer_audit = reconstruct_mpqa_relation_chains(**arguments)
    verified, verifier_audit = independently_reconstruct_mpqa_relation_chains(**arguments)

    assert produced == verified == {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    assert producer_audit["admitted_relation_count"] == 0
    assert verifier_audit["admitted_relation_count"] == 0
    assert producer_audit["rejection_reason_counts"] == {
        "ambiguous_or_missing_target": 1
    }

    duplicate = ANNOTATIONS + (
        '5\t21,29\tstring\tGATE_target\t id="systems"\n'
    )
    arguments = fixture(tmp_path, duplicate)
    produced, producer_audit = reconstruct_mpqa_relation_chains(**arguments)
    verified, verifier_audit = independently_reconstruct_mpqa_relation_chains(**arguments)
    assert produced == verified == {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    assert producer_audit["rejection_reason_counts"] == {
        "ambiguous_or_missing_target": 1
    }
    assert verifier_audit["partial_relation_admission_count"] == 0


def test_repeated_source_identity_remains_an_ordered_relation_position(
    tmp_path: Path,
) -> None:
    repeated_source = ANNOTATIONS.replace(
        'nested-source="w,alice"', 'nested-source="w,alice,alice"'
    )
    arguments = fixture(tmp_path, repeated_source)
    produced, producer_audit = reconstruct_mpqa_relation_chains(**arguments)
    verified, verifier_audit = independently_reconstruct_mpqa_relation_chains(**arguments)

    assert produced == verified
    annotation = produced["private_dev"][0]["annotation"]
    assert [member["annotation_id"] for member in annotation["source_chain"]] == [
        "w",
        "alice",
        "alice",
    ]
    source_edges = [
        edge
        for edge in annotation["edges"]
        if edge["edge_type"] == "nested_source_member"
    ]
    assert [edge["order"] for edge in source_edges] == [0, 1, 2]
    assert source_edges[1]["to"] == source_edges[2]["to"]
    assert producer_audit["admitted_source_member_count"] == 3
    assert verifier_audit["partial_relation_admission_count"] == 0


def test_large_complete_relation_is_compacted_without_dropping_members(
    tmp_path: Path,
) -> None:
    chunks: list[str] = []
    annotations: list[str] = []
    source_ids: list[str] = ["w"]
    target_ids: list[str] = []
    cursor = 0
    line_id = 1
    for index in range(24):
        if chunks:
            chunks.append(" " * 40)
            cursor += 40
        text = f"member{index}"
        start, end = cursor, cursor + len(text)
        chunks.append(text)
        cursor = end
        if index < 12:
            identity = f"agent{index}"
            source_ids.append(identity)
            annotations.append(
                f'{line_id}\t{start},{end}\tstring\tGATE_agent\t id="{identity}"'
            )
        else:
            identity = f"target{index}"
            target_ids.append(identity)
            annotations.append(
                f'{line_id}\t{start},{end}\tstring\tGATE_target\t id="{identity}"'
            )
        line_id += 1
    source = "".join(chunks)
    expression_start = source.index("member0")
    expression_end = expression_start + len("member0")
    annotations.append(
        f'{line_id}\t{expression_start},{expression_end}\tstring\tGATE_attitude\t '
        f'target-link="{",".join(target_ids)}" id="attitude" attitude-type="sentiment-pos"'
    )
    line_id += 1
    annotations.append(
        f'{line_id}\t{expression_start},{expression_end}\tstring\tGATE_direct-subjective\t '
        f'nested-source="{",".join(source_ids)}" attitude-link="attitude"'
    )
    arguments = fixture(
        tmp_path,
        "\n".join(annotations) + "\n",
        source,
    )
    arguments["maximum_characters"] = 256

    produced, producer_audit = reconstruct_mpqa_relation_chains(**arguments)
    verified, verifier_audit = independently_reconstruct_mpqa_relation_chains(**arguments)

    assert produced == verified
    annotation = produced["private_dev"][0]["annotation"]
    assert len(annotation["source_text"]) <= 256
    assert len(annotation["source_chain"]) == 13
    assert len(annotation["attitudes"][0]["targets"]) == 12
    assert all(
        node["target_spans"]
        for node in annotation["source_chain"]
        if node["annotation_id"] != "w"
    )
    assert all(node["target_spans"] for node in annotation["attitudes"][0]["targets"])
    assert producer_audit["partial_relation_admission_count"] == 0
    assert verifier_audit["partial_relation_admission_count"] == 0
