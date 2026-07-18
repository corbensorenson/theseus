from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kerc_masc_event_coreference import (  # noqa: E402
    reconstruct_event_coreference_groups,
)
from kerc_masc_event_coreference_verify import (  # noqa: E402
    independently_reconstruct_event_coreference_groups,
)


GATE = """<?xml version="1.0" encoding="utf-8"?>
<GateDocument>
<TextWithNodes>Alpha <Node id="6"/>run<Node id="9"/> then <Node id="15"/>run<Node id="18"/>.</TextWithNodes>
<AnnotationSet Name="Other Events">
  <Annotation Id="9" Type="Ignored" StartNode="6" EndNode="9"/>
</AnnotationSet>
<AnnotationSet Name="Repeated Event">
  <Annotation Id="1" Type="Running" StartNode="6" EndNode="9"/>
  <Annotation Id="2" Type="Running Again" StartNode="15" EndNode="18"/>
</AnnotationSet>
</GateDocument>
"""

SENTENCES = """<graph xmlns="http://www.xces.org/ns/GrAF/1.0/">
  <region xml:id="s-r0" anchors="0 19"/>
  <node xml:id="s-n0"><link targets="s-r0"/></node>
  <a xml:id="s-a0" label="s" ref="s-n0"/>
</graph>
"""


def fixture(tmp_path: Path, *, target: str = "Alpha run then run.") -> dict:
    original = tmp_path / "original"
    data = tmp_path / "data" / "doc"
    original.mkdir(parents=True)
    data.mkdir(parents=True)
    (original / "fixture.xml").write_text(GATE, encoding="utf-8")
    (data / "example.txt").write_text(target, encoding="utf-8")
    (data / "example-s.xml").write_text(SENTENCES, encoding="utf-8")
    return {
        "original_event_root": original,
        "data_root": tmp_path / "data",
        "document_map": {"fixture.xml": "doc/example"},
        "private_dev_documents": {"doc/example"},
        "private_eval_documents": set(),
        "maximum_characters": 128,
    }


def test_dual_event_coreference_reconstruction_is_identical_and_source_bound(
    tmp_path: Path,
) -> None:
    arguments = fixture(tmp_path)
    produced, producer_audit = reconstruct_event_coreference_groups(**arguments)
    verified, verifier_audit = independently_reconstruct_event_coreference_groups(
        **arguments
    )

    assert produced == verified
    assert producer_audit["observed_group_count"] == 1
    assert verifier_audit["observed_group_count"] == 1
    assert producer_audit["admitted_group_count"] == 1
    assert producer_audit["record_count_by_split"] == {
        "private_train": 0,
        "private_dev": 1,
        "private_eval": 0,
    }
    row = produced["private_dev"][0]
    annotation = row["annotation"]
    assert annotation["annotation_set_name"] == "Repeated Event"
    assert annotation["complete_group_alignment"] is True
    assert len(annotation["mentions"]) == 2
    assert annotation["source_compaction_contract"] == (
        "uniform_radius_mention_centered_source_windows_v1"
    )
    assert annotation["maximum_source_characters"] == 128
    assert annotation["distributed_document_sha256"] == (
        "sha256:" + hashlib.sha256(b"Alpha run then run.").hexdigest()
    )
    assert annotation["sentence_graph_sha256"] == (
        "sha256:" + hashlib.sha256(SENTENCES.encode("utf-8")).hexdigest()
    )
    assert [mention["source_text"] for mention in annotation["mentions"]] == [
        "run",
        "run",
    ]
    assert producer_audit["cooccurrence_inferred_relation_count"] == 0
    assert producer_audit["partial_group_admission_count"] == 0


def test_any_failed_mention_rejects_the_complete_group(tmp_path: Path) -> None:
    arguments = fixture(tmp_path, target="Alpha run then stop.")
    produced, producer_audit = reconstruct_event_coreference_groups(**arguments)
    verified, verifier_audit = independently_reconstruct_event_coreference_groups(
        **arguments
    )

    assert produced == verified == {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    assert producer_audit["observed_group_count"] == 1
    assert verifier_audit["observed_group_count"] == 1
    assert producer_audit["admitted_group_count"] == 0
    assert verifier_audit["admitted_group_count"] == 0
    assert producer_audit["partial_group_admission_count"] == 0
    assert verifier_audit["partial_group_admission_count"] == 0


def test_large_complete_group_is_compacted_without_dropping_mentions(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    data = tmp_path / "data" / "doc"
    original.mkdir(parents=True)
    data.mkdir(parents=True)
    text_parts: list[str] = []
    gate_parts: list[str] = []
    annotations: list[str] = []
    sentence_regions: list[str] = []
    cursor = 0
    for index in range(40):
        prefix = f"Unique context {index} before the observed action "
        mention = f"runs{index}"
        suffix = f" and after it contains independently useful context {index}."
        sentence = prefix + mention + suffix
        if text_parts:
            text_parts.append(" ")
            gate_parts.append(" ")
            cursor += 1
        sentence_start = cursor
        gate_parts.append(prefix)
        cursor += len(prefix)
        start_node = str(cursor)
        gate_parts.append(f'<Node id="{start_node}"/>')
        gate_parts.append(mention)
        cursor += len(mention)
        end_node = str(cursor)
        gate_parts.append(f'<Node id="{end_node}"/>')
        gate_parts.append(suffix)
        cursor += len(suffix)
        text_parts.append(sentence)
        annotations.append(
            f'<Annotation Id="{index}" Type="Running" '
            f'StartNode="{start_node}" EndNode="{end_node}"/>'
        )
        sentence_regions.append(
            f'<region xml:id="s-r{index}" anchors="{sentence_start} {cursor}"/>'
        )
    target = "".join(text_parts)
    gate = (
        "<GateDocument><TextWithNodes>"
        + "".join(gate_parts)
        + "</TextWithNodes><AnnotationSet Name=\"Long Event\">"
        + "".join(annotations)
        + "</AnnotationSet></GateDocument>"
    )
    sentences = (
        '<graph xmlns="http://www.xces.org/ns/GrAF/1.0/">'
        + "".join(sentence_regions)
        + "</graph>"
    )
    (original / "fixture.xml").write_text(gate, encoding="utf-8")
    (data / "example.txt").write_text(target, encoding="utf-8")
    (data / "example-s.xml").write_text(sentences, encoding="utf-8")
    arguments = {
        "original_event_root": original,
        "data_root": tmp_path / "data",
        "document_map": {"fixture.xml": "doc/example"},
        "private_dev_documents": set(),
        "private_eval_documents": {"doc/example"},
        "maximum_characters": 512,
    }

    produced, producer_audit = reconstruct_event_coreference_groups(**arguments)
    verified, verifier_audit = independently_reconstruct_event_coreference_groups(
        **arguments
    )

    assert produced == verified
    row = produced["private_eval"][0]
    assert len(row["annotation"]["mentions"]) == 40
    assert len(row["annotation"]["source_text"]) <= 512
    assert row["annotation"]["uniform_context_radius_characters"] >= 1
    assert producer_audit["partial_group_admission_count"] == 0
    assert verifier_audit["partial_group_admission_count"] == 0
