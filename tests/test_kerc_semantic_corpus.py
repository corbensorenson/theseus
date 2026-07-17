from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_semantic_corpus as producer  # noqa: E402
import kerc_semantic_corpus_verify as verifier  # noqa: E402


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
    candidate["kernel_packet"]["program"]["nodes"][0]["operator"] = "WRONG"
    with pytest.raises(ValueError, match="kernel program replay mismatch"):
        verifier.verify_masc_record(candidate, source, expected)


def test_residual_supervision_matches_packet_mechanics() -> None:
    packet = {
        "residual": {
            "fidelity": "exact",
            "segment_frame": {},
            "token_tags": [],
            "exact_object_handles": ["@E1"],
        }
    }
    supervision = producer.residual_supervision("fixture", packet=packet)
    assert supervision["record_fidelity_label"] == 3
    assert supervision["labels_by_channel"] == {
        "interaction": 0,
        "segment": 0,
        "token": 0,
        "exact": 3,
    }
