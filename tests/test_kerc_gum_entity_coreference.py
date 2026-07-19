from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kerc_gum_entity_coreference import (  # noqa: E402
    reconstruct_gum_entity_coreference,
    selected_source_digest,
)
from kerc_gum_entity_coreference_verify import (  # noqa: E402
    independently_reconstruct_gum_entity_coreference,
)
from kerc_semantic_corpus import gum_entity_coreference_record  # noqa: E402
from kerc_semantic_corpus_verify import (  # noqa: E402
    verify_gum_entity_coreference_record,
)
from kernel_english_protocol import (  # noqa: E402
    compiler_training_io,
    parse_learned_compiler_output,
)


TSV = """#FORMAT=WebAnno TSV 3.2
#T_SP=webanno.custom.Referent|entity|infstat|salience|identity|centering
#T_RL=webanno.custom.Coref|type|BT_webanno.custom.Referent

#Text=Alice saw Bob .
1-1\t0-5\tAlice\tperson[1]\tnew[1]\tnnnnn[1]\tAlice[1]\tcf1[1]\tcoref\t2-1[2_1]\t
1-2\t6-9\tsaw\t_\t_\t_\t_\t_\t_\t_\t
1-3\t10-13\tBob\tperson[3]\tnew[3]\tnnnnn[3]\t_\tcf2[3]\tbridge:entity-associative\t2-3[4_3]\t
1-4\t14-15\t.\t_\t_\t_\t_\t_\t_\t_\t

#Text=She joined Acme .
2-1\t16-19\tShe\tperson[2]\tgiv:act[2]\tsssss[2]\tAlice[2]\tcf1[2]\t_\t_\t
2-2\t20-26\tjoined\t_\t_\t_\t_\t_\t_\t_\t
2-3\t27-31\tAcme\torganization[4]\tnew[4]\tnnnnn[4]\tAcme[4]\tcf2[4]\t_\t_\t
2-4\t32-33\t.\t_\t_\t_\t_\t_\t_\t_\t
"""

CONLLU = """# newdoc id = GUM_academic_fixture
# global.Entity = GRP-etype-infstat-salience-centering-minspan-link-identity
# sent_id = GUM_academic_fixture-1
1\tAlice\tAlice\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(1-person-new-nnnnn-cf1-1-coref)
2\tsaw\tsee\tVERB\tVBD\t_\t0\troot\t_\t_
3\tBob\tBob\tPROPN\tNNP\t_\t2\tobj\t_\tEntity=(2-person-new-nnnnn-cf2-1-sgl)
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_

# sent_id = GUM_academic_fixture-2
1\tShe\tshe\tPRON\tPRP\t_\t2\tnsubj\t_\tEntity=(1-person-giv:act-sssss-cf1-1-ana)
2\tjoined\tjoin\tVERB\tVBD\t_\t0\troot\t_\t_
3\tAcme\tAcme\tPROPN\tNNP\t_\t2\tobj\t_\tEntity=(3-organization-new-nnnnn-cf2-1-sgl)
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
"""

CONLL = """# begin document GUM_academic_fixture
0\tAlice\t(person-1)
1\tsaw\t_
2\tBob\t(person-2)
3\t.\t_
4\tShe\t(person-1)
5\tjoined\t_
6\tAcme\t(organization-3)
7\t.\t_
"""


def fixture(tmp_path: Path) -> tuple[dict, dict]:
    root = tmp_path / "gum"
    (root / "xml").mkdir(parents=True)
    (root / "coref" / "gum" / "tsv").mkdir(parents=True)
    (root / "coref" / "gum" / "conll").mkdir(parents=True)
    (root / "dep").mkdir(parents=True)
    (root / "LICENSE.md").write_text("fixture license", encoding="utf-8")
    document = "GUM_academic_fixture"
    (root / "xml" / f"{document}.xml").write_text(
        f'<text id="{document}" partition="train" type="academic" />\n',
        encoding="utf-8",
    )
    (root / "coref" / "gum" / "tsv" / f"{document}.tsv").write_text(
        TSV, encoding="utf-8"
    )
    (root / "coref" / "gum" / "conll" / f"{document}.conll").write_text(
        CONLL, encoding="utf-8"
    )
    (root / "dep" / f"{document}.conllu").write_text(CONLLU, encoding="utf-8")
    arguments = {
        "source_root": root,
        "allowed_genre_licenses": {"academic": "CC-BY-4.0"},
        "private_dev_documents": set(),
        "private_eval_documents": set(),
        "expected_selected_source_sha256": selected_source_digest(root, [document]),
        "maximum_characters": 2048,
    }
    source = {
        "dataset_id": "fixture/gum",
        "dataset_revision": "fixture-v1",
        "source_url": "https://example.test/gum",
        "license_evidence_url": "https://example.test/license",
        "content_sha256": "sha256:" + "8" * 64,
        "license_spdx": "LicenseRef-GUM-Permissive-Subset",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        ],
    }
    return arguments, source


def test_independent_gum_coreference_parsers_preserve_complete_topology(
    tmp_path: Path,
) -> None:
    arguments, _source = fixture(tmp_path)
    produced, producer_audit = reconstruct_gum_entity_coreference(**arguments)
    verified, verifier_audit = independently_reconstruct_gum_entity_coreference(
        **arguments
    )

    assert produced == verified
    assert producer_audit["counts_by_split"]["private_train"] == {
        "documents": 1,
        "mentions": 4,
        "components": 3,
        "records": 2,
        "identity_records": 1,
        "bridge_records": 1,
    }
    assert verifier_audit["cross_format_topology_by_split"]["private_train"] == {
        "documents_agreeing": 1,
        "component_membership_documents_agreeing": 1,
        "mentions": 4,
        "components": 3,
    }
    identity = next(
        row
        for row in produced["private_train"]
        if row["annotation"]["record_kind"] == "identity_component"
    )
    assert identity["annotation"]["groups"][0]["mention_ids"] == ["1", "2"]
    bridge = next(
        row
        for row in produced["private_train"]
        if row["annotation"]["record_kind"] == "bridge_relation"
    )
    assert len(bridge["annotation"]["groups"]) == 2
    assert bridge["annotation"]["relations"][0]["relation_type"].startswith("bridge:")


def test_conllu_topology_and_component_mutations_fail_closed(tmp_path: Path) -> None:
    arguments, _source = fixture(tmp_path)
    conllu = arguments["source_root"] / "dep" / "GUM_academic_fixture.conllu"
    conllu.write_text(
        CONLLU.replace("(3-organization", "(2-organization"), encoding="utf-8"
    )
    arguments["expected_selected_source_sha256"] = selected_source_digest(
        arguments["source_root"], ["GUM_academic_fixture"]
    )
    with pytest.raises(ValueError, match="TSV/CoNLL-U topology disagreement"):
        independently_reconstruct_gum_entity_coreference(**arguments)

    arguments, _source = fixture(tmp_path / "membership")
    conll = (
        arguments["source_root"]
        / "coref"
        / "gum"
        / "conll"
        / "GUM_academic_fixture.conll"
    )
    conll.write_text(
        CONLL.replace("4\tShe\t(person-1)", "4\tShe\t(person-2)"),
        encoding="utf-8",
    )
    arguments["expected_selected_source_sha256"] = selected_source_digest(
        arguments["source_root"], ["GUM_academic_fixture"]
    )
    with pytest.raises(ValueError, match="component membership disagreement"):
        independently_reconstruct_gum_entity_coreference(**arguments)

    arguments, _source = fixture(tmp_path / "partial")
    tsv = (
        arguments["source_root"] / "coref" / "gum" / "tsv" / "GUM_academic_fixture.tsv"
    )
    tsv.write_text(TSV.replace("2-1[2_1]", "2-1[99_1]"), encoding="utf-8")
    arguments["expected_selected_source_sha256"] = selected_source_digest(
        arguments["source_root"], ["GUM_academic_fixture"]
    )
    with pytest.raises(ValueError, match="unknown mention"):
        reconstruct_gum_entity_coreference(**arguments)


def test_coreference_record_binds_capsules_graph_and_independent_replay(
    tmp_path: Path,
) -> None:
    arguments, source = fixture(tmp_path)
    produced, _ = reconstruct_gum_entity_coreference(**arguments)
    verified, _ = independently_reconstruct_gum_entity_coreference(**arguments)
    row = next(
        item
        for item in produced["private_train"]
        if item["annotation"]["record_kind"] == "identity_component"
    )
    expected = next(
        item
        for item in verified["private_train"]
        if item["source_id"] == row["source_id"]
    )
    record = gum_entity_coreference_record(
        row, source=source, producer_sha256="sha256:" + "1" * 64
    )
    replay = verify_gum_entity_coreference_record(record, source, expected)
    assert replay["source_declared_only"] is True
    assert record["kernel_packet"]["concept_capsules"]["@C0"]["mention_count"] == 2
    assert record["semantic_supervision"]["derived_view_unique_source_credit"] == 0
    compiler_prompt, compiler_target = compiler_training_io(
        packet=record["kernel_packet"],
        source_text=record["source_text"],
        hrl_state=record["hrl_state"],
    )
    assert compiler_prompt["concept_capsules"] == {}
    assert compiler_target["concept_capsules"]["@C0"] == {
        "entity_types": ["person"],
        "source_identity_values": ["Alice"],
        "mention_count": 2,
    }
    assert "stable_identity" not in compiler_target["concept_capsules"]["@C0"]
    assert "provenance" not in compiler_target["concept_capsules"]["@C0"]
    learned = parse_learned_compiler_output(
        json.dumps(compiler_target),
        protected_objects=record["kernel_packet"]["protected_objects"],
        concept_capsules={},
        source_character_length=len(record["source_text"]),
        source=record["source_text"],
        hrl_state=record["hrl_state"],
    )
    assert learned["generated_concept_capsules"]["@C0"]["stable_identity"] == (
        "local.concept.0"
    )
    assert learned["generated_concept_capsules"]["@C0"]["provenance"] == {
        "source": "learned_compiler_output_v1",
        "scope": "packet_local",
        "registry_promotion_allowed": False,
    }
    assert learned["canonical_program"] == record["kernel_packet"]["program"]

    forged = json.loads(json.dumps(record))
    forged["kernel_packet"]["concept_capsules"]["@C0"]["stable_identity"] = (
        "gum.entity.forged.1"
    )
    with pytest.raises(ValueError, match="concept capsule mismatch"):
        verify_gum_entity_coreference_record(forged, source, expected)

    forged = json.loads(json.dumps(record))
    relation = next(
        node
        for node in forged["kernel_packet"]["program"]["nodes"]
        if node["operator"].startswith("ENTITY_RELATION_")
    )
    relation["arguments"][0]["value"] = relation["arguments"][1]["value"]
    with pytest.raises(ValueError, match="relation graph mismatch"):
        verify_gum_entity_coreference_record(forged, source, expected)
