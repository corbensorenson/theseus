from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kerc_gum_discourse_relations import (  # noqa: E402
    reconstruct_gum_discourse_relations,
)
from kerc_gum_discourse_relations_verify import (  # noqa: E402
    independently_reconstruct_gum_discourse_relations,
)
from kerc_semantic_corpus import (  # noqa: E402
    gum_discourse_record,
    gum_scoped_supervision_audit,
)
from kerc_semantic_corpus_verify import (  # noqa: E402
    independently_audit_gum_scoped_supervision,
    verify_gum_discourse_record,
)
from kernel_english_protocol import (  # noqa: E402
    KernelProtocolFault,
    _normalize_segment_frame,
    compiler_training_io,
)


LICENSES = {"academic": "CC-BY-4.0"}
RSD = """1\tRoot statement\t0\t_\t_\tpos1=NN|heading\t0\tROOT\t_\t_
2\tThis follows because evidence exists\t1\t_\t_\tpos1=DT|sid=2\t1\tcausal-result_r\t3:adversative-contrast:0:0:orphan-but-2-gold\tdm-so-1-gold
3\tA contrasting observation\t1\t_\t_\tpos1=DT|sid=3\t1\tjoint-list_m\t_\tdm-and-3-gold
"""

SCOPED_RELATIONS = {
    "contingency-condition_r": ("SCOPE_CONDITION", ["ANTECEDENT", "CONSEQUENT"]),
    "causal-cause_r": ("SCOPE_CONSEQUENCE", ["CAUSE", "RESULT"]),
    "causal-result_r": ("SCOPE_CONSEQUENCE", ["RESULT", "CAUSE"]),
    "explanation-evidence_r": ("SCOPE_EXPLANATION", ["EVIDENCE", "CLAIM"]),
    "adversative-contrast_m": ("SCOPE_CONTRAST", ["LEFT", "RIGHT"]),
    "joint-disjunction_m": ("SCOPE_ALTERNATION", ["MEMBER", "MEMBER"]),
    "joint-sequence_m": ("SCOPE_CONTINUATION", ["PREVIOUS", "NEXT"]),
}


def scoped_row(relation: str, *, split: str = "private_train") -> dict:
    source_text = "Earlier evidence\nLater claim"
    units = [
        {
            "edu_id": 10,
            "text": "Earlier evidence",
            "excerpt_span": [0, 16],
            "tree_depth": 1,
            "features": [],
            "source_row_sha256": "sha256:" + "1" * 64,
        },
        {
            "edu_id": 20,
            "text": "Later claim",
            "excerpt_span": [17, 28],
            "tree_depth": 2,
            "features": [],
            "source_row_sha256": "sha256:" + "2" * 64,
        },
    ]
    annotation = {
        "document_id": "GUM_academic_scope_fixture",
        "anchor_edu_id": 10,
        "genre": "academic",
        "primary_relation": relation,
        "units": units,
        "edges": [
            {
                "edge_kind": "primary",
                "edge_order": 0,
                "child_edu_id": 10,
                "parent_edu_id": 20,
                "relation": relation,
                "raw_depth_fields": [],
                "raw_signal_payload": "",
                "source_annotation_sha256": "sha256:" + "3" * 64,
            }
        ],
    }
    return {
        "split": split,
        "source_text": source_text,
        "source_id": f"gum-erst:scope:{split}:{relation}",
        "source_group": f"gum-document:scope-{split}",
        "license_spdx": "CC-BY-4.0",
        "annotation": annotation,
    }


def scoped_source() -> dict:
    return {
        "dataset_id": "fixture/gum",
        "dataset_revision": "fixture",
        "source_url": "https://example.test/gum",
        "license_evidence_url": "https://example.test/license",
        "content_sha256": "sha256:" + "4" * 64,
        "license_spdx": "LicenseRef-GUM-Permissive-Subset",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        ],
    }


def _xml(document_id: str, partition: str = "train") -> str:
    return (
        f'<text id="{document_id}" partition="{partition}" type="academic" '
        f'sourceURL="https://example.test/{document_id}"><p>fixture</p></text>\n'
    )


def _source_digest(root: Path, documents: list[str]) -> str:
    state = hashlib.sha256()
    paths = [root / "LICENSE.md"]
    for document in sorted(documents):
        paths.extend(
            (
                root / "xml" / f"{document}.xml",
                root / "rst" / "dependencies" / f"{document}.rsd",
            )
        )
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state.update(f"{path.relative_to(root)}\0{digest}\n".encode())
    return "sha256:" + state.hexdigest()


def fixture(tmp_path: Path, *, rsd: str = RSD) -> dict:
    source_root = tmp_path / "gum"
    (source_root / "xml").mkdir(parents=True)
    (source_root / "rst" / "dependencies").mkdir(parents=True)
    (source_root / "LICENSE.md").write_text("fixture license", encoding="utf-8")
    documents = ["GUM_academic_train", "GUM_academic_dev", "GUM_academic_eval"]
    for document in documents:
        (source_root / "xml" / f"{document}.xml").write_text(
            _xml(document), encoding="utf-8"
        )
        (source_root / "rst" / "dependencies" / f"{document}.rsd").write_text(
            rsd, encoding="utf-8"
        )
    # Official heldouts are visible to the parser but must never enter this source set.
    (source_root / "xml" / "GUM_academic_official.xml").write_text(
        _xml("GUM_academic_official", "dev"), encoding="utf-8"
    )
    (source_root / "rst" / "dependencies" / "GUM_academic_official.rsd").write_text(
        RSD, encoding="utf-8"
    )
    return {
        "source_root": source_root,
        "allowed_genre_licenses": LICENSES,
        "private_dev_documents": {"GUM_academic_dev"},
        "private_eval_documents": {"GUM_academic_eval"},
        "expected_selected_source_sha256": _source_digest(source_root, documents),
        "maximum_characters": 2048,
    }


def test_dual_gum_parsers_reconstruct_identical_source_declared_graphs(
    tmp_path: Path,
) -> None:
    arguments = fixture(tmp_path)
    produced, producer_audit = reconstruct_gum_discourse_relations(**arguments)
    verified, verifier_audit = independently_reconstruct_gum_discourse_relations(
        **arguments
    )

    assert produced == verified
    assert producer_audit["record_count_by_split"] == {
        "private_train": 2,
        "private_dev": 2,
        "private_eval": 2,
    }
    assert verifier_audit["official_nontrain_document_admission_count"] == 0
    assert producer_audit["secondary_edge_count_by_split"] == {
        "private_train": 1,
        "private_dev": 1,
        "private_eval": 1,
    }
    relation = next(
        row
        for row in produced["private_dev"]
        if row["annotation"]["primary_relation"] == "causal-result_r"
    )
    assert relation["license_spdx"] == "CC-BY-4.0"
    assert len(relation["annotation"]["units"]) == 3
    assert [edge["relation"] for edge in relation["annotation"]["edges"]] == [
        "causal-result_r",
        "adversative-contrast",
    ]
    assert relation["annotation"]["inferred_relation_count"] == 0


def test_gum_content_license_split_and_edge_mutations_fail_closed(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    path = arguments["source_root"] / "rst" / "dependencies" / "GUM_academic_train.rsd"
    path.write_text(RSD.replace("evidence exists", "evidence changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="selected source content mismatch"):
        reconstruct_gum_discourse_relations(**arguments)
    with pytest.raises(ValueError, match="selected source content mismatch"):
        independently_reconstruct_gum_discourse_relations(**arguments)

    malformed = RSD.replace(
        "3:adversative-contrast:0:0:orphan-but-2-gold",
        "unknown:adversative-contrast",
    )
    arguments = fixture(tmp_path / "malformed", rsd=malformed)
    with pytest.raises(ValueError, match="secondary edge"):
        reconstruct_gum_discourse_relations(**arguments)
    with pytest.raises(ValueError, match="secondary edge"):
        independently_reconstruct_gum_discourse_relations(**arguments)

    arguments = fixture(tmp_path / "overlap")
    arguments["private_eval_documents"] = {"GUM_academic_dev"}
    with pytest.raises(ValueError, match="splits overlap"):
        reconstruct_gum_discourse_relations(**arguments)
    with pytest.raises(ValueError, match="splits overlap"):
        independently_reconstruct_gum_discourse_relations(**arguments)


def test_gum_record_is_typed_replayable_and_rejects_false_nuclearity(
    tmp_path: Path,
) -> None:
    arguments = fixture(tmp_path)
    produced, _ = reconstruct_gum_discourse_relations(**arguments)
    verified, _ = independently_reconstruct_gum_discourse_relations(**arguments)
    row = next(
        item
        for item in produced["private_train"]
        if item["annotation"]["primary_relation"] == "causal-result_r"
    )
    expected = next(
        item for item in verified["private_train"] if item["source_id"] == row["source_id"]
    )
    source = {
        "dataset_id": "fixture/gum",
        "dataset_revision": "fixture",
        "source_url": "https://example.test/gum",
        "license_evidence_url": "https://example.test/license",
        "content_sha256": arguments["expected_selected_source_sha256"],
        "license_spdx": "LicenseRef-GUM-Permissive-Subset",
        "allowed_objectives": [
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        ],
    }
    record = gum_discourse_record(
        row, source=source, producer_sha256="sha256:" + "1" * 64
    )
    replay = verify_gum_discourse_record(record, source, expected)
    assert replay["source_declared_only"] is True
    assert replay["edge_count"] == 2
    assert replay["common_source_binding"] == {
        "dataset_revision": "fixture",
        "license_spdx": "CC-BY-4.0",
        "content_sha256": arguments["expected_selected_source_sha256"],
    }
    assert record["semantic_supervision"]["objective_authority"] == {
        "surface_direct_control_v1": False,
        "surface_to_kernel_program_v1": True,
        "kernel_program_to_answer_packet_v1": True,
        "answer_packet_to_surface_v1": False,
    }
    assert record["semantic_supervision"]["derived_view_unique_source_credit"] == 0
    program_arguments = record["kernel_packet"]["program"]["nodes"][-1]["arguments"]
    answer_arguments = record["answer_packet"]["claims"][0]["arguments"]
    assert {argument["value"]["type"] for argument in program_arguments[:2]} == {
        "node_ref"
    }
    assert {argument["value"]["type"] for argument in answer_arguments[:2]} == {
        "concept"
    }
    assert all(
        str(argument["value"]["value"]).startswith("erst.edu.gum_")
        for argument in answer_arguments[:2]
    )
    forged_license = json.loads(json.dumps(record))
    forged_license["provenance"]["license_spdx"] = "LicenseRef-GUM-Permissive-Subset"
    with pytest.raises(ValueError, match="identity or license mismatch"):
        verify_gum_discourse_record(forged_license, source, expected)
    segment = json.loads(
        json.dumps(record["kernel_packet"]["residual"]["segment_frame"])
    )
    segment["edges"][0]["nuclearity"] = "multinuclear"
    with pytest.raises(KernelProtocolFault, match="KERC_ERST_DISCOURSE_NUCLEARITY_INVALID"):
        _normalize_segment_frame(
            segment,
            source_character_length=len(record["source_text"]),
            path="test.segment",
        )


@pytest.mark.parametrize("relation", sorted(SCOPED_RELATIONS))
def test_human_single_edge_relations_compile_to_exact_scoped_roles(
    relation: str,
) -> None:
    row = scoped_row(relation)
    source = scoped_source()
    record = gum_discourse_record(
        row,
        source=source,
        producer_sha256="sha256:" + "5" * 64,
    )
    replay = verify_gum_discourse_record(record, source, deepcopy(row))

    expected_operator, expected_roles = SCOPED_RELATIONS[relation]
    root = record["kernel_packet"]["program"]["nodes"][-1]
    assert root["operator"] == expected_operator
    assert [argument["role"] for argument in root["arguments"][:2]] == expected_roles
    assert replay["scoped_semantic_disposition"] == "ADMITTED"
    assert replay["scoped_semantic_operator"] == expected_operator.removeprefix(
        "SCOPE_"
    )
    projection = record["semantic_supervision"]["scoped_semantic_projection"]
    assert projection["authority"] == (
        "human_erst_primary_relation_direction_and_endpoint_spans"
    )
    assert projection["complete_sentence_semantics_claimed"] is False
    assert projection["truth_claimed"] is False
    assert projection["derived_view_unique_source_credit"] == 0

    compiler_input, compiler_target = compiler_training_io(
        packet=record["kernel_packet"],
        source_text=record["source_text"],
        hrl_state=record["hrl_state"],
    )
    assert "program" not in compiler_input
    assert compiler_target["program"] == record["kernel_packet"]["program"]
    assert compiler_target["program"]["nodes"][-1]["operator"] == expected_operator


def test_scoped_relation_direction_role_and_authority_mutations_reject() -> None:
    row = scoped_row("causal-result_r")
    source = scoped_source()
    record = gum_discourse_record(
        row,
        source=source,
        producer_sha256="sha256:" + "5" * 64,
    )
    swapped = deepcopy(record)
    arguments = swapped["kernel_packet"]["program"]["nodes"][-1]["arguments"]
    arguments[0]["role"], arguments[1]["role"] = arguments[1]["role"], arguments[0]["role"]
    with pytest.raises(ValueError, match="Kernel graph topology mismatch"):
        verify_gum_discourse_record(swapped, source, deepcopy(row))

    forged_relation = deepcopy(record)
    forged_relation["semantic_supervision"]["scoped_semantic_projection"][
        "source_relation"
    ] = "causal-cause_r"
    with pytest.raises(ValueError, match="supervision authority mismatch"):
        verify_gum_discourse_record(forged_relation, source, deepcopy(row))

    moved_span = deepcopy(record)
    moved_span["kernel_packet"]["program"]["nodes"][0]["source_spans"] = [[1, 16]]
    with pytest.raises(ValueError, match="Kernel graph topology mismatch"):
        verify_gum_discourse_record(moved_span, source, deepcopy(row))


def test_multi_edge_scope_projection_is_explicitly_excluded_without_losing_erst() -> None:
    row = scoped_row("causal-cause_r")
    row["annotation"]["edges"].append(
        {
            "edge_kind": "secondary",
            "edge_order": 1,
            "child_edu_id": 10,
            "parent_edu_id": 20,
            "relation": "context-circumstance",
            "raw_depth_fields": [],
            "raw_signal_payload": "",
            "source_annotation_sha256": "sha256:" + "6" * 64,
        }
    )
    record = gum_discourse_record(
        row,
        source=scoped_source(),
        producer_sha256="sha256:" + "5" * 64,
    )
    projection = record["semantic_supervision"]["scoped_semantic_projection"]
    assert projection["disposition"] == "EXCLUDED"
    assert projection["exclusion_reason"] == (
        "multi_edge_or_nonprimary_neighborhood_has_shared_endpoint_ownership"
    )
    assert all(
        node["operator"].startswith("ERST_")
        for node in record["kernel_packet"]["program"]["nodes"][-2:]
    )
    replay = verify_gum_discourse_record(record, scoped_source(), deepcopy(row))
    assert replay["scoped_semantic_disposition"] == "EXCLUDED"


def test_scoped_supervision_audits_are_independent_and_expose_missing_cells() -> None:
    rows = {
        "private_train": [scoped_row(relation) for relation in SCOPED_RELATIONS],
        "private_dev": [
            scoped_row(relation, split="private_dev") for relation in SCOPED_RELATIONS
        ],
        "private_eval": [
            scoped_row(relation, split="private_eval") for relation in SCOPED_RELATIONS
        ],
    }
    produced = gum_scoped_supervision_audit(rows)
    verified = independently_audit_gum_scoped_supervision(rows)
    assert produced == verified
    assert produced["record_counts_by_split"] == {
        "private_train": 7,
        "private_dev": 7,
        "private_eval": 7,
    }
    assert produced["cross_split_source_group_overlap_count"] == 0
    assert produced["missing_relation_genre_cells_by_split"]["private_dev"]
    assert produced["learned_competence_claimed"] is False
