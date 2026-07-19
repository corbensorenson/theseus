from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import kerc_concept_registry as registry
import kerc_concept_registry_verify as verifier
from kernel_english_protocol import (
    KernelProtocolFault,
    learned_concept_capsule_view,
    materialize_learned_concept_capsules,
)


def _edge(relation: str, start: str, end: str, **metadata: object) -> str:
    assertion = f"/a/[{relation},{start},{end}]"
    payload = {
        "dataset": "/d/wordnet/3.1",
        "license": "cc:by/4.0",
        "sources": [{"contributor": "/s/resource/wordnet/rdf/3.1"}],
        "surfaceStart": start.split("/")[3].replace("_", " "),
        "surfaceEnd": end.split("/")[3].replace("_", " "),
        "weight": 2.0,
        **metadata,
    }
    return "\t".join((assertion, relation, start, end, json.dumps(payload, sort_keys=True)))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "conceptnet.csv.gz"
    rows = sorted(
        [
            _edge(
                "/r/IsA",
                "/c/en/bank/n/wn/finance",
                "/c/en/institution/n/wn/group",
                surfaceText="[[bank]] is a type of [[institution]]",
            ),
            _edge(
                "/r/IsA",
                "/c/en/bank/n/wn/river",
                "/c/en/slope/n/wn/shape",
                surfaceText="[[bank]] is a type of [[slope]]",
            ),
            _edge(
                "/r/Synonym",
                "/c/en/canine/n/wn/animal",
                "/c/en/dog/n/wn/animal",
                surfaceText="[[canine]] is a synonym of [[dog]]",
            ),
            _edge(
                "/r/IsA",
                "/c/es/perro/n/wn/animal",
                "/c/es/animal/n/wn/animal",
            ),
        ]
    )
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(rows) + "\n")
    output = tmp_path / "registry.sqlite3"
    manifest = tmp_path / "manifest.json"
    config = copy.deepcopy(
        json.loads((registry.ROOT / "configs" / "kerc_concept_registry.json").read_text())
    )
    config["source"].update(
        {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "assertion_count": 4,
        }
    )
    config["admission"]["expected_edge_count"] = 3
    config["registry"]["path"] = str(output)
    config["registry"]["manifest_path"] = str(manifest)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path, output, manifest


def test_build_and_independent_full_replay(tmp_path: Path) -> None:
    config, output, manifest_path = _fixture(tmp_path)
    manifest = registry.build(config)
    assert manifest["trigger_state"] == "GREEN"
    assert manifest["source"]["row_count"] == 4
    assert manifest["registry"]["edge_count"] == 3
    assert manifest["registry"]["relation_counts"] == {"/r/IsA": 2, "/r/Synonym": 1}
    assert output.exists() and manifest_path.exists()

    report = verifier.verify(config)
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["producer_authority_reused"] is False
    assert report["training_row_count"] == 0


def test_resolution_preserves_ambiguity_and_unknown(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    registry.build(config)
    with registry.ConceptRegistry(config) as concepts:
        unique = concepts.resolve({"surface": "Canine", "pos": "n"})
        assert unique["status"] == "RESOLVED"
        assert unique["candidate_count"] == 1
        assert unique["selected_identity"].startswith("conceptnet.uri.")
        assert unique["candidates"][0]["relation_count"] == 1

        ambiguous = concepts.resolve({"surface": "bank", "pos": "n"})
        assert ambiguous["status"] == "AMBIGUOUS"
        assert ambiguous["candidate_count"] == 2
        assert ambiguous["selected_identity"] == ""

        disambiguated = concepts.resolve(
            {"surface": "bank", "pos": "n", "sense": "wn/finance"}
        )
        assert disambiguated["status"] == "AMBIGUOUS"
        assert disambiguated["selected_identity"] == ""
        assert disambiguated["candidate_count"] == 2
        assert disambiguated["non_authoritative_hint_match_count"] == 1

        unknown = concepts.resolve({"surface": "not in registry"})
        assert unknown["status"] == "UNRESOLVED"
        assert unknown["candidates"] == []
        with pytest.raises(ValueError, match="forbidden fields"):
            concepts.resolve({"surface": "canine", "stable_identity": "forged"})


def test_protocol_assigns_authority_only_after_unique_resolution(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    registry.build(config)
    with registry.ConceptRegistry(config) as concepts:
        learned = {
            "@C0": {
                "surface_forms": ["canine"],
                "resolution_request": {"surface": "canine", "pos": "n"},
            }
        }
        materialized = materialize_learned_concept_capsules(
            learned, concept_resolver=concepts.resolve
        )
        capsule = materialized["@C0"]
        assert capsule["stable_identity"].startswith("conceptnet.uri.")
        assert capsule["provenance"]["source"] == "kerc_concept_registry_v1"
        assert capsule["registry_resolution"]["status"] == "RESOLVED"
        assert capsule["registry_semantics"]["canonical_surface"] == "canine"
        visible = learned_concept_capsule_view(materialized)
        assert visible == learned

        ambiguous = materialize_learned_concept_capsules(
            {
                "@C0": {
                    "surface_forms": ["bank"],
                    "resolution_request": {"surface": "bank", "pos": "n"},
                }
            },
            concept_resolver=concepts.resolve,
        )["@C0"]
        assert ambiguous["stable_identity"] == "local.concept.0"
        assert ambiguous["registry_resolution"]["status"] == "AMBIGUOUS"
        assert ambiguous["provenance"]["registry_promotion_allowed"] is False

    with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_REGISTRY_UNAVAILABLE"):
        materialize_learned_concept_capsules(learned)
    forged = {"@C0": {"stable_identity": "forged", "surface_forms": ["canine"]}}
    with pytest.raises(KernelProtocolFault, match="KERC_LEARNED_CONCEPT_AUTHORITY_FORBIDDEN"):
        materialize_learned_concept_capsules(forged)


def test_protocol_rejects_malformed_or_authority_claiming_resolver_results() -> None:
    learned = {
        "@C0": {
            "surface_forms": ["bank"],
            "resolution_request": {"surface": "bank", "pos": "n"},
        }
    }
    identity = "conceptnet.uri." + ("a" * 64)
    candidate = {"stable_identity": identity, "canonical_surface": "bank"}

    def response(**overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "status": "AMBIGUOUS",
            "candidate_count": 2,
            "candidates_truncated": True,
            "candidates": [candidate],
            "selected_identity": "",
            "authority_basis": "exact_normalized_surface_has_one_global_identity",
            "non_authoritative_hint_match_count": 1,
            "external_inference_calls": 0,
        }
        value.update(overrides)
        return value

    with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_REGISTRY_EXTERNAL_INFERENCE_FORBIDDEN"):
        materialize_learned_concept_capsules(
            learned,
            concept_resolver=lambda _: response(external_inference_calls=1),
        )
    with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_REGISTRY_RESPONSE_INVALID"):
        materialize_learned_concept_capsules(
            learned,
            concept_resolver=lambda _: response(selected_identity=identity),
        )
    with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_REGISTRY_IDENTITY_INVALID"):
        materialize_learned_concept_capsules(
            learned,
            concept_resolver=lambda _: response(
                candidates=[{"stable_identity": "forged"}]
            ),
        )
    with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_REGISTRY_RESPONSE_INVALID"):
        materialize_learned_concept_capsules(
            learned,
            concept_resolver=lambda _: response(
                candidate_count=1,
                candidates_truncated=False,
            ),
        )


def test_protocol_rejects_unbounded_or_authority_bearing_requests() -> None:
    for request in (
        {"surface": "bank", "stable_identity": "forged"},
        {"surface": "bank", "pos": "invalid"},
        {"surface": "x" * 513},
        {"surface": "bank", "sense": "x" * 257},
    ):
        with pytest.raises(KernelProtocolFault, match="KERC_CONCEPT_RESOLUTION_REQUEST_INVALID"):
            materialize_learned_concept_capsules(
                {"@C0": {"resolution_request": request}},
                concept_resolver=lambda _: pytest.fail("invalid request reached resolver"),
            )


def test_builder_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["source"]["sha256"] = "0" * 64
    config.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        registry.build(config)


def test_verifier_rejects_registry_corruption(tmp_path: Path) -> None:
    config, output, _ = _fixture(tmp_path)
    registry.build(config)
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute(
        "UPDATE relations SET surface_text='corrupted' WHERE assertion_uri=(SELECT MIN(assertion_uri) FROM relations)"
    )
    connection.commit()
    connection.close()
    report = verifier.verify(config)
    assert report["trigger_state"] == "RED"
    assert any(row["kind"] == "relation_mismatch" for row in report["faults"])


def test_verifier_rejects_manifest_authority_corruption(tmp_path: Path) -> None:
    config, _, manifest_path = _fixture(tmp_path)
    registry.build(config)
    manifest = json.loads(manifest_path.read_text())
    manifest["external_inference_calls"] = 1
    manifest["training_row_count"] = 4
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = verifier.verify(config)
    assert report["trigger_state"] == "RED"
    mismatches = {
        row.get("field")
        for row in report["faults"]
        if row["kind"] == "manifest_authority_mismatch"
    }
    assert mismatches == {"external_inference_calls", "training_row_count"}
