#!/usr/bin/env python3
"""Independently verify the KERC ConceptNet/WordNet concept registry.

This verifier does not import or execute the producer. It frames source rows
manually, reconstructs identities and aliases independently, and compares every
admitted relation and registry object against the read-only SQLite artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kerc_concept_registry.json"
DEFAULT_OUT = ROOT / "reports" / "runtime" / "kerc_concept_registry_verification.json"
POS_CODES = {"a", "n", "r", "s", "v"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _concept(uri: str) -> tuple[str, str, str, str, str, str, str]:
    parts = uri.split("/")
    if len(parts) < 4 or parts[0] or parts[1] != "c":
        raise ValueError(f"invalid concept URI {uri}")
    language = parts[2]
    term = unquote(parts[3])
    surface = term.replace("_", " ")
    pos = parts[4] if len(parts) >= 5 and parts[4] in POS_CODES else ""
    sense = "/".join(parts[5 if pos else 4 :])
    identity = f"conceptnet.uri.{hashlib.sha256(uri.encode('utf-8')).hexdigest()}"
    return identity, uri, language, surface, _normalize(surface), pos, sense


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _iter_expected(
    config: dict[str, Any],
    source: Path,
    stats: dict[str, int],
    concepts: dict[str, tuple[str, ...]],
    aliases: set[tuple[str, str, str, str]],
) -> Iterator[tuple[Any, ...]]:
    admission = config["admission"]
    language = str(admission["language"])
    dataset = str(admission["dataset"])
    licenses = {str(row) for row in admission["allowed_licenses"]}
    relations = {str(row) for row in admission["allowed_relations"]}
    minimum_weight = float(admission["minimum_weight"])
    previous = ""
    decoder = json.JSONDecoder()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        for row_number, line in enumerate(handle, 1):
            stats["source_row_count"] = row_number
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"source row {row_number} has {len(fields)} columns")
            assertion, relation, start_uri, end_uri, raw_metadata = fields
            if not (
                start_uri.startswith(f"/c/{language}/")
                and end_uri.startswith(f"/c/{language}/")
            ):
                continue
            metadata, consumed = decoder.raw_decode(raw_metadata)
            if raw_metadata[consumed:].strip():
                raise ValueError(f"source row {row_number} has trailing metadata")
            if str(metadata.get("dataset") or "") != dataset:
                continue
            if str(metadata.get("license") or "") not in licenses:
                continue
            if relation not in relations or float(metadata.get("weight") or 0.0) < minimum_weight:
                continue
            if assertion <= previous:
                raise ValueError("admitted source assertions are not strictly ordered")
            previous = assertion
            start = _concept(start_uri)
            end = _concept(end_uri)
            concepts[start[0]] = start
            concepts[end[0]] = end
            aliases.add((start[4], start[0], start[3], "concept_uri"))
            aliases.add((end[4], end[0], end[3], "concept_uri"))
            for concept, key, kind in (
                (start, "surfaceStart", "surface_start"),
                (end, "surfaceEnd", "surface_end"),
            ):
                surface = metadata.get(key)
                if isinstance(surface, str) and surface.strip():
                    aliases.add((_normalize(surface), concept[0], surface, kind))
            stats["admitted_edge_count"] = stats.get("admitted_edge_count", 0) + 1
            yield (
                assertion,
                relation,
                start[0],
                end[0],
                float(metadata.get("weight") or 0.0),
                str(metadata.get("surfaceText") or ""),
                str(metadata.get("dataset") or ""),
                str(metadata.get("license") or ""),
                hashlib.sha256(_canonical(metadata).encode("utf-8")).hexdigest(),
            )


def _digest(rows: Iterator[tuple[Any, ...]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_canonical(list(row)).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def verify(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("policy") != "project_theseus_kerc_concept_registry_v1":
        return {
            "trigger_state": "RED",
            "faults": [{"kind": "config_policy_mismatch"}],
        }
    source = _path(ROOT, str(config["source"]["path"]))
    registry = _path(ROOT, str(config["registry"]["path"]))
    manifest_path = _path(ROOT, str(config["registry"]["manifest_path"]))
    faults: list[dict[str, Any]] = []
    for path, expected, kind in (
        (source, str(config["source"]["sha256"]), "source_hash"),
        (config_path, None, "config_hash"),
        (registry, None, "registry_hash"),
    ):
        if not path.exists():
            faults.append({"kind": f"{kind}_missing", "path": str(path)})
        elif expected and _file_hash(path) != expected:
            faults.append({"kind": f"{kind}_mismatch", "path": str(path)})
    if faults or not manifest_path.exists():
        if not manifest_path.exists():
            faults.append({"kind": "manifest_missing", "path": str(manifest_path)})
        return {"trigger_state": "RED", "faults": faults}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source_manifest = {
        "path": _display_path(source),
        "sha256": _file_hash(source),
        "row_count": int(config["source"]["assertion_count"]),
        "dataset": config["admission"]["dataset"],
        "license": sorted(config["admission"]["allowed_licenses"]),
    }
    manifest_checks = {
        "policy": (
            manifest.get("policy"),
            "project_theseus_kerc_concept_registry_manifest_v1",
        ),
        "trigger_state": (manifest.get("trigger_state"), "GREEN"),
        "config_sha256": (manifest.get("config_sha256"), _file_hash(config_path)),
        "source": (manifest.get("source"), expected_source_manifest),
        "authority": (manifest.get("authority"), config.get("authority")),
        "claim_ceiling": (manifest.get("claim_ceiling"), config.get("claim_ceiling")),
        "external_inference_calls": (manifest.get("external_inference_calls"), 0),
        "training_row_count": (manifest.get("training_row_count"), 0),
    }
    for field, (observed_value, expected_value) in manifest_checks.items():
        if observed_value != expected_value:
            faults.append(
                {
                    "kind": "manifest_authority_mismatch",
                    "field": field,
                    "expected": expected_value,
                    "observed": observed_value,
                }
            )
    connection = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
    if integrity != "ok":
        faults.append({"kind": "sqlite_integrity_failure", "detail": integrity})
    if foreign_keys:
        faults.append({"kind": "sqlite_foreign_key_failure", "count": len(foreign_keys)})
    schema = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()
    if not schema or schema[0] != config["registry"]["schema_version"]:
        faults.append({"kind": "schema_version_mismatch"})

    concepts: dict[str, tuple[str, ...]] = {}
    aliases: set[tuple[str, str, str, str]] = set()
    stats = {"source_row_count": 0, "admitted_edge_count": 0}
    actual_relations = iter(
        connection.execute(
            "SELECT assertion_uri,relation,start_identity,end_identity,weight,surface_text,dataset,license,metadata_sha256 FROM relations ORDER BY assertion_uri"
        )
    )
    relation_digest = hashlib.sha256()
    relation_count = 0
    for expected in _iter_expected(config, source, stats, concepts, aliases):
        actual = next(actual_relations, None)
        if actual is None or tuple(actual) != expected:
            faults.append(
                {
                    "kind": "relation_mismatch",
                    "index": relation_count,
                    "expected": list(expected),
                    "actual": list(actual) if actual is not None else None,
                }
            )
            break
        relation_digest.update(_canonical(list(actual)).encode("utf-8"))
        relation_digest.update(b"\n")
        relation_count += 1
    if not faults and next(actual_relations, None) is not None:
        faults.append({"kind": "unexpected_extra_relation"})

    expected_concepts = iter(sorted(concepts.values(), key=lambda row: row[0]))
    actual_concepts = iter(
        connection.execute(
            "SELECT identity,concept_uri,language,canonical_surface,normalized_surface,pos,sense FROM concepts ORDER BY identity"
        )
    )
    concept_digest = hashlib.sha256()
    concept_count = 0
    for expected in expected_concepts:
        actual = next(actual_concepts, None)
        if actual is None or tuple(actual) != expected:
            faults.append({"kind": "concept_mismatch", "index": concept_count})
            break
        concept_digest.update(_canonical(list(actual)).encode("utf-8"))
        concept_digest.update(b"\n")
        concept_count += 1
    if next(actual_concepts, None) is not None:
        faults.append({"kind": "unexpected_extra_concept"})

    expected_aliases = iter(sorted(aliases))
    actual_aliases = iter(
        connection.execute(
            "SELECT normalized_surface,identity,surface,source_kind FROM aliases ORDER BY normalized_surface,identity,surface,source_kind"
        )
    )
    alias_digest = hashlib.sha256()
    alias_count = 0
    for expected in expected_aliases:
        actual = next(actual_aliases, None)
        if actual is None or tuple(actual) != expected:
            faults.append({"kind": "alias_mismatch", "index": alias_count})
            break
        alias_digest.update(_canonical(list(actual)).encode("utf-8"))
        alias_digest.update(b"\n")
        alias_count += 1
    if next(actual_aliases, None) is not None:
        faults.append({"kind": "unexpected_extra_alias"})

    relation_counts = {
        relation: count
        for relation, count in connection.execute(
            "SELECT relation,COUNT(*) FROM relations GROUP BY relation ORDER BY relation"
        )
    }
    connection.close()
    observed = {
        "source_row_count": stats["source_row_count"],
        "edge_count": relation_count,
        "concept_count": concept_count,
        "alias_count": alias_count,
        "edge_digest": relation_digest.hexdigest(),
        "concept_digest": concept_digest.hexdigest(),
        "alias_digest": alias_digest.hexdigest(),
        "relation_counts": relation_counts,
        "registry_sha256": _file_hash(registry),
    }
    expected_manifest = manifest.get("registry") or {}
    expected_registry_metadata = {
        "path": _display_path(registry),
        "schema_version": config["registry"]["schema_version"],
        "bytes": registry.stat().st_size,
    }
    for key, expected_value in expected_registry_metadata.items():
        if expected_manifest.get(key) != expected_value:
            faults.append(
                {
                    "kind": "manifest_registry_metadata_mismatch",
                    "field": key,
                    "expected": expected_value,
                    "observed": expected_manifest.get(key),
                }
            )
    for key in (
        "edge_count",
        "concept_count",
        "alias_count",
        "edge_digest",
        "concept_digest",
        "alias_digest",
        "relation_counts",
    ):
        if observed[key] != expected_manifest.get(key):
            faults.append(
                {"kind": "manifest_mismatch", "field": key, "expected": expected_manifest.get(key), "observed": observed[key]}
            )
    if observed["registry_sha256"] != expected_manifest.get("sha256"):
        faults.append({"kind": "manifest_registry_hash_mismatch"})
    if stats["source_row_count"] != int(config["source"]["assertion_count"]):
        faults.append({"kind": "source_row_count_mismatch"})
    if stats["admitted_edge_count"] != int(config["admission"]["expected_edge_count"]):
        faults.append({"kind": "admitted_edge_count_mismatch"})
    return {
        "policy": "project_theseus_kerc_concept_registry_verifier_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "observed": observed,
        "source": {
            "path": _display_path(source),
            "sha256": _file_hash(source),
            "dataset": config["admission"]["dataset"],
            "license": sorted(config["admission"]["allowed_licenses"]),
        },
        "producer_authority_reused": False,
        "training_row_count": 0,
        "external_inference_calls": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    config_path = _path(ROOT, args.config)
    output_path = _path(ROOT, args.out)
    report = verify(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not args.gate or report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
