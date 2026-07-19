#!/usr/bin/env python3
"""Build and query the governed KERC cross-document concept registry.

The learned compiler may ask this registry to resolve a surface/sense query. It
cannot provide or mint an authoritative identity. Ambiguous and unresolved
queries remain explicit; this module never guesses a winner.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kerc_concept_registry.json"
ALLOWED_REQUEST_FIELDS = {"surface", "pos", "sense"}
POS_CODES = {"a", "n", "r", "s", "v"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_concept_uri(uri: str) -> dict[str, str]:
    parts = uri.split("/")
    if len(parts) < 4 or parts[0] or parts[1] != "c":
        raise ValueError(f"invalid ConceptNet concept URI: {uri}")
    language = parts[2]
    raw_term = unquote(parts[3])
    surface = raw_term.replace("_", " ")
    pos = parts[4] if len(parts) >= 5 and parts[4] in POS_CODES else ""
    sense_start = 5 if pos else 4
    sense = "/".join(parts[sense_start:])
    return {
        "uri": uri,
        "language": language,
        "term": raw_term,
        "surface": surface,
        "normalized_surface": normalize_surface(surface),
        "pos": pos,
        "sense": sense,
        "stable_identity": f"conceptnet.uri.{hashlib.sha256(uri.encode('utf-8')).hexdigest()}",
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy") != "project_theseus_kerc_concept_registry_v1":
        raise ValueError("unexpected KERC concept-registry policy")
    return payload


def _source_path(config: dict[str, Any]) -> Path:
    path = Path(str(config["source"]["path"]))
    return path if path.is_absolute() else ROOT / path


def _registry_path(config: dict[str, Any]) -> Path:
    path = Path(str(config["registry"]["path"]))
    return path if path.is_absolute() else ROOT / path


def _manifest_path(config: dict[str, Any]) -> Path:
    path = Path(str(config["registry"]["manifest_path"]))
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _iter_source_rows(
    config: dict[str, Any], stats: dict[str, int]
) -> Iterator[dict[str, Any]]:
    source = _source_path(config)
    admission = config["admission"]
    language = str(admission["language"])
    dataset = str(admission["dataset"])
    licenses = {str(row) for row in admission["allowed_licenses"]}
    relations = {str(row) for row in admission["allowed_relations"]}
    minimum_weight = float(admission["minimum_weight"])
    previous_assertion = ""
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row_number, row in enumerate(reader, 1):
            stats["source_row_count"] = row_number
            if len(row) != 5:
                raise ValueError(f"source row {row_number} has {len(row)} columns")
            assertion_uri, relation, start_uri, end_uri, metadata_raw = row
            if not (
                start_uri.startswith(f"/c/{language}/")
                and end_uri.startswith(f"/c/{language}/")
            ):
                continue
            metadata = json.loads(metadata_raw)
            if str(metadata.get("dataset") or "") != dataset:
                continue
            if str(metadata.get("license") or "") not in licenses:
                continue
            if relation not in relations:
                continue
            if float(metadata.get("weight") or 0.0) < minimum_weight:
                continue
            if assertion_uri <= previous_assertion:
                raise ValueError("admitted ConceptNet assertion URIs are not strictly ordered")
            previous_assertion = assertion_uri
            stats["admitted_edge_count"] = stats.get("admitted_edge_count", 0) + 1
            yield {
                "assertion_uri": assertion_uri,
                "relation": relation,
                "start": parse_concept_uri(start_uri),
                "end": parse_concept_uri(end_uri),
                "metadata": metadata,
            }


def _schema(connection: sqlite3.Connection, schema_version: str) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE concepts (
            identity TEXT PRIMARY KEY,
            concept_uri TEXT NOT NULL UNIQUE,
            language TEXT NOT NULL,
            canonical_surface TEXT NOT NULL,
            normalized_surface TEXT NOT NULL,
            pos TEXT NOT NULL,
            sense TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE aliases (
            normalized_surface TEXT NOT NULL,
            identity TEXT NOT NULL REFERENCES concepts(identity),
            surface TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            PRIMARY KEY (normalized_surface, identity, surface, source_kind)
        ) WITHOUT ROWID;
        CREATE INDEX aliases_lookup ON aliases(normalized_surface, identity);
        CREATE TABLE relations (
            assertion_uri TEXT PRIMARY KEY,
            relation TEXT NOT NULL,
            start_identity TEXT NOT NULL REFERENCES concepts(identity),
            end_identity TEXT NOT NULL REFERENCES concepts(identity),
            weight REAL NOT NULL,
            surface_text TEXT NOT NULL,
            dataset TEXT NOT NULL,
            license TEXT NOT NULL,
            metadata_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX relations_start ON relations(start_identity, relation, end_identity);
        CREATE INDEX relations_end ON relations(end_identity, relation, start_identity);
        """
    )
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        ("schema_version", schema_version),
    )


def _insert_concept(connection: sqlite3.Connection, concept: dict[str, str]) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO concepts VALUES (?,?,?,?,?,?,?)",
        (
            concept["stable_identity"],
            concept["uri"],
            concept["language"],
            concept["surface"],
            concept["normalized_surface"],
            concept["pos"],
            concept["sense"],
        ),
    )
    connection.execute(
        "INSERT OR IGNORE INTO aliases VALUES (?,?,?,?)",
        (
            concept["normalized_surface"],
            concept["stable_identity"],
            concept["surface"],
            "concept_uri",
        ),
    )


def _add_metadata_alias(
    connection: sqlite3.Connection,
    concept: dict[str, str],
    surface: Any,
    source_kind: str,
) -> None:
    if not isinstance(surface, str) or not surface.strip():
        return
    normalized = normalize_surface(surface)
    if not normalized:
        return
    connection.execute(
        "INSERT OR IGNORE INTO aliases VALUES (?,?,?,?)",
        (normalized, concept["stable_identity"], surface, source_kind),
    )


def _digest_query(connection: sqlite3.Connection, query: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(query):
        digest.update(canonical_json(list(row)).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    source = _source_path(config)
    observed_source_sha = sha256_file(source)
    expected_source_sha = str(config["source"]["sha256"])
    if observed_source_sha != expected_source_sha:
        raise ValueError(
            f"ConceptNet source hash mismatch: {observed_source_sha} != {expected_source_sha}"
        )

    registry_path = _registry_path(config)
    manifest_path = _manifest_path(config)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{registry_path.name}.", suffix=".tmp", dir=registry_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    stats: dict[str, int] = {"source_row_count": 0, "admitted_edge_count": 0}
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        _schema(connection, str(config["registry"]["schema_version"]))
        connection.commit()
        connection.execute("BEGIN")
        for edge in _iter_source_rows(config, stats):
            start = edge["start"]
            end = edge["end"]
            metadata = edge["metadata"]
            _insert_concept(connection, start)
            _insert_concept(connection, end)
            _add_metadata_alias(connection, start, metadata.get("surfaceStart"), "surface_start")
            _add_metadata_alias(connection, end, metadata.get("surfaceEnd"), "surface_end")
            connection.execute(
                "INSERT INTO relations VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    edge["assertion_uri"],
                    edge["relation"],
                    start["stable_identity"],
                    end["stable_identity"],
                    float(metadata.get("weight") or 0.0),
                    str(metadata.get("surfaceText") or ""),
                    str(metadata.get("dataset") or ""),
                    str(metadata.get("license") or ""),
                    hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest(),
                ),
            )
        connection.commit()
        if stats["source_row_count"] != int(config["source"]["assertion_count"]):
            raise ValueError("ConceptNet source assertion count mismatch")
        if stats["admitted_edge_count"] != int(config["admission"]["expected_edge_count"]):
            raise ValueError("ConceptNet admitted edge count mismatch")
        edge_digest, edge_count = _digest_query(
            connection,
            "SELECT assertion_uri,relation,start_identity,end_identity,weight,surface_text,dataset,license,metadata_sha256 FROM relations ORDER BY assertion_uri",
        )
        concept_digest, concept_count = _digest_query(
            connection,
            "SELECT identity,concept_uri,language,canonical_surface,normalized_surface,pos,sense FROM concepts ORDER BY identity",
        )
        alias_digest, alias_count = _digest_query(
            connection,
            "SELECT normalized_surface,identity,surface,source_kind FROM aliases ORDER BY normalized_surface,identity,surface,source_kind",
        )
        relation_counts = {
            relation: count
            for relation, count in connection.execute(
                "SELECT relation,COUNT(*) FROM relations GROUP BY relation ORDER BY relation"
            )
        }
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, registry_path)
        manifest = {
            "policy": "project_theseus_kerc_concept_registry_manifest_v1",
            "trigger_state": "GREEN",
            "config": _display_path(config_path),
            "config_sha256": sha256_file(config_path),
            "source": {
                "path": _display_path(source),
                "sha256": observed_source_sha,
                "row_count": stats["source_row_count"],
                "dataset": config["admission"]["dataset"],
                "license": sorted(config["admission"]["allowed_licenses"]),
            },
            "registry": {
                "path": _display_path(registry_path),
                "sha256": sha256_file(registry_path),
                "schema_version": config["registry"]["schema_version"],
                "edge_count": edge_count,
                "concept_count": concept_count,
                "alias_count": alias_count,
                "edge_digest": edge_digest,
                "concept_digest": concept_digest,
                "alias_digest": alias_digest,
                "relation_counts": relation_counts,
                "bytes": registry_path.stat().st_size,
            },
            "authority": config["authority"],
            "claim_ceiling": config["claim_ceiling"],
            "external_inference_calls": 0,
            "training_row_count": 0,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        if temporary.exists():
            temporary.unlink()


class ConceptRegistry:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        *,
        registry_path: Path | None = None,
    ) -> None:
        self.config = _load_config(config_path)
        self.path = registry_path or _registry_path(self.config)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            f"PRAGMA busy_timeout={int(self.config['registry']['busy_timeout_ms'])}"
        )
        observed = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if not observed or observed[0] != self.config["registry"]["schema_version"]:
            self.close()
            raise ValueError("KERC concept registry schema mismatch")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ConceptRegistry":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("concept resolution request must be an object")
        forbidden = sorted(set(request) - ALLOWED_REQUEST_FIELDS)
        if forbidden:
            raise ValueError(f"concept resolution request has forbidden fields: {forbidden}")
        surface = str(request.get("surface") or "").strip()
        if not surface:
            raise ValueError("concept resolution request surface is required")
        normalized = normalize_surface(surface)
        pos = str(request.get("pos") or "")
        sense = str(request.get("sense") or "")
        if pos and pos not in POS_CODES:
            raise ValueError("concept resolution request POS is invalid")
        query = """
            SELECT DISTINCT c.identity,c.concept_uri,c.canonical_surface,c.pos,c.sense
            FROM aliases a JOIN concepts c ON c.identity=a.identity
            WHERE a.normalized_surface=?
        """
        query += " ORDER BY c.identity"
        rows = list(self.connection.execute(query, [normalized]))
        maximum = int(self.config["registry"]["maximum_candidates"])
        candidates = []
        for row in rows[:maximum]:
            candidate = self._candidate(row)
            candidate["non_authoritative_hint_match"] = bool(
                (not pos or row["pos"] == pos)
                and (not sense or row["sense"] == sense)
            )
            candidates.append(candidate)
        status = "UNRESOLVED" if not rows else ("RESOLVED" if len(rows) == 1 else "AMBIGUOUS")
        return {
            "policy": "project_theseus_kerc_concept_resolution_v1",
            "status": status,
            "request": {"surface": surface, "normalized_surface": normalized, "pos": pos, "sense": sense},
            "candidate_count": len(rows),
            "candidates_truncated": len(rows) > maximum,
            "candidates": candidates,
            "selected_identity": candidates[0]["stable_identity"] if status == "RESOLVED" else "",
            "authority_basis": "exact_normalized_surface_has_one_global_identity",
            "non_authoritative_hint_match_count": sum(
                int(candidate["non_authoritative_hint_match"])
                for candidate in candidates
            ),
            "registry_schema_version": self.config["registry"]["schema_version"],
            "external_inference_calls": 0,
        }

    def _candidate(self, row: sqlite3.Row) -> dict[str, Any]:
        limit = int(self.config["registry"]["maximum_relations_per_candidate"])
        relations = [
            {
                "assertion_uri": relation[0],
                "relation": relation[1],
                "direction": relation[2],
                "other_identity": relation[3],
                "other_surface": relation[4],
                "weight": relation[5],
            }
            for relation in self.connection.execute(
                """
                SELECT r.assertion_uri,r.relation,'outgoing',r.end_identity,e.canonical_surface,r.weight
                FROM relations r JOIN concepts e ON e.identity=r.end_identity
                WHERE r.start_identity=?
                UNION ALL
                SELECT r.assertion_uri,r.relation,'incoming',r.start_identity,s.canonical_surface,r.weight
                FROM relations r JOIN concepts s ON s.identity=r.start_identity
                WHERE r.end_identity=?
                ORDER BY 1,3 LIMIT ?
                """,
                (row["identity"], row["identity"], limit),
            )
        ]
        total = self.connection.execute(
            "SELECT (SELECT COUNT(*) FROM relations WHERE start_identity=?) + (SELECT COUNT(*) FROM relations WHERE end_identity=?)",
            (row["identity"], row["identity"]),
        ).fetchone()[0]
        return {
            "stable_identity": row["identity"],
            "concept_uri": row["concept_uri"],
            "canonical_surface": row["canonical_surface"],
            "pos": row["pos"],
            "sense": row["sense"],
            "relation_count": int(total),
            "relations_truncated": int(total) > limit,
            "relations": relations,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resolve")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if args.build:
        print(json.dumps(build(config_path), indent=2))
        return 0
    if args.resolve:
        request = json.loads(args.resolve)
        with ConceptRegistry(config_path) as registry:
            print(json.dumps(registry.resolve(request), indent=2))
        return 0
    manifest = _manifest_path(_load_config(config_path))
    if not manifest.exists():
        print(json.dumps({"trigger_state": "RED", "blockers": ["registry_not_built"]}, indent=2))
        return 2
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
