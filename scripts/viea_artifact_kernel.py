"""SQLite artifact kernel for VIEA / Project Theseus.

Reports are excellent audit artifacts, but a growing intent-to-execution system
needs a durable object substrate. This script builds a small local SQLite store
for VIEA objects and emits JSON/Markdown views over that store.

It is deterministic, local-only, and does not treat architecture scaffolding as
student-learning evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_DB = REPORTS / "viea_artifact_kernel.sqlite"

OBJECT_TYPES = [
    "World",
    "Command",
    "Artifact",
    "Claim",
    "Critique",
    "Tool",
    "Benchmark",
    "Residual",
    "Release",
    "Feedback",
    "ResourceEvent",
    "Primitive",
    "SpecialistModule",
    "CompileTarget",
    "RuntimeAdapter",
]


class ArtifactStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                content_json TEXT NOT NULL,
                source_path TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                version TEXT NOT NULL DEFAULT '0.1.0',
                verification_state TEXT NOT NULL DEFAULT 'unknown',
                release_state TEXT NOT NULL DEFAULT 'internal',
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);

            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                content_json TEXT NOT NULL,
                source_path TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);

            CREATE TABLE IF NOT EXISTS views (
                name TEXT PRIMARY KEY,
                content_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def reset(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM relationships")
        cur.execute("DELETE FROM objects")
        cur.execute("DELETE FROM views")
        self.conn.commit()

    def upsert_object(
        self,
        *,
        object_id: str,
        object_type: str,
        title: str,
        content: Any,
        source_path: str,
        provenance: dict[str, Any] | None = None,
        verification_state: str = "unknown",
        release_state: str = "internal",
        version: str = "0.1.0",
    ) -> str:
        stamp = now()
        self.conn.execute(
            """
            INSERT INTO objects (
                id, type, title, content_json, source_path, provenance_json,
                version, verification_state, release_state, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type,
                title=excluded.title,
                content_json=excluded.content_json,
                source_path=excluded.source_path,
                provenance_json=excluded.provenance_json,
                version=excluded.version,
                verification_state=excluded.verification_state,
                release_state=excluded.release_state,
                updated_utc=excluded.updated_utc
            """,
            (
                object_id,
                object_type,
                title,
                json.dumps(content, sort_keys=True),
                source_path,
                json.dumps(provenance or {}, sort_keys=True),
                version,
                verification_state,
                release_state,
                stamp,
                stamp,
            ),
        )
        return object_id

    def add_relationship(
        self,
        *,
        source_id: str,
        target_id: str,
        relation: str,
        source_path: str,
        content: Any | None = None,
    ) -> str:
        relationship_id = stable_id("rel", source_id, relation, target_id)
        self.conn.execute(
            """
            INSERT INTO relationships (id, source_id, target_id, relation, content_json, source_path, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content_json=excluded.content_json,
                source_path=excluded.source_path
            """,
            (
                relationship_id,
                source_id,
                target_id,
                relation,
                json.dumps(content or {}, sort_keys=True),
                source_path,
                now(),
            ),
        )
        return relationship_id

    def put_view(self, name: str, payload: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO views (name, content_json, created_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                content_json=excluded.content_json,
                created_utc=excluded.created_utc
            """,
            (name, json.dumps(payload, sort_keys=True), now()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def counts_by_type(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT type, COUNT(*) AS n FROM objects GROUP BY type ORDER BY type").fetchall()
        return {str(row["type"]): int(row["n"]) for row in rows}

    def object_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()
        return int(row["n"]) if row else 0

    def relationship_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM relationships").fetchone()
        return int(row["n"]) if row else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default="reports/viea_artifact_kernel.json")
    parser.add_argument("--markdown-out", default="reports/viea_artifact_kernel.md")
    parser.add_argument("--reset", action="store_true", help="Rebuild the local artifact DB from report views.")
    args = parser.parse_args()

    db_path = resolve(args.db)
    store = ArtifactStore(db_path)
    if args.reset:
        store.reset()
    sources = ingest_known_reports(store)
    summary = build_summary(store, db_path=db_path, sources=sources)
    store.put_view("viea_kernel_summary", summary)
    store.commit()
    store.close()

    payload = {
        "policy": "project_theseus_viea_artifact_kernel_v1",
        "created_utc": now(),
        "trigger_state": trigger_state(summary),
        "database": rel_or_abs(db_path),
        "object_types": OBJECT_TYPES,
        "summary": summary,
        "rules": {
            "reports_are_views": "Reports remain audit artifacts, but VIEA objects are also indexed in SQLite for durable routing and feedback.",
            "learning_boundary": "Artifact kernel health is scaffold evidence, not student-learning proof.",
            "external_inference": "No external inference is used by this kernel.",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def ingest_known_reports(store: ArtifactStore) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    sources.extend(ingest_reality_manipulator(store, REPORTS / "reality_manipulator.json"))
    sources.extend(ingest_command_executor(store, REPORTS / "viea_command_executor.json"))
    sources.extend(ingest_plan_compiler(store))
    sources.extend(ingest_tool_registry(store, REPORTS / "tool_registry.json"))
    sources.extend(ingest_deterministic_tool_substrate(store))
    sources.extend(ingest_viea_execution_spine(store, REPORTS / "viea_execution_spine.json"))
    sources.extend(ingest_benchmark_reports(store))
    sources.extend(ingest_growth_reports(store))
    store.commit()
    return sources


def ingest_reality_manipulator(store: ArtifactStore, path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return [source_row(path, False, "missing_or_invalid")]
    source = rel_or_abs(path)
    world = payload.get("world") if isinstance(payload.get("world"), dict) else {}
    world_id = str(world.get("id") or stable_id("world", world.get("name"), source))
    store.upsert_object(
        object_id=world_id,
        object_type="World",
        title=str(world.get("name") or "Reality Manipulator world"),
        content=world,
        source_path=source,
        provenance={"report": source, "policy": payload.get("policy")},
        verification_state=str(payload.get("trigger_state") or "unknown"),
        release_state="internal",
    )
    command = get_path(payload, ["structured_command_layer", "command_contract"], {})
    command_id = str(command.get("id") or stable_id("command", source))
    store.upsert_object(
        object_id=command_id,
        object_type="Command",
        title=str(command.get("title") or "VIEA command contract"),
        content=command,
        source_path=source,
        provenance={"report": source},
        verification_state="verified" if command.get("content_hash") else "incomplete",
        release_state="internal",
    )
    store.add_relationship(source_id=command_id, target_id=world_id, relation="creates", source_path=source)
    graph = payload.get("artifact_graph") if isinstance(payload.get("artifact_graph"), dict) else {}
    for artifact in graph.get("artifacts", []) if isinstance(graph.get("artifacts"), list) else []:
        if not isinstance(artifact, dict):
            continue
        object_id = str(artifact.get("id") or stable_id("artifact", artifact.get("title"), source))
        object_type = artifact_type(str(artifact.get("type") or "Artifact"))
        store.upsert_object(
            object_id=object_id,
            object_type=object_type,
            title=str(artifact.get("title") or object_id),
            content=artifact,
            source_path=source,
            provenance=artifact.get("provenance") if isinstance(artifact.get("provenance"), dict) else {"report": source},
            verification_state=str(artifact.get("current_state") or "active"),
            release_state=str(artifact.get("release_status") or "internal"),
            version=str(artifact.get("version") or "0.1.0"),
        )
        store.add_relationship(source_id=world_id, target_id=object_id, relation="contains", source_path=source)
    for edge in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
        if isinstance(edge, dict) and edge.get("source") and edge.get("target"):
            store.add_relationship(
                source_id=str(edge["source"]),
                target_id=str(edge["target"]),
                relation=str(edge.get("relation") or "related_to"),
                source_path=source,
                content=edge,
            )
    for claim in payload.get("claim_ledger", []) if isinstance(payload.get("claim_ledger"), list) else []:
        if isinstance(claim, dict):
            object_id = str(claim.get("id") or stable_id("claim", claim.get("claim_text"), source))
            store.upsert_object(
                object_id=object_id,
                object_type="Claim",
                title=str(claim.get("claim_text") or object_id)[:160],
                content=claim,
                source_path=source,
                provenance={"report": source},
                verification_state=str(claim.get("support_state") or "unknown"),
                release_state="internal",
            )
            store.add_relationship(source_id=object_id, target_id=world_id, relation="used_in", source_path=source)
    for critique in payload.get("critique_log", []) if isinstance(payload.get("critique_log"), list) else []:
        if isinstance(critique, dict):
            object_id = str(critique.get("id") or stable_id("critique", critique.get("recommendation"), source))
            store.upsert_object(
                object_id=object_id,
                object_type="Critique",
                title=str(critique.get("id") or "critique"),
                content=critique,
                source_path=source,
                provenance={"report": source},
                verification_state=str(critique.get("status") or "open"),
                release_state="internal",
            )
            store.add_relationship(source_id=object_id, target_id=world_id, relation="critiques", source_path=source)
    for target in payload.get("world_runtimes", []) if isinstance(payload.get("world_runtimes"), list) else []:
        if isinstance(target, dict):
            object_id = stable_id("runtime", target.get("target_type"), source)
            store.upsert_object(
                object_id=object_id,
                object_type="RuntimeAdapter",
                title=f"{target.get('target_type')} runtime",
                content=target,
                source_path=source,
                provenance={"report": source},
                verification_state=str(target.get("gate_status") or "unknown"),
                release_state="planning_only" if target.get("risk_tier") == "high" else "internal",
            )
            store.add_relationship(source_id=world_id, target_id=object_id, relation="offers_compile_target", source_path=source)
    for residual in get_path(payload, ["feedback_ratchet", "residual_escrow"], []) or []:
        if isinstance(residual, dict):
            object_id = str(residual.get("id") or stable_id("residual", json.dumps(residual, sort_keys=True)))
            store.upsert_object(
                object_id=object_id,
                object_type="Residual",
                title=str(residual.get("failure_type") or object_id),
                content=residual,
                source_path=source,
                provenance={"report": source},
                verification_state=str(residual.get("severity") or "open"),
                release_state=str(residual.get("promotion_status") or "track"),
            )
    for primitive in payload.get("primitive_registry", []) if isinstance(payload.get("primitive_registry"), list) else []:
        if isinstance(primitive, dict):
            object_id = str(primitive.get("id") or stable_id("primitive", primitive.get("name"), source))
            store.upsert_object(
                object_id=object_id,
                object_type="Primitive",
                title=str(primitive.get("name") or object_id),
                content=primitive,
                source_path=source,
                provenance={"report": source},
                verification_state=str(primitive.get("status") or "candidate"),
                release_state="candidate",
            )
    release_manifest = resolve("reports/reality_manipulator/latest_world/release_manifest.json")
    release = read_json(release_manifest)
    if release:
        release_id = stable_id("release", release.get("release_name"), rel_or_abs(release_manifest))
        store.upsert_object(
            object_id=release_id,
            object_type="Release",
            title=str(release.get("release_name") or "Reality Manipulator release"),
            content=release,
            source_path=rel_or_abs(release_manifest),
            provenance={"report": rel_or_abs(release_manifest)},
            verification_state="draft",
            release_state=str(release.get("release_type") or "internal"),
        )
        store.add_relationship(source_id=world_id, target_id=release_id, relation="summarized_by", source_path=rel_or_abs(release_manifest))
    feedback_path = resolve("reports/reality_manipulator/latest_world/feedback_plan.md")
    if feedback_path.exists():
        feedback_id = stable_id("feedback", rel_or_abs(feedback_path))
        store.upsert_object(
            object_id=feedback_id,
            object_type="Feedback",
            title="Reality Manipulator feedback plan",
            content={"path": rel_or_abs(feedback_path), "text": feedback_path.read_text(encoding="utf-8")[:12000]},
            source_path=rel_or_abs(feedback_path),
            provenance={"report": rel_or_abs(feedback_path)},
            verification_state="planned",
            release_state="internal",
        )
        store.add_relationship(source_id=world_id, target_id=feedback_id, relation="needs_feedback", source_path=rel_or_abs(feedback_path))
    return [source_row(path, True, "ingested")]


def ingest_tool_registry(store: ArtifactStore, path: Path) -> list[dict[str, Any]]:
    registry = read_json(path)
    if not registry:
        return [source_row(path, False, "missing_or_invalid")]
    source = rel_or_abs(path)
    for tool in registry.get("tools", []) if isinstance(registry.get("tools"), list) else []:
        if not isinstance(tool, dict):
            continue
        object_id = stable_id("tool", tool.get("tool_name"), tool.get("version", ""))
        store.upsert_object(
            object_id=object_id,
            object_type="Tool",
            title=str(tool.get("tool_name") or object_id),
            content=tool,
            source_path=source,
            provenance={"report": source},
            verification_state=str(tool.get("lifecycle") or "active"),
            release_state=str(tool.get("risk_tier") or "low"),
        )
    return [source_row(path, True, "ingested")]


def ingest_deterministic_tool_substrate(store: ArtifactStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry_path = REPORTS / "deterministic_tool_registry.json"
    registry = read_json(registry_path)
    if registry:
        source = rel_or_abs(registry_path)
        for tool in registry.get("tools", []) if isinstance(registry.get("tools"), list) else []:
            if not isinstance(tool, dict):
                continue
            object_id = stable_id("tool", tool.get("id"), tool.get("replay_checksum", ""))
            verification_state = "available" if get_path(tool, ["dependency_status", "available"], False) else "dependency_unverified"
            store.upsert_object(
                object_id=object_id,
                object_type="Tool",
                title=str(tool.get("id") or object_id),
                content=tool,
                source_path=source,
                provenance={"report": source, "policy": registry.get("policy")},
                verification_state=verification_state,
                release_state=str(tool.get("trust_tier") or "local"),
            )
        rows.append(source_row(registry_path, True, "ingested"))
    else:
        rows.append(source_row(registry_path, False, "missing_or_invalid"))

    report_path = REPORTS / "deterministic_tool_substrate.json"
    report = read_json(report_path)
    if report:
        source = rel_or_abs(report_path)
        command_id = stable_id("command", "deterministic_tool_substrate", report.get("created_utc", ""))
        store.upsert_object(
            object_id=command_id,
            object_type="Command",
            title="deterministic tool substrate run",
            content={
                "summary": report.get("summary", {}),
                "recommendation": report.get("recommendation", {}),
                "public_benchmark_boundary": report.get("public_benchmark_boundary", {}),
            },
            source_path=source,
            provenance={"report": source, "policy": report.get("policy")},
            verification_state=str(report.get("trigger_state") or "unknown"),
            release_state="internal",
        )
        for result in report.get("tool_results", []) if isinstance(report.get("tool_results"), list) else []:
            if not isinstance(result, dict):
                continue
            artifact_id = stable_id("artifact", result.get("run_id"), result.get("replay_checksum", ""))
            store.upsert_object(
                object_id=artifact_id,
                object_type="Artifact",
                title=f"{result.get('tool_id')} {result.get('case_id')}",
                content=result,
                source_path=source,
                provenance={"report": source, "evidence_ref": result.get("evidence_ref")},
                verification_state="verified" if result.get("verified") else str(result.get("state") or "unknown"),
                release_state="internal",
            )
            store.add_relationship(source_id=command_id, target_id=artifact_id, relation="emits_tool_result", source_path=source)
            claim_id = str(result.get("claim_id") or stable_id("claim", artifact_id))
            store.upsert_object(
                object_id=claim_id,
                object_type="Claim",
                title=f"{result.get('tool_id')} case {result.get('case_id')}",
                content={
                    "claim_id": claim_id,
                    "tool_id": result.get("tool_id"),
                    "case_id": result.get("case_id"),
                    "support_state": "SUPPORTED" if result.get("verified") else "UNSUPPORTED",
                    "evidence_refs": [result.get("evidence_ref")],
                    "replay_checksum": result.get("replay_checksum"),
                },
                source_path=source,
                provenance={"report": source},
                verification_state="SUPPORTED" if result.get("verified") else "UNSUPPORTED",
                release_state="internal",
            )
            store.add_relationship(source_id=claim_id, target_id=artifact_id, relation="supported_by", source_path=source)
        rows.append(source_row(report_path, True, "ingested"))
    else:
        rows.append(source_row(report_path, False, "missing_or_invalid"))

    graph_path = REPORTS / "deterministic_tool_artifact_graph.json"
    graph = read_json(graph_path)
    if graph:
        source = rel_or_abs(graph_path)
        for artifact in graph.get("artifacts", []) if isinstance(graph.get("artifacts"), list) else []:
            if not isinstance(artifact, dict):
                continue
            object_id = str(artifact.get("id") or stable_id("artifact", json.dumps(artifact, sort_keys=True)))
            store.upsert_object(
                object_id=object_id,
                object_type=artifact_type(str(artifact.get("type") or "Artifact")),
                title=str(artifact.get("title") or object_id),
                content=artifact,
                source_path=source,
                provenance={"report": source, "policy": graph.get("policy")},
                verification_state=str(artifact.get("support_state") or "unknown"),
                release_state="internal",
            )
        for edge in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
            if isinstance(edge, dict) and edge.get("source") and edge.get("target"):
                store.add_relationship(
                    source_id=str(edge["source"]),
                    target_id=str(edge["target"]),
                    relation=str(edge.get("relation") or "related_to"),
                    source_path=source,
                    content=edge,
                )
        rows.append(source_row(graph_path, True, "ingested"))
    else:
        rows.append(source_row(graph_path, False, "missing_or_invalid"))
    return rows


def ingest_command_executor(store: ArtifactStore, path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return [source_row(path, False, "missing_or_invalid")]
    source = rel_or_abs(path)
    command_id = str(payload.get("command_contract_id") or stable_id("command", source))
    executor_id = stable_id("artifact", "viea_command_executor", command_id)
    store.upsert_object(
        object_id=executor_id,
        object_type="Artifact",
        title="VIEA command executor run",
        content=payload,
        source_path=source,
        provenance={"report": source, "command_contract_id": command_id},
        verification_state=str(payload.get("trigger_state") or "unknown"),
        release_state="internal",
    )
    if command_id:
        store.add_relationship(source_id=command_id, target_id=executor_id, relation="executed_by", source_path=source)
    for call in payload.get("specialist_calls", []) if isinstance(payload.get("specialist_calls"), list) else []:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or stable_id("call", command_id, call.get("stage"), call.get("arm")))
        store.upsert_object(
            object_id=call_id,
            object_type="Artifact",
            title=f"executor call {call.get('stage')}",
            content=call,
            source_path=source,
            provenance={"report": source, "command_contract_id": command_id},
            verification_state=str(call.get("status") or "unknown"),
            release_state="internal",
        )
        store.add_relationship(source_id=executor_id, target_id=call_id, relation="emits_specialist_call", source_path=source)
    for packet in payload.get("runtime_packets", []) if isinstance(payload.get("runtime_packets"), list) else []:
        if not isinstance(packet, dict):
            continue
        packet_id = stable_id("runtime_packet", packet.get("path"), command_id)
        store.upsert_object(
            object_id=packet_id,
            object_type="RuntimeAdapter",
            title=str(packet.get("file") or "digital runtime packet"),
            content=packet,
            source_path=source,
            provenance={"report": source, "command_contract_id": command_id},
            verification_state="present" if packet.get("exists") else "missing",
            release_state="internal",
        )
        store.add_relationship(source_id=executor_id, target_id=packet_id, relation="writes_runtime_packet", source_path=source)
    for residual in payload.get("residuals", []) if isinstance(payload.get("residuals"), list) else []:
        if not isinstance(residual, dict):
            continue
        residual_id = str(residual.get("id") or stable_id("residual", json.dumps(residual, sort_keys=True)))
        store.upsert_object(
            object_id=residual_id,
            object_type="Residual",
            title=str(residual.get("failure_type") or residual_id),
            content=residual,
            source_path=source,
            provenance={"report": source, "command_contract_id": command_id},
            verification_state=str(residual.get("severity") or "open"),
            release_state=str(residual.get("promotion_status") or "track"),
        )
        store.add_relationship(source_id=executor_id, target_id=residual_id, relation="produces_residual", source_path=source)
    feedback_id = stable_id("feedback", "viea_command_executor", command_id)
    store.upsert_object(
        object_id=feedback_id,
        object_type="Feedback",
        title="VIEA command executor feedback",
        content={
            "gates": payload.get("verification_gates", []),
            "residuals": payload.get("residuals", []),
            "runtime_packets": payload.get("runtime_packets", []),
        },
        source_path=source,
        provenance={"report": source, "command_contract_id": command_id},
        verification_state=str(payload.get("trigger_state") or "unknown"),
        release_state="internal",
    )
    store.add_relationship(source_id=executor_id, target_id=feedback_id, relation="produces_feedback", source_path=source)
    return [source_row(path, True, "ingested")]


def ingest_plan_compiler(store: ArtifactStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    report_path = REPORTS / "theseus_plan_compiler.json"
    dags_path = REPORTS / "theseus_plan_compiled_dags.json"
    report = read_json(report_path)
    dags = read_json(dags_path)
    if not report:
        rows.append(source_row(report_path, False, "missing_or_invalid"))
    if not dags:
        rows.append(source_row(dags_path, False, "missing_or_invalid"))
    if not report or not dags:
        return rows
    source = rel_or_abs(dags_path)
    for goal in dags.get("compiled_goals", []) if isinstance(dags.get("compiled_goals"), list) else []:
        if not isinstance(goal, dict):
            continue
        contract = goal.get("contract") if isinstance(goal.get("contract"), dict) else {}
        command_id = str(contract.get("contract_id") or stable_id("command", goal.get("goal_id"), goal.get("contract_hash", "")))
        store.upsert_object(
            object_id=command_id,
            object_type="Command",
            title=str(goal.get("title") or goal.get("goal_id") or "compiled Theseus goal"),
            content=contract,
            source_path=source,
            provenance={"report": rel_or_abs(report_path), "dags": source, "policy": dags.get("policy")},
            verification_state=str(goal.get("trigger_state") or "unknown"),
            release_state="internal",
        )
        for node in goal.get("nodes", []) if isinstance(goal.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            node_id = stable_id("artifact", node.get("node_id"), node.get("semantic_hash", ""))
            store.upsert_object(
                object_id=node_id,
                object_type="Artifact",
                title=str(node.get("node_id") or "compiled plan node"),
                content={
                    "node_id": node.get("node_id"),
                    "op": node.get("op"),
                    "executor_backend": node.get("executor_backend"),
                    "route": node.get("route"),
                    "vcm_context_slice": node.get("vcm_context_slice"),
                    "execution_packet": node.get("execution_packet"),
                    "tool_requirements": node.get("tool_requirements"),
                },
                source_path=source,
                provenance={"report": rel_or_abs(report_path), "dags": source},
                verification_state="compiled",
                release_state="internal",
            )
            store.add_relationship(source_id=command_id, target_id=node_id, relation="has_plan_node", source_path=source)
            for claim in node.get("claim_objects", []) if isinstance(node.get("claim_objects"), list) else []:
                if not isinstance(claim, dict):
                    continue
                claim_id = str(claim.get("claim_id") or stable_id("claim", node_id, json.dumps(claim, sort_keys=True)))
                store.upsert_object(
                    object_id=claim_id,
                    object_type="Claim",
                    title=str(claim.get("predicate") or claim_id),
                    content=claim,
                    source_path=source,
                    provenance={"report": rel_or_abs(report_path), "dags": source},
                    verification_state=str(claim.get("assurance_level") or "planned"),
                    release_state="internal",
                )
                store.add_relationship(source_id=claim_id, target_id=node_id, relation="planned_evidence_for", source_path=source)
    rows.append(source_row(report_path, True, "ingested"))
    rows.append(source_row(dags_path, True, "ingested"))
    return rows


def ingest_viea_execution_spine(store: ArtifactStore, path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return [source_row(path, False, "missing_or_invalid")]
    source = rel_or_abs(path)
    run_id = str(payload.get("run_id") or stable_id("viea_run", source, payload.get("created_utc", "")))
    command_id = stable_id("command", "viea_execution_spine", run_id)
    store.upsert_object(
        object_id=command_id,
        object_type="Command",
        title="VIEA execution spine private run",
        content={
            "run_id": run_id,
            "summary": payload.get("summary", {}),
            "ab_comparison": payload.get("ab_comparison", {}),
            "boundaries": payload.get("boundaries", {}),
        },
        source_path=source,
        provenance={"report": source, "policy": payload.get("policy")},
        verification_state=str(payload.get("trigger_state") or "unknown"),
        release_state="internal",
    )
    packet = payload.get("compiled_packet") if isinstance(payload.get("compiled_packet"), dict) else {}
    packet_id = stable_id("artifact", "compiled_packet", packet.get("packet_hash", ""), run_id)
    store.upsert_object(
        object_id=packet_id,
        object_type="Artifact",
        title="compiled VIEA local deterministic packet",
        content=packet,
        source_path=source,
        provenance={"report": source, "run_id": run_id},
        verification_state="present" if packet.get("packet_id") else "missing",
        release_state="internal",
    )
    store.add_relationship(source_id=command_id, target_id=packet_id, relation="executes_packet", source_path=source)

    for result in payload.get("compiled_execution_results", []) if isinstance(payload.get("compiled_execution_results"), list) else []:
        if not isinstance(result, dict):
            continue
        artifact_id = stable_id("artifact", result.get("run_id"), result.get("case_id"), result.get("replay_checksum", ""))
        store.upsert_object(
            object_id=artifact_id,
            object_type="Artifact",
            title=f"{result.get('tool_id')} {result.get('case_id')}",
            content=result,
            source_path=source,
            provenance={"report": source, "evidence_ref": result.get("evidence_ref")},
            verification_state="verified" if result.get("verified") else str(result.get("state") or "unknown"),
            release_state="internal",
        )
        store.add_relationship(source_id=command_id, target_id=artifact_id, relation="emits_executed_tool_result", source_path=source)

    vcm_path = resolve(get_path(payload, ["outputs", "vcm_artifacts"], ""))
    if vcm_path.exists():
        for row in read_jsonl(vcm_path):
            artifact_id = stable_id("artifact", row.get("vcm_address"), row.get("replay_checksum", ""))
            store.upsert_object(
                object_id=artifact_id,
                object_type="Artifact",
                title=f"VCM tool artifact {row.get('case_id')}",
                content=row,
                source_path=rel_or_abs(vcm_path),
                provenance={"report": source, "run_id": run_id},
                verification_state=str(row.get("support_state") or "unknown"),
                release_state="internal",
            )
            store.add_relationship(source_id=command_id, target_id=artifact_id, relation="writes_vcm_artifact", source_path=rel_or_abs(vcm_path))

    traces_path = resolve(get_path(payload, ["outputs", "learning_traces"], ""))
    if traces_path.exists():
        trace_count = 0
        for row in read_jsonl(traces_path):
            trace_count += 1
            feedback_id = stable_id("feedback", row.get("trace_id"), row.get("evidence_ref", ""))
            store.upsert_object(
                object_id=feedback_id,
                object_type="Feedback",
                title=f"tool-use learning trace {row.get('selected_tool')}",
                content=row,
                source_path=rel_or_abs(traces_path),
                provenance={"report": source, "run_id": run_id},
                verification_state=str(row.get("verifier_outcome") or "unknown"),
                release_state="training_candidate" if get_path(row, ["training_eligibility", "eligible"], False) else "internal",
            )
            store.add_relationship(source_id=command_id, target_id=feedback_id, relation="emits_tool_use_feedback", source_path=rel_or_abs(traces_path))
        if trace_count == 0:
            store.upsert_object(
                object_id=stable_id("feedback", "empty_viea_tool_use_trace", run_id),
                object_type="Feedback",
                title="empty VIEA tool-use trace",
                content={"run_id": run_id, "trace_count": 0},
                source_path=rel_or_abs(traces_path),
                provenance={"report": source},
                verification_state="empty",
                release_state="internal",
            )

    training_evidence_path = resolve(get_path(payload, ["outputs", "training_evidence"], ""))
    if training_evidence_path.exists():
        for row in read_jsonl(training_evidence_path):
            evidence_id = str(row.get("evidence_id") or stable_id("feedback", row.get("trace_id"), "training_evidence"))
            store.upsert_object(
                object_id=evidence_id,
                object_type="Feedback",
                title=f"tool-use training evidence {row.get('selected_tool')}",
                content=row,
                source_path=rel_or_abs(training_evidence_path),
                provenance={"report": source, "run_id": run_id},
                verification_state=str(row.get("support_state") or "unknown"),
                release_state="training_evidence",
            )
            store.add_relationship(source_id=command_id, target_id=evidence_id, relation="emits_training_evidence", source_path=rel_or_abs(training_evidence_path))

    residuals = payload.get("residuals") if isinstance(payload.get("residuals"), dict) else {}
    for residual in residuals.get("rows", []) if isinstance(residuals.get("rows"), list) else []:
        if not isinstance(residual, dict):
            continue
        residual_id = str(residual.get("residual_id") or stable_id("residual", json.dumps(residual, sort_keys=True)))
        store.upsert_object(
            object_id=residual_id,
            object_type="Residual",
            title=f"{residual.get('category')} {residual.get('tool_id')}",
            content=residual,
            source_path=source,
            provenance={"report": source, "run_id": run_id},
            verification_state=str(residual.get("state") or "open"),
            release_state="repair_target",
        )
        store.add_relationship(source_id=command_id, target_id=residual_id, relation="produces_residual", source_path=source)

    loop = payload.get("loop_closure_candidates") if isinstance(payload.get("loop_closure_candidates"), dict) else {}
    for candidate in loop.get("candidates", []) if isinstance(loop.get("candidates"), list) else []:
        if not isinstance(candidate, dict):
            continue
        tool_id = str(candidate.get("candidate_id") or stable_id("tool", candidate.get("tool_id"), run_id))
        store.upsert_object(
            object_id=tool_id,
            object_type="Tool",
            title=f"loop candidate {candidate.get('tool_id')}",
            content=candidate,
            source_path=source,
            provenance={"report": source, "run_id": run_id},
            verification_state=str(candidate.get("status") or "candidate"),
            release_state="candidate",
        )
        store.add_relationship(source_id=command_id, target_id=tool_id, relation="proposes_loop_closure_tool", source_path=source)

    procedural = payload.get("verified_procedural_tools") if isinstance(payload.get("verified_procedural_tools"), dict) else {}
    for tool in procedural.get("tools", []) if isinstance(procedural.get("tools"), list) else []:
        if not isinstance(tool, dict):
            continue
        object_id = str(tool.get("procedural_tool_id") or stable_id("tool", tool.get("tool_id"), run_id))
        store.upsert_object(
            object_id=object_id,
            object_type="Tool",
            title=f"verified procedural tool {tool.get('tool_id')}",
            content=tool,
            source_path=source,
            provenance={"report": source, "run_id": run_id},
            verification_state=str(tool.get("status") or "candidate"),
            release_state="procedural_tool" if tool.get("promotion_allowed") else "candidate",
        )
        store.add_relationship(source_id=command_id, target_id=object_id, relation="emits_procedural_tool_record", source_path=source)

    return [source_row(path, True, "ingested")]


def ingest_benchmark_reports(store: ArtifactStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        REPORTS / "learning_scoreboard.json",
        REPORTS / "broad_transfer_matrix.json",
        REPORTS / "transfer_generalization_audit.json",
        REPORTS / "candidate_promotion_gate.json",
        REPORTS / "benchmaxx_curriculum.json",
    ]:
        payload = read_json(path)
        if not payload:
            rows.append(source_row(path, False, "missing_or_invalid"))
            continue
        object_id = stable_id("benchmark", path.name, payload.get("policy", ""))
        store.upsert_object(
            object_id=object_id,
            object_type="Benchmark",
            title=path.stem,
            content=payload,
            source_path=rel_or_abs(path),
            provenance={"report": rel_or_abs(path)},
            verification_state=str(payload.get("trigger_state") or "unknown"),
            release_state="calibration_or_governance",
        )
        rows.append(source_row(path, True, "ingested"))
    return rows


def ingest_growth_reports(store: ArtifactStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, object_type, title in [
        (REPORTS / "architecture_guidance_loop.json", "Artifact", "architecture guidance loop"),
        (REPORTS / "viea_command_executor.json", "Artifact", "command contract executor"),
        (REPORTS / "digital_runtime_adapter.json", "RuntimeAdapter", "digital runtime adapter"),
        (REPORTS / "private_repo_repair_curriculum.json", "Benchmark", "private repo repair curriculum"),
        (REPORTS / "workflow_tool_compiler_v2.json", "Tool", "workflow-to-tool compiler v2"),
        (REPORTS / "symliquid_substrate_map.json", "Artifact", "SymLiquid substrate map"),
        (REPORTS / "teacher_architect_loop.json", "Artifact", "teacher architect loop"),
        (REPORTS / "feedback_ratchet.json", "Feedback", "feedback ratchet"),
        (REPORTS / "viea_autonomy_spine.json", "Artifact", "VIEA autonomy spine"),
        (REPORTS / "feedback_action_queue.json", "Feedback", "feedback action queue"),
        (REPORTS / "viea_action_executor.json", "Feedback", "VIEA action executor"),
        (REPORTS / "resource_governor.json", "ResourceEvent", "resource governor"),
        (REPORTS / "performance_optimizer.json", "ResourceEvent", "performance optimizer"),
        (REPORTS / "broad_transfer_action_queue.json", "Benchmark", "broad transfer action queue"),
        (REPORTS / "repo_repair_main_curriculum.json", "Benchmark", "repo repair main curriculum"),
        (REPORTS / "viea_repo_repair_learner.json", "Benchmark", "VIEA repo repair learner"),
        (REPORTS / "teacher_architect_closure.json", "Artifact", "teacher architect closure"),
        (REPORTS / "teacher_architect_experiment_runner.json", "Artifact", "teacher architect experiment runner"),
        (REPORTS / "symliquid_state_engine_queue.json", "Artifact", "SymLiquid state engine queue"),
        (REPORTS / "symliquid_state_engine.json", "Artifact", "SymLiquid state engine"),
    ]:
        payload = read_json(path)
        if not payload:
            rows.append(source_row(path, False, "missing_or_invalid"))
            continue
        store.upsert_object(
            object_id=stable_id(object_type, title, payload.get("policy", "")),
            object_type=object_type,
            title=title,
            content=payload,
            source_path=rel_or_abs(path),
            provenance={"report": rel_or_abs(path)},
            verification_state=str(payload.get("trigger_state") or "unknown"),
            release_state="internal",
        )
        rows.append(source_row(path, True, "ingested"))
    return rows


def build_summary(store: ArtifactStore, *, db_path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    counts = store.counts_by_type()
    required_v0_types = [
        "Artifact",
        "Claim",
        "Critique",
        "Release",
        "Feedback",
        "ResourceEvent",
    ]
    required_present = {object_type: counts.get(object_type, 0) > 0 for object_type in required_v0_types}
    return {
        "db_path": rel_or_abs(db_path),
        "object_count": store.object_count(),
        "relationship_count": store.relationship_count(),
        "counts_by_type": counts,
        "required_object_types_present": required_present,
        "missing_required_object_types": [key for key, present in required_present.items() if not present],
        "source_count": len(sources),
        "ingested_source_count": sum(1 for row in sources if row.get("ingested")),
        "sources": sources,
        "promotion_evidence": False,
        "score_semantics": "artifact kernel scaffolding only; student learning remains in broad transfer reports",
    }


def trigger_state(summary: dict[str, Any]) -> str:
    missing = summary.get("missing_required_object_types") or []
    if missing:
        return "YELLOW"
    return "GREEN"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# VIEA Artifact Kernel",
        "",
        f"- trigger_state: `{payload['trigger_state']}`",
        f"- database: `{payload['database']}`",
        f"- objects: `{summary['object_count']}`",
        f"- relationships: `{summary['relationship_count']}`",
        "",
        "## Counts By Type",
        "",
    ]
    for key, value in summary["counts_by_type"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Missing Required Types", ""])
    missing = summary.get("missing_required_object_types") or []
    lines.append(", ".join(f"`{item}`" for item in missing) if missing else "None.")
    lines.extend(["", "## Rule", "", payload["rules"]["learning_boundary"], ""])
    return "\n".join(lines)


def artifact_type(raw: str) -> str:
    mapping = {
        "world": "World",
        "command": "Command",
        "claim": "Claim",
        "critique": "Critique",
        "tool": "Tool",
        "benchmark": "Benchmark",
        "residual": "Residual",
        "release": "Release",
        "feedback": "Feedback",
        "primitive": "Primitive",
        "arm": "SpecialistModule",
        "compile_target": "CompileTarget",
    }
    return mapping.get(raw.lower(), "Artifact")


def source_row(path: Path, ingested: bool, status: str) -> dict[str, Any]:
    return {
        "path": rel_or_abs(path),
        "exists": path.exists(),
        "ingested": bool(ingested),
        "status": status,
    }


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix.lower()}_{digest}"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
