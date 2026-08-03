#!/usr/bin/env python3
"""Materialize deterministic minimal parent/target archives for S3 adequacy."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy as adequacy  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_materialization.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v2.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_materialization_v1"


class ContentClient(Protocol):
    request_count: int
    response_digests: list[str]

    def get_file(self, repository: str, revision: str, path: str) -> bytes | None: ...


class GitHubCliContentClient:
    def __init__(self) -> None:
        self.request_count = 0
        self.response_digests: list[str] = []

    def get_file(self, repository: str, revision: str, path: str) -> bytes | None:
        endpoint = f"repos/{repository}/contents/{path}"
        completed = subprocess.run(
            ["gh", "api", "-X", "GET", endpoint, "-f", f"ref={revision}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=False,
            timeout=120,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            if "HTTP 404" in stderr or "Not Found" in stderr:
                self.request_count += 1
                self.response_digests.append(hashlib.sha256(completed.stderr).hexdigest())
                return None
            raise RuntimeError(f"github_content_error:{repository}:{revision}:{path}:{stderr[-1000:]}")
        self.request_count += 1
        self.response_digests.append(hashlib.sha256(completed.stdout).hexdigest())
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or payload.get("type") != "file":
            raise RuntimeError(f"github_content_not_file:{repository}:{revision}:{path}")
        encoded = str(payload.get("content") or "").replace("\n", "")
        return base64.b64decode(encoded, validate=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        try:
            report = materialize(config_path, client=GitHubCliContentClient())
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "faults": sorted(set(report["faults"] + [f"{type(exc).__name__}:{exc}"[:4000]])),
                "partial_archive_set_admitted": False,
            }
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("config_policy_invalid")
    state = config.get("state")
    if state not in {
        "FIXED_BEFORE_SOURCE_BYTE_RETRIEVAL",
        "AMENDED_AND_FIXED_BEFORE_RENEWED_SOURCE_BYTE_RETRIEVAL",
    }:
        faults.append("config_state_invalid")
    loaded: dict[str, dict[str, Any]] = {}
    for path_key, hash_key in (
        ("adequacy_preregistration", "adequacy_preregistration_sha256"),
        ("source_candidates", "source_candidates_sha256"),
        ("metadata_report", "metadata_report_sha256"),
    ):
        path = resolve(str(config.get(path_key) or ""))
        if not path.is_file() or sha256_file(path) != str(config.get(hash_key) or ""):
            faults.append(f"binding_invalid:{path_key}")
        else:
            loaded[path_key] = read_json(path)
    metadata = loaded.get("metadata_report", {})
    if metadata.get("trigger_state") != "GREEN" or metadata.get("selection_admitted") is not True:
        faults.append("metadata_selection_not_green")
    if len(adequacy.dictionaries(metadata.get("rows"))) != 18:
        faults.append("metadata_row_count_invalid")
    if state == "AMENDED_AND_FIXED_BEFORE_RENEWED_SOURCE_BYTE_RETRIEVAL":
        for path_key, hash_key in (
            ("prior_source_failure_report", "prior_source_failure_report_sha256"),
            ("prior_source_failure_audit", "prior_source_failure_audit_sha256"),
        ):
            path = resolve(str(config.get(path_key) or ""))
            if not path.is_file() or sha256_file(path) != str(config.get(hash_key) or ""):
                faults.append(f"binding_invalid:{path_key}")
    policy = adequacy.mapping(config.get("archive_policy"))
    expected_policy = {
        "selected_source_and_license_files_only": True,
        "deterministic_tar_gzip": True,
        "mtime": 0,
        "uid": 0,
        "gid": 0,
        "mode": "0644",
        "symlinks_allowed": False,
        "parent_missing_target_added_selected_path_allowed": True,
        "target_missing_selected_path_allowed": False,
    }
    if policy != expected_policy:
        faults.append("archive_policy_invalid")
    authority = adequacy.mapping(config.get("authority"))
    if authority.get("public_source_file_retrieval_authorized") is not True:
        faults.append("source_retrieval_not_authorized")
    if any(
        authority.get(key) is not False
        for key in authority
        if key != "public_source_file_retrieval_authorized"
    ):
        faults.append("authority_boundary_invalid")
    return {
        "policy": POLICY,
        "stage": "materialization_preflight",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "archive_set_admitted": False,
        "partial_archive_set_admitted": False,
        "counters": zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
    }


def materialize(config_path: Path, *, client: ContentClient) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = read_json(config_path)
    candidates = read_json(resolve(str(config["source_candidates"])))
    metadata = read_json(resolve(str(config["metadata_report"])))
    metadata_by_index = {
        int(row["index"]): row for row in adequacy.dictionaries(metadata.get("rows"))
    }
    output_directory = resolve(str(config["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for selected in adequacy.dictionaries(candidates.get("candidates")):
        index = int(selected["index"])
        meta = metadata_by_index[index]
        row, row_faults = materialize_pair(
            client,
            selected,
            meta,
            output_directory,
        )
        rows.append(row)
        faults.extend(f"task_{index}:{fault}" for fault in row_faults)
    admitted = not faults and len(rows) == 18
    counters = zero_counters()
    counters["network_source_calls"] = client.request_count
    counters["source_archives_materialized"] = sum(
        len(adequacy.mapping(row.get("archives"))) for row in rows
    )
    return {
        **before,
        "stage": "source_materialization_complete",
        "trigger_state": "GREEN" if admitted else "RED",
        "faults": sorted(set(faults)),
        "archive_set_admitted": admitted,
        "partial_archive_set_admitted": False,
        "rows": rows,
        "transport": {
            "provider": "GitHub_public_contents_API_via_gh_cli",
            "request_count": client.request_count,
            "response_digest_chain_sha256": adequacy.stable_hash(client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "counters": counters,
    }


def materialize_pair(
    client: ContentClient,
    selected: dict[str, Any],
    metadata: dict[str, Any],
    output_directory: Path,
) -> tuple[dict[str, Any], list[str]]:
    index = int(selected["index"])
    repository = str(selected["repository"])
    parent_revision = str(metadata["parent_revision"])
    target_revision = str(metadata["target_revision"])
    source_paths = [str(value) for value in selected["selected_source_paths"]]
    license_path = str(metadata["license"]["path"])
    requested_paths = list(dict.fromkeys(source_paths + [license_path]))
    pair: dict[str, dict[str, bytes]] = {"parent": {}, "target": {}}
    faults: list[str] = []
    for role, revision in (("parent", parent_revision), ("target", target_revision)):
        for path in requested_paths:
            validate_member_path(path)
            content = client.get_file(repository, revision, path)
            if content is not None:
                pair[role][path] = content
        if license_path not in pair[role]:
            faults.append(f"{role}_license_missing")
    for path in source_paths:
        if path not in pair["target"]:
            faults.append(f"target_selected_source_missing:{path}")
    if not any(path in pair["parent"] for path in source_paths):
        faults.append("parent_has_no_selected_source")
    if all(
        pair["parent"].get(path) == pair["target"].get(path)
        for path in source_paths
    ):
        faults.append("selected_source_bytes_unchanged")
    root = f"semantic-adequacy-{index:02d}"
    archive_rows: dict[str, Any] = {}
    if not faults:
        for role in ("parent", "target"):
            archive = output_directory / f"{root}-{role}.tar.gz"
            write_deterministic_archive(archive, root, pair[role])
            archive_rows[role] = {
                "path": relative(archive),
                "sha256": sha256_file(archive),
                "root": root,
                "members": [
                    {
                        "path": path,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                    }
                    for path, content in sorted(pair[role].items())
                ],
            }
    return {
        "index": index,
        "repository": repository,
        "parent_revision": parent_revision,
        "target_revision": target_revision,
        "stratum": selected["stratum"],
        "selected_source_paths": source_paths,
        "license_path": license_path,
        "archives": archive_rows,
        "faults": faults,
    }, faults


def write_deterministic_archive(
    path: Path,
    root: str,
    files: dict[str, bytes],
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member_path, content in sorted(files.items()):
            validate_member_path(member_path)
            info = tarfile.TarInfo(str(PurePosixPath(root) / member_path))
            info.size = len(content)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(buffer.getvalue())


def validate_member_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value or "\\" in value:
        raise ValueError(f"unsafe_archive_member:{value}")


def zero_counters() -> dict[str, int]:
    return {
        "network_source_calls": 0,
        "source_archives_materialized": 0,
        "parent_target_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def artifact(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "archive_set_admitted": report.get("archive_set_admitted"),
        "row_count": len(adequacy.dictionaries(report.get("rows"))),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
