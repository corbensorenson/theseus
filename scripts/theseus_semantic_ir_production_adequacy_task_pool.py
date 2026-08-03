#!/usr/bin/env python3
"""Materialize the blind Semantic-IR adequacy task and candidate-prompt pool.

Evaluator-custody manifests retain source provenance and archive bindings.  The
separate candidate packets contain the exact serialized prompt and nothing else
that reaches generation.  Pool sealing fails closed if the production language
cannot represent a selected causal slice or if any answer-identifying metadata
reaches an actual prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402
import theseus_semantic_ir_production as production  # noqa: E402
import theseus_semantic_ir_production_canary as canary  # noqa: E402


MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
CONSTRUCT_REVIEW = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_source_construct_review_v2.json"
CONSTRUCT_REVIEW_SHA256 = "bf2780ba8e38e2d9959aaee4603ca3e7d67907e9f8dda855291a72827476e053"
EVALUATOR_QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
EVALUATOR_QUALIFICATION_SHA256 = "448544a147595413b0d8d0c7523d9442571651a7f11b49d38b9e5a5c9eb9c35a"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
TASK_PREFIX = "theseus_semantic_ir_production_adequacy_task_"
PACKET_PREFIX = "theseus_semantic_ir_production_adequacy_candidate_packet_"
POLICY = "project_theseus_semantic_ir_production_adequacy_task_pool_v1"
PACKET_POLICY = "project_theseus_semantic_ir_production_adequacy_candidate_packet_v1"
MODEL_CONTEXT_TOKENS = 262_144

FORBIDDEN_KEYS = {
    *p2a.FORBIDDEN_TASK_FIELDS,
    "repository", "pull_request", "pull_request_url", "target", "target_archive",
    "target_revision", "target_sha256", "oracle", "evaluator", "benchmark_card",
    "answer_family", "decoder_fields",
}


SPECS: tuple[dict[str, Any], ...] = (
    {"request": "Make cache-entry listing handle legacy manifests that omit size_bytes by reporting the associated data-file size when available and zero when it is absent. Preserve explicit sizes and existing provider and format filtering. Modify only src/prioris_mcp/storage.py.", "require": "A legacy manifest without size_bytes reports its associated data-file byte size, or zero when that file is absent.", "preserve": "An explicit size_bytes value and existing provider and format filtering remain unchanged."},
    {"request": "Preserve the first-occurrence order of instance identifiers when transformed features are reindexed, including inputs whose identifier column is reordered or contains duplicates. Preserve distinct identifiers and the rest of transformation behavior. Modify only sktime/transformations/tsfresh.py.", "require": "Transformed features are reindexed in stable first-occurrence identifier order even when identifiers repeat.", "preserve": "Distinct identifier values and all unrelated transformation behavior remain unchanged."},
    {"request": "Make get_current_value tolerate a device with no mapping for a requested parameter by using the parameter's own code. Preserve the exact mapped code when a device mapping exists. Modify only tests/components/compit/conftest.py.", "require": "A missing device-to-parameter mapping falls back to the requested parameter code.", "preserve": "A present device mapping retains and uses its exact mapped code."},
    {"request": "Ensure path resolution distinguishes an omitted explicit path from an explicitly supplied empty path. Preserve precedence of explicit input over stored configuration and the built-in fallback. Modify only src/lintle/cli.py.", "require": "An explicitly supplied empty path wins, while an omitted path falls through to configured or built-in defaults.", "preserve": "Non-empty explicit, configured, and fallback path precedence remains unchanged."},
    {"request": "When a patterned polygon has a saved fill color, use that valid saved color for its generated extrusion companion without allowing fill-pattern and fill-color to conflict. Preserve expressions, unrelated paint fields, and the default color fallback. Modify only backend/app/modules/catalog/maps/style_json.py.", "require": "A valid saved fill color is used by the generated extrusion companion for a patterned polygon.", "preserve": "Expressions, unrelated paint fields, mutual exclusion, and default-color fallback behavior remain stable."},
    {"request": "Make get_kind resolve named F2Py kind selectors to the numeric kind whose C type matches the declared type mapping. Preserve numeric selectors and unmapped behavior without accepting a mismatched typedef. Modify only numpy/f2py/auxfuncs.py.", "require": "A mapped named kind resolves to the matching numeric kind for its declared type specification.", "preserve": "Numeric kinds, unmapped kinds, and mismatched type mappings retain safe existing behavior."},
    {"request": "Update RangeFrame.sort_ranges so requested keys and descending directions determine a stable range ordering while every non-key value stays attached to its original row. Preserve indexes and reject invalid descending keys. Modify only pyranges1/range_frame/range_frame.py.", "require": "Requested sort keys and descending directions produce the declared stable range order.", "preserve": "Every non-key value remains attached to its original row, indexes are preserved, and invalid descending keys fail safely."},
    {"request": "Make shaped-axis rectilinearity detection invariant to a constant coordinate offset so equivalent low- and high-offset sweeps receive the same orientation decision. Preserve non-serpentine data and finite-value handling. Modify only src/qplot/tools/worker.py.", "require": "Equivalent sweeps differing only by a constant coordinate offset receive the same rectilinearity decision.", "preserve": "Non-serpentine data, invalid-value handling, and zero-span behavior remain stable."},
    {"request": "Ensure ingest_file never emits a text chunk larger than the configured maximum, even when the input has no useful structural separators. Preserve all meaningful text exactly once, avoid empty trailing fragments, and retain source metadata. Modify only linafish/ingest.py.", "require": "Every emitted chunk respects the configured maximum for oversized boundary cases.", "preserve": "Meaningful text is neither lost nor duplicated, empty trailing chunks are avoided, and source metadata remains attached."},
    {"request": "Make hf_env_offline honor an active temporary force-offline window before ordinary environment checks or reachability work. Preserve normal environment-variable behavior when the force window is inactive. Modify only studio/backend/utils/utils.py.", "require": "An active force-offline window reports offline without consulting network reachability.", "preserve": "Inactive force state retains the existing HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE behavior."},
    {"request": "Allow the fallback module to import when Tavily is not installed. If tavily_extract is invoked without that optional dependency, raise the local missing-dependency error; preserve callable behavior when Tavily is installed. Modify only src/mdfetch/fallback.py.", "require": "The module imports without Tavily and fallback use then raises the declared local missing-dependency error.", "preserve": "Installed-Tavily extraction and the existing fallback error behavior remain callable."},
    {"request": "Before converting a challenge writeup, remove one duplicated leading Markdown H1 because the platform already renders the challenge title. Preserve distinct headings, title-free content, and the remainder of the writeup. Modify only ed_push.py.", "require": "A duplicated leading H1 is removed exactly once before conversion.", "preserve": "Distinct headings, title-free content, body content, and upload behavior remain unchanged."},
    {"request": "Treat the management response for a missing message-session collection as an empty result. Preserve successful collections, ordinary empty responses, and unrelated management errors. Modify only sdk/servicebus/azure-servicebus/azure/servicebus/_common/mgmt_handlers.py.", "require": "The declared not-found management response returns an empty session list.", "preserve": "Success, no-content, and unrelated error responses retain their existing behavior."},
    {"request": "Make repository cloning noninteractive and bounded, and return a safe failure when the clone command exits nonzero instead of yielding a source reference or waiting for credentials. Preserve the successful clone path. Modify only src/dd_license_attribution/artifact_management/source_code_manager.py.", "require": "A nonzero clone result returns a bounded failure without producing a source reference or prompting for credentials.", "preserve": "A successful clone and existing reference-discovery behavior remain unchanged."},
    {"request": "Add bounded ordered caching to S3 Express credential refreshes: concurrent refreshes for one bucket must deduplicate, hits must become newest, overflow must evict exactly the oldest entry, and failures must not poison the cache. Modify only aiobotocore/utils.py.", "require": "Credential refreshes are deduplicated per bucket and successful credentials enter a bounded recency-ordered cache.", "preserve": "Cache hits move to newest, overflow evicts only the oldest entry, and failed refreshes leave no poisoned entry."},
    {"request": "Normalize recognizer target bounds into one logical coordinate space before overlap comparison so physical-pixel and logical targets agree under non-unit scaling. Add src/openframe/recognize/coords.py and update src/openframe/recognize/locator.py; preserve unit-scale behavior and bounded dimensions.", "require": "A shared coordinate-normalization unit makes physical and logical target bounds agree under non-unit scaling and is used by overlap comparison.", "preserve": "Unit-scale behavior, round-trip bounds within rounding tolerance, and positive bounded dimensions remain stable."},
    {"request": "Add dependency-aware definition ordering so the sorter cannot move a definition below a decorator, default expression, or other import-time read that depends on it. Add src/funcsort/ordering.py and integrate it into src/funcsort/sorter.py; preserve deterministic content and safe behavior for dependency cycles.", "require": "The sorter respects load-time definition dependencies while fitting its preferred ordering.", "preserve": "Dependency cycles or infeasible constraints retain a deterministic safe order without changing statement content."},
    {"request": "Centralize instrument-type validation and use the same allowlist in configuration objects and every database-write path. Preserve all currently allowed types and reject unknown types consistently. Modify only librae/config/symbols.py and librae/db/timescale_writer.py.", "require": "Configuration and all database-write paths validate instrument types through one shared allowlist.", "preserve": "Every currently allowed type remains accepted and unknown types fail consistently without duplicated divergent constants."},
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    report = materialize_pool(write_artifacts=not args.preflight_only)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "state": report["state"],
        "task_count": report["task_count"],
        "sealed_packet_count": report["sealed_packet_count"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def materialize_pool(*, write_artifacts: bool) -> dict[str, Any]:
    faults = binding_faults()
    materialization = read_json(MATERIALIZATION)
    qualification = read_json(EVALUATOR_QUALIFICATION)
    rows = dictionaries(materialization.get("rows"))
    if len(rows) != 18 or len(SPECS) != 18:
        faults.append("panel_shape_invalid")
    if qualification.get("trigger_state") != "GREEN" or qualification.get(
        "panel_admitted_for_task_packet_materialization"
    ) is not True:
        faults.append("evaluator_qualification_not_green")
    missing_parent: dict[int, list[str]] = {}
    for row in rows:
        present = {
            str(member.get("path") or "")
            for member in dictionaries(mapping(mapping(row.get("archives")).get("parent")).get("members"))
        }
        missing = sorted(set(strings(row.get("selected_source_paths"))) - present)
        if missing:
            missing_parent[integer(row.get("index"))] = missing
    if missing_parent and not production_supports_create_file():
        faults.extend(
            f"production_ir_cannot_create_selected_path:task_{index:02d}:{path}"
            for index, paths in missing_parent.items() for path in paths
        )

    task_rows: list[dict[str, Any]] = []
    if write_artifacts and not faults:
        for row, spec in zip(rows, SPECS, strict=True):
            task_rows.append(materialize_task(row, spec))
        faults.extend(
            fault for row in task_rows for fault in strings(row.get("faults"))
        )

    sealed = sum(row.get("trigger_state") == "GREEN" for row in task_rows)
    green = not faults and sealed == 18
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION" if green else "INVALID_NOT_SEALED",
        "materialization": artifact(MATERIALIZATION),
        "construct_review": artifact(CONSTRUCT_REVIEW),
        "evaluator_qualification": artifact(EVALUATOR_QUALIFICATION),
        "task_pool_owner": artifact(Path(__file__).resolve()),
        "task_count": len(rows),
        "sealed_packet_count": sealed,
        "missing_parent_selected_paths": {
            f"task_{index:02d}": paths for index, paths in sorted(missing_parent.items())
        },
        "production_create_file_supported": production_supports_create_file(),
        "rows": task_rows,
        "faults": sorted(set(faults)),
        "information_flow": {
            "candidate_receives_exact_serialized_packet_only": True,
            "execution_manifest_candidate_visible": False,
            "repository_pr_revision_target_evaluator_or_oracle_candidate_visible": False,
            "recursive_forbidden_key_audit_required": True,
            "actual_serialized_prompt_audited": True,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "completion_boundary": {
            "model_declared_context_window_tokens": MODEL_CONTEXT_TOKENS,
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_context_boundary_hit_invalidates_observation": True,
        },
        "counters": {
            "candidate_or_control_calls": 0,
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": (
            "A GREEN seal establishes only that all 18 independently qualified causal slices "
            "have production-representable, parent-source-only candidate prompts with recursive "
            "anti-cheating checks and physical context headroom. It does not establish model "
            "competence, a Semantic-IR effect, D1, D2, training value, serving, or book support."
        ),
    }


def materialize_task(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    index = integer(row.get("index"))
    suffix = f"{index:02d}"
    parent = mapping(mapping(row.get("archives")).get("parent"))
    target = mapping(mapping(row.get("archives")).get("target"))
    selected = strings(row.get("selected_source_paths"))
    sources = archive_sources(parent, selected)
    existing = [path for path in selected if path in sources]
    missing = [path for path in selected if path not in sources]
    task_path = ROOT / "configs" / f"{TASK_PREFIX}{suffix}.json"
    packet_path = ROOT / "configs" / f"{PACKET_PREFIX}{suffix}.json"
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": f"semantic-ir-adequacy-{suffix}",
        "campaign_index": index,
        "partition": "semantic_ir_production_implementation_adequacy",
        "family": "licensed_source_disjoint_python_causal_repair",
        "natural_request": spec["request"],
        "source_archive": str(parent.get("path") or ""),
        "source_archive_sha256": str(parent.get("sha256") or ""),
        "source_archive_root": str(parent.get("root") or ""),
        "source_provenance": {
            "url": "https://github.com/" + str(row.get("repository") or ""),
            "revision": str(row.get("parent_revision") or ""),
            "license_spdx": license_spdx(index),
        },
        "obligations": [
            {"id": "O1", "kind": "require", "text": spec["require"]},
            {"id": "O2", "kind": "preserve", "text": spec["preserve"]},
            {"id": "O3", "kind": "non_goal", "text": "Modify only the declared allowed effect paths; do not alter unrelated behavior or files."},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
        "allowed_effect_paths": selected,
        "candidate_visible_context": {
            "reads": [
                {"path": path, "start_line": 1, "end_line": len(sources[path].splitlines())}
                for path in existing
            ],
            "searches": [],
            "full_selected_parent_sources": True,
            "missing_allowed_effect_paths": missing,
            "project_selected_character_or_token_cap": None,
        },
        "visible_verifier": {
            "command": [
                "python3", "-c",
                "import ast,sys; [ast.parse(open(p, encoding='utf-8').read(), filename=p) for p in sys.argv[1:]]",
                *selected,
            ],
            "timeout_seconds": 60,
            "answer_specific": False,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": [{"marker": "SyntaxError", "obligation_ids": ["O1", "O2", "O3"]}],
        "semantic_ir_contract": {
            "version": production.HEADER,
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 1_000_000,
            "maximum_units": len(selected) + 4,
            "create_file_allowed_only_for_declared_missing_effect_paths": True,
            "role_partition_source_target_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One implementation-adequacy observation only; no treatment, D1, D2, serving, training, or book-support claim.",
    }
    p2a.write_json(task_path, task)
    task_audit = p4.audit_task(task_path)
    faults = [] if task_audit.get("trigger_state") == "GREEN" else strings(task_audit.get("faults"))
    prompt = ""
    symbols: dict[str, Any] = {}
    if not faults:
        with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-packet-{suffix}-") as directory:
            root = Path(directory) / "source"
            p2a.extract_source_archive(p2a.resolve(task["source_archive"]), root, task["source_archive_root"])
            symbols = p4s.semantic_scope_symbol_table(root, task)
            exact_count = len(dictionaries(symbols.get("nodes")))
            task["semantic_ir_contract"]["maximum_symbol_nodes"] = exact_count
            task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = exact_count
            p2a.write_json(task_path, task)
            common = canary.render_common_context(root, task, symbols)
            if missing:
                common += "\n[MISSING_ALLOWED_EFFECT_PATHS]\n" + "\n".join(missing)
            prompt = production.render_prompt(task, common)
    packet = {
        "policy": PACKET_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": task["opaque_task_id"],
        "serialized_prompt": prompt,
    }
    packet_faults = audit_candidate_packet(packet, row, target)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes >= MODEL_CONTEXT_TOKENS:
        packet_faults.append("physical_context_residual_not_proven_by_utf8_upper_bound")
    faults.extend(packet_faults)
    p2a.write_json(packet_path, packet)
    return {
        "index": index,
        "opaque_task_id": task["opaque_task_id"],
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": sha256_text(prompt),
        "serialized_prompt_utf8_bytes": prompt_bytes,
        "conservative_minimum_residual_tokens": MODEL_CONTEXT_TOKENS - prompt_bytes,
        "full_parent_source_path_count": len(existing),
        "declared_missing_effect_path_count": len(missing),
        "semantic_scope_node_count": len(dictionaries(symbols.get("nodes"))),
        "target_archive_or_source_visible": False,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
    }


def audit_candidate_packet(packet: dict[str, Any], row: dict[str, Any], target: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    for path, key in recursive_keys(packet):
        if key.lower() in FORBIDDEN_KEYS:
            faults.append(f"forbidden_candidate_key:{path}")
    serialized = json.dumps(packet, sort_keys=True)
    forbidden_values = [
        str(row.get("repository") or ""), str(row.get("parent_revision") or ""),
        str(row.get("target_revision") or ""), str(target.get("path") or ""),
        str(target.get("sha256") or ""), "pull request", "github.com/",
    ]
    for value in forbidden_values:
        if value and value.lower() in serialized.lower():
            faults.append("forbidden_candidate_value:" + sha256_text(value)[:12])
    prompt = str(packet.get("serialized_prompt") or "")
    for field in ("natural_request", "obligations", "allowed_effect_paths"):
        if not prompt:
            faults.append(f"candidate_prompt_missing:{field}")
    return sorted(set(faults))


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (MATERIALIZATION, MATERIALIZATION_SHA256, "materialization"),
        (CONSTRUCT_REVIEW, CONSTRUCT_REVIEW_SHA256, "construct_review"),
        (EVALUATOR_QUALIFICATION, EVALUATOR_QUALIFICATION_SHA256, "evaluator_qualification"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"binding_invalid:{label}")
    return faults


def production_supports_create_file() -> bool:
    return bool(
        getattr(production, "CREATE_FILE_OPERATION", "") == "CREATE_FILE"
        and getattr(production, "CREATE_FILE_SUPPORTED", False) is True
    )


def archive_sources(receipt: dict[str, Any], selected: list[str]) -> dict[str, str]:
    archive_path = p2a.resolve(str(receipt.get("path") or ""))
    if p2a.sha256_file(archive_path) != str(receipt.get("sha256") or ""):
        raise ValueError("parent archive digest mismatch")
    root = str(receipt.get("root") or "")
    result: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
            if not member.isfile() or not member.name.startswith(root + "/"):
                continue
            logical = member.name[len(root) + 1 :]
            if logical in selected:
                handle = archive.extractfile(member)
                result[logical] = (handle.read() if handle else b"").decode("utf-8")
    return result


def license_spdx(index: int) -> str:
    candidates = read_json(ROOT / "configs" / "theseus_semantic_ir_production_adequacy_source_candidates_v3.json")
    rows = flatten_candidate_rows(candidates)
    row = next((value for value in rows if integer(value.get("campaign_index")) == index), {})
    return str(row.get("license_spdx") or "UNKNOWN")


def flatten_candidate_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = dictionaries(value.get("candidates"))
    predecessor = value.get("predecessor_registry")
    if predecessor:
        rows = [*flatten_candidate_rows(read_json(p2a.resolve(str(predecessor)))), *rows]
    replacement = mapping(value.get("replacement_candidate"))
    if replacement:
        replace_index = integer(replacement.get("campaign_index"))
        rows = [row for row in rows if integer(row.get("campaign_index")) != replace_index]
        rows.append(replacement)
    return sorted(rows, key=lambda row: integer(row.get("campaign_index")))


def recursive_keys(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            rows.append((path, str(key)))
            rows.extend(recursive_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(recursive_keys(child, f"{prefix}[{index}]"))
    return rows


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def strings(value: Any) -> list[str]:
    return [str(item) for item in value or [] if isinstance(item, str)]


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
