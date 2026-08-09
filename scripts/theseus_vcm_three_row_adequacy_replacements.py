#!/usr/bin/env python3
"""Replace the three VCM rows invalidated by host/toolchain adequacy walls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_content_language_replacements as content  # noqa: E402
import theseus_vcm_language_replacements as base  # noqa: E402
import theseus_vcm_source_materialization as source  # noqa: E402

POLICY = "project_theseus_vcm_three_row_adequacy_replacements_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_three_row_adequacy_replacements.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        retry = p2a.mapping(config.get("transport_retry_policy"))
        ledger = source.SourceLedger(p2a.resolve(args.checkpoint or config["checkpoint"]), config_path, retry)
        client = source.SourceClient(ledger, retry)
        try:
            report = acquire(config_path, ledger, client, retry)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "THREE_ROW_REPLACEMENT_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "replacement_set_admitted": False,
            }
        report = source.finalize_receipt(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{binding.get('id')}")

    panel_path = p2a.resolve(str(config.get("source_panel") or ""))
    producer_path = p2a.resolve(str(config.get("matched_verifier_report") or ""))
    audit_path = p2a.resolve(str(config.get("matched_verifier_audit") or ""))
    panel = p2a.read_json(panel_path) if panel_path.is_file() else {}
    producer = p2a.read_json(producer_path) if producer_path.is_file() else {}
    audit = p2a.read_json(audit_path) if audit_path.is_file() else {}
    slots = p2a.dicts(config.get("replacement_slots"))
    indices = [integer(row.get("index")) for row in slots]

    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True or panel.get("assembled_task_count") != 62:
        faults.append("source_panel_trigger_invalid")
    if producer.get("trigger_state") != "GREEN" or producer.get("panel_admitted") is not False:
        faults.append("matched_verifier_producer_trigger_invalid")
    if audit.get("trigger_state") != "GREEN" or audit.get("panel_admitted") is not False:
        faults.append("matched_verifier_audit_trigger_invalid")
    expected_dispositions = {integer(row.get("index")): str(row.get("disposition")) for row in p2a.dicts(config.get("required_dispositions"))}
    for name, report in (("producer", producer), ("audit", audit)):
        observed = {integer(row.get("index")): str(row.get("disposition")) for row in p2a.dicts(report.get("rows"))}
        if observed != expected_dispositions:
            faults.append(f"matched_verifier_{name}_dispositions_invalid")
    if indices != [12, 13, 35] or len(set(indices)) != 3:
        faults.append("replacement_slot_cardinality_invalid")
    panel_rows = {integer(row.get("index")): row for row in p2a.dicts(panel.get("assembled_rows"))}
    for slot in slots:
        index = integer(slot.get("index")); old = panel_rows.get(index, {})
        if (
            slot.get("panel") != old.get("panel")
            or slot.get("query_language") != old.get("query_language")
            or slot.get("rejected_repository") != old.get("repository")
            or slot.get("rejected_title_sha256") != old.get("natural_language_request_sha256")
        ):
            faults.append(f"replacement_slot_binding_invalid:{index}")
    if [integer(v) for v in config.get("frozen_qualified_indices", [])] != [16, 25, 56]:
        faults.append("qualified_row_freeze_invalid")

    authority = p2a.mapping(config.get("authority"))
    allowed = {
        "public_metadata_queries_authorized", "public_source_file_retrieval_authorized",
        "public_pr_title_metadata_retrieval_authorized", "local_language_scope_classification_authorized",
        "selected_content_static_language_scan_authorized",
    }
    if any(authority.get(key) is not True for key in allowed) or any(value is not False for key, value in authority.items() if key not in allowed):
        faults.append("authority_boundary_invalid")
    counters = source.zero_counters()
    counters.update({"public_metadata_selection_requests": 0, "local_language_scope_classification_calls": 0})
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "THREE_ROW_ADEQUACY_REPLACEMENT_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "source_panel": artifact(panel_path),
        "matched_verifier_report": artifact(producer_path),
        "matched_verifier_audit": artifact(audit_path),
        "replacement_set_admitted": False,
        "selected_repository_count": 0,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "qualified_rows_rerun": False,
        "frozen_qualified_indices": [16, 25, 56],
        "counters": counters,
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config_path: Path, ledger: source.SourceLedger, client: source.SourceClient, retry: dict[str, Any]) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    panel = p2a.read_json(p2a.resolve(config["source_panel"]))
    effective = json.loads(json.dumps(p2a.read_json(p2a.resolve(config["base_replacement_config"]))))
    effective["replacement_slots"] = [
        {
            "index": row["index"], "panel": row["panel"], "query_language": row["query_language"],
            "rejected_title_sha256": row["rejected_title_sha256"],
        }
        for row in config["replacement_slots"]
    ]
    effective["output_directory"] = config["output_directory"]
    effective["transport_retry_policy"] = config["transport_retry_policy"]
    original_preflight = base.preflight
    original_materialize = source.materialize_row
    original_prior = base.v1.tracked_prior_repositories
    forbidden = [(row["name"], int(row["start"], 16), int(row["end"], 16)) for row in config["forbidden_unicode_scripts"]]
    binary = set(config["binary_extensions"])
    denied = {str(row.get("repository") or "") for row in p2a.dicts(panel.get("assembled_rows"))}

    def materialize_with_scope(*args: Any, **kwargs: Any):
        row, faults, size = original_materialize(*args, **kwargs)
        if not faults and content.selected_content_violations(row, forbidden, binary):
            faults = [*faults, "selected_content_natural_language_out_of_scope"]
        return row, faults, size

    base.preflight = lambda _path: {**before, "trigger_state": "GREEN"}
    source.materialize_row = materialize_with_scope
    base.v1.tracked_prior_repositories = lambda _path: sorted(denied | set(original_prior(config_path)))
    try:
        with tempfile.TemporaryDirectory(prefix="theseus_vcm_three_row_replacements_", dir="/private/tmp") as tmp:
            temp = Path(tmp) / "effective.json"
            p2a.write_json(temp, effective)
            report = base.acquire(temp, ledger, client, retry)
    finally:
        base.preflight = original_preflight
        source.materialize_row = original_materialize
        base.v1.tracked_prior_repositories = original_prior

    if report.get("trigger_state") == "GREEN":
        receipts = []
        for row in p2a.dicts(report.get("replacement_rows")):
            violations = content.selected_content_violations(row, forbidden, binary)
            receipts.append({
                "index": row.get("index"), "repository": row.get("repository"),
                "selected_content_english_scope_passed": not violations, "violations": violations,
            })
        if any(not row["selected_content_english_scope_passed"] for row in receipts):
            raise RuntimeError("post_materialization_content_scope_drift")
        report["content_language_receipts"] = receipts
        report["state"] = "THREE_ADEQUACY_REPLACEMENTS_ENGLISH_AND_SOURCE_BOUND"
    elif report.get("trigger_state") == "RED":
        report["state"] = "THREE_ROW_REPLACEMENT_POOL_EXHAUSTED_NO_PARTIAL_ADMISSION"
    return {
        **report,
        "policy": POLICY,
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "qualified_rows_rerun": False,
        "frozen_qualified_indices": [16, 25, 56],
        "maximum_inference": config.get("maximum_inference"),
    }


def artifact(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)} if path.is_file() else {"path": p2a.rel(path), "sha256": ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "replacement_set_admitted", "selected_repository_count",
        "qualified_rows_rerun", "frozen_qualified_indices", "source_content_retrieval_opened",
        "candidate_packet_materialization_opened", "hidden_evaluation_opened", "faults",
        "rejection_counts", "counters",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
