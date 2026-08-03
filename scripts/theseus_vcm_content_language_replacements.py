#!/usr/bin/env python3
"""Replace seven VCM slots with English-title and English-selected-byte tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_language_replacements as base  # noqa: E402
import theseus_vcm_source_materialization as source  # noqa: E402

POLICY = "project_theseus_vcm_content_language_replacements_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_content_language_replacements.json"


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
        try: report = acquire(config_path, ledger, client, retry)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {**report, "trigger_state":"PAUSED", "state":"CONTENT_LANGUAGE_REPLACEMENT_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION", "faults":[f"{type(exc).__name__}:{exc}"[:4000]], "replacement_set_admitted":False}
        report = source.finalize_receipt(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(base.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN","PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path); faults=[]
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path=p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path)!=binding.get("sha256"): faults.append(f"binding_invalid:{binding.get('id')}")
    audit_path=p2a.resolve(config.get("content_language_audit","")); audit=p2a.read_json(audit_path) if audit_path.is_file() else {}
    panel_path=p2a.resolve(config.get("source_panel","")); panel=p2a.read_json(panel_path) if panel_path.is_file() else {}
    slots=p2a.dicts(config.get("replacement_slots")); indices=[int(row.get("index") or 0) for row in slots]
    if audit.get("trigger_state")!="RED" or audit.get("violation_indices")!=indices or audit.get("source_panel_admission_remains_valid") is not False: faults.append("content_language_trigger_invalid")
    panel_rows={int(row.get("index") or 0):row for row in p2a.dicts(panel.get("assembled_rows"))}
    for slot in slots:
        old=panel_rows.get(int(slot.get("index") or 0),{})
        if slot.get("panel")!=old.get("panel") or slot.get("query_language")!=old.get("query_language") or slot.get("rejected_repository")!=old.get("repository"): faults.append(f"slot_shape_invalid:{slot.get('index')}")
    if len(slots)!=7 or len(set(indices))!=7: faults.append("slot_cardinality_invalid")
    authority=p2a.mapping(config.get("authority")); allowed={"public_metadata_queries_authorized","public_source_file_retrieval_authorized","public_pr_title_metadata_retrieval_authorized","local_language_scope_classification_authorized","selected_content_static_language_scan_authorized"}
    if any(authority.get(key) is not True for key in allowed) or any(value is not False for key,value in authority.items() if key not in allowed): faults.append("authority_boundary_invalid")
    counters=source.zero_counters(); counters.update({"public_metadata_selection_requests":0,"local_language_scope_classification_calls":0})
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"SEVEN_CONTENT_LANGUAGE_REPLACEMENT_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT","faults":sorted(set(faults)),"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"replacement_set_admitted":False,"selected_repository_count":0,"source_content_retrieval_opened":False,"candidate_packet_materialization_opened":False,"hidden_evaluation_opened":False,"counters":counters,"maximum_inference":config.get("maximum_inference")}


def acquire(config_path: Path, ledger: source.SourceLedger, client: source.SourceClient, retry: dict[str, Any]) -> dict[str, Any]:
    before=preflight(config_path)
    if before["trigger_state"]!="GREEN": return before
    config=p2a.read_json(config_path); panel=p2a.read_json(p2a.resolve(config["source_panel"]))
    effective=json.loads(json.dumps(p2a.read_json(p2a.resolve(config["base_replacement_config"]))))
    effective["replacement_slots"]=config["replacement_slots"]
    effective["output_directory"]=config["output_directory"]
    effective["transport_retry_policy"]=config["transport_retry_policy"]
    original_preflight=base.preflight; original_materialize=source.materialize_row; original_prior=base.v1.tracked_prior_repositories
    forbidden=[(row["name"],int(row["start"],16),int(row["end"],16)) for row in config["forbidden_unicode_scripts"]]
    denied={str(row.get("repository") or "") for row in p2a.dicts(panel.get("assembled_rows"))}

    def materialize_with_scope(*args: Any, **kwargs: Any):
        row, faults, size=original_materialize(*args, **kwargs)
        if not faults:
            violations=selected_content_violations(row, forbidden, set(config["binary_extensions"]))
            if violations: faults=[*faults,"selected_content_natural_language_out_of_scope"]
        return row,faults,size

    base.preflight=lambda _path: {**before,"trigger_state":"GREEN"}
    source.materialize_row=materialize_with_scope
    base.v1.tracked_prior_repositories=lambda _path: sorted(denied | set(original_prior(config_path)))
    try:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="theseus_vcm_content_replacements_",dir="/private/tmp") as tmp:
            temp=Path(tmp)/"effective.json"; p2a.write_json(temp,effective)
            report=base.acquire(temp,ledger,client,retry)
    finally:
        base.preflight=original_preflight; source.materialize_row=original_materialize; base.v1.tracked_prior_repositories=original_prior
    if report.get("trigger_state")=="GREEN":
        receipts=[]
        for row in p2a.dicts(report.get("replacement_rows")):
            violations=selected_content_violations(row,forbidden,set(config["binary_extensions"]))
            receipts.append({"index":row.get("index"),"repository":row.get("repository"),"selected_content_english_scope_passed":not violations,"violations":violations})
        if any(not row["selected_content_english_scope_passed"] for row in receipts): raise RuntimeError("post_materialization_content_scope_drift")
        report["content_language_receipts"]=receipts
        report["state"]="SEVEN_ENGLISH_TITLE_AND_SELECTED_CONTENT_REPLACEMENTS_BOUND"
    return {**report,"policy":POLICY,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"maximum_inference":config.get("maximum_inference")}


def selected_content_violations(row: dict[str, Any], ranges: list[tuple[str,int,int]], binary: set[str]) -> list[dict[str, Any]]:
    violations=[]
    for role,raw in p2a.mapping(row.get("archives")).items():
        archive=p2a.mapping(raw); path=p2a.resolve(str(archive.get("path") or "")); root=str(archive.get("root") or "")
        with tarfile.open(path,"r:gz") as handle:
            for member in handle.getmembers():
                if not member.isfile(): continue
                relative=member.name[len(root)+1:] if member.name.startswith(root+"/") else member.name
                if relative.lower().startswith(("license","copying")) or Path(relative).suffix.lower() in binary: continue
                extracted=handle.extractfile(member); payload=extracted.read() if extracted else b""
                try:text=payload.decode("utf-8")
                except UnicodeDecodeError:continue
                hits={name:sum(start<=ord(char)<=end for char in text) for name,start,end in ranges}; hits={k:v for k,v in hits.items() if v}
                if hits: violations.append({"role":role,"path":relative,"scripts":hits})
    return violations

if __name__=="__main__": raise SystemExit(main())
