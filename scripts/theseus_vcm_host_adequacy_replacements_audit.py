#!/usr/bin/env python3
"""Role-separately audit host-adequate VCM replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_content_language_replacements as content  # noqa:E402
import theseus_vcm_host_adequacy_replacements as host  # noqa:E402
import theseus_vcm_source_panel_audit as panel_audit  # noqa:E402

POLICY="project_theseus_vcm_host_adequacy_replacements_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_host_adequacy_replacements_audit.json";ROLES={"parent_source","target_source","parent_verifier","target_verifier"}

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));parser.add_argument("--out",default="");args=parser.parse_args();path=p2a.resolve(args.config);config=p2a.read_json(path);report=audit(path);p2a.write_json(p2a.resolve(args.out or config["report"]),report);print(json.dumps(summary(report),indent=2,sort_keys=True));return 0 if report["trigger_state"]=="GREEN" else 2

def audit(config_path:Path=DEFAULT_CONFIG)->dict[str,Any]:
    config=p2a.read_json(config_path);faults=[]
    if config.get("policy")!=POLICY:faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("bindings")):
        path=p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path)!=binding.get("sha256"):faults.append(f"binding_invalid:{binding.get('id')}")
    panel_path=p2a.resolve(config["source_panel"]);producer_path=p2a.resolve(config["producer_report"]);panel=p2a.read_json(panel_path);producer=p2a.read_json(producer_path)
    if panel.get("trigger_state")!="GREEN" or panel.get("source_panel_admitted") is not True:faults.append("source_panel_invalid")
    if producer.get("trigger_state")!="GREEN" or producer.get("replacement_set_admitted") is not True:faults.append("producer_invalid")
    rows=p2a.dicts(producer.get("replacement_rows"));old=p2a.dicts(panel.get("assembled_rows"))[12] if len(p2a.dicts(panel.get("assembled_rows")))>=13 else {};row=rows[0] if len(rows)==1 else {}
    if len(rows)!=1 or int(row.get("index") or 0)!=13:faults.append("replacement_denominator_invalid")
    if row.get("panel")!=old.get("panel") or row.get("query_language")!=old.get("query_language") or row.get("replacement_for_title_sha256")!=old.get("natural_language_request_sha256"):faults.append("replacement_slot_binding_invalid")
    current={str(item.get("repository") or "") for item in p2a.dicts(panel.get("assembled_rows"))}
    if not row.get("repository") or row.get("repository") in current:faults.append("replacement_repository_not_source_disjoint")
    title=str(row.get("natural_language_request") or "")
    if hashlib.sha256(title.encode()).hexdigest()!=row.get("natural_language_request_sha256"):faults.append("title_hash_invalid")
    titles=p2a.dicts(producer.get("language_classification_receipts"));title_receipt=titles[0] if len(titles)==1 else {}
    if title_receipt.get("repository")!=row.get("repository") or title_receipt.get("title_sha256")!=row.get("natural_language_request_sha256") or title_receipt.get("dominant_language")!="en" or title_receipt.get("accepted_english") is not True or title_receipt.get("forbidden_unicode_scripts")!=[]:faults.append("title_language_receipt_invalid")
    contents=p2a.dicts(producer.get("content_language_receipts"));content_receipt=contents[0] if len(contents)==1 else {}
    host_receipts=p2a.dicts(producer.get("host_compatibility_receipts"));host_receipt=host_receipts[0] if len(host_receipts)==1 else {}
    if content_receipt.get("repository")!=row.get("repository") or content_receipt.get("selected_content_english_scope_passed") is not True or content_receipt.get("violations")!=[]:faults.append("content_language_receipt_invalid")
    if host_receipt.get("repository")!=row.get("repository") or host_receipt.get("host_platform")!="darwin" or host_receipt.get("explicit_unsupported_host") is not False or host_receipt.get("hits")!=[] or int(host_receipt.get("scanned_file_count") or 0)<1:faults.append("host_compatibility_receipt_invalid")
    ranges=[(r["name"],int(r["start"],16),int(r["end"],16)) for r in config["forbidden_unicode_scripts"]];binary=set(config["binary_extensions"])
    if content.selected_content_violations(row,ranges,binary):faults.append("selected_content_language_invalid")
    archives=p2a.mapping(row.get("archives"));hashes={};members=total_bytes=0
    if set(archives)!=ROLES or row.get("faults")!=[]:faults.append("archive_set_invalid")
    for role in sorted(ROLES):
        receipt=p2a.mapping(archives.get(role));path=p2a.resolve(str(receipt.get("path") or ""));archive_faults,member_hashes,member_bytes=panel_audit.audit_archive(path,receipt);faults.extend(f"{role}:{fault}" for fault in archive_faults);hashes[role]=member_hashes;members+=len(member_hashes);total_bytes+=member_bytes
    source_changed=panel_audit.selected_paths_changed(row,hashes,"source");verifier_changed=panel_audit.selected_paths_changed(row,hashes,"verifier")
    if not source_changed:faults.append("source_unchanged")
    if not verifier_changed:faults.append("verifier_unchanged")
    counters=p2a.mapping(producer.get("counters"));forbidden=["D1_cases_consumed","D2_cases_consumed","candidate_or_control_calls","external_inference_calls","local_model_calls","parent_target_or_evaluator_executions","teacher_calls","training_rows_written"]
    if any(counters.get(key)!=0 for key in forbidden):faults.append("forbidden_counter_nonzero")
    if producer.get("qualified_rows_rerun") is not False or producer.get("frozen_qualified_indices")!=[16,25,56]:faults.append("qualified_row_freeze_invalid")
    authority=p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()):faults.append("authority_boundary_invalid")
    green=not faults
    return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"HOST_ADEQUATE_REPLACEMENT_ROLE_SEPARATELY_REDERIVED" if green else "HOST_ADEQUATE_REPLACEMENT_AUDIT_FAILED","faults":sorted(set(faults)),"replacement_set_admitted":green,"replacement_index":13,"repository":row.get("repository"),"source_disjoint_from_current_panel":row.get("repository") not in current,"archive_receipt_count":len(archives),"member_receipt_count":members,"total_member_bytes":total_bytes,"source_changed":source_changed,"verifier_changed":verifier_changed,"explicit_unsupported_host":host_receipt.get("explicit_unsupported_host"),"qualified_rows_rerun":False,"frozen_qualified_indices":[16,25,56],"parent_target_or_evaluator_executions":0,"local_model_calls":0,"external_reference_calls":0,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"producer_report":{"path":p2a.rel(producer_path),"sha256":p2a.sha256_file(producer_path)},"maximum_inference":config.get("maximum_inference")}

def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","replacement_set_admitted","replacement_index","repository","source_disjoint_from_current_panel","archive_receipt_count","source_changed","verifier_changed","explicit_unsupported_host","qualified_rows_rerun","parent_target_or_evaluator_executions","local_model_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
