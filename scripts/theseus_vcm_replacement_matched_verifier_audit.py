#!/usr/bin/env python3
"""Role-separately audit replacement common-evaluator receipts."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_replacement_matched_verifier as owner  # noqa:E402
POLICY="project_theseus_vcm_replacement_matched_verifier_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_matched_verifier.json"
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","audited_task_count","qualified_task_count","inconclusive_task_count","faults")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=owner.preflight(path,verify_store=False);report_path=p2a.resolve(cfg["report"]);report=p2a.read_json(report_path) if report_path.is_file() else {}
 if report.get("trigger_state")!="GREEN" or report.get("state")!="THREE_REPLACEMENT_MATCHED_VERIFIERS_EXECUTED_WITH_SCOPED_DISPOSITIONS":faults.append("producer_invalid")
 if report.get("panel_admitted") is not False or report.get("partial_panel_admission_forbidden") is not True or report.get("target_production_transplant_count")!=0 or report.get("network_enabled_calls")!=0:faults.append("producer_policy_invalid")
 rows=[];executions=installs=0
 configured={int(r["index"]):r for r in p2a.dicts(cfg.get("rows"))}
 for actual in p2a.dicts(report.get("rows")):
  index=int(actual.get("index") or 0);expected=configured.get(index,{});sides={s:p2a.mapping(actual.get(s)) for s in ("parent","target")}
  if actual.get("repository")!=expected.get("repository") or actual.get("manager")!=expected.get("manager"):faults.append(f"task_{index}:identity_invalid")
  if set(p2a.strings(actual.get("common_evaluator_paths")))&set(p2a.strings(expected.get("forbidden_transplant_paths"))):faults.append(f"task_{index}:transplant_overlap")
  for side,receipt in sides.items():
   if receipt:
    executions+=1
    if receipt.get("network_denied") is not True or receipt.get("stdout_complete") is not True or receipt.get("stderr_complete") is not True or receipt.get("project_selected_output_cap") is not None:faults.append(f"task_{index}:{side}_policy_invalid")
    for stream in ("stdout","stderr"):
     payload=str(receipt.get(stream) or "").encode()
     if len(payload)!=receipt.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest()!=receipt.get(f"{stream}_sha256"):faults.append(f"task_{index}:{side}:{stream}_identity_invalid")
  if expected.get("manager")=="uv" and p2a.mapping(actual.get("environment")).get("sync"):installs+=1
  disposition=owner.base.derive_disposition(sides,p2a.strings(actual.get("faults")))
  if disposition!=actual.get("disposition"):faults.append(f"task_{index}:disposition_invalid")
  rows.append({"index":index,"parent_returncode":sides["parent"].get("returncode"),"target_returncode":sides["target"].get("returncode"),"disposition":disposition})
 if [r["index"] for r in rows]!=owner.EXPECTED:faults.append("denominator_invalid")
 qualified=sum(r["disposition"]=="QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS" for r in rows)
 if report.get("qualified_task_count")!=qualified or report.get("parent_target_or_evaluator_executions")!=executions or report.get("package_installations")!=installs:faults.append("counter_rederivation_invalid")
 expected_store=p2a.mapping(p2a.mapping(bound.get("sources")).get("environment")).get("retained_shared_store");before=p2a.mapping(report.get("retained_store_before"));after=p2a.mapping(report.get("retained_store_after"))
 if before!=after or before!=expected_store or report.get("retained_store_unchanged") is not True:faults.append("store_identity_invalid")
 if any(v is not False for v in p2a.mapping(cfg.get("audit_authority")).values()):faults.append("audit_authority_invalid")
 green=not faults
 return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"THREE_REPLACEMENT_MATCHED_VERIFIERS_ROLE_SEPARATELY_REDERIVED" if green else "REPLACEMENT_MATCHED_VERIFIER_AUDIT_FAILED","faults":sorted(set(faults)),"audited_task_count":len(rows),"qualified_task_count":qualified,"inconclusive_task_count":len(rows)-qualified,"rows":rows,"retained_store_identity_rederived":before==after==expected_store,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":p2a.rel(report_path),"sha256":p2a.sha256_file(report_path) if report_path.is_file() else ""},"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
