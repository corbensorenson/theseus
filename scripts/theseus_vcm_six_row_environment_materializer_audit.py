#!/usr/bin/env python3
"""Role-separated audit of the generic six-row environment materializer."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_six_row_environment_materializer as producer  # noqa:E402
POLICY="project_theseus_vcm_six_row_environment_materializer_audit_v1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_six_row_environment_materializer.json"
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=producer.preflight(path);report=p2a.read_json(p2a.resolve(str(cfg.get("report") or "")));audited=[]
 if cfg.get("audit_policy")!=POLICY:faults.append("audit_policy_invalid")
 if report.get("trigger_state")!="GREEN" or report.get("state")!="K2_05_SIX_ROW_ENVIRONMENTS_MATERIALIZED_WITH_SCOPED_DISPOSITIONS" or report.get("panel_admitted") is not False:faults.append("producer_state_invalid")
 observed_store=producer.tree_identity(bound["store"])
 if observed_store!=report.get("retained_shared_store"):faults.append("retained_store_identity_invalid")
 for row in p2a.dicts(report.get("rows")):
  index=int(row.get("index") or 0);receipts=p2a.mapping(row.get("receipts"));disposition=str(row.get("disposition") or "")
  for name,raw in receipts.items():
   receipt=p2a.mapping(raw)
   for stream in ("stdout","stderr"):
    if stream in receipt:
     payload=str(receipt.get(stream) or "").encode()
     if len(payload)!=receipt.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest()!=receipt.get(f"{stream}_sha256") or receipt.get(f"{stream}_complete") is not True:faults.append(f"diagnostic_invalid:{index}:{name}:{stream}")
   if receipt.get("project_selected_output_cap") is not None:faults.append(f"output_cap_invalid:{index}:{name}")
  online=p2a.mapping(receipts.get("online_sync") or receipts.get("online_fetch"));offline=p2a.mapping(receipts.get("offline_sync") or receipts.get("offline_fetch"))
  if disposition=="ENVIRONMENT_MATERIALIZATION_QUALIFIED":
   if online.get("returncode")!=0 or offline.get("returncode")!=0 or online.get("network_denied") is not False or offline.get("network_denied") is not True:faults.append(f"qualified_receipt_invalid:{index}")
   if row.get("manager")=="uv" and receipts.get("online_environment")!=receipts.get("offline_environment"):faults.append(f"environment_replay_mismatch:{index}")
  elif not disposition.startswith("INCONCLUSIVE_"):faults.append(f"unknown_disposition:{index}")
  audited.append({"index":index,"manager":row.get("manager"),"disposition":disposition,"online_returncode":online.get("returncode"),"offline_returncode":offline.get("returncode")})
 if [r["index"] for r in audited]!=[12,13,16,25,35,56]:faults.append("audited_denominator_invalid")
 qualified=sum(r["disposition"]=="ENVIRONMENT_MATERIALIZATION_QUALIFIED" for r in audited)
 if report.get("qualified_task_count")!=qualified or report.get("inconclusive_task_count")!=len(audited)-qualified:faults.append("producer_counts_invalid")
 for key in ("source_build_executions","project_installations","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","teacher_calls"):
  if report.get(key)!=0:faults.append(f"zero_boundary_invalid:{key}")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"K2_05_SIX_ROW_ENVIRONMENT_MATERIALIZATION_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_SIX_ROW_ENVIRONMENT_MATERIALIZATION_AUDIT_FAILED","faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":cfg["report"],"sha256":p2a.sha256_file(p2a.resolve(str(cfg["report"])))},"audited_task_count":len(audited),"qualified_task_count":qualified,"inconclusive_task_count":len(audited)-qualified,"rows":audited,"retained_shared_store":observed_store,"panel_admitted":False,"audit_kind":"role-separated rederivation","network_or_dependency_execution_performed":False,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("audit_maximum_inference")}
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","audited_task_count","qualified_task_count","inconclusive_task_count")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
if __name__=="__main__":raise SystemExit(main())
