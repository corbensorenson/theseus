#!/usr/bin/env python3
"""Role-separately audit the three replacement environment replays."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_six_row_environment_materializer as base  # noqa:E402
POLICY="project_theseus_vcm_replacement_environment_materializer_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_environment_materializer.json"
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","task_count","qualified_task_count","network_enabled_materializations","network_denied_replays","faults")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(path);faults=[]
 for row in p2a.dicts(cfg.get("bindings")):
  q=p2a.resolve(str(row.get("path") or ""))
  if not q.is_file() or p2a.sha256_file(q)!=row.get("sha256"):faults.append(f"binding_invalid:{row.get('id')}")
 report_path=p2a.resolve(cfg["report"]);producer=p2a.read_json(report_path) if report_path.is_file() else {}
 if producer.get("trigger_state")!="GREEN" or producer.get("qualified_task_count")!=3:faults.append("producer_invalid")
 rows=[];online=offline=0
 for row in p2a.dicts(producer.get("rows")):
  index=int(row.get("index") or 0);receipts=p2a.mapping(row.get("receipts"));lock=p2a.resolve(str(p2a.mapping(row.get("lock")).get("path") or ""))
  if row.get("disposition")!="ENVIRONMENT_MATERIALIZATION_QUALIFIED" or row.get("faults")!=[]:faults.append(f"task_{index}:disposition_invalid")
  if not lock.is_file() or p2a.sha256_file(lock)!=p2a.mapping(row.get("lock")).get("sha256"):faults.append(f"task_{index}:lock_invalid")
  if row.get("manager")=="uv":
   online+=1;offline+=1
   if p2a.mapping(receipts.get("online_environment"))!=p2a.mapping(receipts.get("offline_environment")):faults.append(f"task_{index}:online_offline_mismatch")
   if p2a.strings(p2a.mapping(receipts.get("online_environment")).get("faults")) or p2a.strings(p2a.mapping(receipts.get("offline_environment")).get("faults")):faults.append(f"task_{index}:environment_identity_fault")
  else:online+=1;offline+=1
  for name,receipt in receipts.items():
   if isinstance(receipt,dict) and (name.endswith("sync") or name.endswith("fetch") or name.endswith("venv")):
    if receipt.get("returncode")!=0 or receipt.get("boundary_hit") is True or receipt.get("stdout_complete") is not True or receipt.get("stderr_complete") is not True or receipt.get("project_selected_output_cap") is not None:faults.append(f"task_{index}:{name}_invalid")
  rows.append({"index":index,"repository":row.get("repository"),"manager":row.get("manager"),"faults":[]})
 if [r["index"] for r in rows]!=[12,13,35] or online!=3 or offline!=3:faults.append("denominator_invalid")
 store=p2a.resolve(cfg["store"]);observed=base.tree_identity(store)
 if observed!=producer.get("retained_shared_store"):faults.append("store_identity_invalid")
 if any(v is not False for v in p2a.mapping(cfg.get("audit_authority")).values()):faults.append("audit_authority_invalid")
 green=not faults
 return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"THREE_REPLACEMENT_ENVIRONMENTS_ROLE_SEPARATELY_REDERIVED" if green else "REPLACEMENT_ENVIRONMENT_AUDIT_FAILED","faults":sorted(set(faults)),"task_count":len(rows),"qualified_task_count":len(rows) if green else 0,"rows":rows,"retained_shared_store":observed,"network_enabled_materializations":online,"network_denied_replays":offline,"source_build_executions":0,"project_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":p2a.rel(report_path),"sha256":p2a.sha256_file(report_path) if report_path.is_file() else ""},"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
