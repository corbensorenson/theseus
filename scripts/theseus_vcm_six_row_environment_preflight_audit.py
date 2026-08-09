#!/usr/bin/env python3
"""Role-separated rederivation of the six-row environment fit preflight."""
from __future__ import annotations
import argparse, json, re, shutil, sys, tomllib
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
POLICY="project_theseus_vcm_six_row_environment_preflight_audit_v1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_six_row_environment_preflight.json"
def count(manager:str,path:Path)->int:
 return len(tomllib.loads(path.read_text())["package"]) if manager=="cargo" else len(re.findall(r"^[A-Za-z0-9_.-]+==[^\\\s]+",path.read_text(),re.MULTILINE))
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(path);faults=[];owner=p2a.resolve(str(cfg.get("audit_owner") or ""))
 if cfg.get("audit_policy")!=POLICY or owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("audit_owner_sha256"):faults.append("audit_owner_binding_invalid")
 report=p2a.read_json(p2a.resolve(str(cfg["report"])))
 if report.get("trigger_state")!="GREEN" or report.get("execution_ready") is not True or report.get("panel_admitted") is not False:faults.append("producer_state_invalid")
 builder=p2a.read_json(p2a.resolve(str(p2a.mapping(p2a.mapping(cfg["sources"])["instrument_builder"])["path"])))
 coeffs=p2a.mapping(p2a.mapping(builder["batch_preflight"])["manager_projection"]);rows=[]
 for raw in p2a.dicts(cfg["rows"]):
  manager=str(raw["manager"]);lock=p2a.resolve(str(raw["lock"]));n=count(manager,lock);coef=int(p2a.mapping(coeffs[manager])["observed_upper_bytes_per_entry"]);rows.append({"index":raw["index"],"manager":manager,"package_count":n,"upper_bytes_per_entry":coef,"projected_store_or_environment_bytes":n*coef})
 shared=sum(r["projected_store_or_environment_bytes"] for r in rows);largest=max(r["projected_store_or_environment_bytes"] for r in rows);required=shared+largest+int(p2a.mapping(cfg["limits"])["projected_temporary_bytes"])
 for key,value in (("rows",rows),("projected_shared_store_upper_bytes",shared),("projected_largest_disposable_environment_bytes",largest),("required_incremental_peak_bytes",required)):
  if report.get(key)!=value:faults.append(f"projection_rederivation_invalid:{key}")
 if report.get("network_or_dependency_execution_performed") is not False or any(report.get(k)!=0 for k in ("package_installations","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls")):faults.append("zero_execution_boundary_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"K2_05_SIX_ROW_ENVIRONMENT_PREFLIGHT_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_SIX_ROW_ENVIRONMENT_PREFLIGHT_AUDIT_FAILED","faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":cfg["report"],"sha256":p2a.sha256_file(p2a.resolve(str(cfg["report"])))},"rows":rows,"required_incremental_peak_bytes":required,"current_host_free_bytes":shutil.disk_usage(ROOT).free,"minimum_free_bytes_after_execution":p2a.mapping(cfg["limits"])["minimum_free_bytes_after_execution"],"execution_ready_rederived":shutil.disk_usage(ROOT).free-required>=int(p2a.mapping(cfg["limits"])["minimum_free_bytes_after_execution"]),"audit_kind":"role-separated rederivation","network_or_dependency_execution_performed":False,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("audit_maximum_inference")}
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
if __name__=="__main__":raise SystemExit(main())
