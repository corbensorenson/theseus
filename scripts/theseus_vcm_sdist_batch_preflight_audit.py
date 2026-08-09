#!/usr/bin/env python3
"""Role-separated audit for the manifest-driven sdist batch static preflight."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_sdist_batch_preflight as producer  # noqa:E402
POLICY="project_theseus_vcm_sdist_batch_preflight_audit_v1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_preflight.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","audited_row_count","eligible_row_count","inconclusive_row_count","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG,*,execution:dict[str,Any]|None=None)->dict[str,Any]:
 cfg,bound,faults=producer.preflight(path);report=execution if execution is not None else p2a.read_json(p2a.resolve(cfg["report"]));audited=[]
 if cfg.get("audit_policy")!=POLICY:faults.append("audit_policy_invalid")
 if report.get("trigger_state")!="GREEN" or report.get("state")!="TASK13_TRANSITIVE_SDIST_BATCH_STATIC_PREFLIGHT_COMPLETE":faults.append("producer_state_invalid")
 limits=p2a.mapping(cfg["limits"]);allowed=p2a.strings(cfg["allowed_build_backend_prefixes"])
 for row in p2a.dicts(report.get("rows")):
  retained=p2a.mapping(row.get("retained_sdist"));path2=p2a.resolve(str(retained.get("path") or ""));inspection,errs=producer.inspect_archive(path2,limits,allowed) if path2.is_file() else ({},["retained_sdist_missing"]);faults.extend(f"{row.get('name')}:{e}" for e in errs)
  if not path2.is_file() or p2a.sha256_file(path2)!=retained.get("sha256") or inspection!=row.get("inspection"):faults.append(f"row_rederivation_invalid:{row.get('name')}")
  disposition="ELIGIBLE_FOR_NETWORK_DENIED_BATCH_BUILD" if not errs and inspection.get("eligible") is True else "INCONCLUSIVE_IMPLEMENTATION_UNTRUSTED_BUILD_PREFLIGHT"
  if row.get("disposition")!=disposition:faults.append(f"disposition_invalid:{row.get('name')}")
  audited.append({"name":row.get("name"),"version":row.get("version"),"build_backend":inspection.get("build_backend"),"build_requirements":inspection.get("build_requirements"),"disposition":disposition})
 eligible=sum(r["disposition"]=="ELIGIBLE_FOR_NETWORK_DENIED_BATCH_BUILD" for r in audited)
 if len(audited)!=3 or report.get("eligible_row_count")!=eligible:faults.append("denominator_or_count_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"TASK13_TRANSITIVE_SDIST_BATCH_ROLE_SEPARATELY_REDERIVED" if not faults else "TASK13_TRANSITIVE_SDIST_BATCH_AUDIT_FAILED","faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":cfg["report"],"sha256":p2a.sha256_file(p2a.resolve(cfg["report"])) if execution is None else None},"audited_row_count":len(audited),"eligible_row_count":eligible,"inconclusive_row_count":len(audited)-eligible,"rows":audited,"audit_kind":"role-separated rederivation","network_or_build_execution_performed":False,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
