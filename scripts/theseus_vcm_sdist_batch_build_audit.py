#!/usr/bin/env python3
"""Role-separated audit of the manifest-driven network-denied wheel batch."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_sdist_batch_build as producer  # noqa:E402
POLICY="project_theseus_vcm_sdist_batch_build_audit_v1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_build.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","audited_row_count","qualified_row_count","inconclusive_row_count","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG,*,execution:dict[str,Any]|None=None)->dict[str,Any]:
 cfg,_,faults=producer.preflight(path);report=execution if execution is not None else p2a.read_json(p2a.resolve(cfg["report"]));audited=[]
 if cfg.get("audit_policy")!=POLICY:faults.append("audit_policy_invalid")
 if report.get("trigger_state")!="GREEN" or report.get("state")!="TASK13_NETWORK_DENIED_TRANSITIVE_SDIST_BATCH_BUILD_COMPLETE":faults.append("producer_state_invalid")
 for row in p2a.dicts(report.get("rows")):
  receipts=p2a.mapping(row.get("receipts"));build=p2a.mapping(receipts.get("sandbox_build"));retained=p2a.mapping(receipts.get("retained_wheel"));wheel=p2a.resolve(str(retained.get("path") or ""));observed,errs=producer.inspect_wheel(wheel,str(row.get("name") or ""),str(row.get("version") or "")) if wheel.is_file() else ({},["retained_wheel_missing"]);faults.extend(f"{row.get('name')}:{e}" for e in errs)
  if not wheel.is_file() or p2a.sha256_file(wheel)!=retained.get("sha256") or observed!=receipts.get("wheel"):faults.append(f"wheel_rederivation_invalid:{row.get('name')}")
  command=p2a.strings(build.get("command"));valid=build.get("returncode")==0 and command and command[0]=="/usr/bin/sandbox-exec" and "--offline" in command
  if row.get("disposition")=="QUALIFIED_NETWORK_DENIED_SDIST_WHEEL_BUILD" and not valid:faults.append(f"build_receipt_invalid:{row.get('name')}")
  for receipt_name,receipt in receipts.items():
   rec=p2a.mapping(receipt)
   for stream in ("stdout","stderr"):
    payload=str(rec.get(stream) or "").encode();
    if stream in rec and (len(payload)!=rec.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest()!=rec.get(f"{stream}_sha256")):faults.append(f"diagnostic_invalid:{row.get('name')}:{receipt_name}:{stream}")
  audited.append({"name":row.get("name"),"version":row.get("version"),"disposition":row.get("disposition"),"wheel_sha256":retained.get("sha256"),"native_member_count":len(p2a.strings(observed.get("native_members")))})
 qualified=sum(r["disposition"]=="QUALIFIED_NETWORK_DENIED_SDIST_WHEEL_BUILD" for r in audited)
 if len(audited)!=3 or report.get("qualified_row_count")!=qualified:faults.append("denominator_or_count_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"TASK13_NETWORK_DENIED_SDIST_BATCH_BUILD_ROLE_SEPARATELY_REDERIVED" if not faults else "TASK13_NETWORK_DENIED_SDIST_BATCH_BUILD_AUDIT_FAILED","faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":cfg["report"],"sha256":p2a.sha256_file(p2a.resolve(cfg["report"])) if execution is None else None},"audited_row_count":len(audited),"qualified_row_count":qualified,"inconclusive_row_count":len(audited)-qualified,"rows":audited,"audit_kind":"role-separated rederivation","network_or_build_execution_performed":False,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
