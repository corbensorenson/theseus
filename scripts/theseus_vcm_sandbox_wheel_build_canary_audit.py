#!/usr/bin/env python3
"""Role-separated audit of the retained network-denied canary wheel."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_sandbox_wheel_build_canary as producer  # noqa:E402
POLICY="project_theseus_vcm_sandbox_wheel_build_canary_audit_v1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_sandbox_wheel_build_canary.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG,*,execution:dict[str,Any]|None=None)->dict[str,Any]:
 cfg,bound,faults=producer.preflight(path);report=execution if execution is not None else p2a.read_json(p2a.resolve(cfg["report"]));
 if cfg.get("audit_policy")!=POLICY:faults.append("audit_policy_invalid")
 if report.get("trigger_state")!="GREEN" or report.get("state")!="NETWORK_DENIED_EXACT_SDIST_WHEEL_BUILD_QUALIFIED":faults.append("producer_state_invalid")
 retained=p2a.mapping(p2a.mapping(report.get("receipt")).get("retained_wheel"));wheel=p2a.resolve(str(retained.get("path") or ""));observed,errs=producer.inspect_wheel(wheel) if wheel.is_file() else ({},["retained_wheel_missing"]);faults.extend(errs)
 if not wheel.is_file() or p2a.sha256_file(wheel)!=retained.get("sha256") or wheel.stat().st_size!=retained.get("bytes"):faults.append("retained_wheel_identity_invalid")
 claimed=p2a.mapping(p2a.mapping(report.get("receipt")).get("wheel"));
 if observed!=claimed:faults.append("wheel_inspection_rederivation_mismatch")
 build=p2a.mapping(p2a.mapping(report.get("receipt")).get("sandbox_build"));
 if build.get("returncode")!=0 or "--offline" not in p2a.strings(build.get("command")) or not p2a.strings(build.get("command")) or p2a.strings(build.get("command"))[0]!="/usr/bin/sandbox-exec":faults.append("network_denied_build_receipt_invalid")
 for key in ("evaluator_package_installations","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","teacher_calls"):
  if report.get(key)!=0:faults.append(f"zero_boundary_invalid:{key}")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"NETWORK_DENIED_WHEEL_BUILD_ROLE_SEPARATELY_REDERIVED" if not faults else "NETWORK_DENIED_WHEEL_BUILD_AUDIT_FAILED","faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":cfg["report"],"sha256":p2a.sha256_file(p2a.resolve(cfg["report"])) if execution is None else None},"wheel":observed,"audit_kind":"role-separated rederivation","network_or_build_execution_performed":False,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
